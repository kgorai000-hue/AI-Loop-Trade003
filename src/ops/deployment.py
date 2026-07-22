from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.core.config import PROJECT_ROOT


@dataclass
class ModelRecord:
    model_id: str
    version: str
    created_at: int
    created_by: str
    metrics: dict[str, float]
    status: str
    artifact_path: str
    config_hash: str = ""
    data_hash: str = ""


class ModelRegistry:
    """Model version registry (Lesson 20.7.4)."""

    def __init__(self, db_path: str | Path = "data/model_registry.sqlite") -> None:
        raw = str(db_path)
        self._memory_mode = raw == ":memory:"
        if self._memory_mode:
            self.db_path = raw
            self._conn = sqlite3.connect(raw, check_same_thread=False)
        else:
            path = Path(db_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)
            self._conn = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    created_by TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    config_hash TEXT NOT NULL DEFAULT '',
                    data_hash TEXT NOT NULL DEFAULT ''
                )
                """
        )
        conn.commit()

    def register(self, record: ModelRecord) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.model_id,
                record.version,
                record.created_at,
                record.created_by,
                json.dumps(record.metrics),
                record.status,
                record.artifact_path,
                record.config_hash,
                record.data_hash,
            ),
        )
        conn.commit()

    def activate(self, model_id: str) -> None:
        conn = self._connect()
        conn.execute("UPDATE models SET status='retired' WHERE status='production'")
        conn.execute(
            "UPDATE models SET status='production' WHERE model_id=?",
            (model_id,),
        )
        conn.commit()

    def get_production(self) -> ModelRecord | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM models WHERE status='production' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return ModelRecord(
            model_id=row[0],
            version=row[1],
            created_at=row[2],
            created_by=row[3],
            metrics=json.loads(row[4]),
            status=row[5],
            artifact_path=row[6],
            config_hash=row[7],
            data_hash=row[8],
        )


class CanaryDeployment:
    """Canary deployment controller (Lesson 20.7.3)."""

    PROMOTION_STEPS = (0.05, 0.25, 0.50, 1.0)

    def __init__(
        self,
        stable_predict: Callable[[dict[str, Any]], dict[str, Any]],
        canary_predict: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        initial_weight: float = 0.05,
    ) -> None:
        self.stable_predict = stable_predict
        self.canary_predict = canary_predict
        self.canary_weight = initial_weight

    def get_signal(self, market_data: dict[str, Any]) -> dict[str, Any]:
        if random.random() < self.canary_weight:
            signal = self.canary_predict(market_data)
            signal["model_version"] = "canary"
        else:
            signal = self.stable_predict(market_data)
            signal["model_version"] = "stable"
        return signal

    def promote_canary(self, new_weight: float) -> float:
        if new_weight > self.canary_weight:
            self.canary_weight = min(new_weight, 1.0)
        return self.canary_weight

    def next_promotion_step(self) -> float | None:
        for step in self.PROMOTION_STEPS:
            if step > self.canary_weight:
                return step
        return None

    def rollback(self) -> None:
        self.canary_weight = 0.0


class AutoRollback:
    """Automatic rollback on degradation (Lesson 20.7.5)."""

    def __init__(self, thresholds: dict[str, float], registry: ModelRegistry) -> None:
        self.thresholds = thresholds
        self.registry = registry
        self.metrics_buffer: list[dict[str, float]] = []
        self.rollback_triggered = False
        self.last_reason = ""

    def check_and_rollback(self, current_metrics: dict[str, float]) -> bool:
        if current_metrics.get("error_rate", 0.0) > self.thresholds.get("max_error_rate", 0.05):
            self.trigger_rollback("error rate too high")
            return True

        if current_metrics.get("latency_p99", 0.0) > self.thresholds.get("max_latency", 5000.0):
            self.trigger_rollback("latency too high")
            return True

        self.metrics_buffer.append(current_metrics)
        if len(self.metrics_buffer) >= 10:
            recent_sharpe = sum(m.get("sharpe", 0.0) for m in self.metrics_buffer[-10:]) / 10.0
            if recent_sharpe < self.thresholds.get("min_sharpe", 0.5):
                self.trigger_rollback(f"Sharpe degraded: {recent_sharpe:.2f}")
                return True

        return False

    def trigger_rollback(self, reason: str) -> None:
        self.rollback_triggered = True
        self.last_reason = reason
        prod = self.registry.get_production()
        if prod:
            stable_id = prod.model_id.replace("canary", "stable")
            self.registry.activate(stable_id)
