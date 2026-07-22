from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from src.core.config import PROJECT_ROOT
from src.execution.execution_types import ExecutionLogRecord, ExecutionPipelineReport


class ExecutionTelemetryStore:
    """Type-B execution data closed loop (Lesson 19.6.2, 19.8 Stage 2)."""

    def __init__(self, db_path: str | Path) -> None:
        raw = str(db_path)
        self._memory_mode = raw == ":memory:"
        if self._memory_mode:
            self.db_path = raw
            self._conn = sqlite3.connect(raw, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_log (
                record_id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                canonical_symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                expected_price REAL NOT NULL,
                average_fill_price REAL NOT NULL,
                requested_lots REAL NOT NULL,
                filled_lots REAL NOT NULL,
                slippage_pct REAL NOT NULL,
                latency_ms REAL NOT NULL,
                fill_ratio REAL NOT NULL,
                commission_jpy REAL NOT NULL,
                dry_run INTEGER NOT NULL,
                status TEXT NOT NULL,
                child_orders INTEGER NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_execution_log_ts
            ON execution_log(timestamp)
            """
        )
        conn.commit()

    def record(self, entry: ExecutionLogRecord) -> str:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO execution_log (
                record_id, timestamp, symbol, canonical_symbol, side, order_type,
                expected_price, average_fill_price, requested_lots, filled_lots,
                slippage_pct, latency_ms, fill_ratio, commission_jpy, dry_run,
                status, child_orders, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.record_id,
                entry.timestamp,
                entry.symbol,
                entry.canonical_symbol,
                entry.side,
                entry.order_type,
                entry.expected_price,
                entry.average_fill_price,
                entry.requested_lots,
                entry.filled_lots,
                entry.slippage_pct,
                entry.latency_ms,
                entry.fill_ratio,
                entry.commission_jpy,
                1 if entry.dry_run else 0,
                entry.status,
                entry.child_orders,
                entry.reason,
            ),
        )
        conn.commit()
        return entry.record_id

    def recent(self, limit: int = 50) -> list[ExecutionLogRecord]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM execution_log
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            ExecutionLogRecord(
                record_id=row["record_id"],
                timestamp=row["timestamp"],
                symbol=row["symbol"],
                canonical_symbol=row["canonical_symbol"],
                side=row["side"],
                order_type=row["order_type"],
                expected_price=row["expected_price"],
                average_fill_price=row["average_fill_price"],
                requested_lots=row["requested_lots"],
                filled_lots=row["filled_lots"],
                slippage_pct=row["slippage_pct"],
                latency_ms=row["latency_ms"],
                fill_ratio=row["fill_ratio"],
                commission_jpy=row["commission_jpy"],
                dry_run=bool(row["dry_run"]),
                status=row["status"],
                child_orders=row["child_orders"],
                reason=row["reason"],
            )
            for row in rows
        ]

    def summarize(self, limit: int = 100) -> ExecutionPipelineReport:
        records = self.recent(limit)
        if not records:
            return ExecutionPipelineReport()

        avg_slip = sum(r.slippage_pct for r in records) / len(records)
        avg_fill = sum(r.fill_ratio for r in records) / len(records)
        avg_latency = sum(r.latency_ms for r in records) / len(records)
        partial = sum(1 for r in records if r.status == "partial")

        warnings: list[str] = []
        if avg_slip > 0.05:
            warnings.append(f"avg slippage {avg_slip:.3f}% exceeds 0.05%")
        if avg_fill < 0.95:
            warnings.append(f"avg fill ratio {avg_fill:.0%} below 95%")

        return ExecutionPipelineReport(
            records=records,
            avg_slippage_pct=avg_slip,
            avg_fill_ratio=avg_fill,
            avg_latency_ms=avg_latency,
            partial_fill_count=partial,
            warnings=warnings,
        )


def new_record_id() -> str:
    return str(uuid.uuid4())[:12]
