from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RiskAuditEntry:
    id: int
    timestamp: int
    symbol: str
    requested_lots: float
    approved_lots: float
    decision: str
    reason: str
    drawdown_pct: float
    drawdown_level: str
    total_exposure_pct: float


class RiskAuditStore:
    """Immutable-style audit trail for Risk Agent decisions (Lesson 15.5)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    requested_lots REAL NOT NULL,
                    approved_lots REAL NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    drawdown_level TEXT NOT NULL,
                    total_exposure_pct REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    peak_equity REAL NOT NULL,
                    circuit_breaker_active INTEGER NOT NULL DEFAULT 0,
                    circuit_breaker_until INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                )
                """
            )

    def record(
        self,
        *,
        symbol: str,
        requested_lots: float,
        approved_lots: float,
        decision: str,
        reason: str,
        drawdown_pct: float,
        drawdown_level: str,
        total_exposure_pct: float,
    ) -> int:
        ts = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO risk_audit(
                    timestamp, symbol, requested_lots, approved_lots, decision,
                    reason, drawdown_pct, drawdown_level, total_exposure_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    symbol,
                    requested_lots,
                    approved_lots,
                    decision,
                    reason,
                    drawdown_pct,
                    drawdown_level,
                    total_exposure_pct,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[RiskAuditEntry]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, symbol, requested_lots, approved_lots,
                       decision, reason, drawdown_pct, drawdown_level, total_exposure_pct
                FROM risk_audit ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            RiskAuditEntry(
                id=row[0],
                timestamp=row[1],
                symbol=row[2],
                requested_lots=row[3],
                approved_lots=row[4],
                decision=row[5],
                reason=row[6],
                drawdown_pct=row[7],
                drawdown_level=row[8],
                total_exposure_pct=row[9],
            )
            for row in rows
        ]

    def load_state(self) -> tuple[float, bool, int]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT peak_equity, circuit_breaker_active, circuit_breaker_until FROM risk_state WHERE id = 1"
            ).fetchone()
        if not row:
            return 0.0, False, 0
        return float(row[0]), bool(row[1]), int(row[2])

    def save_state(
        self,
        *,
        peak_equity: float,
        circuit_breaker_active: bool,
        circuit_breaker_until: int,
    ) -> None:
        ts = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO risk_state(id, peak_equity, circuit_breaker_active, circuit_breaker_until, updated_at)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    peak_equity = excluded.peak_equity,
                    circuit_breaker_active = excluded.circuit_breaker_active,
                    circuit_breaker_until = excluded.circuit_breaker_until,
                    updated_at = excluded.updated_at
                """,
                (peak_equity, int(circuit_breaker_active), circuit_breaker_until, ts),
            )
