from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RegimeHistoryEntry:
    symbol: str
    timeframe: str
    detected_label: str
    confirmed_label: str
    switched: bool
    reason: str
    timestamp: int
    max_probability: float = 0.0
    adx: float = 0.0
    degradation_level: int = 0


class RegimeHistoryStore:
    """Persist regime decisions, confirmation, attribution (Lesson 12-13)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    detected_label TEXT NOT NULL,
                    confirmed_label TEXT NOT NULL,
                    switched INTEGER NOT NULL,
                    reason TEXT,
                    timestamp INTEGER NOT NULL,
                    max_probability REAL DEFAULT 0,
                    adx REAL DEFAULT 0,
                    degradation_level INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_regime_history_symbol_ts ON regime_history(symbol, timestamp)"
            )
            self._migrate_columns(conn)

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(regime_history)")}
        migrations = {
            "max_probability": "ALTER TABLE regime_history ADD COLUMN max_probability REAL DEFAULT 0",
            "adx": "ALTER TABLE regime_history ADD COLUMN adx REAL DEFAULT 0",
            "degradation_level": "ALTER TABLE regime_history ADD COLUMN degradation_level INTEGER DEFAULT 0",
        }
        for name, sql in migrations.items():
            if name not in cols:
                conn.execute(sql)

    def last_confirmed_label(self, symbol: str, timeframe: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT confirmed_label FROM regime_history
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (symbol, timeframe),
            ).fetchone()
        return row[0] if row else None

    def recent_raw_labels(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT detected_label FROM regime_history
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (symbol, timeframe, limit),
            ).fetchall()
        return [row[0] for row in reversed(rows)]

    def recent_adx_values(self, symbol: str, timeframe: str, limit: int) -> list[float]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT adx FROM regime_history
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (symbol, timeframe, limit),
            ).fetchall()
        return [float(row[0]) for row in reversed(rows) if row[0] is not None]

    def days_since_last_switch(self, symbol: str, timeframe: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT timestamp FROM regime_history
                WHERE symbol = ? AND timeframe = ? AND switched = 1
                ORDER BY timestamp DESC LIMIT 1
                """,
                (symbol, timeframe),
            ).fetchone()
        if not row:
            return 999
        return max(0, int((time.time() - int(row[0])) / 86400))

    def detect_adx_oscillation(
        self,
        symbol: str,
        timeframe: str,
        *,
        threshold: float,
        band: float,
        lookback: int,
    ) -> bool:
        values = self.recent_adx_values(symbol, timeframe, lookback)
        if len(values) < 3:
            return False
        crossings = 0
        for val in values:
            if abs(val - threshold) <= band:
                crossings += 1
        alternations = 0
        for prev, curr in zip(values, values[1:]):
            if (prev - threshold) * (curr - threshold) < 0:
                alternations += 1
        return alternations >= 2 or crossings >= len(values) - 1

    def apply_confirmation(
        self,
        symbol: str,
        timeframe: str,
        detected_label: str,
        confirm_days: int,
    ) -> tuple[str, bool]:
        """Return confirmed label and whether switch is still pending."""
        last = self.last_confirmed_label(symbol, timeframe)
        if confirm_days <= 1:
            return detected_label, False

        recent = self.recent_raw_labels(symbol, timeframe, confirm_days - 1)
        pending = last is not None and detected_label != last

        if len(recent) < confirm_days - 1:
            if pending:
                return last or detected_label, True
            return last or detected_label, False

        if all(label == detected_label for label in recent):
            return detected_label, False

        if pending:
            return "uncertain", True
        return last or detected_label, False

    def record(
        self,
        symbol: str,
        timeframe: str,
        detected_label: str,
        confirmed_label: str,
        reason: str,
        *,
        max_probability: float = 0.0,
        adx: float = 0.0,
        degradation_level: int = 0,
    ) -> RegimeHistoryEntry:
        switched = False
        last = self.last_confirmed_label(symbol, timeframe)
        if last is not None and last != confirmed_label and confirmed_label != "uncertain":
            switched = True

        ts = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO regime_history(
                    symbol, timeframe, detected_label, confirmed_label, switched, reason,
                    timestamp, max_probability, adx, degradation_level
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    timeframe,
                    detected_label,
                    confirmed_label,
                    int(switched),
                    reason,
                    ts,
                    max_probability,
                    adx,
                    degradation_level,
                ),
            )

        return RegimeHistoryEntry(
            symbol=symbol,
            timeframe=timeframe,
            detected_label=detected_label,
            confirmed_label=confirmed_label,
            switched=switched,
            reason=reason,
            timestamp=ts,
            max_probability=max_probability,
            adx=adx,
            degradation_level=degradation_level,
        )

    def switch_stats(self, symbol: str | None = None, days: int = 365) -> dict[str, float]:
        cutoff = int(time.time()) - days * 86400
        query = """
            SELECT COUNT(*), SUM(switched), MIN(timestamp), MAX(timestamp)
            FROM regime_history WHERE timestamp >= ?
        """
        params: list[object] = [cutoff]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(query, params).fetchone()

        count = int(row[0] or 0)
        switches = int(row[1] or 0)
        if count == 0:
            return {"observations": 0, "switches": 0, "switch_rate": 0.0, "avg_duration_days": 0.0}

        span_days = max(1, (int(row[3]) - int(row[2])) / 86400)
        avg_duration = span_days / max(switches, 1)
        return {
            "observations": float(count),
            "switches": float(switches),
            "switch_rate": switches / count,
            "avg_duration_days": avg_duration,
            "span_days": span_days,
        }
