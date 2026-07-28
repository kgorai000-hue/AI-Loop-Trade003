from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.core.config import PROJECT_ROOT


@dataclass
class BarRecord:
    symbol: str
    timeframe: str
    time: int
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    real_volume: int
    source: str = "mt5"
    ingested_at: int | None = None


class OHLCVStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            self.db_path = PROJECT_ROOT / self.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    tick_volume INTEGER NOT NULL,
                    spread INTEGER NOT NULL,
                    real_volume INTEGER NOT NULL,
                    PRIMARY KEY (symbol, timeframe, time)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_time
                ON ohlcv(symbol, timeframe, time)
                """
            )
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(ohlcv)").fetchall()}
        migrations = {
            "source": "ALTER TABLE ohlcv ADD COLUMN source TEXT NOT NULL DEFAULT 'mt5'",
            "ingested_at": "ALTER TABLE ohlcv ADD COLUMN ingested_at INTEGER",
        }
        for name, sql in migrations.items():
            if name in columns:
                continue
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as exc:
                # Concurrent init / already-migrated race on Windows.
                if "duplicate column" not in str(exc).lower():
                    raise

    def upsert_bars(self, bars: Iterable[BarRecord]) -> int:
        now = int(time.time())
        rows = [
            (
                bar.symbol,
                bar.timeframe,
                bar.time,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.tick_volume,
                bar.spread,
                bar.real_volume,
                bar.source,
                bar.ingested_at if bar.ingested_at is not None else now,
            )
            for bar in bars
        ]
        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO ohlcv (
                    symbol, timeframe, time, open, high, low, close,
                    tick_volume, spread, real_volume, source, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, time) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    tick_volume=excluded.tick_volume,
                    spread=excluded.spread,
                    real_volume=excluded.real_volume,
                    source=excluded.source,
                    ingested_at=excluded.ingested_at
                """,
                rows,
            )
        return len(rows)

    def get_last_bar_time(self, symbol: str, timeframe: str) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(time) AS last_time
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                """,
                (symbol, timeframe),
            ).fetchone()
        if row is None or row["last_time"] is None:
            return None
        return int(row["last_time"])

    def count_bars(self, symbol: str | None = None, timeframe: str | None = None) -> int:
        query = "SELECT COUNT(*) AS cnt FROM ohlcv WHERE 1=1"
        params: list[Any] = []

        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe is not None:
            query += " AND timeframe = ?"
            params.append(timeframe)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["cnt"]) if row else 0

    def get_summary(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    symbol,
                    timeframe,
                    COUNT(*) AS bars,
                    MIN(time) AS first_time,
                    MAX(time) AS last_time,
                    MAX(ingested_at) AS last_ingested_at
                FROM ohlcv
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
                """
            ).fetchall()

        summary: list[dict[str, Any]] = []
        for row in rows:
            summary.append(
                {
                    "symbol": row["symbol"],
                    "timeframe": row["timeframe"],
                    "bars": row["bars"],
                    "first_time": datetime.fromtimestamp(row["first_time"], tz=timezone.utc),
                    "last_time": datetime.fromtimestamp(row["last_time"], tz=timezone.utc),
                    "last_ingested_at": (
                        datetime.fromtimestamp(row["last_ingested_at"], tz=timezone.utc)
                        if row["last_ingested_at"]
                        else None
                    ),
                }
            )
        return summary

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT time, open, high, low, close, tick_volume, spread, real_volume,
                       source, ingested_at
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
                """,
                (symbol, timeframe, count),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_all_bars(self, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT time, open, high, low, close, tick_volume, spread, real_volume,
                       source, ingested_at
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time ASC
                """,
                (symbol, timeframe),
            ).fetchall()
        return [dict(row) for row in rows]
