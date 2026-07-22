from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.core.config import PROJECT_ROOT


@dataclass
class SpreadSnapshot:
    symbol: str
    bid: float
    ask: float
    spread_points: int
    spread_price: float
    timestamp: int


class SpreadStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            self.db_path = PROJECT_ROOT / self.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spread_snapshots (
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    spread_points INTEGER NOT NULL,
                    spread_price REAL NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
                """
            )

    def save_snapshots(self, snapshots: Iterable[SpreadSnapshot]) -> int:
        rows = [
            (s.symbol, s.timestamp, s.bid, s.ask, s.spread_points, s.spread_price)
            for s in snapshots
        ]
        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO spread_snapshots
                (symbol, timestamp, bid, ask, spread_points, spread_price)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def latest_by_symbol(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                """
                SELECT s.*
                FROM spread_snapshots s
                INNER JOIN (
                    SELECT symbol, MAX(timestamp) AS max_ts
                    FROM spread_snapshots
                    GROUP BY symbol
                ) latest
                ON s.symbol = latest.symbol AND s.timestamp = latest.max_ts
                ORDER BY s.symbol
                """
            ).fetchall()


def snapshot_from_market_info(info, timestamp: int | None = None) -> SpreadSnapshot:
    return SpreadSnapshot(
        symbol=info.symbol,
        bid=info.bid,
        ask=info.ask,
        spread_points=info.spread_points,
        spread_price=info.spread_price,
        timestamp=timestamp or int(time.time()),
    )
