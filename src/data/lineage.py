from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT


@dataclass
class SyncRunRecord:
    id: int | None = None
    started_at: int = 0
    finished_at: int | None = None
    source: str = "mt5"
    broker_server: str | None = None
    symbols_count: int = 0
    timeframes: list[str] = field(default_factory=list)
    fetched_total: int = 0
    stored_total: int = 0
    rejected_total: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "running"


class SyncRunStore:
    """Lineage tracking for data ingest runs (Lesson 6.7 traceability)."""

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
                CREATE TABLE IF NOT EXISTS sync_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at INTEGER NOT NULL,
                    finished_at INTEGER,
                    source TEXT NOT NULL DEFAULT 'mt5',
                    broker_server TEXT,
                    symbols_count INTEGER,
                    timeframes TEXT,
                    fetched_total INTEGER DEFAULT 0,
                    stored_total INTEGER DEFAULT 0,
                    rejected_total INTEGER DEFAULT 0,
                    errors TEXT,
                    status TEXT NOT NULL DEFAULT 'running'
                )
                """
            )

    def start_run(
        self,
        source: str,
        broker_server: str | None,
        symbols_count: int,
        timeframes: list[str],
    ) -> int:
        now = int(time.time())
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sync_runs (
                    started_at, source, broker_server, symbols_count, timeframes, status
                ) VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (now, source, broker_server, symbols_count, json.dumps(timeframes)),
            )
            return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        fetched_total: int,
        stored_total: int,
        rejected_total: int,
        errors: list[str],
        status: str = "completed",
    ) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sync_runs SET
                    finished_at = ?,
                    fetched_total = ?,
                    stored_total = ?,
                    rejected_total = ?,
                    errors = ?,
                    status = ?
                WHERE id = ?
                """,
                (
                    now,
                    fetched_total,
                    stored_total,
                    rejected_total,
                    json.dumps(errors) if errors else None,
                    status,
                    run_id,
                ),
            )

    def get_recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sync_runs ORDER BY started_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
