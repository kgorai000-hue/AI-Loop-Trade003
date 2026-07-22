from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMAuditEntry:
    id: int
    timestamp: int
    provider: str
    model: str
    temperature: float
    input_prompt: str
    output_raw: str
    output_parsed: str
    action_taken: str
    final_decision: str
    success: bool
    error: str | None = None


class LLMAuditStore:
    """Audit trail for all LLM calls (Lesson 14.5)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    input_prompt TEXT NOT NULL,
                    output_raw TEXT,
                    output_parsed TEXT,
                    action_taken TEXT NOT NULL,
                    final_decision TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT
                )
                """
            )

    def record(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        input_prompt: str,
        output_raw: str,
        output_parsed: str,
        action_taken: str,
        final_decision: str,
        success: bool,
        error: str | None = None,
    ) -> int:
        ts = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO llm_audit(
                    timestamp, provider, model, temperature, input_prompt,
                    output_raw, output_parsed, action_taken, final_decision, success, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    provider,
                    model,
                    temperature,
                    input_prompt,
                    output_raw,
                    output_parsed,
                    action_taken,
                    final_decision,
                    int(success),
                    error,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[LLMAuditEntry]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, provider, model, temperature, input_prompt,
                       output_raw, output_parsed, action_taken, final_decision, success, error
                FROM llm_audit ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            LLMAuditEntry(
                id=row[0],
                timestamp=row[1],
                provider=row[2],
                model=row[3],
                temperature=row[4],
                input_prompt=row[5],
                output_raw=row[6] or "",
                output_parsed=row[7] or "",
                action_taken=row[8],
                final_decision=row[9],
                success=bool(row[10]),
                error=row[11],
            )
            for row in rows
        ]
