from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.llm_research.types import SymbolSentimentFeature


class SentimentFeatureStore:
    """Persist LLM sentiment features for optional downstream use."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    event_count INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    dominant_event_type TEXT,
                    timestamp INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_ts ON sentiment_features(symbol, timestamp)"
            )

    def upsert_batch(self, features: list[SymbolSentimentFeature]) -> int:
        ts = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            for feature in features:
                conn.execute(
                    """
                    INSERT INTO sentiment_features(
                        symbol, sentiment_score, event_count, confidence,
                        dominant_event_type, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feature.symbol,
                        feature.sentiment_score,
                        feature.event_count,
                        feature.confidence,
                        feature.dominant_event_type,
                        ts,
                    ),
                )
        return len(features)

    def latest_for_symbols(self, symbols: list[str]) -> dict[str, SymbolSentimentFeature]:
        result: dict[str, SymbolSentimentFeature] = {}
        with sqlite3.connect(self.db_path) as conn:
            for symbol in symbols:
                row = conn.execute(
                    """
                    SELECT symbol, sentiment_score, event_count, confidence, dominant_event_type
                    FROM sentiment_features WHERE symbol = ?
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (symbol,),
                ).fetchone()
                if row:
                    result[symbol] = SymbolSentimentFeature(
                        symbol=row[0],
                        sentiment_score=row[1],
                        event_count=row[2],
                        confidence=row[3],
                        dominant_event_type=row[4] or "other",
                    )
        return result
