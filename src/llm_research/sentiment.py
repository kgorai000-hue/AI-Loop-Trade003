from __future__ import annotations

import json
import logging
from collections import defaultdict

from src.llm_research.client import LLMClient
from src.llm_research.guard import LLMGuardError, assert_not_execution_path, validate_sentiment_payload
from src.llm_research.prompts import SENTIMENT_SYSTEM, build_sentiment_prompt
from src.llm_research.types import SentimentAnalysis, SentimentKeyPoint, SymbolSentimentFeature

logger = logging.getLogger(__name__)


def parse_sentiment_response(raw: str, *, headline: str = "", news_id: str = "") -> SentimentAnalysis:
    data = json.loads(raw)
    data = validate_sentiment_payload(data)
    assert_not_execution_path("feature_input_only")

    key_points = [
        SentimentKeyPoint(
            topic=str(point.get("topic", "unknown")),
            sentiment=float(point.get("sentiment", 0.0)),
            value=str(point.get("value", "")),
        )
        for point in data.get("key_points", [])
    ]

    return SentimentAnalysis(
        symbol=str(data["symbol"]),
        event_type=str(data["event_type"]),
        sentiment_overall=float(data["sentiment_overall"]),
        key_points=key_points,
        trading_signal=str(data.get("trading_signal", "neutral")),
        confidence=float(data["confidence"]),
        source_headline=headline,
        news_id=news_id,
    )


def analyze_news_item(
    client: LLMClient,
    headline: str,
    body: str,
    *,
    news_id: str = "",
) -> SentimentAnalysis:
    prompt = build_sentiment_prompt(headline, body)
    raw = client.complete(SENTIMENT_SYSTEM, prompt)
    return parse_sentiment_response(raw, headline=headline, news_id=news_id)


def aggregate_symbol_features(analyses: list[SentimentAnalysis]) -> list[SymbolSentimentFeature]:
    grouped: dict[str, list[SentimentAnalysis]] = defaultdict(list)
    for analysis in analyses:
        grouped[analysis.symbol].append(analysis)

    features: list[SymbolSentimentFeature] = []
    for symbol, items in grouped.items():
        if symbol == "UNKNOWN":
            continue
        weights = [max(a.confidence, 0.1) for a in items]
        total_w = sum(weights)
        score = sum(a.sentiment_overall * w for a, w in zip(items, weights)) / total_w
        confidence = sum(weights) / len(weights)
        event_types = [a.event_type for a in items]
        dominant = max(set(event_types), key=event_types.count)
        features.append(
            SymbolSentimentFeature(
                symbol=symbol,
                sentiment_score=round(score, 4),
                event_count=len(items),
                confidence=round(confidence, 4),
                dominant_event_type=dominant,
            )
        )
    return features
