from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SentimentKeyPoint:
    topic: str
    sentiment: float
    value: str


@dataclass
class SentimentAnalysis:
    symbol: str
    event_type: str
    sentiment_overall: float
    key_points: list[SentimentKeyPoint]
    trading_signal: str
    confidence: float
    source_headline: str = ""
    news_id: str = ""


@dataclass
class SymbolSentimentFeature:
    symbol: str
    sentiment_score: float
    event_count: int
    confidence: float
    dominant_event_type: str = "other"


@dataclass
class LLMResearchReport:
    sentiment_features: list[SymbolSentimentFeature] = field(default_factory=list)
    analyses: list[SentimentAnalysis] = field(default_factory=list)
    filtered_news_count: int = 0
    analyzed_count: int = 0
    skipped_count: int = 0
    audit_ids: list[int] = field(default_factory=list)
    provider: str = "mock"
    strategy_report: str | None = None
