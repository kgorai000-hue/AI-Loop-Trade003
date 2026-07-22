"""LLM research layer — sentiment and reports only, never execution (Lesson 14)."""

from src.llm_research.audit import LLMAuditEntry, LLMAuditStore
from src.llm_research.client import LLMClient, MockLLMClient, OpenAIClient, build_llm_client
from src.llm_research.guard import LLMGuardError, assert_not_execution_path, validate_sentiment_payload
from src.llm_research.news import NewsItem, filter_news_for_symbols, load_news_file
from src.llm_research.report import generate_strategy_diagnostic_report
from src.llm_research.sentiment import aggregate_symbol_features, analyze_news_item, parse_sentiment_response
from src.llm_research.store import SentimentFeatureStore
from src.llm_research.types import LLMResearchReport, SentimentAnalysis, SymbolSentimentFeature

__all__ = [
    "LLMAuditEntry",
    "LLMAuditStore",
    "LLMClient",
    "LLMGuardError",
    "LLMResearchReport",
    "MockLLMClient",
    "NewsItem",
    "OpenAIClient",
    "SentimentAnalysis",
    "SentimentFeatureStore",
    "SymbolSentimentFeature",
    "aggregate_symbol_features",
    "analyze_news_item",
    "assert_not_execution_path",
    "build_llm_client",
    "filter_news_for_symbols",
    "generate_strategy_diagnostic_report",
    "load_news_file",
    "parse_sentiment_response",
    "validate_sentiment_payload",
]
