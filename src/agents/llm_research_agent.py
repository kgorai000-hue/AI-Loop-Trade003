from __future__ import annotations

import json
import logging
from pathlib import Path

from src.core.config import AppConfig
from src.core.types import SymbolStatsReport
from src.llm_research.audit import LLMAuditStore
from src.llm_research.client import build_llm_client
from src.llm_research.guard import LLMGuardError, assert_not_execution_path
from src.llm_research.news import filter_news_for_symbols, load_news_file
from src.llm_research.prompts import SENTIMENT_SYSTEM, build_sentiment_prompt
from src.llm_research.report import generate_strategy_diagnostic_report
from src.llm_research.sentiment import aggregate_symbol_features, parse_sentiment_response
from src.llm_research.store import SentimentFeatureStore
from src.llm_research.types import LLMResearchReport, SentimentAnalysis

logger = logging.getLogger(__name__)


class LLMResearchAgent:
    """Research assistant only — features and reports, never direct trading (Lesson 14)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg = config.llm_research
        self.client = build_llm_client(
            self.cfg.provider,
            api_key=self.cfg.openai_api_key,
            model=self.cfg.model,
            temperature=self.cfg.temperature,
        )
        self.audit = LLMAuditStore(config.storage.path)
        self.feature_store = SentimentFeatureStore(config.storage.path)

    def analyze(
        self,
        symbols: list[str] | None = None,
        research_reports: list[SymbolStatsReport] | None = None,
    ) -> LLMResearchReport:
        if not self.cfg.enabled:
            return LLMResearchReport(provider=self.cfg.provider)

        symbols = symbols or self.config.symbols
        research_reports = research_reports or []
        assert_not_execution_path("feature_input_only")

        news_path = Path(self.cfg.news_path) / self.cfg.news_file
        all_news = load_news_file(news_path)
        filtered = filter_news_for_symbols(
            all_news,
            symbols,
            keyword_filter=self.cfg.keyword_filter,
            max_items=self.cfg.max_news_per_run,
        )

        analyses: list[SentimentAnalysis] = []
        audit_ids: list[int] = []
        skipped = 0

        for item in filtered:
            prompt = build_sentiment_prompt(item.headline, item.body)
            try:
                raw = self.client.complete(SENTIMENT_SYSTEM, prompt)
                analysis = parse_sentiment_response(raw, headline=item.headline, news_id=item.id)
                analyses.append(analysis)
                if self.cfg.audit_enabled:
                    audit_ids.append(
                        self.audit.record(
                            provider=self.cfg.provider,
                            model=self.cfg.model,
                            temperature=self.cfg.temperature,
                            input_prompt=prompt[:2000],
                            output_raw=raw[:4000],
                            output_parsed=json.dumps(
                                {
                                    "symbol": analysis.symbol,
                                    "sentiment": analysis.sentiment_overall,
                                    "confidence": analysis.confidence,
                                }
                            ),
                            action_taken="feature_input_only",
                            final_decision="no trade executed",
                            success=True,
                        )
                    )
            except (LLMGuardError, ValueError, json.JSONDecodeError) as exc:
                skipped += 1
                logger.warning("LLMResearchAgent skip news %s: %s", item.id, exc)
                if self.cfg.audit_enabled:
                    audit_ids.append(
                        self.audit.record(
                            provider=self.cfg.provider,
                            model=self.cfg.model,
                            temperature=self.cfg.temperature,
                            input_prompt=prompt[:2000],
                            output_raw="",
                            output_parsed="",
                            action_taken="feature_input_only",
                            final_decision="rejected by guard",
                            success=False,
                            error=str(exc),
                        )
                    )

        features = aggregate_symbol_features(analyses)
        if features:
            self.feature_store.upsert_batch(features)

        strategy_report = None
        if self.cfg.generate_strategy_report and research_reports:
            strategy_report = generate_strategy_diagnostic_report(
                strategy_name="MT5_loop Pipeline",
                period="latest run",
                research_reports=research_reports,
            )

        return LLMResearchReport(
            sentiment_features=features,
            analyses=analyses,
            filtered_news_count=len(filtered),
            analyzed_count=len(analyses),
            skipped_count=skipped,
            audit_ids=audit_ids,
            provider=self.cfg.provider,
            strategy_report=strategy_report,
        )

    def latest_features(self, symbols: list[str]) -> dict[str, float]:
        stored = self.feature_store.latest_for_symbols(symbols)
        return {sym: feat.sentiment_score for sym, feat in stored.items()}
