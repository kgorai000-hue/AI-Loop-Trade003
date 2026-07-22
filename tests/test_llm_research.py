from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.agents.llm_research_agent import LLMResearchAgent
from src.core.config import LLMResearchConfig, load_config
from src.llm_research.client import MockLLMClient
from src.llm_research.guard import LLMGuardError, assert_not_execution_path, validate_sentiment_payload
from src.llm_research.news import filter_news_for_symbols, load_news_file
from src.llm_research.report import generate_strategy_diagnostic_report
from src.llm_research.sentiment import aggregate_symbol_features, parse_sentiment_response
from src.llm_research.store import SentimentFeatureStore
from src.llm_research.types import SentimentAnalysis


class LLMResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_files: list[Path] = []

    def tearDown(self) -> None:
        import gc

        gc.collect()
        for path in self._tmp_files:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def test_validate_sentiment_rejects_out_of_range(self) -> None:
        with self.assertRaises(LLMGuardError):
            validate_sentiment_payload(
                {
                    "symbol": "EURUSD",
                    "event_type": "macro",
                    "sentiment_overall": 1.5,
                    "confidence": 0.8,
                }
            )

    def test_validate_sentiment_rejects_execute_order(self) -> None:
        with self.assertRaises(LLMGuardError):
            validate_sentiment_payload(
                {
                    "symbol": "GOLD",
                    "event_type": "macro",
                    "sentiment_overall": 0.2,
                    "confidence": 0.7,
                    "notes": "please execute buy now",
                }
            )

    def test_assert_not_execution_path_blocks_execute(self) -> None:
        with self.assertRaises(LLMGuardError):
            assert_not_execution_path("execute market order")

    def test_mock_client_returns_valid_json(self) -> None:
        client = MockLLMClient()
        raw = client.complete("system", "Apple beats revenue expectations")
        data = json.loads(raw)
        analysis = parse_sentiment_response(raw, headline="Apple beats revenue")
        self.assertIn("symbol", data)
        self.assertGreaterEqual(analysis.confidence, 0.0)

    def test_filter_news_for_symbols(self) -> None:
        news_path = Path(__file__).resolve().parents[1] / "data" / "news" / "sample_news.json"
        items = load_news_file(news_path)
        matched = filter_news_for_symbols(items, ["EURUSD", "GOLD"])
        symbols = {sym for item in matched for sym in item.symbols}
        self.assertTrue({"EURUSD", "GOLD"} & symbols)

    def test_aggregate_symbol_features(self) -> None:
        analyses = [
            SentimentAnalysis(
                symbol="EURUSD",
                event_type="policy",
                sentiment_overall=0.3,
                key_points=[],
                trading_signal="bullish",
                confidence=0.7,
            ),
            SentimentAnalysis(
                symbol="EURUSD",
                event_type="macro",
                sentiment_overall=-0.1,
                key_points=[],
                trading_signal="mixed",
                confidence=0.6,
            ),
        ]
        features = aggregate_symbol_features(analyses)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].symbol, "EURUSD")
        self.assertEqual(features[0].event_count, 2)

    def test_sentiment_feature_store_roundtrip(self) -> None:
        db_path = Path(__file__).resolve().parent / "_tmp_sentiment_test.db"
        self._tmp_files.append(db_path)
        if db_path.exists():
            db_path.unlink()
        store = SentimentFeatureStore(db_path)
        from src.llm_research.types import SymbolSentimentFeature

        store.upsert_batch(
            [
                SymbolSentimentFeature(
                    symbol="GOLD",
                    sentiment_score=-0.4,
                    event_count=1,
                    confidence=0.65,
                    dominant_event_type="macro",
                )
            ]
        )
        latest = store.latest_for_symbols(["GOLD"])
        self.assertIn("GOLD", latest)
        self.assertAlmostEqual(latest["GOLD"].sentiment_score, -0.4)
        del store

    def test_strategy_diagnostic_report(self) -> None:
        from src.core.types import MarketRegime, SymbolStatsReport

        reports = [
            SymbolStatsReport(
                symbol="GOLD",
                timeframe="D1",
                bars=100,
                cumulative_log_return=0.05,
                annualized_return=0.12,
                daily_volatility=0.01,
                annualized_volatility=0.35,
                autocorr_lag1=0.0,
                autocorr_lag5=0.0,
                skewness=0.0,
                excess_kurtosis=0.0,
                price_stationary=False,
                return_stationary=True,
                vol_autocorr=0.0,
                regime=MarketRegime.BULL,
                tail_warning="",
                rsi=55.0,
            )
        ]
        text = generate_strategy_diagnostic_report("Test", "2024", reports)
        self.assertIn("GOLD", text)
        self.assertIn("Risk Flags", text)

    def test_llm_research_agent_mock_run(self) -> None:
        db_path = Path(__file__).resolve().parent / "_tmp_llm_agent_test.db"
        self._tmp_files.append(db_path)
        if db_path.exists():
            db_path.unlink()
        config = load_config()
        config.llm_research = LLMResearchConfig(
            enabled=True,
            provider="mock",
            model="mock",
            temperature=0.0,
            openai_api_key=None,
            news_path=str(Path(__file__).resolve().parents[1] / "data" / "news"),
            news_file="sample_news.json",
            keyword_filter=True,
            max_news_per_run=10,
            audit_enabled=True,
            generate_strategy_report=False,
            use_as_feature=False,
            sentiment_nudge_strength=0.05,
            min_confidence=0.5,
        )
        config.storage.path = str(db_path)
        agent = LLMResearchAgent(config)
        report = agent.analyze(symbols=["EURUSD", "GOLD", "#USSPX500"])
        self.assertGreater(report.filtered_news_count, 0)
        self.assertGreater(report.analyzed_count, 0)
        self.assertTrue(report.audit_ids)
        del agent


if __name__ == "__main__":
    unittest.main()
