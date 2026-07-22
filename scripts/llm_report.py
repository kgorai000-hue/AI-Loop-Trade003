from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_research_agent import LLMResearchAgent
from src.core.config import load_config
from src.llm_research.audit import LLMAuditStore


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_lesson_content() -> None:
    print("\n=== Lesson 14: LLM Applications in Quant ===")
    print("  Use cases : news sentiment, strategy diagnostics, research reports")
    print("  Safety    : mock default, JSON validation, audit SQLite, no execution path")
    print("  Feature   : optional sentiment nudge (use_as_feature in settings.local.yaml)")
    print("  Provider  : mock (offline) or openai via settings.local.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM research layer report (Lesson 14)")
    parser.add_argument("--paper-only", action="store_true", help="Lesson content only")
    parser.add_argument("--enable", action="store_true", help="Force llm_research.enabled for this run")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    if args.enable:
        config.llm_research.enabled = True
    setup_logging(config.log_level)

    agent = LLMResearchAgent(config)
    report = agent.analyze(symbols=config.symbols[:6])

    print(f"\n=== LLM Research Run ===")
    print(f"  provider       : {report.provider}")
    print(f"  news filtered  : {report.filtered_news_count}")
    print(f"  analyzed       : {report.analyzed_count}")
    print(f"  skipped        : {report.skipped_count}")

    for feat in report.sentiment_features:
        print(
            f"  {feat.symbol}: score={feat.sentiment_score:+.3f} "
            f"events={feat.event_count} type={feat.dominant_event_type}"
        )

    for analysis in report.analyses[:3]:
        print(
            f"  [{analysis.news_id}] {analysis.symbol} "
            f"sentiment={analysis.sentiment_overall:+.2f} signal={analysis.trading_signal}"
        )

    audit = LLMAuditStore(config.storage.path)
    recent = audit.recent(limit=3)
    if recent:
        print(f"\n=== Recent Audit ({len(recent)}) ===")
        for entry in recent:
            status = "OK" if entry.success else "FAIL"
            print(f"  #{entry.id} {status} {entry.provider} -> {entry.final_decision}")

    if report.strategy_report:
        print("\n=== Strategy Diagnostic (excerpt) ===")
        for line in report.strategy_report.splitlines()[:12]:
            print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
