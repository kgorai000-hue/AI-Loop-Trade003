from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.regime_agent import RegimeAgent
from src.agents.resilience_agent import ResilienceAgent
from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.regime.misjudgment import expected_monthly_return, lag_cost_table, misjudgment_cost
from src.regime.resilience import DegradationLevel, LEVEL_NAMES


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_lesson_content() -> None:
    print("\n=== Lesson 13: 5 Misjudgment Patterns ===")
    patterns = [
        "false_positive   - ranging misread as trending",
        "false_negative   - trending misread as ranging (2020 Feb story)",
        "delayed          - correct direction, wrong timing",
        "oversensitive    - too many switches, cost drag",
        "boundary_oscillation - ADX flicker near threshold",
    ]
    for line in patterns:
        print(f"  {line}")

    print("\n=== Lag Cost Table (15% crash in 5 days) ===")
    for days, row in lag_cost_table().items():
        print(
            f"  lag {days}d: loss at confirm {row['loss_at_confirm_pct']:.0%} "
            f"saved {row['saved_pct']:.0%} ({row['note']})"
        )

    print("\n=== Accuracy Impact (70% detection) ===")
    impact = expected_monthly_return(0.70)
    print(f"  monthly return     : {impact['monthly_return']:.2%}")
    print(f"  perfect return     : {impact['perfect_monthly_return']:.2%}")
    print(f"  return reduction   : {impact['return_reduction_pct']:.0%}")

    cost = misjudgment_cost(direct_loss=0.07, opportunity_cost=0.05, switch_count=24, switch_cost_pct=0.005)
    print(f"\n=== Misjudgment Cost Example ===")
    print(f"  direct + opportunity + switches = {cost.total:.1%}")

    print("\n=== Degradation Levels (Meta Agent) ===")
    for level in DegradationLevel:
        print(f"  Level {level.value}: {LEVEL_NAMES[level]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regime misjudgment and resilience report (Lesson 13)")
    parser.add_argument("--symbol", action="append", help="Symbols to assess")
    parser.add_argument("--paper-only", action="store_true", help="Lesson content only")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    setup_logging(config.log_level)
    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    regime_agent = RegimeAgent(config, store)
    resilience_agent = ResilienceAgent(config)

    try:
        connector.connect()
        symbols = args.symbol or config.symbols[:6]
        regime_map = {}
        for symbol in symbols:
            assessment = regime_agent.assess(symbol)
            if assessment is not None:
                regime_map[symbol] = assessment

        market = regime_agent.assess_market_proxy()
        report = resilience_agent.build_report(market, regime_map)

        print(f"\n=== Resilience Report ===")
        print(f"  degradation level : {report.level_name} ({report.degradation_level})")
        print(f"  position scale    : {report.position_scale_multiplier:.0%}")
        if report.warnings:
            print("  warnings:")
            for warning in report.warnings[:5]:
                print(f"    - {warning}")

        print(f"\n=== Symbol Regime Health ===")
        for symbol, assessment in regime_map.items():
            flags = []
            if assessment.uncertain:
                flags.append("uncertain")
            if assessment.misjudgment_pattern:
                flags.append(assessment.misjudgment_pattern)
            flag_str = ", ".join(flags) if flags else "ok"
            print(
                f"  {symbol:12} label={assessment.regime_label:14} "
                f"max_p={assessment.max_probability:.0%} scale={assessment.position_scale:.0%} [{flag_str}]"
            )

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
