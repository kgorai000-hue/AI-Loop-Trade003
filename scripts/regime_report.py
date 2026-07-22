from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.regime_agent import RegimeAgent
from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.regime.detection import evaluate_regime_value, rule_based_detect


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_paper_exercises() -> None:
    print("\n=== Lesson 12 Paper Exercise: Periods A-D ===")
    periods = [
        ("A", 35.0, 0.18, 0.12),
        ("B", 15.0, 0.08, -0.02),
        ("C", 28.0, 0.45, -0.25),
        ("D", 22.0, 0.12, 0.03),
    ]
    for name, adx, vol, ret in periods:
        result = rule_based_detect(
            adx=adx,
            annualized_volatility=vol,
            recent_return=ret,
            asset_correlation=0.5,
        )
        print(f"  Period {name}: {result.label:16} | {result.reason}")

    print("\n=== Lesson 12 Rule Scenarios 1-4 ===")
    scenarios = [
        ("1", 32, 0.22, 0.08, 0.4),
        ("2", 18, 0.12, -0.01, 0.3),
        ("3", 25, 0.38, -0.15, 0.85),
        ("4", 23, 0.18, 0.03, 0.5),
    ]
    for name, adx, vol, ret, corr in scenarios:
        result = rule_based_detect(
            adx=adx,
            annualized_volatility=vol,
            recent_return=ret,
            asset_correlation=corr,
        )
        print(f"  Scenario {name}: {result.label:16} | weights trend={result.strategy_weights.get('trend', 0):.0%} mr={result.strategy_weights.get('mean_reversion', 0):.0%}")

    print("\n=== Lesson 12 Value Evaluation ===")
    value = evaluate_regime_value(
        return_without=0.08,
        return_with=0.15,
        switch_count=24,
        switch_cost_pct=0.005,
    )
    print(f"  Return improvement : {value['return_improvement']:.1%}")
    print(f"  Switch cost (24x)  : {value['switch_cost_total']:.1%}")
    print(f"  Net value          : {value['net_value']:.1%}")
    print("  Note: frequent switching can erase gains; use confirm_days + soft weights.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regime detection report (Lesson 12)")
    parser.add_argument("--symbol", action="append", help="Symbols to assess")
    parser.add_argument("--paper-only", action="store_true", help="Print paper exercises only")
    args = parser.parse_args()

    print_paper_exercises()
    if args.paper_only:
        return 0

    config = load_config()
    setup_logging(config.log_level)
    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    agent = RegimeAgent(config, store)

    try:
        connector.connect()
        symbols = args.symbol or config.symbols[:6]

        print(f"\n=== Regime Config ===")
        rc = config.regime
        print(f"  method            : {rc.method}")
        print(f"  confirm_days      : {rc.confirm_days}")
        print(f"  crisis vol/corr   : {rc.crisis_vol_threshold:.0%} / {rc.crisis_correlation_threshold:.2f}")
        print(f"  transition        : {rc.transition_strategy}")

        print(f"\n=== Live Assessments ({len(symbols)}) ===")
        for symbol in symbols:
            assessment = agent.assess(symbol)
            if assessment is None:
                print(f"  {symbol:12} (insufficient data)")
                continue
            probs = ", ".join(f"{k}={v:.0%}" for k, v in assessment.probabilities.items())
            weights = ", ".join(f"{k}={v:.0%}" for k, v in assessment.strategy_weights.items())
            print(
                f"  {symbol:12} {assessment.regime_label:16} "
                f"ADX={assessment.adx:.1f} vol={assessment.annualized_volatility:.1%} "
                f"corr={assessment.asset_correlation:.2f}"
            )
            print(f"    probs   : {probs}")
            print(f"    weights : {weights}")
            print(f"    route   : {assessment.selected_strategy.value} scale={assessment.position_scale:.0%}")

        stats = agent.switch_stats()
        print(f"\n=== Switch History (365d) ===")
        print(f"  observations      : {stats['observations']:.0f}")
        print(f"  switches          : {stats['switches']:.0f}")
        print(f"  switch rate       : {stats['switch_rate']:.1%}")
        print(f"  avg duration (d)  : {stats['avg_duration_days']:.1f}")

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
