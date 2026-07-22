from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.loop_engine import run_full_loop_matrix, run_parameter_loop, save_loop_report
from src.backtest.parameter_spaces import default_parameter_specs
from src.core.config import load_config
from src.data.store import OHLCVStore


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_report_summary(report) -> None:
    print(f"\n=== Loop Report: {report.symbol} {report.timeframe} {report.strategy} ===")
    print(f"  baseline WF Sharpe : {report.baseline_metrics.get('wf_avg_test_sharpe', 0):.2f}")
    print(f"  trials             : {len(report.trials)}")
    print(f"  adopted            : {report.adopted or '(none)'}")
    if report.stopped_early:
        print(f"  stopped early      : {report.stop_reason}")

    adopted = [t for t in report.trials if t.adopted]
    if adopted:
        print("\n  Adopted trials:")
        for t in adopted:
            print(
                f"    {t.parameter}={t.value} WF={t.metrics.get('wf_avg_test_sharpe', 0):.2f} "
                f"MDD={t.metrics.get('max_drawdown_pct', 0):.1f}%"
            )

    hard = [t for t in report.trials if t.verdict == "hard_stop"]
    if hard:
        print(f"\n  Hard stops: {len(hard)} (showing up to 3)")
        for t in hard[:3]:
            print(f"    {t.parameter}={t.value}: {', '.join(t.reasons[:2])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Loop engineering: one-parameter-at-a-time optimization (loop criteria PDF)"
    )
    parser.add_argument("--symbol", action="append", help="Symbol(s) to optimize")
    parser.add_argument("--timeframe", help="Override signal timeframe")
    parser.add_argument(
        "--strategy",
        choices=["trend_following", "mean_reversion", "feature_score"],
        default="feature_score",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run loop for all three backtest strategies",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List default parameter search spaces and exit",
    )
    args = parser.parse_args()

    if args.list_params:
        print("=== Default parameter search spaces ===")
        for spec in sorted(default_parameter_specs(), key=lambda s: (s.strategy, s.priority)):
            print(
                f"  [{spec.priority}] {spec.strategy:18} {spec.name:30} "
                f"values={spec.values}"
            )
        return 0

    config = load_config()
    setup_logging(config.log_level)
    if not config.loop_engineering.enabled:
        print("ERROR: loop_engineering.enabled is false in settings.yaml")
        return 1

    store = OHLCVStore(config.storage.path)
    symbols = args.symbol or [config.symbols[0]]
    timeframe = args.timeframe or config.stats.signal_timeframe
    strategies = ["trend_following", "mean_reversion", "feature_score"]
    if not args.all_strategies:
        strategies = [args.strategy]

    print("=== Loop Engineering Parameter Optimization ===")
    print(f"  symbols    : {', '.join(symbols)}")
    print(f"  timeframe  : {timeframe}")
    print(f"  strategies : {', '.join(strategies)}")
    print("  method     : one parameter at a time, OOS/WF primary (not in-sample Sharpe)")

    exit_code = 0
    if len(symbols) == 1 and len(strategies) == 1:
        report = run_parameter_loop(config, store, symbols[0], strategies[0], timeframe)
        path = save_loop_report(report, config)
        print_report_summary(report)
        print(f"\n  saved: {path}")
        exit_code = 0 if report.adopted or report.trials else 1
    else:
        reports = run_full_loop_matrix(config, store, symbols, strategies, timeframe)
        for report in reports:
            path = save_loop_report(report, config)
            print_report_summary(report)
            print(f"  saved: {path}")
        exit_code = 0 if reports else 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
