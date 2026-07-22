from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.backtest_agent import BacktestAgent
from src.agents.regime_agent import RegimeAgent
from src.core.config import load_config
from src.data.store import OHLCVStore


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_gate_report(result, decay_factor: float) -> None:
    gate = result.quality_gate
    status = "PASS" if gate.passed else "FAIL"
    perf = result.backtest.performance
    print(f"\n--- {result.strategy_name} [{status}] ---")
    print(f"  total return      : {perf.total_return:7.2%}")
    print(f"  ann return        : {perf.annualized_return:7.2%}")
    print(f"  Sharpe            : {perf.sharpe_ratio:5.2f}")
    print(f"  max drawdown      : {perf.max_drawdown:6.2%}")
    print(f"  cost/trade        : {result.backtest.cost_per_trade_pct:.3f}%")
    print(f"  OOS train/test    : {result.oos.train_return:.2%} / {result.oos.test_return:.2%} (ratio={result.oos.oos_ratio:.2f})")
    wf = result.walk_forward_summary
    print(f"  Walk-forward      : {int(wf['rounds'])} rounds, avg test ret={wf['avg_test_return']:.2%}")
    mc = result.monte_carlo
    print(f"  Monte Carlo       : P5={mc.percentile_5:.2%} P50={mc.percentile_50:.2%} prob+={mc.prob_positive:.1%}")
    print(f"  Expected live     : {gate.live_expected_return:.2%} (BT x {decay_factor} - hidden cost)")

    if result.param_sensitivity:
        for ps in result.param_sensitivity:
            stable = "stable" if ps.stable else "UNSTABLE"
            print(f"  Param {ps.parameter:8} : {stable} (max change {ps.max_change_pct:.0%})")

    failed = [c for c in gate.checks if not c.passed]
    if failed:
        print("  Failed gates:")
        for check in failed:
            print(f"    [{check.check_id}] {check.name}: {check.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest validation with quality gates (Lesson 07)"
    )
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", help="Override signal timeframe")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    agent = BacktestAgent(config, store)
    regime_agent = RegimeAgent(config, store)

    regime = regime_agent.assess(args.symbol)
    market_regime = regime.regime if regime else None
    timeframe = args.timeframe or config.stats.signal_timeframe

    try:
        report = agent.validate_symbol(args.symbol, timeframe, market_regime)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\n=== Backtest Validation: {report.symbol} ({report.timeframe}) ===")
    print("  Method: T signal -> T+1 execution, cost-aware, no look-ahead")
    print(f"  Round-trip cost : {report.cost_per_trade_pct:.3f}% per trade change")
    print(f"  Bonferroni (3 strategies): p-threshold = {0.05 / config.backtest.strategies_tested:.6f}")

    all_pass = True
    for result in report.strategies:
        print_gate_report(result, config.backtest.live_decay_factor)
        if not result.quality_gate.passed:
            all_pass = False

    print(f"\n=== Summary ===")
    passed = sum(1 for r in report.strategies if r.quality_gate.passed)
    print(f"  strategies passed : {passed}/{len(report.strategies)}")
    print("  Note: quality gates are advisory (report-only, no pipeline block).")

    if not all_pass:
        print("\n  Recommendation: do NOT deploy strategies with failed gates to live trading.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
