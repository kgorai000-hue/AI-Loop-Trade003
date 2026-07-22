from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.costs import estimate_round_trip_cost_pct
from src.backtest.engine import run_backtest
from src.backtest.strategies import build_mean_reversion_signals, build_trend_signals
from src.core.config import load_config
from src.core.history import periods_per_year_for_timeframe
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.features.indicators import bars_to_arrays
from src.strategies.pairs import backtest_pair_returns


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_report(name: str, bt) -> None:
    perf = bt.performance
    print(
        f"  {name:18} ret={perf.total_return:7.2%} sharpe={perf.sharpe_ratio:5.2f} "
        f"mdd={perf.max_drawdown:6.2%} round_trips={perf.trades:4d}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest classical strategies (Lesson 05/07, T+1 execution)"
    )
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", help="Override signal timeframe")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    engine = FeatureEngine(config)
    cost_pct = estimate_round_trip_cost_pct(config)
    rf = config.indicators.risk_free_rate

    timeframe = args.timeframe or config.stats.signal_timeframe
    periods = periods_per_year_for_timeframe(timeframe, config.trading.trading_days_per_year)
    bars = store.get_recent_bars(args.symbol, timeframe, config.history_bars_for(timeframe))
    if len(bars) < config.stats.min_bars:
        print(f"ERROR: insufficient bars for {args.symbol}")
        return 1

    _, _, _, closes, _ = bars_to_arrays(bars)
    closes_arr = np.array(closes, dtype=float)
    series = engine.compute_series(bars)
    adx = np.array(series["adx"]) if series else np.array([])

    trend_sig = build_trend_signals(closes_arr, adx, config)
    mr_sig = build_mean_reversion_signals(bars, config)

    trend_bt = run_backtest(
        closes_arr, trend_sig, "trend_following", cost_pct, rf, periods, zero_means_flat=True
    )
    mr_bt = run_backtest(closes_arr, mr_sig, "mean_reversion", cost_pct, rf, periods)

    print(f"\n=== Strategy Backtest: {args.symbol} ({timeframe}) ===")
    print("  Method: T signal -> T+1 execution, cost-aware (Lesson 07)")
    print(f"  Round-trip cost: {cost_pct:.3f}% per position change\n")
    print_report("trend_following", trend_bt)
    print_report("mean_reversion", mr_bt)

    print("\n=== Pairs Backtest (legacy spread model) ===")
    cfg = config.strategies
    for pair in cfg.pairs:
        if len(pair) != 2:
            continue
        sym_a, sym_b = pair[0], pair[1]
        bars_a = store.get_recent_bars(sym_a, timeframe, config.history_bars_for(timeframe))
        bars_b = store.get_recent_bars(sym_b, timeframe, config.history_bars_for(timeframe))
        if len(bars_a) < cfg.pair_lookback + 10 or len(bars_b) < cfg.pair_lookback + 10:
            print(f"  {sym_a}/{sym_b:12} insufficient data")
            continue
        _, _, _, ca, _ = bars_to_arrays(bars_a)
        _, _, _, cb, _ = bars_to_arrays(bars_b)
        pair_ret = backtest_pair_returns(ca, cb, cfg.pair_lookback, cfg.pair_zscore_entry)
        from src.stats.performance import evaluate_returns

        pair_perf = evaluate_returns(pair_ret, rf, periods, trade_pnls=None)
        print(
            f"  {sym_a}/{sym_b:12} ret={pair_perf.total_return:7.2%} "
            f"sharpe={pair_perf.sharpe_ratio:5.2f} mdd={pair_perf.max_drawdown:6.2%}"
        )

    print("\n  For full validation (Walk-Forward, Monte Carlo, gates):")
    print(f"  python scripts/run_backtest_validation.py --symbol {args.symbol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
