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
from src.backtest.strategies import (
    build_feature_score_signals,
    build_mean_reversion_signals,
    build_trend_signals,
)
from src.core.config import load_config
from src.core.history import periods_per_year_for_timeframe
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.features.indicators import bars_to_arrays
from src.stats.beta import compute_hedge_ratio, decompose_returns, estimate_hedge_cost_annual
from src.stats.returns import log_returns


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_decomposition(name: str, decomp) -> None:
    print(f"\n--- {name} ---")
    print(f"  Beta              : {decomp.beta:.2f}")
    print(f"  Alpha (annual)    : {decomp.alpha_annual:.2%}")
    print(f"  R-squared         : {decomp.r_squared:.2f}")
    print(f"  Total return      : {decomp.total_return:.2%}")
    print(f"  Benchmark return  : {decomp.benchmark_total_return:.2%}")
    print(f"  Beta contribution : {decomp.beta_contribution:.2%} ({decomp.beta_pct:.0f}% of total)")
    print(f"  Alpha contribution: {decomp.alpha_contribution:.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Beta decomposition and hedge analysis (Lesson 08)")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", help="Override analysis timeframe")
    parser.add_argument("--capital", type=float, help="Capital JPY for hedge cost estimate")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    engine = FeatureEngine(config)
    hedging = config.hedging

    timeframe = args.timeframe or config.stats.signal_timeframe
    benchmark = hedging.market_benchmark

    bars = store.get_recent_bars(args.symbol, timeframe, config.history_bars_for(timeframe))
    bench_bars = store.get_recent_bars(benchmark, timeframe, config.history_bars_for(timeframe))
    if len(bars) < config.stats.min_bars or len(bench_bars) < config.stats.min_bars:
        print(f"ERROR: insufficient data for {args.symbol} or {benchmark}")
        return 1

    _, _, _, closes, _ = bars_to_arrays(bars)
    _, _, _, bench_closes, _ = bars_to_arrays(bench_bars)
    closes_arr = np.array(closes, dtype=float)
    cost_pct = estimate_round_trip_cost_pct(config)
    rf = config.indicators.risk_free_rate
    periods = periods_per_year_for_timeframe(timeframe, config.trading.trading_days_per_year)

    series = engine.compute_series(bars)
    adx = np.array(series["adx"]) if series else np.array([])

    strategies = [
        ("trend_following", build_trend_signals(closes_arr, adx, config)),
        ("mean_reversion", build_mean_reversion_signals(bars, config)),
        ("feature_score", build_feature_score_signals(bars, config)),
    ]

    bench_returns = np.array(log_returns(bench_closes))
    sym_rets = np.array(log_returns(closes))

    print(f"\n=== Beta Report: {args.symbol} vs {benchmark} ({timeframe}) ===")
    print("  Lesson 08: return decomposition, hedge ratio, retail cost check")

    sym_decomp = decompose_returns(sym_rets, bench_returns, rf, periods)
    print(f"\n--- {args.symbol} buy-and-hold ---")
    print(f"  Beta              : {sym_decomp.beta:.2f}")
    print(f"  R-squared         : {sym_decomp.r_squared:.2f}")

    for name, signals in strategies:
        bt = run_backtest(closes_arr, signals, name, cost_pct, rf, periods)
        if not bt.returns:
            continue
        decomp = decompose_returns(np.array(bt.returns), bench_returns, rf, periods)
        print_decomposition(name, decomp)

    capital = args.capital or 1_000_000.0
    long_notional = capital * config.risk.max_single_position_pct / 100.0
    ratio = compute_hedge_ratio(
        long_notional,
        sym_decomp.beta,
        hedging.hedge_instrument_beta,
    )
    cost = estimate_hedge_cost_annual(
        ratio.hedge_notional,
        capital,
        hedging.retail_borrow_rate_annual,
        hedging.trading_cost_annual_pct,
    )

    print(f"\n=== Hedge Analysis (Lesson 8.2-8.4) ===")
    print(f"  Long notional (example)     : {long_notional:,.0f} JPY")
    print(f"  Symbol beta                 : {sym_decomp.beta:.2f}")
    print(f"  Dollar-neutral short        : {ratio.dollar_neutral_hedge_notional:,.0f} JPY")
    print(f"  Beta-neutral short          : {ratio.hedge_notional:,.0f} JPY")
    print(f"  Net beta (dollar-neutral)   : {ratio.net_beta_dollar_neutral:,.0f}")
    print(f"  Net beta (beta-neutral)     : {ratio.net_beta_beta_neutral:,.0f}")
    print(f"  Hedge cost / capital        : {cost['cost_pct_of_capital']:.2%}")
    print(f"  Breakeven gross alpha       : {cost['breakeven_alpha_pct']:.2%}")
    print(f"  Retail market-neutral viable: {hedging.retail_viable}")
    print("\n  Note: CFD short on MT5 demo; live hedge costs may differ.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
