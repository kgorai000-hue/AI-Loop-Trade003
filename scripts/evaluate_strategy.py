from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.regime_agent import RegimeAgent
from src.backtest.costs import estimate_round_trip_cost_pct
from src.backtest.engine import run_backtest
from src.backtest.strategies import build_feature_score_signals
from src.core.config import load_config
from src.core.history import periods_per_year_for_timeframe
from src.data.store import OHLCVStore
from src.features.indicators import bars_to_arrays


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate feature-based strategy (Lesson 04/07)")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", help="Override signal timeframe")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    regime_agent = RegimeAgent(config, store)

    timeframe = args.timeframe or config.stats.signal_timeframe
    bars = store.get_recent_bars(args.symbol, timeframe, config.history_bars_for(timeframe))
    if len(bars) < config.stats.min_bars:
        print(f"ERROR: insufficient bars for {args.symbol}")
        return 1

    regime = regime_agent.assess(args.symbol)
    market_regime = regime.regime if regime else None

    _, _, _, closes, _ = bars_to_arrays(bars)
    closes_arr = np.array(closes, dtype=float)
    signals = build_feature_score_signals(bars, config, market_regime)
    cost_pct = estimate_round_trip_cost_pct(config)

    result = run_backtest(
        closes_arr,
        signals,
        "feature_score",
        cost_pct,
        config.indicators.risk_free_rate,
        periods_per_year_for_timeframe(timeframe, config.trading.trading_days_per_year),
    )
    perf = result.performance

    print(f"\n=== Strategy Evaluation: {args.symbol} ({timeframe}) ===")
    print(f"  method            : T+1 execution, cost-aware (Lesson 07)")
    print(f"  regime used       : {market_regime.value if market_regime else 'unknown'}")
    print(f"  score threshold   : {config.indicators.signal_score_threshold}")
    print(f"  round-trip cost   : {cost_pct:.3f}% per trade change")
    print(f"  total return      : {perf.total_return:.2%}")
    print(f"  annualized return : {perf.annualized_return:.2%}")
    print(f"  Sharpe ratio      : {perf.sharpe_ratio:.2f}")
    print(f"  Sortino ratio     : {perf.sortino_ratio:.2f}")
    print(f"  max drawdown      : {perf.max_drawdown:.2%}")
    print(f"  Calmar ratio      : {perf.calmar_ratio:.2f}")
    print(f"  round-trips       : {perf.trades}")
    print(f"  position changes  : {len(result.trades)}")
    print(f"  win rate          : {perf.win_rate:.2%}")
    print("\n  For full validation: python scripts/run_backtest_validation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
