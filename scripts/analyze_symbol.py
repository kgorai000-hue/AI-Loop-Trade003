from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.research_agent import ResearchAgent
from src.core.config import load_config
from src.data.store import OHLCVStore


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Statistical analysis for a symbol (Lesson 03)")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", help="Override analysis timeframe (default from config)")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    store = OHLCVStore(config.storage.path)
    agent = ResearchAgent(config, store)

    report = agent.analyze_symbol(args.symbol, args.timeframe)
    if report is None:
        print(f"ERROR: insufficient data for {args.symbol}")
        return 1

    print(f"\n=== Symbol Analysis: {report.symbol} ({report.timeframe}) ===")
    print(f"  bars                  : {report.bars}")
    print(f"  cumulative log return : {report.cumulative_log_return:.4f}")
    print(f"  annualized return     : {report.annualized_return:.2%}")
    print(f"  daily volatility      : {report.daily_volatility:.4f}")
    print(f"  annualized volatility : {report.annualized_volatility:.2%}")
    print(f"  autocorr lag-1        : {report.autocorr_lag1:.3f}")
    print(f"  autocorr lag-5        : {report.autocorr_lag5:.3f}")
    print(f"  skewness              : {report.skewness:.3f}")
    print(f"  excess kurtosis       : {report.excess_kurtosis:.3f}")
    print(f"  price stationary      : {report.price_stationary}")
    print(f"  return stationary     : {report.return_stationary}")
    print(f"  vol autocorr          : {report.vol_autocorr:.3f}")
    print(f"  regime                : {report.regime.value}")
    print(f"  tail warning          : {report.tail_warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
