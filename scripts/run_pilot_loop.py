"""One-off pilot runner: reduced scope for practical runtime on local PC."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_parameter_loop import print_report_summary, setup_logging
from src.backtest.loop_engine import run_parameter_loop, save_loop_report
from src.core.config import load_config
from src.data.store import OHLCVStore


def main() -> int:
    symbol = "WTI"
    timeframe = "H1"
    strategy = "feature_score"
    pilot_bars = 3000
    mc_sims = 100

    config = load_config()
    config.data.history_years = None
    config.data.history_bars_by_timeframe = {timeframe: pilot_bars}
    config.history_bars = pilot_bars
    config.backtest.monte_carlo_simulations = mc_sims
    setup_logging(config.log_level)

    store = OHLCVStore(config.storage.path)
    bars = store.get_recent_bars(symbol, timeframe, pilot_bars)
    print("=== WTI Pilot Loop (restart) ===")
    print(f"  symbol     : {symbol}")
    print(f"  timeframe  : {timeframe}")
    print(f"  strategy   : {strategy}")
    print(f"  bars       : {len(bars)} / {pilot_bars} (pilot subset; DB has more)")
    print(f"  MC sims    : {mc_sims} (pilot speed)")
    print("  method     : one parameter at a time, OOS/WF primary")
    sys.stdout.flush()

    report = run_parameter_loop(config, store, symbol, strategy, timeframe)
    path = save_loop_report(report, config)
    print_report_summary(report)
    print(f"\n  saved: {path}")

    rejects = [t for t in report.trials if t.verdict == "reject"]
    if rejects:
        print(f"\n  Rejected trials: {len(rejects)} (showing up to 5)")
        for t in rejects[:5]:
            print(f"    {t.parameter}={t.value}: {', '.join(t.reasons[:2])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
