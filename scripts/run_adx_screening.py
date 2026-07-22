"""Screen adx_trend_threshold across all symbols (M30 screening mode)."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.agents.backtest_agent import BacktestAgent
from src.agents.regime_agent import RegimeAgent
from src.core.config import load_config
from src.data.store import OHLCVStore

ADX_VALUES = (18, 20, 22, 24, 26, 28, 30)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ADX trend threshold grid for all symbols")
    parser.add_argument("--timeframe", default="M30")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--mc-sims", type=int, default=100)
    parser.add_argument("--symbol", action="append")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    timeframe = args.timeframe.upper()

    config.data.history_years = None
    config.data.history_bars_by_timeframe = {timeframe: args.bars}
    config.history_bars = args.bars
    config.backtest.monte_carlo_simulations = args.mc_sims

    store = OHLCVStore(config.storage.path)
    symbols = args.symbol or config.symbols
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = (
        PROJECT_ROOT
        / f"data/reports/screening_{timeframe}_{args.bars}bars_nogate_adxgrid_{stamp}.xlsx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== ADX trend_threshold grid ===")
    print(f"  symbols    : {len(symbols)}")
    print(f"  timeframe  : {timeframe}")
    print(f"  bars       : {args.bars}")
    print(f"  MC sims    : {args.mc_sims}")
    print(f"  strategy   : trend_following only")
    print(f"  ADX values : {', '.join(str(v) for v in ADX_VALUES)}")
    print(f"  output     : {out_path}")

    regime_agent = RegimeAgent(config, store)
    rows: list[dict] = []
    errors: list[str] = []

    for adx in ADX_VALUES:
        config.strategies.adx_trend_threshold = float(adx)
        agent = BacktestAgent(config, store)
        print(f"\n=== ADX={adx} ===", flush=True)

        for idx, symbol in enumerate(symbols, start=1):
            print(f"  [{idx}/{len(symbols)}] {symbol} ...", flush=True)
            try:
                regime = regime_agent.assess(symbol)
                market_regime = regime.regime if regime else None
                result = agent.validate_strategy(
                    symbol, "trend_following", timeframe, market_regime
                )
                perf = result.backtest.performance
                wf = result.walk_forward_summary
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "bars": args.bars,
                        "strategy": "trend_following",
                        "adx_trend_threshold": adx,
                        "total_return_pct": round(perf.total_return * 100, 2),
                        "ann_return_pct": round(perf.annualized_return * 100, 2),
                        "sharpe": round(perf.sharpe_ratio, 3),
                        "max_drawdown_pct": round(perf.max_drawdown * 100, 2),
                        "trades": perf.trades,
                        "win_rate_pct": round(perf.win_rate * 100, 1),
                        "wf_rounds": int(wf.get("rounds", 0)),
                        "wf_avg_test_sharpe": round(float(wf.get("avg_test_sharpe", 0.0)), 3),
                        "wf_positive_rounds_pct": round(
                            float(wf.get("positive_rounds_pct", 0.0)) * 100, 1
                        ),
                        "oos_ratio": round(result.oos.oos_ratio, 3),
                        "expected_live_pct": round(
                            result.quality_gate.live_expected_return * 100, 2
                        ),
                        "mc_prob_positive_pct": round(
                            result.monte_carlo.prob_positive * 100, 1
                        ),
                        "mc_p5_pct": round(result.monte_carlo.percentile_5 * 100, 2),
                    }
                )
                print(
                    f"      ret={perf.total_return:.2%} Sharpe={perf.sharpe_ratio:.3f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"ADX={adx} {symbol}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                print(f"      ERROR: {exc}", flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        print("ERROR: no results")
        return 1

    # Best ADX per symbol by total return
    best_idx = detail.groupby("symbol")["total_return_pct"].idxmax()
    best = detail.loc[best_idx].sort_values("total_return_pct", ascending=False)

    pivot = detail.pivot_table(
        index="symbol",
        columns="adx_trend_threshold",
        values="total_return_pct",
        aggfunc="first",
    )

    meta = pd.DataFrame(
        [
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "timeframe": timeframe,
                "bars": args.bars,
                "mc_sims": args.mc_sims,
                "strategy": "trend_following",
                "adx_values": ",".join(str(v) for v in ADX_VALUES),
                "error_count": len(errors),
                "note": "Gates OFF; ADX grid on trend_following only",
            }
        ]
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        best.to_excel(writer, sheet_name="BestPerSymbol", index=False)
        pivot.to_excel(writer, sheet_name="ReturnPivot")
        detail.to_excel(writer, sheet_name="AllTrials", index=False)
        meta.to_excel(writer, sheet_name="RunInfo", index=False)
        if errors:
            pd.DataFrame({"error": errors}).to_excel(writer, sheet_name="Errors", index=False)

    print("\n=== Best ADX per symbol (by return) ===")
    print(
        best[
            [
                "symbol",
                "adx_trend_threshold",
                "total_return_pct",
                "sharpe",
                "wf_avg_test_sharpe",
                "max_drawdown_pct",
                "mc_prob_positive_pct",
            ]
        ].to_string(index=False)
    )
    print(f"\n  report saved : {out_path}")
    print(f"  errors       : {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
