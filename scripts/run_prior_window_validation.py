"""Evaluate adopted params on a prior bar window (default: bars before latest 2000)."""
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

from scripts.adopted_params import SYMBOL_ORDER, apply_adopted_params
from src.agents.backtest_agent import BacktestAgent
from src.agents.regime_agent import RegimeAgent
from src.core.config import load_config
from src.data.store import OHLCVStore

STRATEGIES = ("trend_following", "mean_reversion", "feature_score")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def slice_prior_window(bars: list[dict], window: int, skip_recent: int) -> list[dict]:
    """Take `window` bars ending `skip_recent` bars before the latest bar."""
    if skip_recent < 0 or window <= 0:
        raise ValueError("window and skip_recent must be positive / non-negative")
    end = len(bars) - skip_recent
    start = end - window
    if start < 0 or end <= 0:
        raise ValueError(
            f"need at least {window + skip_recent} bars, have {len(bars)}"
        )
    return bars[start:end]


def _bar_time_iso(bar: dict) -> str:
    ts = int(bar["time"])
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prior-window validation with adopted maps")
    parser.add_argument("--timeframe", default="M30")
    parser.add_argument("--bars", type=int, default=2000, help="Window size")
    parser.add_argument(
        "--skip-recent",
        type=int,
        default=2000,
        help="Skip this many most-recent bars (0 = evaluate latest --bars window)",
    )
    parser.add_argument("--mc-sims", type=int, default=100)
    parser.add_argument("--symbol", action="append")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=list(STRATEGIES),
        help="Limit strategies (repeatable). Default: all three",
    )
    args = parser.parse_args()
    strategies = tuple(args.strategy) if args.strategy else STRATEGIES

    config = load_config()
    setup_logging(config.log_level)
    timeframe = args.timeframe.upper()
    config.data.history_years = None
    config.data.history_bars_by_timeframe = {timeframe: args.bars + args.skip_recent}
    config.history_bars = args.bars + args.skip_recent
    config.backtest.monte_carlo_simulations = args.mc_sims

    store = OHLCVStore(config.storage.path)
    symbols = args.symbol or list(SYMBOL_ORDER)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = (
        PROJECT_ROOT
        / f"data/reports/prior_window_{timeframe}_{args.bars}bars_skip{args.skip_recent}_{stamp}.xlsx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    window_label = (
        f"latest {args.bars}" if args.skip_recent == 0 else f"prior {args.bars} (skip {args.skip_recent})"
    )
    print("=== Window validation (adopted maps fixed) ===")
    print(f"  symbols     : {len(symbols)}")
    print(f"  timeframe   : {timeframe}")
    print(f"  window      : {window_label}")
    print(f"  MC sims     : {args.mc_sims}")
    print(f"  strategies  : {', '.join(strategies)}")
    print("  params      : adopted maps (no re-tune)")
    print(f"  output      : {out_path}")

    regime_agent = RegimeAgent(config, store)
    rows: list[dict] = []
    errors: list[str] = []
    windows: list[dict] = []

    for idx, symbol in enumerate(symbols, start=1):
        all_bars = store.get_all_bars(symbol, timeframe)
        print(f"\n[{idx}/{len(symbols)}] {symbol}  stored={len(all_bars)} ...", flush=True)
        try:
            bars = slice_prior_window(all_bars, args.bars, args.skip_recent)
        except ValueError as exc:
            msg = f"{symbol}: {exc}"
            errors.append(msg)
            print(f"  ERROR: {exc}", flush=True)
            continue

        windows.append(
            {
                "symbol": symbol,
                "stored_bars": len(all_bars),
                "window_bars": len(bars),
                "skip_recent": args.skip_recent,
                "window_start": _bar_time_iso(bars[0]),
                "window_end": _bar_time_iso(bars[-1]),
            }
        )
        print(
            f"  window { _bar_time_iso(bars[0]) } → { _bar_time_iso(bars[-1]) } "
            f"({len(bars)} bars)",
            flush=True,
        )

        apply_adopted_params(config, symbol)
        agent = BacktestAgent(config, store)
        try:
            regime = regime_agent.assess(symbol)
            market_regime = regime.regime if regime else None
        except Exception:  # noqa: BLE001
            market_regime = None

        for strategy in strategies:
            try:
                result = agent.validate_strategy(
                    symbol,
                    strategy,
                    timeframe,
                    market_regime,
                    bars=bars,
                )
                perf = result.backtest.performance
                wf = result.walk_forward_summary
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": strategy,
                        "window_bars": len(bars),
                        "skip_recent": args.skip_recent,
                        "window_start": _bar_time_iso(bars[0]),
                        "window_end": _bar_time_iso(bars[-1]),
                        "total_return_pct": round(perf.total_return * 100, 2),
                        "ann_return_pct": round(perf.annualized_return * 100, 2),
                        "sharpe": round(perf.sharpe_ratio, 3),
                        "max_drawdown_pct": round(perf.max_drawdown * 100, 2),
                        "trades": perf.trades,
                        "win_rate_pct": round(perf.win_rate * 100, 1),
                        "wf_avg_test_sharpe": round(float(wf.get("avg_test_sharpe", 0.0)), 3),
                        "oos_ratio": round(result.oos.oos_ratio, 3),
                        "mc_prob_positive_pct": round(result.monte_carlo.prob_positive * 100, 1),
                        "gate_passed": bool(result.quality_gate.passed),
                        "expected_live_pct": round(
                            result.quality_gate.live_expected_return * 100, 2
                        ),
                    }
                )
                print(
                    f"    {strategy:16} ret={perf.total_return:.2%} "
                    f"Sharpe={perf.sharpe_ratio:.3f} round_trips={perf.trades} "
                    f"win={perf.win_rate:.1%} "
                    f"gate={'PASS' if result.quality_gate.passed else 'FAIL'}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{symbol} {strategy}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                print(f"    {strategy:16} ERROR: {exc}", flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        print("ERROR: no results")
        return 1

    # Best strategy per symbol by return
    best_idx = detail.groupby("symbol")["total_return_pct"].idxmax()
    best = detail.loc[best_idx].sort_values("total_return_pct", ascending=False)

    pivot = detail.pivot_table(
        index="symbol",
        columns="strategy",
        values="total_return_pct",
        aggfunc="first",
    )

    meta = pd.DataFrame(
        [
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "timeframe": timeframe,
                "window_bars": args.bars,
                "skip_recent": args.skip_recent,
                "mc_sims": args.mc_sims,
                "note": "Adopted maps fixed; prior window = bars before latest skip_recent",
                "error_count": len(errors),
            }
        ]
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        best.to_excel(writer, sheet_name="BestPerSymbol", index=False)
        pivot.to_excel(writer, sheet_name="ReturnPivot")
        detail.to_excel(writer, sheet_name="AllStrategies", index=False)
        pd.DataFrame(windows).to_excel(writer, sheet_name="Windows", index=False)
        meta.to_excel(writer, sheet_name="RunInfo", index=False)
        if errors:
            pd.DataFrame({"error": errors}).to_excel(writer, sheet_name="Errors", index=False)

    print("\n=== Best strategy per symbol (prior window, by return) ===")
    print(
        best[
            [
                "symbol",
                "strategy",
                "total_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "trades",
                "gate_passed",
                "mc_prob_positive_pct",
            ]
        ].to_string(index=False)
    )
    print("\n=== Return pivot (%) ===")
    print(pivot.to_string())
    print(f"\n  report saved : {out_path}")
    print(f"  errors       : {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
