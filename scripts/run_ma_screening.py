"""Screen trend MA lengths one-at-a-time (PDF: do not change both together).

Phase 1: fix ma_long=20, grid ma_short
Phase 2: fix ma_short to per-symbol best from phase 1, grid ma_long

ADX trend / sideways stay on adopted maps.
"""
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

from scripts.screening_params import BEST_ADX_SIDEWAYS, BEST_ADX_TREND
from src.agents.backtest_agent import BacktestAgent
from src.agents.regime_agent import RegimeAgent
from src.core.config import load_config
from src.data.store import OHLCVStore

MA_SHORT_VALUES = (3, 5, 8, 10)
MA_LONG_VALUES = (15, 20, 30, 40)
DEFAULT_MA_LONG = 20
DEFAULT_MA_SHORT = 5


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _apply_adx_maps(config, symbol: str) -> tuple[float, float]:
    trend = float(BEST_ADX_TREND.get(symbol, config.strategies.adx_trend_threshold))
    sideways = float(BEST_ADX_SIDEWAYS.get(symbol, config.strategies.adx_sideways_threshold))
    config.strategies.adx_trend_threshold = trend
    config.strategies.adx_sideways_threshold = sideways
    return trend, sideways


def _run_trial(
    *,
    agent: BacktestAgent,
    regime_agent: RegimeAgent,
    symbol: str,
    timeframe: str,
    bars: int,
    ma_short: int,
    ma_long: int,
    phase: str,
) -> dict:
    regime = regime_agent.assess(symbol)
    market_regime = regime.regime if regime else None
    result = agent.validate_strategy(symbol, "trend_following", timeframe, market_regime)
    perf = result.backtest.performance
    wf = result.walk_forward_summary
    return {
        "phase": phase,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "strategy": "trend_following",
        "ma_short": ma_short,
        "ma_long": ma_long,
        "adx_trend_threshold": agent.config.strategies.adx_trend_threshold,
        "adx_sideways_threshold": agent.config.strategies.adx_sideways_threshold,
        "total_return_pct": round(perf.total_return * 100, 2),
        "ann_return_pct": round(perf.annualized_return * 100, 2),
        "sharpe": round(perf.sharpe_ratio, 3),
        "max_drawdown_pct": round(perf.max_drawdown * 100, 2),
        "trades": perf.trades,
        "win_rate_pct": round(perf.win_rate * 100, 1),
        "wf_rounds": int(wf.get("rounds", 0)),
        "wf_avg_test_sharpe": round(float(wf.get("avg_test_sharpe", 0.0)), 3),
        "wf_positive_rounds_pct": round(float(wf.get("positive_rounds_pct", 0.0)) * 100, 1),
        "oos_ratio": round(result.oos.oos_ratio, 3),
        "expected_live_pct": round(result.quality_gate.live_expected_return * 100, 2),
        "mc_prob_positive_pct": round(result.monte_carlo.prob_positive * 100, 1),
        "mc_p5_pct": round(result.monte_carlo.percentile_5 * 100, 2),
    }


def _best_by_return(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    best_idx = detail.groupby(group_cols)["total_return_pct"].idxmax()
    return detail.loc[best_idx].sort_values("total_return_pct", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trend MA short/long sequential grid")
    parser.add_argument("--timeframe", default="M30")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--mc-sims", type=int, default=100)
    parser.add_argument("--symbol", action="append")
    parser.add_argument(
        "--phase",
        choices=("both", "short", "long"),
        default="both",
        help="both=short then long; short=ma_short only; long=ma_long with default short=5",
    )
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
        / f"data/reports/screening_{timeframe}_{args.bars}bars_nogate_magrid_{stamp}.xlsx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Trend MA grid (one-at-a-time) ===")
    print(f"  symbols    : {len(symbols)}")
    print(f"  timeframe  : {timeframe}")
    print(f"  bars       : {args.bars}")
    print(f"  MC sims    : {args.mc_sims}")
    print("  strategy   : trend_following only")
    print("  ADX        : per-symbol best trend + sideways maps (fixed)")
    print(f"  phase      : {args.phase}")
    print(f"  ma_short   : {', '.join(str(v) for v in MA_SHORT_VALUES)} (long fixed={DEFAULT_MA_LONG})")
    print(f"  ma_long    : {', '.join(str(v) for v in MA_LONG_VALUES)} (short = best from phase1 or {DEFAULT_MA_SHORT})")
    print("  note       : ATR not screened (stops unused in backtest signals)")
    print(f"  output     : {out_path}")

    regime_agent = RegimeAgent(config, store)
    rows: list[dict] = []
    errors: list[str] = []
    best_short_map: dict[str, int] = {s: DEFAULT_MA_SHORT for s in symbols}

    if args.phase in ("both", "short"):
        config.strategies.trend_ma_long = DEFAULT_MA_LONG
        for ma_short in MA_SHORT_VALUES:
            if ma_short >= DEFAULT_MA_LONG:
                continue
            config.strategies.trend_ma_short = int(ma_short)
            print(f"\n=== phase1 ma_short={ma_short} (ma_long={DEFAULT_MA_LONG}) ===", flush=True)
            for idx, symbol in enumerate(symbols, start=1):
                trend, sideways = _apply_adx_maps(config, symbol)
                agent = BacktestAgent(config, store)
                print(
                    f"  [{idx}/{len(symbols)}] {symbol} "
                    f"(ADX={trend:g}/{sideways:g}) ...",
                    flush=True,
                )
                try:
                    row = _run_trial(
                        agent=agent,
                        regime_agent=regime_agent,
                        symbol=symbol,
                        timeframe=timeframe,
                        bars=args.bars,
                        ma_short=ma_short,
                        ma_long=DEFAULT_MA_LONG,
                        phase="ma_short",
                    )
                    rows.append(row)
                    print(
                        f"      ret={row['total_return_pct']:.2f}% Sharpe={row['sharpe']:.3f} "
                        f"trades={row['trades']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = f"ma_short={ma_short} {symbol}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    print(f"      ERROR: {exc}", flush=True)

        phase1 = pd.DataFrame([r for r in rows if r["phase"] == "ma_short"])
        if not phase1.empty:
            best_short = _best_by_return(phase1, ["symbol"])
            best_short_map = {
                str(r.symbol): int(r.ma_short) for r in best_short.itertuples(index=False)
            }
            print("\n=== Best ma_short per symbol (phase1) ===")
            print(
                best_short[
                    ["symbol", "ma_short", "ma_long", "total_return_pct", "sharpe", "trades"]
                ].to_string(index=False)
            )

    if args.phase in ("both", "long"):
        for ma_long in MA_LONG_VALUES:
            print(f"\n=== phase2 ma_long={ma_long} (ma_short=per-symbol best) ===", flush=True)
            for idx, symbol in enumerate(symbols, start=1):
                ma_short = int(best_short_map.get(symbol, DEFAULT_MA_SHORT))
                if ma_short >= ma_long:
                    print(
                        f"  [{idx}/{len(symbols)}] {symbol} ... SKIP "
                        f"(short={ma_short} >= long={ma_long})",
                        flush=True,
                    )
                    continue
                config.strategies.trend_ma_short = ma_short
                config.strategies.trend_ma_long = int(ma_long)
                trend, sideways = _apply_adx_maps(config, symbol)
                agent = BacktestAgent(config, store)
                print(
                    f"  [{idx}/{len(symbols)}] {symbol} "
                    f"(MA={ma_short}/{ma_long} ADX={trend:g}/{sideways:g}) ...",
                    flush=True,
                )
                try:
                    row = _run_trial(
                        agent=agent,
                        regime_agent=regime_agent,
                        symbol=symbol,
                        timeframe=timeframe,
                        bars=args.bars,
                        ma_short=ma_short,
                        ma_long=ma_long,
                        phase="ma_long",
                    )
                    rows.append(row)
                    print(
                        f"      ret={row['total_return_pct']:.2f}% Sharpe={row['sharpe']:.3f} "
                        f"trades={row['trades']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = f"ma_long={ma_long} {symbol}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    print(f"      ERROR: {exc}", flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        print("ERROR: no results")
        return 1

    phase1_detail = detail[detail["phase"] == "ma_short"]
    phase2_detail = detail[detail["phase"] == "ma_long"]

    best_short_df = (
        _best_by_return(phase1_detail, ["symbol"]) if not phase1_detail.empty else pd.DataFrame()
    )
    best_long_df = (
        _best_by_return(phase2_detail, ["symbol"]) if not phase2_detail.empty else pd.DataFrame()
    )
    # Final adopted = best of phase2 when available, else phase1
    if not best_long_df.empty:
        final_best = best_long_df.copy()
    else:
        final_best = best_short_df.copy()

    pivot_short = (
        phase1_detail.pivot_table(
            index="symbol", columns="ma_short", values="total_return_pct", aggfunc="first"
        )
        if not phase1_detail.empty
        else pd.DataFrame()
    )
    pivot_long = (
        phase2_detail.pivot_table(
            index="symbol", columns="ma_long", values="total_return_pct", aggfunc="first"
        )
        if not phase2_detail.empty
        else pd.DataFrame()
    )

    meta = pd.DataFrame(
        [
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "timeframe": timeframe,
                "bars": args.bars,
                "mc_sims": args.mc_sims,
                "strategy": "trend_following",
                "phase": args.phase,
                "ma_short_values": ",".join(str(v) for v in MA_SHORT_VALUES),
                "ma_long_values": ",".join(str(v) for v in MA_LONG_VALUES),
                "error_count": len(errors),
                "note": (
                    "Gates OFF; ADX maps fixed; MA one-at-a-time "
                    "(short w/ long=20, then long w/ best short); ATR skipped"
                ),
            }
        ]
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if not final_best.empty:
            final_best.to_excel(writer, sheet_name="BestPerSymbol", index=False)
        if not best_short_df.empty:
            best_short_df.to_excel(writer, sheet_name="BestMaShort", index=False)
        if not best_long_df.empty:
            best_long_df.to_excel(writer, sheet_name="BestMaLong", index=False)
        if not pivot_short.empty:
            pivot_short.to_excel(writer, sheet_name="ShortReturnPivot")
        if not pivot_long.empty:
            pivot_long.to_excel(writer, sheet_name="LongReturnPivot")
        detail.to_excel(writer, sheet_name="AllTrials", index=False)
        meta.to_excel(writer, sheet_name="RunInfo", index=False)
        if errors:
            pd.DataFrame({"error": errors}).to_excel(writer, sheet_name="Errors", index=False)

    print("\n=== Best MA per symbol (final) ===")
    cols = [
        "symbol",
        "ma_short",
        "ma_long",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "trades",
        "mc_prob_positive_pct",
    ]
    print(final_best[cols].to_string(index=False))
    print(f"\n  report saved : {out_path}")
    print(f"  errors       : {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
