"""Screen signal_score_threshold for feature_score (M30 screening mode)."""
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

# Default config is 0.15; PDF / parameter_spaces uses 0.05–0.30 step 0.05
SCORE_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25)

_LOG_THRESHOLD_RE = __import__("re").compile(r"^=== threshold=([\d.]+) ===")
_LOG_SYMBOL_RE = __import__("re").compile(r"^\s+\[\d+/\d+\]\s+(\S+)\s+\.\.\.")
_LOG_RESULT_RE = __import__("re").compile(
    r"ret=([-\d.]+%)\s+Sharpe=([-\d.]+)\s+trades=(\d+)"
)


def _pct_to_float(text: str) -> float:
    return round(float(text.rstrip("%")), 2)


def _read_log_text(log_path: Path) -> str:
    raw = log_path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_scoregrid_log(
    log_path: Path,
    *,
    timeframe: str,
    bars: int,
) -> tuple[list[dict], set[tuple[float, str]]]:
    """Rebuild completed trial rows from a prior run log (for resume)."""
    if not log_path.is_file():
        return [], set()

    rows: list[dict] = []
    completed: set[tuple[float, str]] = set()
    current_threshold: float | None = None
    pending_symbol: str | None = None

    for raw_line in _read_log_text(log_path).splitlines():
        line = raw_line.strip()
        threshold_match = _LOG_THRESHOLD_RE.match(line)
        if threshold_match:
            current_threshold = float(threshold_match.group(1))
            pending_symbol = None
            continue

        symbol_match = _LOG_SYMBOL_RE.match(raw_line)
        if symbol_match and current_threshold is not None:
            pending_symbol = symbol_match.group(1)
            continue

        result_match = _LOG_RESULT_RE.search(line)
        if result_match and current_threshold is not None and pending_symbol:
            symbol = pending_symbol
            key = (current_threshold, symbol)
            if key in completed:
                pending_symbol = None
                continue
            completed.add(key)
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars": bars,
                    "strategy": "feature_score",
                    "signal_score_threshold": current_threshold,
                    "total_return_pct": _pct_to_float(result_match.group(1)),
                    "ann_return_pct": None,
                    "sharpe": round(float(result_match.group(2)), 3),
                    "max_drawdown_pct": None,
                    "trades": int(result_match.group(3)),
                    "win_rate_pct": None,
                    "wf_rounds": None,
                    "wf_avg_test_sharpe": None,
                    "wf_positive_rounds_pct": None,
                    "oos_ratio": None,
                    "expected_live_pct": None,
                    "mc_prob_positive_pct": None,
                    "mc_p5_pct": None,
                    "resumed_from_log": True,
                }
            )
            pending_symbol = None

    return rows, completed


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="signal_score_threshold grid for feature_score")
    parser.add_argument("--timeframe", default="M30")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--mc-sims", type=int, default=100)
    parser.add_argument("--symbol", action="append")
    parser.add_argument(
        "--resume-log",
        type=Path,
        help="Skip (threshold, symbol) pairs already present in a prior run log",
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
        / f"data/reports/screening_{timeframe}_{args.bars}bars_nogate_scoregrid_{stamp}.xlsx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== signal_score_threshold grid ===")
    print(f"  symbols    : {len(symbols)}")
    print(f"  timeframe  : {timeframe}")
    print(f"  bars       : {args.bars}")
    print(f"  MC sims    : {args.mc_sims}")
    print("  strategy   : feature_score only")
    print(f"  thresholds : {', '.join(str(v) for v in SCORE_VALUES)}")
    if args.resume_log:
        print(f"  resume log : {args.resume_log}")
    print(f"  output     : {out_path}")

    regime_agent = RegimeAgent(config, store)
    rows: list[dict] = []
    errors: list[str] = []
    completed: set[tuple[float, str]] = set()
    if args.resume_log:
        prior_rows, completed = parse_scoregrid_log(
            args.resume_log, timeframe=timeframe, bars=args.bars
        )
        rows.extend(prior_rows)
        print(f"  resumed    : {len(completed)} completed trials from log", flush=True)

    for threshold in SCORE_VALUES:
        config.indicators.signal_score_threshold = float(threshold)
        agent = BacktestAgent(config, store)
        print(f"\n=== threshold={threshold} ===", flush=True)

        for idx, symbol in enumerate(symbols, start=1):
            if (float(threshold), symbol) in completed:
                print(f"  [{idx}/{len(symbols)}] {symbol} ... SKIP (log)", flush=True)
                continue
            print(f"  [{idx}/{len(symbols)}] {symbol} ...", flush=True)
            try:
                regime = regime_agent.assess(symbol)
                market_regime = regime.regime if regime else None
                result = agent.validate_strategy(
                    symbol, "feature_score", timeframe, market_regime
                )
                perf = result.backtest.performance
                wf = result.walk_forward_summary
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "bars": args.bars,
                        "strategy": "feature_score",
                        "signal_score_threshold": threshold,
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
                        "resumed_from_log": False,
                    }
                )
                print(
                    f"      ret={perf.total_return:.2%} Sharpe={perf.sharpe_ratio:.3f} "
                    f"trades={perf.trades}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"threshold={threshold} {symbol}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                print(f"      ERROR: {exc}", flush=True)

    detail = pd.DataFrame(rows)
    if detail.empty:
        print("ERROR: no results")
        return 1

    # Prefer positive-trade trials; if all zero trades, still pick best return
    best_rows: list[dict] = []
    for symbol, group in detail.groupby("symbol"):
        with_trades = group[group["trades"] > 0]
        pool = with_trades if not with_trades.empty else group
        best_rows.append(pool.loc[pool["total_return_pct"].idxmax()].to_dict())
    best = pd.DataFrame(best_rows).sort_values("total_return_pct", ascending=False)

    pivot_ret = detail.pivot_table(
        index="symbol",
        columns="signal_score_threshold",
        values="total_return_pct",
        aggfunc="first",
    )
    pivot_trades = detail.pivot_table(
        index="symbol",
        columns="signal_score_threshold",
        values="trades",
        aggfunc="first",
    )

    meta = pd.DataFrame(
        [
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "timeframe": timeframe,
                "bars": args.bars,
                "mc_sims": args.mc_sims,
                "strategy": "feature_score",
                "thresholds": ",".join(str(v) for v in SCORE_VALUES),
                "resume_log": str(args.resume_log) if args.resume_log else "",
                "resumed_trials": len(completed),
                "error_count": len(errors),
                "note": "Gates OFF; feature_score only; best prefers trades>0 then max return",
            }
        ]
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        best.to_excel(writer, sheet_name="BestPerSymbol", index=False)
        pivot_ret.to_excel(writer, sheet_name="ReturnPivot")
        pivot_trades.to_excel(writer, sheet_name="TradesPivot")
        detail.to_excel(writer, sheet_name="AllTrials", index=False)
        meta.to_excel(writer, sheet_name="RunInfo", index=False)
        if errors:
            pd.DataFrame({"error": errors}).to_excel(writer, sheet_name="Errors", index=False)

    print("\n=== Best threshold per symbol (by return, prefer trades>0) ===")
    print(
        best[
            [
                "symbol",
                "signal_score_threshold",
                "total_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "trades",
                "mc_prob_positive_pct",
            ]
        ].to_string(index=False)
    )
    print(f"\n  report saved : {out_path}")
    print(f"  errors       : {len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
