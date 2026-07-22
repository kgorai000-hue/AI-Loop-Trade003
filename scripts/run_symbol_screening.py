"""Run backtest validation for all symbols sequentially (screening mode)."""
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

from scripts.screening_params import BEST_MR_BB_BY_SYMBOL, BEST_MR_RSI_BY_SYMBOL
from src.agents.backtest_agent import BacktestAgent, ValidationReport
from src.agents.regime_agent import RegimeAgent
from src.core.config import load_config
from src.data.store import OHLCVStore

SCREENING_BARS = 2000
SCREENING_MC_SIMS = 100


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def apply_screening_config(
    config,
    bars: int,
    mc_sims: int,
    timeframe: str,
    *,
    mr_rsi: tuple[float, float] | None = None,
    bb_entry: tuple[float, float] | None = None,
):
    config.data.history_years = None
    config.data.history_bars_by_timeframe = {timeframe: bars}
    config.history_bars = bars
    config.backtest.monte_carlo_simulations = mc_sims
    if mr_rsi is not None:
        oversold, overbought = mr_rsi
        config.strategies.mr_rsi_oversold = oversold
        config.strategies.mr_rsi_overbought = overbought
    if bb_entry is not None:
        low, high = bb_entry
        config.strategies.mr_bb_entry_low = low
        config.strategies.mr_bb_entry_high = high
    return config


def validate_symbol_sequential(
    config,
    store: OHLCVStore,
    symbols: list[str],
    timeframe: str,
    *,
    ignore_gates: bool = False,
    rsi_by_symbol: dict[str, tuple[float, float]] | None = None,
    bb_by_symbol: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[ValidationReport | None], list[str]]:
    agent = BacktestAgent(config, store)
    regime_agent = RegimeAgent(config, store)
    reports: list[ValidationReport | None] = []
    errors: list[str] = []

    for idx, symbol in enumerate(symbols, start=1):
        notes: list[str] = []
        if rsi_by_symbol and symbol in rsi_by_symbol:
            oversold, overbought = rsi_by_symbol[symbol]
            config.strategies.mr_rsi_oversold = oversold
            config.strategies.mr_rsi_overbought = overbought
            notes.append(f"RSI={oversold:g}/{overbought:g}")
        if bb_by_symbol and symbol in bb_by_symbol:
            low, high = bb_by_symbol[symbol]
            config.strategies.mr_bb_entry_low = low
            config.strategies.mr_bb_entry_high = high
            notes.append(f"BB={low:g}/{high:g}")
        note = f" {' '.join(notes)}" if notes else ""
        print(
            f"\n[{idx}/{len(symbols)}] Validating {symbol} {timeframe}{note} ...",
            flush=True,
        )
        try:
            regime = regime_agent.assess(symbol)
            market_regime = regime.regime if regime else None
            report = agent.validate_symbol(symbol, timeframe, market_regime)
            reports.append(report)
            if ignore_gates:
                best = max(
                    report.strategies,
                    key=lambda s: s.backtest.performance.sharpe_ratio,
                )
                print(
                    f"  -> gates OFF | best={best.strategy_name} "
                    f"Sharpe={best.backtest.performance.sharpe_ratio:.3f} "
                    f"ret={best.backtest.performance.total_return:.2%}",
                    flush=True,
                )
            else:
                n_pass = sum(1 for s in report.strategies if s.quality_gate.passed)
                print(f"  -> {n_pass}/3 strategies passed quality gate", flush=True)
        except ValueError as exc:
            msg = f"{symbol}: {exc}"
            errors.append(msg)
            reports.append(None)
            print(f"  -> ERROR: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = f"{symbol}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            reports.append(None)
            print(f"  -> ERROR: {exc}", flush=True)

    return reports, errors


def build_summary_rows(
    symbols: list[str],
    reports: list[ValidationReport | None],
    timeframe: str,
    bars: int,
    *,
    ignore_gates: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for symbol, report in zip(symbols, reports):
        if report is None:
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars": bars,
                    "status": "ERROR",
                    "n_pass": 0,
                    "best_strategy": "",
                    "best_sharpe": None,
                    "best_wf_test_sharpe": None,
                    "trend_gate": "ERROR",
                    "mean_rev_gate": "ERROR",
                    "feature_gate": "ERROR",
                }
            )
            continue

        if ignore_gates:
            passed = {s.strategy_name: True for s in report.strategies}
        else:
            passed = {s.strategy_name: s.quality_gate.passed for s in report.strategies}
        sharpes = {s.strategy_name: s.backtest.performance.sharpe_ratio for s in report.strategies}
        wf_sharpes = {
            s.strategy_name: float(s.walk_forward_summary.get("avg_test_sharpe", 0.0))
            for s in report.strategies
        }
        best = max(sharpes, key=sharpes.get)
        gate_label = "N/A" if ignore_gates else None
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": bars,
                "status": "OK",
                "n_pass": sum(1 for ok in passed.values() if ok),
                "best_strategy": best,
                "best_sharpe": round(sharpes[best], 3),
                "best_wf_test_sharpe": round(wf_sharpes[best], 3),
                "trend_gate": gate_label or ("PASS" if passed.get("trend_following") else "FAIL"),
                "mean_rev_gate": gate_label or ("PASS" if passed.get("mean_reversion") else "FAIL"),
                "feature_gate": gate_label or ("PASS" if passed.get("feature_score") else "FAIL"),
            }
        )
    return rows


def build_detail_rows(
    symbols: list[str],
    reports: list[ValidationReport | None],
    timeframe: str,
    bars: int,
    *,
    ignore_gates: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for symbol, report in zip(symbols, reports):
        if report is None:
            continue
        for s in report.strategies:
            perf = s.backtest.performance
            wf = s.walk_forward_summary
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars": bars,
                    "strategy": s.strategy_name,
                    "gate": "N/A" if ignore_gates else ("PASS" if s.quality_gate.passed else "FAIL"),
                    "gate_passes": (
                        len(s.quality_gate.checks)
                        if ignore_gates
                        else sum(1 for c in s.quality_gate.checks if c.passed)
                    ),
                    "gate_total": len(s.quality_gate.checks),
                    "total_return_pct": round(perf.total_return * 100, 2),
                    "ann_return_pct": round(perf.annualized_return * 100, 2),
                    "sharpe": round(perf.sharpe_ratio, 3),
                    "max_drawdown_pct": round(perf.max_drawdown * 100, 2),
                    "trades": perf.trades,
                    "win_rate_pct": round(perf.win_rate * 100, 1),
                    "wf_rounds": int(wf.get("rounds", 0)),
                    "wf_avg_test_sharpe": round(float(wf.get("avg_test_sharpe", 0.0)), 3),
                    "wf_positive_rounds_pct": round(float(wf.get("positive_rounds_pct", 0.0)) * 100, 1),
                    "oos_ratio": round(s.oos.oos_ratio, 3),
                    "oos_test_return_pct": round(s.oos.test_return * 100, 2),
                    "expected_live_pct": round(s.quality_gate.live_expected_return * 100, 2),
                    "mc_prob_positive_pct": round(s.monte_carlo.prob_positive * 100, 1),
                    "mc_p5_pct": round(s.monte_carlo.percentile_5 * 100, 2),
                }
            )
    return rows


def build_gate_failure_rows(
    symbols: list[str],
    reports: list[ValidationReport | None],
    *,
    ignore_gates: bool = False,
) -> list[dict]:
    if ignore_gates:
        return []
    rows: list[dict] = []
    for symbol, report in zip(symbols, reports):
        if report is None:
            continue
        for s in report.strategies:
            for check in s.quality_gate.checks:
                if not check.passed:
                    rows.append(
                        {
                            "symbol": symbol,
                            "strategy": s.strategy_name,
                            "check_id": check.check_id,
                            "check_name": check.name,
                            "detail": check.detail,
                        }
                    )
    return rows


def export_excel(
    output_path: Path,
    summary: list[dict],
    detail: list[dict],
    gate_failures: list[dict],
    errors: list[str],
    meta: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(detail).to_excel(writer, sheet_name="StrategyDetail", index=False)
        if gate_failures:
            pd.DataFrame(gate_failures).to_excel(writer, sheet_name="GateFailures", index=False)
        if errors:
            pd.DataFrame({"error": errors}).to_excel(writer, sheet_name="Errors", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="RunInfo", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen all symbols sequentially and export Excel report")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")
    parser.add_argument("--bars", type=int, default=SCREENING_BARS, help="Bars per symbol")
    parser.add_argument("--mc-sims", type=int, default=SCREENING_MC_SIMS, help="Monte Carlo simulations")
    parser.add_argument("--output", help="Output xlsx path")
    parser.add_argument("--symbol", action="append", help="Limit symbol(s)")
    parser.add_argument(
        "--ignore-gates",
        action="store_true",
        help="Disable quality-gate pass/fail filtering; report all strategies as eligible",
    )
    parser.add_argument(
        "--mr-rsi",
        help="Mean-reversion RSI bands as oversold,overbought (e.g. 25,75)",
    )
    parser.add_argument(
        "--bb-entry",
        help="Mean-reversion BB entry bands as low,high (e.g. 0.20,0.80)",
    )
    parser.add_argument(
        "--use-best-rsi-map",
        action="store_true",
        help="Apply per-symbol best RSI from prior M30 screening (overrides --mr-rsi)",
    )
    parser.add_argument(
        "--use-best-bb-map",
        action="store_true",
        help="Apply per-symbol best BB entry from prior M30 screening (overrides --bb-entry)",
    )
    args = parser.parse_args()

    mr_rsi: tuple[float, float] | None = None
    if args.mr_rsi:
        parts = [p.strip() for p in args.mr_rsi.split(",")]
        if len(parts) != 2:
            print("ERROR: --mr-rsi must be oversold,overbought (e.g. 25,75)")
            return 1
        mr_rsi = (float(parts[0]), float(parts[1]))

    bb_entry: tuple[float, float] | None = None
    if args.bb_entry:
        parts = [p.strip() for p in args.bb_entry.split(",")]
        if len(parts) != 2:
            print("ERROR: --bb-entry must be low,high (e.g. 0.20,0.80)")
            return 1
        bb_entry = (float(parts[0]), float(parts[1]))
        if not (0.0 <= bb_entry[0] < bb_entry[1] <= 1.0):
            print("ERROR: --bb-entry requires 0 <= low < high <= 1")
            return 1

    rsi_by_symbol: dict[str, tuple[float, float]] | None = None
    if args.use_best_rsi_map:
        rsi_by_symbol = BEST_MR_RSI_BY_SYMBOL
        mr_rsi = None

    bb_by_symbol: dict[str, tuple[float, float]] | None = None
    if args.use_best_bb_map:
        bb_by_symbol = BEST_MR_BB_BY_SYMBOL
        bb_entry = None

    config = load_config()
    setup_logging(config.log_level)
    apply_screening_config(
        config,
        args.bars,
        args.mc_sims,
        args.timeframe.upper(),
        mr_rsi=mr_rsi,
        bb_entry=bb_entry,
    )

    store = OHLCVStore(config.storage.path)
    symbols = args.symbol or config.symbols
    timeframe = args.timeframe.upper()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gate_tag = "_nogate" if args.ignore_gates else ""
    if args.use_best_rsi_map:
        rsi_tag = "_rsibest"
    elif mr_rsi:
        rsi_tag = f"_rsi{int(mr_rsi[0])}{int(mr_rsi[1])}"
    else:
        rsi_tag = ""
    if args.use_best_bb_map:
        bb_tag = "_bbbest"
    elif bb_entry:
        bb_tag = f"_bb{int(bb_entry[0]*100):02d}{int(bb_entry[1]*100):02d}"
    else:
        bb_tag = ""
    output_path = Path(
        args.output
        or PROJECT_ROOT
        / f"data/reports/screening_{timeframe}_{args.bars}bars{gate_tag}{rsi_tag}{bb_tag}_{stamp}.xlsx"
    )

    print("=== Symbol Screening (sequential) ===")
    print(f"  symbols    : {len(symbols)}")
    print(f"  timeframe  : {timeframe}")
    print(f"  bars       : {args.bars}")
    print(f"  MC sims    : {args.mc_sims}")
    print(f"  gates      : {'OFF (ignored)' if args.ignore_gates else 'ON'}")
    if args.use_best_rsi_map:
        print("  MR RSI     : per-symbol best map (from prior M30 screening)")
    elif mr_rsi:
        print(f"  MR RSI     : {mr_rsi[0]}/{mr_rsi[1]} (oversold/overbought)")
    if args.use_best_bb_map:
        print("  BB entry   : per-symbol best map (from prior M30 screening)")
    elif bb_entry:
        print(f"  BB entry   : {bb_entry[0]}/{bb_entry[1]} (low/high)")
    else:
        print(
            f"  BB entry   : {config.strategies.mr_bb_entry_low}/"
            f"{config.strategies.mr_bb_entry_high} (default)"
        )
    print(f"  output     : {output_path}")

    reports, errors = validate_symbol_sequential(
        config,
        store,
        symbols,
        timeframe,
        ignore_gates=args.ignore_gates,
        rsi_by_symbol=rsi_by_symbol,
        bb_by_symbol=bb_by_symbol,
    )

    summary = build_summary_rows(
        symbols, reports, timeframe, args.bars, ignore_gates=args.ignore_gates
    )
    detail = build_detail_rows(
        symbols, reports, timeframe, args.bars, ignore_gates=args.ignore_gates
    )
    gate_failures = build_gate_failure_rows(
        symbols, reports, ignore_gates=args.ignore_gates
    )

    note = "Screening mode: quality gates DISABLED for this run"
    if not args.ignore_gates:
        note = "Screening mode: not for final adoption; use 9000+ bars for full validation"

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "bars_per_symbol": str(args.bars),
        "monte_carlo_simulations": str(args.mc_sims),
        "symbol_count": str(len(symbols)),
        "error_count": str(len(errors)),
        "quality_gates": "OFF" if args.ignore_gates else "ON",
        "mr_rsi_mode": "best_map" if args.use_best_rsi_map else ("uniform" if mr_rsi else "default"),
        "mr_rsi_oversold": str(mr_rsi[0]) if mr_rsi else ("map" if args.use_best_rsi_map else "default"),
        "mr_rsi_overbought": str(mr_rsi[1]) if mr_rsi else ("map" if args.use_best_rsi_map else "default"),
        "bb_entry_mode": "best_map" if args.use_best_bb_map else ("uniform" if bb_entry else "default"),
        "bb_entry_low": str(bb_entry[0]) if bb_entry else ("map" if args.use_best_bb_map else str(config.strategies.mr_bb_entry_low)),
        "bb_entry_high": str(bb_entry[1]) if bb_entry else ("map" if args.use_best_bb_map else str(config.strategies.mr_bb_entry_high)),
        "cells_pass_ge1": str(sum(1 for r in summary if r.get("n_pass", 0) >= 1)),
        "note": note,
    }

    export_excel(output_path, summary, detail, gate_failures, errors, meta)

    print("\n=== Screening Complete ===")
    print(f"  report saved : {output_path}")
    print(f"  errors       : {len(errors)}")
    if args.ignore_gates:
        print(f"  gates        : OFF (all strategies reported without gate filter)")
    else:
        print(f"  >=1 pass     : {meta['cells_pass_ge1']}/{len(symbols)}")
    return 1 if len(errors) == len(symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
