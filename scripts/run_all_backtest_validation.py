from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.backtest_agent import BacktestAgent, ValidationReport
from src.agents.regime_agent import RegimeAgent
from src.backtest.gate_registry import GateRegistry
from src.core.config import load_config
from src.data.store import OHLCVStore

DEFAULT_TIMEFRAMES = ("M15", "M30", "H1", "H4", "D1")


@dataclass
class SymbolTimeframeResult:
    symbol: str
    timeframe: str
    trend_pass: bool = False
    mean_rev_pass: bool = False
    feature_pass: bool = False
    best_sharpe: float = 0.0
    best_strategy: str = ""
    n_pass: int = 0
    error: str | None = None


@dataclass
class ValidationMatrix:
    timeframes: list[str] = field(default_factory=list)
    results: list[SymbolTimeframeResult] = field(default_factory=list)

    def for_timeframe(self, timeframe: str) -> list[SymbolTimeframeResult]:
        return [r for r in self.results if r.timeframe == timeframe]

    def for_symbol(self, symbol: str) -> list[SymbolTimeframeResult]:
        return [r for r in self.results if r.symbol == symbol]


def gate_label(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def parse_timeframes(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_TIMEFRAMES)
    return [tf.strip().upper() for tf in raw.split(",") if tf.strip()]


def validate_matrix(
    config,
    store: OHLCVStore,
    symbols: list[str],
    timeframes: list[str],
) -> tuple[ValidationMatrix, list[ValidationReport]]:
    agent = BacktestAgent(config, store)
    regime_agent = RegimeAgent(config, store)
    matrix = ValidationMatrix(timeframes=timeframes)
    reports: list[ValidationReport] = []

    for timeframe in timeframes:
        for symbol in symbols:
            row = SymbolTimeframeResult(symbol=symbol, timeframe=timeframe)
            try:
                regime = regime_agent.assess(symbol)
                market_regime = regime.regime if regime else None
                report = agent.validate_symbol(symbol, timeframe, market_regime)
                reports.append(report)
            except ValueError as exc:
                row.error = str(exc)
                matrix.results.append(row)
                continue
            except Exception as exc:
                row.error = f"{type(exc).__name__}: {exc}"
                matrix.results.append(row)
                continue

            passed = {r.strategy_name: r.quality_gate.passed for r in report.strategies}
            sharpes = {r.strategy_name: r.backtest.performance.sharpe_ratio for r in report.strategies}
            row.trend_pass = passed.get("trend_following", False)
            row.mean_rev_pass = passed.get("mean_reversion", False)
            row.feature_pass = passed.get("feature_score", False)
            row.n_pass = sum(1 for ok in passed.values() if ok)
            row.best_strategy = max(sharpes, key=sharpes.get)
            row.best_sharpe = sharpes[row.best_strategy]
            matrix.results.append(row)

    return matrix, reports


def save_gate_cache(
    config,
    reports: list[ValidationReport],
    timeframes: list[str],
) -> None:
    signal_tf = config.stats.signal_timeframe
    if signal_tf not in timeframes:
        return
    tf_reports = [r for r in reports if r.timeframe == signal_tf]
    if not tf_reports:
        return
    registry = GateRegistry.merge_reports(tf_reports, signal_tf)
    registry.save(config.backtest.gate_cache_path)
    print(
        f"\nGate registry saved: {config.backtest.gate_cache_path} "
        f"({registry.summary(enabled=True).passed_count}/{len(registry.entries)} passed, TF={signal_tf})"
    )


def print_timeframe_section(matrix: ValidationMatrix, timeframe: str) -> None:
    rows = matrix.for_timeframe(timeframe)
    print(f"\n=== Timeframe: {timeframe} ===")
    print(f"{'Symbol':14} {'Trend':6} {'MeanRev':7} {'Feature':7} {'BestSharpe':10} Note")
    print("-" * 78)

    validated = 0
    any_pass = 0
    for row in rows:
        if row.error:
            print(f"{row.symbol:14} ERROR    -        -         -          {row.error}")
            continue
        validated += 1
        if row.n_pass > 0:
            any_pass += 1
        note = f"{row.n_pass}/3 pass, best={row.best_strategy}({row.best_sharpe:.2f})"
        print(
            f"{row.symbol:14} "
            f"{gate_label(row.trend_pass):6} "
            f"{gate_label(row.mean_rev_pass):7} "
            f"{gate_label(row.feature_pass):7} "
            f"{row.best_sharpe:10.2f} "
            f"{note}"
        )

    print(f"  -> validated {validated}/{len(rows)}, >=1 gate pass: {any_pass}")


def print_pass_matrix(matrix: ValidationMatrix, symbols: list[str]) -> None:
    print("\n=== Pass Count Matrix (symbol x timeframe, max 3) ===")
    header = f"{'Symbol':14}" + "".join(f"{tf:>6}" for tf in matrix.timeframes)
    print(header)
    print("-" * (14 + 6 * len(matrix.timeframes)))

    best_cells: list[tuple[str, str, int, float]] = []

    for symbol in symbols:
        cells = []
        for tf in matrix.timeframes:
            match = [r for r in matrix.results if r.symbol == symbol and r.timeframe == tf]
            if not match:
                cells.append("  -  ")
                continue
            row = match[0]
            if row.error:
                cells.append(" ERR ")
            else:
                cells.append(f"{row.n_pass:>3}/3")
                if row.n_pass > 0:
                    best_cells.append((symbol, tf, row.n_pass, row.best_sharpe))
        print(f"{symbol:14}" + "".join(f"{c:>6}" for c in cells))

    print("\n=== Best Combinations (>=1 gate pass) ===")
    if not best_cells:
        print("  (none)")
        return
    best_cells.sort(key=lambda x: (x[2], x[3]), reverse=True)
    for symbol, tf, n_pass, sharpe in best_cells[:15]:
        print(f"  {symbol:14} {tf:4} {n_pass}/3 pass  best_sharpe={sharpe:.2f}")


def print_summary(matrix: ValidationMatrix, symbols: list[str]) -> None:
    total = len(matrix.results)
    errors = [r for r in matrix.results if r.error]
    ok = [r for r in matrix.results if not r.error]
    full_pass = [r for r in ok if r.n_pass == 3]
    any_pass = [r for r in ok if r.n_pass > 0]

    print("\n=== Overall Summary ===")
    print(f"  symbol x timeframe cells : {total}")
    print(f"  errors                   : {len(errors)}")
    print(f"  cells with >=1 pass      : {len(any_pass)}")
    print(f"  cells with 3/3 pass      : {len(full_pass)}")

    if errors:
        print("\n  Errors (first 10):")
        for row in errors[:10]:
            print(f"    {row.symbol} {row.timeframe}: {row.error}")


def print_data_availability(
    store: OHLCVStore,
    symbols: list[str],
    timeframes: list[str],
    config,
) -> list[tuple[str, str, int, int]]:
    """Report stored bar counts; return shortfalls (actual < requested)."""
    print("\n=== Data Availability (per-timeframe targets) ===")
    shortfalls: list[tuple[str, str, int, int]] = []
    for timeframe in timeframes:
        requested = config.history_bars_for(timeframe)
        for symbol in symbols:
            bars = store.get_recent_bars(symbol, timeframe, requested)
            actual = len(bars)
            flag = "OK" if actual >= requested else "LOW"
            if actual < requested:
                shortfalls.append((symbol, timeframe, actual, requested))
            print(f"  [{flag:3}] {symbol:14} {timeframe:4} {actual:5}/{requested}")
    if shortfalls:
        print(
            f"\n  {len(shortfalls)} cell(s) below target "
            "(MT5/broker history limit) - validating with available bars."
        )
    else:
        print("\n  All cells meet target bar count.")
    return shortfalls


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest validation across all symbols and timeframes"
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(DEFAULT_TIMEFRAMES),
        help=f"Comma-separated timeframes (default: {','.join(DEFAULT_TIMEFRAMES)})",
    )
    parser.add_argument("--symbol", action="append", help="Limit to symbol(s)")
    args = parser.parse_args()

    config = load_config()
    store = OHLCVStore(config.storage.path)
    symbols = args.symbol or config.symbols
    timeframes = parse_timeframes(args.timeframes)

    print("=== Backtest Validation: ALL SYMBOLS x TIMEFRAMES ===")
    print(f"Symbols    : {len(symbols)}")
    print(f"Timeframes : {', '.join(timeframes)}")
    print(f"History    : {config.data.history_years or 'n/a'} years (fallback bars={config.data.history_bars})")
    for tf in timeframes:
        print(f"  {tf:4} target -> {config.history_bars_for(tf)} bars")
    print("Strategies : trend_following, mean_reversion, feature_score")
    print("Gates      : quality gate (saves cache for signal_timeframe)")

    shortfalls = print_data_availability(store, symbols, timeframes, config)

    matrix, reports = validate_matrix(config, store, symbols, timeframes)

    for timeframe in timeframes:
        print_timeframe_section(matrix, timeframe)

    print_pass_matrix(matrix, symbols)
    print_summary(matrix, symbols)
    save_gate_cache(config, reports, timeframes)

    if shortfalls:
        print("\n=== Bar Count Shortfalls (actual < requested) ===")
        for symbol, tf, actual, requested in shortfalls:
            print(f"  {symbol:14} {tf:4} {actual}/{requested} bars")

    has_errors = any(r.error for r in matrix.results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
