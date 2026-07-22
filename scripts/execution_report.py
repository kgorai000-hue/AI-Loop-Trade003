from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.types import SignalSide
from src.execution.simulator import ConservativeExecutionSimulator
from src.execution.telemetry import ExecutionTelemetryStore
from src.market.symbol_info import MarketSymbolInfo, MarketType
from src.market.symbol_registry import SymbolRegistry


def print_lesson_content() -> None:
    print("\n=== Lesson 19: Execution System ===")
    print("  Real execution = price + slippage + latency + fill probability + friction")

    print("\n=== Close-Price Fantasy vs Bid/Ask (19.3.2) ===")
    config = load_config()
    sim = ConservativeExecutionSimulator(config)
    symbol_info = MarketSymbolInfo(
        symbol="EURUSD",
        market_type=MarketType.FOREX,
        bid=1.08500,
        ask=1.08510,
        spread_points=10,
        spread_price=0.00010,
        point=0.00001,
        digits=5,
        contract_size=100_000,
        tick_size=0.00001,
        tick_value=1.0,
        tick_value_profit=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        commission=0.0,
        commission_mode=0,
        swap_long=0.0,
        swap_short=0.0,
        currency_base="EUR",
        currency_profit="USD",
        currency_margin="EUR",
        trade_mode=4,
        volume_real=0.0,
    )
    comparison = sim.compare_close_vs_realistic(
        close_price=1.08505,
        symbol_info=symbol_info,
        side=SignalSide.BUY,
        lots=1.0,
        daily_volatility=0.008,
    )
    print(f"  fantasy (close)     : {comparison['fantasy_price']:.5f} slip={comparison['fantasy_slippage_pct']:.3f}%")
    print(f"  realistic (bid/ask) : {comparison['realistic_price']:.5f} slip={comparison['realistic_slippage_pct']:.3f}%")
    print(f"  execution bias      : {comparison['bias_pct']:.3f}%")

    print("\n=== Symbol Normalization (19.9) ===")
    registry = SymbolRegistry(config)
    for broker in ("#USSPX500", "EURUSD", "GOLD"):
        canonical = registry.to_canonical(broker)
        roundtrip = registry.to_broker(canonical)
        print(f"  {broker:12} -> {canonical} -> {roundtrip}")

    print("\n=== 3-Stage Evolution Path (19.8) ===")
    stages = [
        ("Stage 1", "Conservative simulator filters 80% of unrealistic strategies"),
        ("Stage 2", "Small live capital collects Type-B execution logs"),
        ("Stage 3", "Calibrate simulator against live slippage/fill samples"),
    ]
    for stage, desc in stages:
        print(f"  {stage}: {desc}")

    print("\n=== Type A vs Type B Data (19.5) ===")
    print("  Type A: market environment (order book, tick) - approximation only")
    print("  Type B: your agent's fills - required for calibration and RL")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execution system report (Lesson 19)")
    parser.add_argument("--paper-only", action="store_true")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    ex = config.execution
    print("\n=== Configured Execution System ===")
    print(f"  enabled             : {ex.enabled}")
    print(f"  simulator mode      : {ex.simulator_mode}")
    print(f"  latency ms          : {ex.latency_ms}")
    print(f"  max child orders    : {ex.max_child_orders}")
    print(f"  use bid/ask         : {ex.use_bid_ask_not_close}")
    print(f"  telemetry db        : {ex.telemetry_db_path}")

    store = ExecutionTelemetryStore(ex.telemetry_db_path)
    summary = store.summarize(limit=20)
    if summary.records:
        print("\n=== Recent Execution Telemetry ===")
        print(f"  records             : {len(summary.records)}")
        print(f"  avg slippage        : {summary.avg_slippage_pct:.4f}%")
        print(f"  avg fill ratio      : {summary.avg_fill_ratio:.0%}")
        print(f"  avg latency         : {summary.avg_latency_ms:.0f} ms")
        print(f"  partial fills       : {summary.partial_fill_count}")
    else:
        print("\n=== Recent Execution Telemetry ===")
        print("  (no records yet — run dry_run_pipeline to populate)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
