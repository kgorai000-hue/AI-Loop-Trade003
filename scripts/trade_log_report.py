from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.trading_log.store import LiveTradeLogStore

APPENDIX_A_CHECKLIST = [
    ("Order", "order_id, symbol, side, order_type, order_price, order_qty, submit_ts"),
    ("Fill", "fill_id, order_id, fill_price, fill_qty, fill_ts (ms)"),
    ("Metrics", "expected_price, slippage, latency_ms, fill_ratio, commission, tax, realized_pnl"),
    ("Market", "bar OHLC/VWAP, atr_5min, bid1/ask1"),
    ("Agent", "agent_id/version, action, target_position, confidence"),
]


def print_lesson_content() -> None:
    print("\n=== Appendix A: Live Trading Logging Standards ===")
    print("  Closed loop: order -> fill -> metrics (+ market snapshot + agent meta)\n")

    print("=== Required Fields (A1) ===")
    for level, fields in APPENDIX_A_CHECKLIST:
        print(f"  [{level:7}] {fields}")

    print("\n=== Storage Layout ===")
    print("  SQLite tables: orders, fills, trade_metrics, market_snapshots, agent_decisions")
    print("  Config: trade_log.enabled, trade_log.db_path, trade_log.agent_version")


def main() -> int:
    parser = argparse.ArgumentParser(description="Appendix A trade log report")
    parser.add_argument("--paper-only", action="store_true", help="Show standards checklist only")
    parser.add_argument("--limit", type=int, default=10, help="Recent orders to display")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    print("\n=== Configuration ===")
    print(f"  enabled       : {config.trade_log.enabled}")
    print(f"  db_path       : {config.trade_log.db_path}")
    print(f"  agent_version : {config.trade_log.agent_version}")

    if not config.trade_log.enabled:
        print("\n  trade_log.enabled is false - enable in config/settings.yaml")
        return 0

    store = LiveTradeLogStore(config.trade_log.db_path)
    summary = store.summarize(args.limit)
    print("\n=== Summary ===")
    print(f"  orders (sample) : {summary.order_count}")
    print(f"  total fills     : {summary.fill_count}")
    print(f"  avg slippage    : {summary.avg_slippage_pct:.4f}%")
    print(f"  avg fill ratio  : {summary.avg_fill_ratio:.2%}")
    print(f"  avg latency     : {summary.avg_latency_ms:.0f} ms")
    print(f"  total commission: {summary.total_commission:.2f} JPY")

    orders = store.recent_orders(args.limit)
    if not orders:
        print("\n=== Recent Orders ===")
        print("  (empty - run `python main.py run` in dry_run mode to populate)")
        return 0

    print(f"\n=== Recent Orders (last {len(orders)}) ===")
    for order in orders:
        fills = store.fills_for_order(order.order_id)
        print(
            f"  {order.order_id[:8]}… {order.side:4} {order.symbol:10} "
            f"qty={order.order_qty:.2f} status={order.status} fills={len(fills)} "
            f"dry_run={order.dry_run}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
