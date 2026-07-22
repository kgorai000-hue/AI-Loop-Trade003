from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.execution.costs.impact import almgren_chriss_total_cost
from src.execution.costs.slippage import estimate_slippage_from_orderbook, linear_slippage, sqrt_slippage
from src.execution.costs.tradability import evaluate_strategy_tradability, opportunity_cost
from src.execution.costs.types import OrderBookLevel


def print_lesson_content() -> None:
    print("\n=== Lesson 18: Trading Costs and Tradability ===")
    print("  Explicit: commission | Implicit: slippage, impact, opportunity cost")

    print("\n=== Slippage Models ===")
    order_size = 5_000_000
    adv = 100_000_000
    sigma = 0.02
    print(f"  linear : {linear_slippage(order_size, adv, 0.3):.4%}")
    print(f"  sqrt   : {sqrt_slippage(order_size, adv, sigma, 1.0):.4%}")

    print("\n=== Order Book Walk (Lesson 18.2.3) ===")
    bids = [
        OrderBookLevel(100.00, 1000),
        OrderBookLevel(99.95, 2000),
        OrderBookLevel(99.90, 3000),
    ]
    asks = [
        OrderBookLevel(100.05, 1000),
        OrderBookLevel(100.10, 2000),
        OrderBookLevel(100.15, 3000),
    ]
    for size in (500, 1500, 4000):
        slip = estimate_slippage_from_orderbook(size, bids, asks, side="buy")
        print(f"  buy {size:5d} units -> slippage {slip:.4%}")

    print("\n=== Almgren-Chriss Tradeoff ===")
    for urgency in (0.5, 1.0, 2.0):
        ac = almgren_chriss_total_cost(participation=0.02, sigma=0.02, urgency=urgency)
        print(
            f"  urgency={urgency:.1f}: temp={ac['temporary_impact']:.4%} "
            f"perm={ac['permanent_impact']:.4%} vol={ac['volatility_risk']:.4%}"
        )

    print("\n=== Turnover Shredder (300% daily turnover) ===")
    scenarios = [
        ("low turnover", 0.5, 0.08),
        ("medium", 1.0, 0.08),
        ("high (300%)", 3.0, 0.08),
    ]
    gross_alpha = 15.0
    for name, turnover, cost_per_trade in scenarios:
        report = evaluate_strategy_tradability(
            name, gross_alpha, turnover, cost_per_trade, trading_days=252
        )
        print(
            f"  {name:16} gross={report.gross_alpha_pct:.0f}% "
            f"annual_cost={report.annual_cost_pct:.0f}% "
            f"net={report.net_alpha_pct:.0f}% tradable={report.tradable}"
        )

    print("\n=== Signal Decay vs Execution Delay ===")
    gross = 0.5
    for delay in (5, 30, 120):
        opp = opportunity_cost(gross, 120.0, float(delay))
        print(f"  delay={delay:3d}min -> opportunity cost {opp:.3f}% (gross {gross:.2f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Trading costs and tradability report (Lesson 18)")
    parser.add_argument("--paper-only", action="store_true")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    costs = config.costs
    print("\n=== Configured Cost Estimator ===")
    print(f"  enabled             : {costs.enabled}")
    print(f"  slippage model      : {costs.slippage_model}")
    print(f"  default ADV         : {costs.default_adv_notional:,.0f}")
    print(f"  max order/ADV       : {costs.max_order_adv_ratio:.2%}")
    print(f"  block untradable    : {costs.block_untradable}")
    print(f"  signal halflife     : {costs.signal_decay_halflife_minutes:.0f} min")
    print(f"  execution delay     : {costs.execution_delay_minutes:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
