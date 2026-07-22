from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.risk.budget import inverse_drawdown_weights, scale_weights_to_budget
from src.risk.drawdown import evaluate_drawdown
from src.risk.kelly import bayesian_kelly, full_kelly, kelly_sample_discount
from src.risk.stops import atr_stop_distance, fixed_stop_distance, vol_stop_distance


def print_lesson_content() -> None:
    print("\n=== Lesson 15: Risk Control and Money Management ===")
    print("  3 layers : position / portfolio / system")
    print("  Kelly    : full -> half -> Bayesian conservative + sample discount")
    print("  Stops    : fixed / ATR / volatility-adjusted")
    print("  Drawdown : warning 5% / stop 10% / circuit 15%")

    print("\n=== Kelly Paper Exercise ===")
    scenarios = [
        ("A", 0.60, 1.0),
        ("B", 0.55, 1.5),
        ("C", 0.50, 1.0),
        ("D", 0.45, 2.0),
        ("E", 0.70, 0.5),
    ]
    for name, p, b in scenarios:
        k = full_kelly(p, b)
        print(f"  {name}: p={p:.0%} b={b:.1f} -> Kelly {k:.1%}, Half {k/2:.1%}")

    print("\n=== Bayesian Kelly (60W/40L example) ===")
    result = bayesian_kelly(60, 40, 0.02, 0.015)
    print(f"  win rate est     : {result['p_estimate']:.1%}")
    p_low, p_high = result["p_interval"]
    print(f"  win rate 90% CI: [{p_low:.1%}, {p_high:.1%}]")
    print(f"  Kelly mean       : {result['kelly_mean']:.1%}")
    print(f"  Kelly conservative: {result['kelly_conservative']:.1%}")
    print(f"  recommendation   : {result['recommendation']:.1%}")

    print("\n=== Risk Budget (inverse drawdown) ===")
    drawdowns = {"A": 0.25, "B": 0.12, "C": 0.08}
    weights = inverse_drawdown_weights(drawdowns)
    scaled = scale_weights_to_budget(weights, max_portfolio_drawdown=0.15, strategy_drawdowns=drawdowns)
    for name in drawdowns:
        print(f"  {name}: raw={weights[name]:.1%} scaled={scaled[name]:.1%} (dd={drawdowns[name]:.0%})")

    print("\n=== Stop Loss Comparison ===")
    rows = [
        ("AAPL-like", 180.0, 0.015, 2.7),
        ("TSLA-like", 250.0, 0.035, 8.8),
        ("SPY-like", 450.0, 0.008, 3.6),
    ]
    for label, entry, daily_vol, atr in rows:
        fixed = entry - fixed_stop_distance(entry, 5.0)
        atr_stop = entry - atr_stop_distance(atr, 2.0)
        vol_stop = entry - vol_stop_distance(entry, daily_vol, 2.5)
        print(
            f"  {label}: fixed5%={fixed:.1f} atrx2={atr_stop:.1f} volx2.5={vol_stop:.1f}"
        )

    print("\n=== Drawdown Scenario ===")
    for dd in (0.0, 6.0, 12.0, 16.0):
        state = evaluate_drawdown(dd)
        print(
            f"  dd={dd:4.1f}% -> {state.level.value:8} action={state.action.value} "
            f"scale={state.position_scale:.0%} new_ok={state.new_positions_allowed}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk control and money management report (Lesson 15)")
    parser.add_argument("--paper-only", action="store_true", help="Lesson content only")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    print("\n=== Configured Limits ===")
    risk = config.risk
    print(f"  max single position : {risk.max_single_position_pct:.1f}%")
    print(f"  max symbol exposure : {risk.max_symbol_exposure_pct:.1f}%")
    print(f"  max sector exposure : {risk.max_sector_exposure_pct:.1f}%")
    print(f"  max total exposure  : {risk.max_total_exposure_pct:.1f}%")
    print(f"  drawdown warning    : {risk.drawdown_warning_pct:.1f}%")
    print(f"  drawdown stop       : {risk.drawdown_stop_pct:.1f}%")
    print(f"  drawdown circuit    : {risk.drawdown_circuit_pct:.1f}%")

    if config.decision.trade_wins is not None and config.decision.trade_losses is not None:
        discount = kelly_sample_discount(config.decision.trade_wins + config.decision.trade_losses)
        print(f"\n  Kelly sample discount: {discount:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
