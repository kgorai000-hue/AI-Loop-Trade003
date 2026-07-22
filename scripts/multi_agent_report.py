from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.arbitration import vote_on_proposal
from src.agents.registry import evolution_stages, standard_agent_registry
from src.core.config import load_config
from src.core.types import (
    MarketRegime,
    RegimeAssessment,
    SignalMode,
    SignalSide,
    StrategyKind,
    TradeSignal,
)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_design_exercise() -> None:
    """Lesson 11 design exercise: evolution order for struggling single agent."""
    print("\n=== Lesson 11 Design Exercise ===")
    print("  Performance: 25% return, 18% max DD, range-market losses, 15% slippage")
    print("\n  Recommended evolution order:")
    print("    1. Risk Agent first   - 18% drawdown too high, protect capital")
    print("    2. Regime Agent next  - fix range-market losses")
    print("    3. Execution Agent last - optimize 15% slippage drag")
    print("\n  Rationale: protect capital before improving returns.")


def print_registry() -> None:
    print("\n=== Agent Registry (Lesson 11.4) ===")
    for spec in standard_agent_registry():
        resp = ", ".join(spec.responsibilities[:2])
        not_resp = ", ".join(spec.not_responsible[:2])
        print(f"  {spec.name:16} | {resp}")
        print(f"  {'':16}   NOT: {not_resp}")
        print(f"  {'':16}   metric: {spec.metric}")


def print_evolution() -> None:
    print("\n=== Evolution Path (Lesson 11.6) ===")
    for stage, desc in evolution_stages().items():
        print(f"  Stage {stage}: {desc}")


def print_voting_demo() -> None:
    print("\n=== Arbitration Demo (Lesson 11.3 voting) ===")
    signal = TradeSignal(
        symbol="AAPL",
        side=SignalSide.BUY,
        timeframe="H1",
        strength=0.8,
        mode=SignalMode.MOMENTUM,
        strategy=StrategyKind.TREND_FOLLOWING,
        reason="demo signal",
    )
    regime = RegimeAssessment(
        symbol="AAPL",
        regime=MarketRegime.BULL,
        annualized_volatility=0.2,
        recent_return=0.05,
        recommended_mode=SignalMode.MOMENTUM,
        reason="trend confirmed",
        selected_strategy=StrategyKind.TREND_FOLLOWING,
    )
    # Scenario: high exposure triggers risk -1, regime +1, signal +1 => net +1
    result = vote_on_proposal(signal, regime, exposure_pct=55.0, max_exposure_pct=60.0)
    print(f"  Signal Agent : +1 (buy proposal)")
    print(f"  Risk Agent   : +1 (exposure 55% < 60%)")
    print(f"  Regime Agent : +1 (trend aligned)")
    print(f"  Net score    : {result.net_score} -> {'APPROVED' if result.approved else 'REJECTED'}")

    tight = vote_on_proposal(signal, regime, exposure_pct=62.0, max_exposure_pct=60.0)
    print(f"\n  With exposure 62% (over limit):")
    for vote in tight.votes:
        print(f"    {vote.agent:12} vote={vote.vote:+d} | {vote.reason}")
    print(f"  Net score    : {tight.net_score} -> {'APPROVED' if tight.approved else 'REJECTED'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent architecture report (Lesson 11)")
    parser.add_argument("--demo-only", action="store_true", help="Print lesson content only")
    args = parser.parse_args()

    print_design_exercise()
    print_registry()
    print_evolution()
    print_voting_demo()

    if args.demo_only:
        return 0

    config = load_config()
    setup_logging(config.log_level)
    ma = config.multi_agent

    print(f"\n=== Config ===")
    print(f"  enabled           : {ma.enabled}")
    print(f"  evolution_stage   : {ma.evolution_stage} - {evolution_stages().get(ma.evolution_stage, '?')}")
    print(f"  arbitration_mode  : {ma.arbitration_mode}")
    print(f"  parallel_analysis : {ma.parallel_analysis}")
    print(f"  agent_timeout     : {ma.agent_timeout_seconds}s")
    print(f"  circuit_breaker   : {ma.circuit_breaker_threshold} failures")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
