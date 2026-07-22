from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.online.alpha_decay import expected_annual_return_from_ic, projected_ic
from src.online.decision import RetrainDecisionEngine
from src.online.drift import calculate_psi, sliding_accuracy_drift
from src.online.ewm_model import ExponentialMovingModel, effective_lookback_days
from src.online.scheduler import AdaptiveUpdateScheduler
from src.online.signal_adaptation import dynamic_signal_threshold
from src.online.strategy_lifecycle import inverse_vol_strategy_weights, sharpe_proportional_weights


def print_lesson_content() -> None:
    print("\n=== Lesson 17: Online Learning and Strategy Evolution ===")
    print("  Concept drift: static models decay as markets change")
    print("  Modes: sliding window / incremental / exponential forgetting")

    print("\n=== Alpha Decay Table (IC=0.05, 5%/month) ===")
    for month in (0, 6, 12, 18, 24):
        ic = projected_ic(0.05, 0.05, month)
        ann = expected_annual_return_from_ic(ic)
        print(f"  month {month:2d}: IC={ic:.3f} expected ann return={ann:.1%}")

    print("\n=== Effective Lookback (lambda) ===")
    for lam in (0.90, 0.95, 0.99):
        print(f"  lambda={lam:.2f} -> ~{effective_lookback_days(lam)} days")

    print("\n=== Dynamic Threshold (Level 1) ===")
    for label, mean, std in [("normal", 0.30, 0.15), ("high vol", 0.35, 0.25), ("low vol", 0.28, 0.08)]:
        history = list(np.random.normal(mean, std, 30))
        threshold = dynamic_signal_threshold(history, k=1.5)
        print(f"  {label}: threshold={threshold:.3f}")

    print("\n=== Strategy Weights (Lesson 17.3) ===")
    sharpes = {"A": 1.6, "B": 0.75, "C": 1.25}
    vols = {"A": 0.05, "B": 0.08, "C": 0.08}
    inv = inverse_vol_strategy_weights(vols)
    sh = sharpe_proportional_weights(sharpes)
    for name in sharpes:
        print(f"  {name}: inv_vol={inv[name]:.0%} sharpe_prop={sh[name]:.0%}")

    print("\n=== Retrain Decision Matrix ===")
    engine = RetrainDecisionEngine()
    scenarios = [
        ("A continue", False, -0.05),
        ("B retrain", True, -0.40),
        ("C observe", True, -0.10),
        ("D pause", False, -0.35),
    ]
    for label, drift, ic_change in scenarios:
        decision = engine.decide(drift_detected=drift, ic_change=ic_change)
        print(f"  {label}: {decision.action.value} - {decision.reason[:50]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Online learning and evolution report (Lesson 17)")
    parser.add_argument("--paper-only", action="store_true")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    ol = config.online_learning
    print("\n=== Configured Online Learning ===")
    print(f"  enabled             : {ol.enabled}")
    print(f"  decay factor        : {ol.decay_factor}")
    print(f"  PSI threshold       : {ol.psi_threshold}")
    print(f"  apply dyn threshold : {ol.apply_dynamic_threshold}")

    model = ExponentialMovingModel(decay_factor=ol.decay_factor)
    x = np.array([1.0, 0.5, -0.2])
    for target in (0.3, 0.5, 0.1):
        pred, err = model.update(x, target, learning_rate=ol.learning_rate)
        print(f"  EWM update: pred={pred:.3f} err={err:.3f}")
    print(f"  effective lookback  : {model.get_effective_lookback()} days")

    scheduler = AdaptiveUpdateScheduler(ol.min_update_interval_days, ol.max_update_interval_days)
    should, reason = scheduler.should_update(day=25, recent_errors=[0.6, 0.55, 0.52, 0.48, 0.42])
    print(f"  scheduler day 25    : update={should} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
