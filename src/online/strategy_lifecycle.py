from __future__ import annotations

from src.online.types import LifecycleStage, StrategyLifecycleState


def sharpe_proxy(annual_return: float, annual_vol: float) -> float:
    if annual_vol <= 0:
        return 0.0
    return annual_return / annual_vol


def classify_lifecycle_stage(
    sharpe: float,
    *,
    incubation_min: float = 1.0,
    maturity_min: float = 1.5,
    decay_max: float = 0.8,
) -> LifecycleStage:
    if sharpe >= maturity_min:
        return LifecycleStage.MATURITY
    if sharpe >= incubation_min:
        return LifecycleStage.VALIDATION
    if sharpe < decay_max:
        return LifecycleStage.DECAY
    return LifecycleStage.INCUBATION


def capital_weight_for_stage(stage: LifecycleStage) -> float:
    return {
        LifecycleStage.INCUBATION: 0.0,
        LifecycleStage.VALIDATION: 0.08,
        LifecycleStage.MATURITY: 0.30,
        LifecycleStage.DECAY: 0.02,
    }[stage]


def inverse_vol_strategy_weights(
    strategy_vols: dict[str, float],
) -> dict[str, float]:
    if not strategy_vols:
        return {}
    inv = {name: 1.0 / max(vol, 1e-6) for name, vol in strategy_vols.items()}
    total = sum(inv.values())
    return {name: weight / total for name, weight in inv.items()}


def sharpe_proportional_weights(strategy_sharpes: dict[str, float]) -> dict[str, float]:
    positive = {name: max(sh, 0.0) for name, sh in strategy_sharpes.items()}
    total = sum(positive.values())
    if total <= 0:
        share = 1.0 / len(strategy_sharpes)
        return {name: share for name in strategy_sharpes}
    return {name: sh / total for name, sh in positive.items()}


def build_strategy_lifecycle_states(
    strategy_metrics: dict[str, tuple[float, float]],
    *,
    incubation_min: float = 1.0,
    maturity_min: float = 1.5,
    decay_max: float = 0.8,
) -> list[StrategyLifecycleState]:
    states: list[StrategyLifecycleState] = []
    for strategy, (ann_return, ann_vol) in strategy_metrics.items():
        sh = sharpe_proxy(ann_return, ann_vol)
        stage = classify_lifecycle_stage(
            sh,
            incubation_min=incubation_min,
            maturity_min=maturity_min,
            decay_max=decay_max,
        )
        states.append(
            StrategyLifecycleState(
                strategy=strategy,
                stage=stage,
                sharpe_proxy=round(sh, 3),
                capital_weight=capital_weight_for_stage(stage),
                note=f"return={ann_return:.1%} vol={ann_vol:.1%}",
            )
        )
    return states
