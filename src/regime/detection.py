from __future__ import annotations

from dataclasses import dataclass

from src.core.types import MarketRegime, SignalMode, StrategyKind


@dataclass
class RegimeDetectionResult:
    """Rule-based / probabilistic regime output (Lesson 12)."""
    primary_regime: MarketRegime
    label: str
    probabilities: dict[str, float]
    strategy_weights: dict[str, float]
    is_transition: bool
    reason: str
    asset_correlation: float = 0.0
    detection_method: str = "rule_based"


def rule_based_detect(
    *,
    adx: float,
    annualized_volatility: float,
    recent_return: float,
    asset_correlation: float,
    crisis_vol_threshold: float = 0.30,
    crisis_correlation_threshold: float = 0.80,
    ranging_vol_max: float = 0.15,
    trending_adx_min: float = 25.0,
    trending_return_min: float = 0.05,
    ranging_adx_max: float = 20.0,
) -> RegimeDetectionResult:
    """Lesson 12.2 rule-based detection with crisis priority."""
    ret = recent_return
    vol = annualized_volatility
    corr = asset_correlation

    if vol >= crisis_vol_threshold and corr >= crisis_correlation_threshold:
        probs = {"trending": 0.05, "mean_reverting": 0.05, "crisis": 0.85, "transition": 0.05}
        weights = adjusted_strategy_weights(probs)
        return RegimeDetectionResult(
            primary_regime=MarketRegime.CRISIS,
            label="crisis",
            probabilities=probs,
            strategy_weights=weights,
            is_transition=False,
            reason=f"crisis: vol {vol:.1%} >= {crisis_vol_threshold:.0%} and corr {corr:.2f} >= {crisis_correlation_threshold:.2f}",
            asset_correlation=corr,
        )

    if vol >= crisis_vol_threshold:
        probs = {"trending": 0.10, "mean_reverting": 0.10, "crisis": 0.75, "transition": 0.05}
        weights = adjusted_strategy_weights(probs)
        return RegimeDetectionResult(
            primary_regime=MarketRegime.CRISIS,
            label="crisis",
            probabilities=probs,
            strategy_weights=weights,
            is_transition=False,
            reason=f"crisis: vol {vol:.1%} >= {crisis_vol_threshold:.0%}",
            asset_correlation=corr,
        )

    if adx >= trending_adx_min and ret >= trending_return_min:
        strength = min(1.0, (adx - trending_adx_min) / 20 + (ret - trending_return_min) / 0.10)
        p_trend = 0.55 + 0.35 * strength
        probs = {
            "trending": p_trend,
            "mean_reverting": max(0.05, 1.0 - p_trend - 0.10),
            "crisis": 0.05,
            "transition": 0.10,
        }
        weights = adjusted_strategy_weights(probs)
        return RegimeDetectionResult(
            primary_regime=MarketRegime.BULL,
            label="trending",
            probabilities=_normalize_probs(probs),
            strategy_weights=weights,
            is_transition=False,
            reason=f"trending: ADX {adx:.1f} >= {trending_adx_min} and return {ret:.1%} >= {trending_return_min:.0%}",
            asset_correlation=corr,
        )

    if adx <= ranging_adx_max and vol <= ranging_vol_max:
        p_mr = 0.60
        probs = {
            "trending": 0.15,
            "mean_reverting": p_mr,
            "crisis": 0.05,
            "transition": 0.20,
        }
        weights = adjusted_strategy_weights(probs)
        return RegimeDetectionResult(
            primary_regime=MarketRegime.SIDEWAYS,
            label="mean_reverting",
            probabilities=probs,
            strategy_weights=weights,
            is_transition=False,
            reason=f"ranging: ADX {adx:.1f} <= {ranging_adx_max} and vol {vol:.1%} <= {ranging_vol_max:.0%}",
            asset_correlation=corr,
        )

    probs = {
        "trending": 0.30,
        "mean_reverting": 0.30,
        "crisis": 0.10,
        "transition": 0.30,
    }
    weights = adjusted_strategy_weights(probs, transition_risk_first=True)
    return RegimeDetectionResult(
        primary_regime=MarketRegime.SIDEWAYS,
        label="transition",
        probabilities=probs,
        strategy_weights=weights,
        is_transition=True,
        reason=f"transition: ADX {adx:.1f}, vol {vol:.1%}, return {ret:.1%} in gray zone",
        asset_correlation=corr,
    )


def volatility_cluster_label(annualized_volatility: float) -> tuple[str, SignalMode]:
    """Lesson 12.2 vol clustering bands."""
    vol = annualized_volatility
    if vol < 0.15:
        return "low_vol_ranging", SignalMode.MEAN_REVERSION
    if vol < 0.25:
        return "normal", SignalMode.MOMENTUM
    if vol < 0.35:
        return "high_vol_trending", SignalMode.MOMENTUM
    return "extreme_crisis", SignalMode.NONE


def adjusted_strategy_weights(
    probabilities: dict[str, float],
    *,
    crisis_amplify: float = 1.5,
    transition_risk_first: bool = False,
) -> dict[str, float]:
    """Map state probabilities to strategy weights (Lesson 12.2 HMM insight)."""
    trend_p = probabilities.get("trending", 0.0)
    mr_p = probabilities.get("mean_reverting", 0.0)
    crisis_p = min(1.0, probabilities.get("crisis", 0.0) * crisis_amplify)
    trans_p = probabilities.get("transition", 0.0)

    defensive = crisis_p + (trans_p * 0.5 if transition_risk_first else trans_p * 0.2)
    offensive = max(0.0, 1.0 - defensive)
    if trend_p + mr_p <= 0:
        trend_w = offensive * 0.5
        mr_w = offensive * 0.5
    else:
        mix = trend_p + mr_p
        trend_w = offensive * (trend_p / mix)
        mr_w = offensive * (mr_p / mix)

    total = trend_w + mr_w + defensive
    if total <= 0:
        return {"trend": 0.33, "mean_reversion": 0.33, "defensive": 0.34}
    return {
        "trend": trend_w / total,
        "mean_reversion": mr_w / total,
        "defensive": defensive / total,
    }


def select_strategy_from_weights(
    weights: dict[str, float],
    is_transition: bool,
    uncertain_scale: float,
) -> tuple[StrategyKind, SignalMode, float]:
    trend_w = weights.get("trend", 0.0)
    mr_w = weights.get("mean_reversion", 0.0)
    defensive = weights.get("defensive", 0.0)

    if defensive >= 0.5:
        return StrategyKind.CRISIS_HALT, SignalMode.NONE, 0.0

    if is_transition:
        return StrategyKind.UNCERTAIN, SignalMode.NONE, uncertain_scale * 0.5

    if trend_w >= mr_w and trend_w >= 0.45:
        return StrategyKind.TREND_FOLLOWING, SignalMode.MOMENTUM, min(1.0, trend_w * 1.2)
    if mr_w > trend_w and mr_w >= 0.45:
        return StrategyKind.MEAN_REVERSION, SignalMode.MEAN_REVERSION, min(1.0, mr_w * 1.2)

    return StrategyKind.UNCERTAIN, SignalMode.MOMENTUM, uncertain_scale


def evaluate_regime_value(
    return_without: float,
    return_with: float,
    switch_count: int,
    switch_cost_pct: float,
) -> dict[str, float]:
    """Lesson 12.3: net value = improvement - switch costs."""
    improvement = return_with - return_without
    switch_cost = switch_count * switch_cost_pct
    net_value = improvement - switch_cost
    return {
        "return_improvement": improvement,
        "switch_cost_total": switch_cost,
        "net_value": net_value,
    }


def _normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        n = len(probs)
        return {k: 1.0 / n for k in probs}
    return {k: v / total for k, v in probs.items()}
