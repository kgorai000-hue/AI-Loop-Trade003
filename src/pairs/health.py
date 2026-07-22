"""Pair health gate: classify R1–R5 from spread metrics + optional leg regimes."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import RegimeAssessment, StrategyKind
from src.pairs.spread import SpreadSnapshot, beta_drift_ratio
from src.pairs.states import PairRegime
from src.regime.states import FiveState


@dataclass
class PairHealthThresholds:
    max_half_life_bars: float = 48.0  # ~1 day of M30
    weaken_half_life_mult: float = 2.0
    max_beta_drift: float = 0.35
    break_beta_drift: float = 0.60
    max_abs_trend_slope: float = 0.002
    break_abs_trend_slope: float = 0.005
    min_zero_cross_rate: float = 0.05
    high_vol_percentile_proxy: float = 0.0  # unused placeholder; use vol ratio
    vol_high_mult: float = 1.5  # vs median of recent spread std — approximate via phi/vol
    min_spread_vol: float = 1e-6


@dataclass
class PairHealthResult:
    regime: str
    allow_entry: bool
    size_scale: float
    z_entry_mult: float
    reasons: list[str] = field(default_factory=list)
    beta: float = 1.0
    half_life: float | None = None
    zscore: float | None = None
    beta_drift: float = 0.0
    single_regime_note: str = ""


def classify_pair_health(
    snapshot: SpreadSnapshot,
    *,
    closes_a,
    closes_b,
    thresholds: PairHealthThresholds | None = None,
    beta_short_window: int = 40,
    beta_long_window: int = 120,
    baseline_half_life: float | None = None,
    leg_a_regime: RegimeAssessment | None = None,
    leg_b_regime: RegimeAssessment | None = None,
    recent_spread_vols: list[float] | None = None,
) -> PairHealthResult:
    th = thresholds or PairHealthThresholds()
    reasons: list[str] = []

    # --- R5: event / exec / stress on either leg (hard stop) ---
    r5_hit, r5_note = _event_risk(leg_a_regime, leg_b_regime)
    if r5_hit:
        return PairHealthResult(
            regime=PairRegime.R5_EVENT_EXEC.value,
            allow_entry=False,
            size_scale=0.0,
            z_entry_mult=99.0,
            reasons=r5_note,
            beta=snapshot.beta,
            half_life=snapshot.half_life,
            zscore=snapshot.zscore,
            single_regime_note=_single_note(leg_a_regime, leg_b_regime),
        )

    drift = beta_drift_ratio(
        closes_a,
        closes_b,
        short_window=beta_short_window,
        long_window=beta_long_window,
    )
    hl = snapshot.half_life
    abs_trend = abs(snapshot.trend_slope)

    # --- R4: structural break ---
    if drift >= th.break_beta_drift:
        reasons.append(f"beta drift {drift:.2f} >= break {th.break_beta_drift}")
    if abs_trend >= th.break_abs_trend_slope:
        reasons.append(f"spread trend {snapshot.trend_slope:.4f} structural")
    if hl is None and snapshot.phi is not None and snapshot.phi >= 0.99:
        reasons.append(f"AR(1) phi={snapshot.phi:.3f} non-mean-reverting")
    if hl is not None and hl > th.max_half_life_bars * 3:
        reasons.append(f"half-life {hl:.1f} extreme")

    if reasons:
        return PairHealthResult(
            regime=PairRegime.R4_STRUCTURAL_BREAK.value,
            allow_entry=False,
            size_scale=0.0,
            z_entry_mult=99.0,
            reasons=reasons,
            beta=snapshot.beta,
            half_life=hl,
            zscore=snapshot.zscore,
            beta_drift=drift,
            single_regime_note=_single_note(leg_a_regime, leg_b_regime),
        )

    # --- R3: weakening ---
    weaken: list[str] = []
    ref_hl = baseline_half_life or th.max_half_life_bars
    if hl is not None and hl > ref_hl * th.weaken_half_life_mult:
        weaken.append(f"half-life {hl:.1f} > {th.weaken_half_life_mult}x ref")
    if hl is not None and hl > th.max_half_life_bars:
        weaken.append(f"half-life {hl:.1f} > max {th.max_half_life_bars}")
    if drift >= th.max_beta_drift:
        weaken.append(f"beta drift {drift:.2f}")
    if abs_trend >= th.max_abs_trend_slope:
        weaken.append(f"spread trend {snapshot.trend_slope:.4f}")
    if snapshot.zero_cross_rate < th.min_zero_cross_rate:
        weaken.append(f"zero-cross rate {snapshot.zero_cross_rate:.2f} low")

    if weaken:
        return PairHealthResult(
            regime=PairRegime.R3_WEAKENING.value,
            allow_entry=False,
            size_scale=0.0,
            z_entry_mult=1.5,
            reasons=weaken,
            beta=snapshot.beta,
            half_life=hl,
            zscore=snapshot.zscore,
            beta_drift=drift,
            single_regime_note=_single_note(leg_a_regime, leg_b_regime),
        )

    # --- R2 vs R1: volatility of spread ---
    high_vol = False
    if recent_spread_vols and len(recent_spread_vols) >= 5:
        med = float(sorted(recent_spread_vols)[len(recent_spread_vols) // 2])
        if med > th.min_spread_vol and snapshot.spread_vol >= med * th.vol_high_mult:
            high_vol = True
            reasons.append(
                f"spread vol {snapshot.spread_vol:.4f} >= {th.vol_high_mult}x median"
            )
    elif snapshot.spread_vol > 0 and hl is not None and hl > ref_hl * 0.8:
        # mild proxy when no history: elevated half-life near limit + decent vol
        if snapshot.zero_cross_rate > 0.15:
            high_vol = True
            reasons.append("elevated chop / half-life near limit")

    if high_vol:
        return PairHealthResult(
            regime=PairRegime.R2_VOLATILE_MR.value,
            allow_entry=True,
            size_scale=0.5,
            z_entry_mult=1.25,
            reasons=reasons or ["volatile mean reversion"],
            beta=snapshot.beta,
            half_life=hl,
            zscore=snapshot.zscore,
            beta_drift=drift,
            single_regime_note=_single_note(leg_a_regime, leg_b_regime),
        )

    reasons.append("stable mean reversion")
    return PairHealthResult(
        regime=PairRegime.R1_STABLE_MR.value,
        allow_entry=True,
        size_scale=1.0,
        z_entry_mult=1.0,
        reasons=reasons,
        beta=snapshot.beta,
        half_life=hl,
        zscore=snapshot.zscore,
        beta_drift=drift,
        single_regime_note=_single_note(leg_a_regime, leg_b_regime),
    )


def _event_risk(
    leg_a: RegimeAssessment | None,
    leg_b: RegimeAssessment | None,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    for name, leg in (("A", leg_a), ("B", leg_b)):
        if leg is None:
            continue
        if leg.regime_label == FiveState.STRESS.value:
            notes.append(f"leg {name} stress")
        if leg.selected_strategy == StrategyKind.CRISIS_HALT and leg.regime_label in (
            FiveState.STRESS.value,
            FiveState.HIGH_VOL_CHOP.value,
        ):
            # Only hard-stop on stress/chop halt — not on uncertain alone for pairs
            if leg.regime_label == FiveState.STRESS.value:
                notes.append(f"leg {name} crisis halt")
        if leg.regime_label == FiveState.HIGH_VOL_CHOP.value:
            notes.append(f"leg {name} high_vol_chop (exec risk)")
    # R5 if either stress or both legs chop
    stress = any("stress" in n or "crisis" in n for n in notes)
    chop_count = sum(1 for n in notes if "high_vol_chop" in n)
    if stress or chop_count >= 2:
        return True, notes or ["event/exec risk"]
    return False, []


def _single_note(
    leg_a: RegimeAssessment | None,
    leg_b: RegimeAssessment | None,
) -> str:
    """Reference-only annotation from single-symbol regimes (not a hard gate)."""
    la = leg_a.regime_label if leg_a else "?"
    lb = leg_b.regime_label if leg_b else "?"
    return f"legs={la}/{lb}"
