from __future__ import annotations

import logging

import numpy as np

from src.core.config import AppConfig
from src.core.types import RegimeAssessment, StrategyKind
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.regime.five_state import FiveStateThresholds, classify_five_state
from src.regime.health import assess_regime_health
from src.regime.history import RegimeHistoryStore
from src.regime.resilience import determine_degradation_level
from src.regime.scores import compute_regime_scores
from src.regime.states import (
    FiveState,
    five_state_to_market_regime,
    mode_for_five_state,
    position_scale_for_five_state,
    strategy_for_five_state,
    strategy_weights_for_five_state,
)
from src.stats.returns import log_returns
from src.stats.risk import volatility

logger = logging.getLogger(__name__)


class RegimeAgent:
    """Five-state regime router (stable/high-vol trend, range, chop, stress)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.stats = config.stats
        self.regime_cfg = config.regime
        self.resilience_cfg = config.resilience
        self.engine = FeatureEngine(config)
        self.history = RegimeHistoryStore(config.storage.path)

    def assess(self, symbol: str, timeframe: str | None = None) -> RegimeAssessment | None:
        timeframe = timeframe or self.stats.analysis_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < self.stats.min_bars:
            return None

        closes = np.array([float(bar["close"]) for bar in bars], dtype=float)
        returns = log_returns(closes)
        if len(returns) < self.stats.vol_window:
            return None

        ann_vol = float(
            volatility(
                returns[-self.stats.vol_window :],
                annualize=True,
                trading_days=self.config.trading.trading_days_per_year,
            )
        )
        recent_return = float(np.sum(returns[-self.stats.vol_window :]))

        series = self.engine.compute_series(bars)
        adx_series = series["adx"] if series else []
        adx_val = float(adx_series[-1]) if len(adx_series) and not np.isnan(adx_series[-1]) else 0.0
        atr = series["atr"] if series else None
        atr_pct = None
        atr_pct_median = None
        if atr is not None and len(atr) and closes[-1] > 0:
            atr_arr = np.asarray(atr, dtype=float)
            valid = ~np.isnan(atr_arr)
            if valid.any():
                atr_pcts = atr_arr[valid] / closes[valid]
                atr_pct = float(atr_pcts[-1])
                atr_pct_median = float(np.median(atr_pcts[-min(60, len(atr_pcts)) :]))

        asset_corr = self._asset_correlation(symbol, timeframe)
        scores = compute_regime_scores(
            closes,
            trend_lookback=self.regime_cfg.trend_lookback,
            er_lookback=self.regime_cfg.er_lookback,
            vol_window=self.stats.vol_window,
            vol_hist_window=self.regime_cfg.vol_hist_window,
            trading_days=self.config.trading.trading_days_per_year,
            asset_correlation=asset_corr,
            atr_pct=atr_pct,
            atr_pct_median=atr_pct_median,
        )

        previous = self.history.last_confirmed_label(symbol, timeframe)
        thresholds = self._thresholds()
        classified = classify_five_state(scores, previous=previous, thresholds=thresholds)

        confirm_bars = max(1, int(self.regime_cfg.confirm_bars))
        confirmed_label, pending_switch = self.history.apply_confirmation(
            symbol,
            timeframe,
            classified.label,
            confirm_bars,
        )

        if pending_switch and confirmed_label == "uncertain":
            effective = FiveState.UNCERTAIN.value
        else:
            effective = confirmed_label

        max_prob = float(classified.confidence)
        uncertain = effective == FiveState.UNCERTAIN.value or pending_switch
        uncertain_reasons: list[str] = []
        if pending_switch:
            uncertain_reasons.append("confirmation pending")
        if classified.label == FiveState.UNCERTAIN.value:
            uncertain_reasons.append("gray zone scores")

        switch_stats = self.history.switch_stats(symbol, days=30)
        span_weeks = max(switch_stats.get("span_days", 7) / 7, 1)
        switches_per_week = switch_stats.get("switches", 0.0) / span_weeks

        health = assess_regime_health(
            symbol,
            switch_stats,
            max_switches_per_week=self.resilience_cfg.max_switches_per_week,
            min_regime_duration_days=self.resilience_cfg.min_regime_duration_days,
            adx_oscillation=False,
            max_probability=max_prob,
            clear_prob_threshold=self.resilience_cfg.clear_prob_threshold,
            uncertain_prob_threshold=self.resilience_cfg.uncertain_prob_threshold,
        )

        degradation = determine_degradation_level(
            max_probability=max_prob,
            clear_prob_threshold=self.resilience_cfg.clear_prob_threshold,
            uncertain_prob_threshold=self.resilience_cfg.uncertain_prob_threshold,
            health_fail=not health.healthy,
            data_quality_ok=True,
            force_defensive=uncertain,
        )

        history_entry = self.history.record(
            symbol,
            timeframe,
            classified.raw_label,
            effective,
            classified.reason,
            max_probability=max_prob,
            adx=adx_val,
            degradation_level=int(degradation),
        )
        if history_entry.switched:
            logger.info(
                "RegimeAgent switch %s: %s -> %s (%s)",
                symbol,
                history_entry.detected_label,
                history_entry.confirmed_label,
                classified.reason,
            )

        selected = strategy_for_five_state(effective)
        mode = mode_for_five_state(effective)
        scale = position_scale_for_five_state(
            effective,
            high_vol_trend_scale=self.regime_cfg.high_vol_trend_scale,
        )
        weights = strategy_weights_for_five_state(effective)
        primary = five_state_to_market_regime(effective)

        if selected == StrategyKind.CRISIS_HALT:
            mode = mode_for_five_state(FiveState.STRESS.value)
            scale = 0.0

        reason = f"{classified.reason} | confirmed={effective}"
        if uncertain_reasons:
            reason += f" | uncertain: {', '.join(uncertain_reasons[:2])}"

        return RegimeAssessment(
            symbol=symbol,
            regime=primary,
            annualized_volatility=ann_vol,
            recent_return=recent_return,
            recommended_mode=mode,
            reason=reason,
            adx=adx_val,
            position_scale=scale,
            selected_strategy=selected,
            trend_confirmed=effective in (
                FiveState.STABLE_TREND.value,
                FiveState.HIGH_VOL_TREND.value,
            ),
            sideways_confirmed=effective == FiveState.STABLE_RANGE.value,
            regime_label=effective,
            probabilities={
                effective: max_prob,
                "other": max(0.0, 1.0 - max_prob),
            },
            strategy_weights=weights,
            is_transition=classified.is_transition or uncertain,
            asset_correlation=asset_corr,
            detection_method="five_state",
            vol_cluster=(
                "extreme"
                if scores.vol_percentile >= 0.9
                else "high"
                if scores.vol_percentile >= 0.7
                else "normal"
                if scores.vol_percentile >= 0.3
                else "low"
            ),
            uncertain=uncertain,
            max_probability=max_prob,
            degradation_level=int(degradation),
            health_warnings=health.warnings,
            misjudgment_pattern=health.misjudgment_pattern.value if health.misjudgment_pattern else None,
            switches_per_week=switches_per_week,
            avg_regime_duration_days=switch_stats.get("avg_duration_days", 0.0),
            adx_oscillation=False,
            uncertain_reasons=uncertain_reasons,
        )

    def assess_market_proxy(self) -> RegimeAssessment | None:
        proxy = self.regime_cfg.correlation_benchmark
        if proxy not in self.config.symbols:
            proxy = self.config.symbols[0]
        return self.assess(proxy)

    def switch_stats(self, symbol: str | None = None) -> dict[str, float]:
        return self.history.switch_stats(symbol)

    def _thresholds(self) -> FiveStateThresholds:
        cfg = self.regime_cfg
        return FiveStateThresholds(
            enter_trend_score=cfg.enter_trend_score,
            enter_er_trend=cfg.enter_er_trend,
            enter_vol_max_for_stable=cfg.enter_vol_max_for_stable,
            exit_trend_score=cfg.exit_trend_score,
            exit_er_trend=cfg.exit_er_trend,
            high_vol_enter=cfg.high_vol_enter,
            stress_vol=cfg.stress_vol,
            enter_range_abs_trend=cfg.enter_range_abs_trend,
            enter_er_range_max=cfg.enter_er_range_max,
            exit_range_abs_trend=cfg.exit_range_abs_trend,
            stress_corr=cfg.stress_corr,
            stress_spread=cfg.stress_spread,
        )

    def _asset_correlation(self, symbol: str, timeframe: str) -> float:
        benchmark = self.regime_cfg.correlation_benchmark
        if benchmark == symbol:
            return 1.0

        lookback = self.regime_cfg.correlation_lookback
        bars_sym = self.store.get_recent_bars(symbol, timeframe, lookback + 5)
        bars_bench = self.store.get_recent_bars(benchmark, timeframe, lookback + 5)
        if len(bars_sym) < lookback or len(bars_bench) < lookback:
            return 0.0

        closes_sym = [float(b["close"]) for b in bars_sym]
        closes_bench = [float(b["close"]) for b in bars_bench]
        rets_sym = log_returns(closes_sym)[-lookback:]
        rets_bench = log_returns(closes_bench)[-lookback:]
        n = min(len(rets_sym), len(rets_bench))
        if n < 5:
            return 0.0
        corr = float(np.corrcoef(rets_sym[-n:], rets_bench[-n:])[0, 1])
        if np.isnan(corr):
            return 0.0
        return corr
