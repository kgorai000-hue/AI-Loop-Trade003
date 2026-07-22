from __future__ import annotations

import logging

import numpy as np

from src.core.config import AppConfig
from src.core.types import MarketRegime, RegimeAssessment, SignalMode, StrategyKind
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.regime.detection import (
    RegimeDetectionResult,
    rule_based_detect,
    select_strategy_from_weights,
    volatility_cluster_label,
)
from src.regime.health import assess_regime_health, detect_uncertain_triggers
from src.regime.history import RegimeHistoryStore
from src.regime.probabilistic import ProbabilisticRegimeDetector
from src.regime.resilience import determine_degradation_level, uncertain_position_scale
from src.stats.returns import log_returns
from src.stats.timeseries import detect_regime, regime_to_signal_mode
from src.strategies.strategy_selection import select_strategy_kind

logger = logging.getLogger(__name__)


class RegimeAgent:
    """Regime router with misjudgment resilience (Lesson 12-13)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.stats = config.stats
        self.strategies = config.strategies
        self.regime_cfg = config.regime
        self.resilience_cfg = config.resilience
        self.engine = FeatureEngine(config)
        self.history = RegimeHistoryStore(config.storage.path)
        self._gmm: ProbabilisticRegimeDetector | None = None

    def assess(self, symbol: str, timeframe: str | None = None) -> RegimeAssessment | None:
        timeframe = timeframe or self.stats.analysis_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < self.stats.min_bars:
            return None

        closes = [float(bar["close"]) for bar in bars]
        returns = log_returns(closes)
        if len(returns) < self.stats.vol_window:
            return None

        legacy_regime, ann_vol, recent_return = detect_regime(
            returns,
            vol_crisis=self.stats.regime_vol_crisis,
            vol_bull_max=self.stats.regime_vol_bull_max,
            lookback=self.stats.vol_window,
            trading_days=self.config.trading.trading_days_per_year,
        )

        series = self.engine.compute_series(bars)
        adx_series = series["adx"] if series else []
        adx_val = float(adx_series[-1]) if len(adx_series) and not np.isnan(adx_series[-1]) else 0.0

        asset_corr = self._asset_correlation(symbol, timeframe)
        vol_cluster, _ = volatility_cluster_label(ann_vol)

        if self.regime_cfg.method == "gmm":
            detection = self._detect_gmm(returns, adx_val, ann_vol, asset_corr)
        else:
            detection = rule_based_detect(
                adx=adx_val,
                annualized_volatility=ann_vol,
                recent_return=recent_return,
                asset_correlation=asset_corr,
                crisis_vol_threshold=self.regime_cfg.crisis_vol_threshold,
                crisis_correlation_threshold=self.regime_cfg.crisis_correlation_threshold,
                ranging_vol_max=self.regime_cfg.ranging_vol_max,
                trending_adx_min=self.regime_cfg.trending_adx_min,
                trending_return_min=self.regime_cfg.trending_return_min,
                ranging_adx_max=self.regime_cfg.ranging_adx_max,
            )

        max_prob = max(detection.probabilities.values()) if detection.probabilities else 0.0
        confirmed_label, pending_switch = self.history.apply_confirmation(
            symbol,
            timeframe,
            detection.label,
            self.regime_cfg.confirm_days,
        )

        post_switch_days = self.history.days_since_last_switch(symbol, timeframe)
        adx_oscillation = self.history.detect_adx_oscillation(
            symbol,
            timeframe,
            threshold=self.regime_cfg.trending_adx_min,
            band=self.regime_cfg.adx_boundary_high - self.regime_cfg.trending_adx_min,
            lookback=self.resilience_cfg.oscillation_lookback,
        )

        uncertain, uncertain_reasons = detect_uncertain_triggers(
            max_probability=max_prob,
            uncertain_prob_threshold=self.resilience_cfg.uncertain_prob_threshold,
            adx=adx_val,
            adx_boundary_low=self.regime_cfg.adx_boundary_low,
            adx_boundary_high=self.regime_cfg.adx_boundary_high,
            annualized_volatility=ann_vol,
            ranging_vol_max=self.regime_cfg.ranging_vol_max,
            trending_adx_min=self.regime_cfg.trending_adx_min,
            pending_switch=pending_switch,
            post_switch_days=post_switch_days,
            post_switch_uncertain_days=self.regime_cfg.post_switch_uncertain_days,
        )

        if confirmed_label == "uncertain":
            uncertain = True
            uncertain_reasons.append("confirmation pending")

        switch_stats = self.history.switch_stats(symbol, days=30)
        span_weeks = max(switch_stats.get("span_days", 7) / 7, 1)
        switches_per_week = switch_stats.get("switches", 0.0) / span_weeks

        health = assess_regime_health(
            symbol,
            switch_stats,
            max_switches_per_week=self.resilience_cfg.max_switches_per_week,
            min_regime_duration_days=self.resilience_cfg.min_regime_duration_days,
            adx_oscillation=adx_oscillation,
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
            force_defensive=uncertain and self.resilience_cfg.uncertain_strategy == "worst_case",
        )

        history_entry = self.history.record(
            symbol,
            timeframe,
            detection.label,
            confirmed_label,
            detection.reason,
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
                detection.reason,
            )

        selected, mode, scale, trend_ok, sideways_ok, adx_reason = select_strategy_kind(
            detection.primary_regime if legacy_regime != MarketRegime.CRISIS else MarketRegime.CRISIS,
            adx_series,
            closes,
            self.strategies,
            self.stats.vol_window,
            self.config.trading.trading_days_per_year,
            self.stats.regime_vol_crisis,
        )

        weight_selected, weight_mode, weight_scale = select_strategy_from_weights(
            detection.strategy_weights,
            detection.is_transition or uncertain,
            self.strategies.uncertain_position_scale,
        )

        effective_label = confirmed_label
        if uncertain:
            effective_label = "uncertain"

        if detection.label == "crisis" or confirmed_label == "crisis":
            selected = StrategyKind.CRISIS_HALT
            mode = SignalMode.NONE
            scale = 0.0
        elif effective_label == "uncertain":
            selected = weight_selected
            mode = weight_mode if weight_mode != SignalMode.NONE else SignalMode.MOMENTUM
            scale = uncertain_position_scale(
                min(scale, weight_scale),
                self.resilience_cfg.uncertain_strategy,
            )
        elif confirmed_label == "trending":
            selected = StrategyKind.TREND_FOLLOWING
            mode = SignalMode.MOMENTUM
            scale = max(scale, weight_scale)
        elif confirmed_label == "mean_reverting":
            selected = StrategyKind.MEAN_REVERSION
            mode = SignalMode.MEAN_REVERSION
            scale = max(scale, weight_scale)
        elif confirmed_label == "transition":
            selected = weight_selected
            mode = weight_mode if weight_mode != SignalMode.NONE else SignalMode.MOMENTUM
            scale = min(scale, weight_scale)

        defensive = detection.strategy_weights.get("defensive", 0.0)
        scale *= max(0.0, 1.0 - defensive * 0.5)

        if selected.value == "crisis_halt":
            mode = SignalMode.NONE
        elif mode == SignalMode.NONE and detection.primary_regime != MarketRegime.CRISIS:
            mode = regime_to_signal_mode(detection.primary_regime)

        reason = f"{detection.reason} | confirmed={confirmed_label} | {adx_reason}"
        if uncertain:
            reason += f" | uncertain: {', '.join(uncertain_reasons[:2])}"

        return RegimeAssessment(
            symbol=symbol,
            regime=detection.primary_regime,
            annualized_volatility=ann_vol,
            recent_return=recent_return,
            recommended_mode=mode,
            reason=reason,
            adx=adx_val,
            position_scale=scale if selected.value != "crisis_halt" else 0.0,
            selected_strategy=selected,
            trend_confirmed=trend_ok,
            sideways_confirmed=sideways_ok,
            regime_label=effective_label,
            probabilities=detection.probabilities,
            strategy_weights=detection.strategy_weights,
            is_transition=detection.is_transition or effective_label in ("transition", "uncertain"),
            asset_correlation=asset_corr,
            detection_method=detection.detection_method,
            vol_cluster=vol_cluster,
            uncertain=uncertain,
            max_probability=max_prob,
            degradation_level=int(degradation),
            health_warnings=health.warnings,
            misjudgment_pattern=health.misjudgment_pattern.value if health.misjudgment_pattern else None,
            switches_per_week=switches_per_week,
            avg_regime_duration_days=switch_stats.get("avg_duration_days", 0.0),
            adx_oscillation=adx_oscillation,
            uncertain_reasons=uncertain_reasons,
        )

    def assess_market_proxy(self) -> RegimeAssessment | None:
        proxy = self.regime_cfg.correlation_benchmark
        if proxy not in self.config.symbols:
            proxy = self.config.symbols[0]
        return self.assess(proxy)

    def switch_stats(self, symbol: str | None = None) -> dict[str, float]:
        return self.history.switch_stats(symbol)

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

    def _detect_gmm(
        self,
        returns: np.ndarray,
        adx: float,
        ann_vol: float,
        asset_correlation: float,
    ) -> RegimeDetectionResult:
        window = self.stats.vol_window
        features = []
        for idx in range(window, len(returns) + 1):
            chunk = returns[idx - window : idx]
            vol = float(np.std(chunk, ddof=1) * np.sqrt(self.config.trading.trading_days_per_year))
            ret = float(np.sum(chunk))
            features.append([ret, vol, adx])

        matrix = np.asarray(features, dtype=float)
        if len(matrix) < self.regime_cfg.gmm_min_samples:
            return rule_based_detect(
                adx=adx,
                annualized_volatility=ann_vol,
                recent_return=float(np.sum(returns[-window:])),
                asset_correlation=asset_correlation,
                crisis_vol_threshold=self.regime_cfg.crisis_vol_threshold,
                crisis_correlation_threshold=self.regime_cfg.crisis_correlation_threshold,
                ranging_vol_max=self.regime_cfg.ranging_vol_max,
                trending_adx_min=self.regime_cfg.trending_adx_min,
                trending_return_min=self.regime_cfg.trending_return_min,
                ranging_adx_max=self.regime_cfg.ranging_adx_max,
            )

        if self._gmm is None:
            self._gmm = ProbabilisticRegimeDetector()
        try:
            self._gmm.fit(matrix)
            return self._gmm.to_detection_result(
                matrix,
                threshold=self.regime_cfg.probability_threshold,
                asset_correlation=asset_correlation,
            )
        except (ValueError, RuntimeError) as exc:
            logger.debug("GMM regime fallback to rules: %s", exc)
            return rule_based_detect(
                adx=adx,
                annualized_volatility=ann_vol,
                recent_return=float(np.sum(returns[-window:])),
                asset_correlation=asset_correlation,
                crisis_vol_threshold=self.regime_cfg.crisis_vol_threshold,
                crisis_correlation_threshold=self.regime_cfg.crisis_correlation_threshold,
                ranging_vol_max=self.regime_cfg.ranging_vol_max,
                trending_adx_min=self.regime_cfg.trending_adx_min,
                trending_return_min=self.regime_cfg.trending_return_min,
                ranging_adx_max=self.regime_cfg.ranging_adx_max,
            )
