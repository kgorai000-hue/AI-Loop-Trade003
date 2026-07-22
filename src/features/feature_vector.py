from __future__ import annotations

from dataclasses import dataclass

from src.core.config import AppConfig
from src.core.types import MarketRegime, SignalMode
from src.features.indicators import IndicatorSnapshot, bars_to_arrays, compute_indicators, latest_snapshot


@dataclass
class FeatureVector:
    symbol: str
    timeframe: str
    snapshot: IndicatorSnapshot
    score: float
    mode: SignalMode
    components: dict[str, float]


class FeatureEngine:
    """Build feature vectors and regime-weighted scores (Lesson 04)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.indicators = config.indicators

    def build_from_bars(
        self,
        symbol: str,
        bars: list[dict],
        timeframe: str,
        regime: MarketRegime | None = None,
    ) -> FeatureVector | None:
        if len(bars) < self.config.stats.min_bars:
            return None

        opens, highs, lows, closes, volumes = bars_to_arrays(bars)
        snapshot = latest_snapshot(
            opens,
            highs,
            lows,
            closes,
            volumes,
            **self._indicator_kwargs(),
        )
        if snapshot is None:
            return None

        mode = self._mode_for_regime(regime)
        score, components = self._score(snapshot, mode, volumes)
        return FeatureVector(
            symbol=symbol,
            timeframe=timeframe,
            snapshot=snapshot,
            score=score,
            mode=mode,
            components=components,
        )

    def compute_series(self, bars: list[dict]) -> dict[str, object] | None:
        if len(bars) < self.config.stats.min_bars:
            return None
        opens, highs, lows, closes, volumes = bars_to_arrays(bars)
        return compute_indicators(
            opens,
            highs,
            lows,
            closes,
            volumes,
            **self._indicator_kwargs(),
        )

    def _indicator_kwargs(self) -> dict:
        cfg = self.indicators
        return {
            "macd_fast": cfg.macd_fast,
            "macd_slow": cfg.macd_slow,
            "macd_signal": cfg.macd_signal,
            "macd_histogram_double": cfg.macd_histogram_double,
            "rsi_period": cfg.rsi_period,
            "bb_period": cfg.bb_period,
            "bb_std": cfg.bb_std,
            "atr_period": cfg.atr_period,
            "adx_period": self.config.strategies.adx_period,
        }

    def _mode_for_regime(self, regime: MarketRegime | None) -> SignalMode:
        if regime == MarketRegime.CRISIS:
            return SignalMode.NONE
        if regime == MarketRegime.BULL:
            return SignalMode.MOMENTUM
        if regime == MarketRegime.SIDEWAYS:
            return SignalMode.MEAN_REVERSION
        return SignalMode.MOMENTUM

    def _score(
        self,
        snap: IndicatorSnapshot,
        mode: SignalMode,
        volumes,
    ) -> tuple[float, dict[str, float]]:
        vol_change = 0.0
        if len(volumes) >= 2 and volumes[-2] > 0:
            vol_change = float((volumes[-1] - volumes[-2]) / volumes[-2])

        if mode == SignalMode.MOMENTUM:
            components = {
                "macd_diff": _clamp(snap.macd_diff / max(abs(snap.ma20), 1e-8), -1, 1),
                "macd_hist_delta": _clamp(snap.macd_histogram_delta * 100, -1, 1),
                "ema_trend": _clamp((snap.ema_fast - snap.ema_slow) / max(abs(snap.ma20), 1e-8), -1, 1),
                "atr_breakout": _clamp(snap.atr_pct * 10, 0, 1),
            }
            weights = {"macd_diff": 0.35, "macd_hist_delta": 0.25, "ema_trend": 0.25, "atr_breakout": 0.15}
        elif mode == SignalMode.MEAN_REVERSION:
            components = {
                "rsi_centered": _clamp((50 - snap.rsi) / 50, -1, 1),
                "bb_position": _clamp(0.5 - snap.bb_position, -1, 1),
                "rsi_delta": _clamp(-snap.rsi_delta / 10, -1, 1),
            }
            weights = {"rsi_centered": 0.4, "bb_position": 0.4, "rsi_delta": 0.2}
        else:
            return 0.0, {}

        score = sum(components[k] * weights[k] for k in weights)
        components["volume_change"] = vol_change
        return score, components


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
