from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.history import periods_per_year_for_timeframe
from src.core.types import SymbolStatsReport
from src.data.store import OHLCVStore
from src.features.divergence import detect_divergence
from src.features.feature_vector import FeatureEngine
from src.features.indicators import bars_to_arrays
from src.stats.returns import annualize_return, cumulative_return, log_returns
from src.stats.risk import excess_kurtosis, skewness, tail_warning, volatility
from src.stats.timeseries import (
    adf_stationarity,
    autocorrelation,
    detect_regime,
    volatility_autocorrelation,
)

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Statistical + technical feature analysis (Lesson 03-04)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.stats = config.stats
        self.engine = FeatureEngine(config)

    def analyze_symbol(
        self,
        symbol: str,
        timeframe: str | None = None,
    ) -> SymbolStatsReport | None:
        timeframe = timeframe or self.stats.analysis_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < self.stats.min_bars:
            logger.debug("Insufficient bars for %s %s: %d", symbol, timeframe, len(bars))
            return None

        closes = [float(bar["close"]) for bar in bars]
        returns = log_returns(closes)
        if len(returns) < self.stats.min_bars - 1:
            return None

        periods = periods_per_year_for_timeframe(
            timeframe, self.config.trading.trading_days_per_year
        )
        cum_log = cumulative_return(returns, method="log")
        ann_return = annualize_return(cum_log, len(returns), periods)

        daily_vol = volatility(returns, annualize=False)
        ann_vol = volatility(returns, annualize=True, trading_days=periods)

        ac1 = autocorrelation(returns, lag=1)
        ac5 = autocorrelation(returns, lag=5 if len(returns) > 5 else 1)
        skew = skewness(returns)
        kurt = excess_kurtosis(returns)
        price_adf = adf_stationarity(closes)
        return_adf = adf_stationarity(returns)
        vol_ac = volatility_autocorrelation(returns, self.stats.vol_window)
        regime, _, _ = detect_regime(
            returns,
            vol_crisis=self.stats.regime_vol_crisis,
            vol_bull_max=self.stats.regime_vol_bull_max,
            lookback=self.stats.vol_window,
            trading_days=periods,
        )

        features = self.engine.build_from_bars(symbol, bars, timeframe, regime)
        series = self.engine.compute_series(bars)
        macd_div = None
        rsi_div = None
        if series is not None:
            import numpy as np

            closes_arr = np.array(closes, dtype=float)
            lookback = self.config.indicators.divergence_lookback
            macd_div_type = detect_divergence(closes_arr, series["macd_diff"], lookback)
            rsi_div_type = detect_divergence(closes_arr, series["rsi"], lookback)
            macd_div = macd_div_type.value if macd_div_type else None
            rsi_div = rsi_div_type.value if rsi_div_type else None

        snap = features.snapshot if features else None

        return SymbolStatsReport(
            symbol=symbol,
            timeframe=timeframe,
            bars=len(bars),
            cumulative_log_return=cum_log,
            annualized_return=ann_return,
            daily_volatility=daily_vol,
            annualized_volatility=ann_vol,
            autocorr_lag1=ac1,
            autocorr_lag5=ac5,
            skewness=skew,
            excess_kurtosis=kurt,
            price_stationary=bool(price_adf["stationary"]),
            return_stationary=bool(return_adf["stationary"]),
            vol_autocorr=vol_ac,
            regime=regime,
            tail_warning=tail_warning(skew, kurt),
            rsi=snap.rsi if snap else None,
            macd_diff=snap.macd_diff if snap else None,
            macd_histogram=snap.macd_histogram if snap else None,
            bb_position=snap.bb_position if snap else None,
            atr=snap.atr if snap else None,
            atr_pct=snap.atr_pct if snap else None,
            macd_divergence=macd_div,
            rsi_divergence=rsi_div,
        )

    def analyze_all(self, symbols: list[str] | None = None) -> list[SymbolStatsReport]:
        symbols = symbols or self.config.symbols
        reports: list[SymbolStatsReport] = []
        for symbol in symbols:
            report = self.analyze_symbol(symbol)
            if report is not None:
                reports.append(report)
        return reports
