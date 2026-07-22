from __future__ import annotations

import numpy as np

from src.core.config import AppConfig
from src.data.quality import check_data_quality
from src.data.store import BarRecord, OHLCVStore
from src.features.indicators import compute_indicators, latest_atr_from_bars, sma
from src.stats.returns import log_returns
from src.stats.risk import volatility


class DataManager:
    """
    Unified data access layer (Lesson 21.2 Step 1).
    MT5-only source; wraps OHLCVStore + indicator computation.
    """

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store

    def get_history(
        self,
        symbol: str,
        timeframe: str | None = None,
        bars: int | None = None,
    ) -> list[dict]:
        timeframe = timeframe or self.config.stats.analysis_timeframe
        bars = bars or self.config.history_bars_for(timeframe)
        return self.store.get_recent_bars(symbol, timeframe, bars)

    def get_latest(self, symbols: list[str], timeframe: str | None = None) -> dict[str, float]:
        timeframe = timeframe or self.config.stats.signal_timeframe
        prices: dict[str, float] = {}
        for symbol in symbols:
            bars = self.store.get_recent_bars(symbol, timeframe, 1)
            if bars:
                prices[symbol] = float(bars[-1]["close"])
        return prices

    def calculate_indicators(
        self,
        symbol: str,
        timeframe: str | None = None,
        bars: int | None = None,
    ) -> dict[str, float]:
        raw = self.get_history(symbol, timeframe, bars)
        if len(raw) < 30:
            return {}

        opens = np.array([float(b["open"]) for b in raw])
        highs = np.array([float(b["high"]) for b in raw])
        lows = np.array([float(b["low"]) for b in raw])
        closes = np.array([float(b["close"]) for b in raw])
        volumes = np.array([float(b["tick_volume"]) for b in raw])

        ind = compute_indicators(
            opens,
            highs,
            lows,
            closes,
            volumes,
            macd_fast=self.config.indicators.macd_fast,
            macd_slow=self.config.indicators.macd_slow,
            macd_signal=self.config.indicators.macd_signal,
            rsi_period=self.config.indicators.rsi_period,
            bb_period=self.config.indicators.bb_period,
            atr_period=self.config.indicators.atr_period,
            adx_period=14,
        )

        idx = len(closes) - 1
        rets = log_returns(closes)
        ann_vol = volatility(rets, annualize=True) if len(rets) else 0.0
        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)

        return {
            "close": float(closes[idx]),
            "rsi": float(ind["rsi"][idx]) if not np.isnan(ind["rsi"][idx]) else 50.0,
            "adx": float(ind["adx"][idx]) if not np.isnan(ind["adx"][idx]) else 20.0,
            "atr": latest_atr_from_bars(raw, self.config.indicators.atr_period),
            "volatility": ann_vol,
            "sma_20": float(sma20[idx]) if not np.isnan(sma20[idx]) else float(closes[idx]),
            "sma_50": float(sma50[idx]) if idx >= 49 and not np.isnan(sma50[idx]) else float(closes[idx]),
        }

    def validate(self, symbol: str, timeframe: str | None = None) -> tuple[bool, list[str]]:
        timeframe = timeframe or self.config.stats.analysis_timeframe
        raw = self.get_history(symbol, timeframe, self.config.stats.min_bars)
        if not raw:
            return False, ["no bars in store"]

        records = [
            BarRecord(
                symbol=symbol,
                timeframe=timeframe,
                time=int(b["time"]),
                open=float(b["open"]),
                high=float(b["high"]),
                low=float(b["low"]),
                close=float(b["close"]),
                tick_volume=int(b["tick_volume"]),
                spread=int(b["spread"]),
                real_volume=int(b["real_volume"]),
            )
            for b in raw
        ]

        report = check_data_quality(records, symbol, timeframe, self.config.data.quality)
        errors = list(report.anomalies)
        if not report.is_valid:
            errors.append(f"missing_rate={report.missing_rate_pct:.1f}%")
        return report.is_valid, errors
