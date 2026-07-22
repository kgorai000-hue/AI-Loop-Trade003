from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import PositionAction
from src.data.store import OHLCVStore
from src.features.indicators import bars_to_arrays, latest_snapshot

logger = logging.getLogger(__name__)


class PositionAgent:
    """Manage exits with ATR-based stops when enabled (Lesson 04)."""

    def __init__(
        self,
        config: AppConfig,
        connector: MT5Connector,
        store: OHLCVStore,
    ) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.risk = config.risk
        self.indicators = config.indicators

    def review_positions(self) -> list[PositionAction]:
        import MetaTrader5 as mt5

        actions: list[PositionAction] = []
        positions = mt5.positions_get()
        if not positions:
            return actions

        for pos in positions:
            action = self._evaluate_position(pos)
            if action is not None:
                actions.append(action)

        return actions

    def _evaluate_position(self, pos) -> PositionAction | None:
        symbol = str(pos.symbol)
        open_price = float(pos.price_open)
        current = float(pos.price_current)
        if open_price <= 0:
            return None

        is_sell = pos.type == 1
        price_move = open_price - current if is_sell else current - open_price

        if self.risk.use_atr_stops:
            atr_stop = self._atr_stop_distance(symbol)
            if atr_stop is not None and atr_stop > 0:
                if price_move <= -atr_stop:
                    return PositionAction(
                        symbol=symbol,
                        action="exit_atr_stop_loss",
                        reason=f"price move {price_move:.5f} <= -{atr_stop:.5f} (ATR x {self.indicators.atr_stop_multiplier})",
                    )
                if price_move >= atr_stop * (self.risk.take_profit_pct / self.risk.stop_loss_pct):
                    return PositionAction(
                        symbol=symbol,
                        action="exit_atr_take_profit",
                        reason=f"price move {price_move:.5f} >= ATR-based target",
                    )
                return None

        pnl_pct = (price_move / open_price) * 100.0
        if pnl_pct <= -self.risk.stop_loss_pct:
            return PositionAction(
                symbol=symbol,
                action="exit_stop_loss",
                reason=f"PnL {pnl_pct:.2f}% <= -{self.risk.stop_loss_pct:.2f}%",
            )

        if pnl_pct >= self.risk.take_profit_pct:
            return PositionAction(
                symbol=symbol,
                action="exit_take_profit",
                reason=f"PnL {pnl_pct:.2f}% >= {self.risk.take_profit_pct:.2f}%",
            )

        return None

    def _atr_stop_distance(self, symbol: str) -> float | None:
        bars = self.store.get_recent_bars(
            symbol,
            self.config.stats.signal_timeframe,
            self.config.history_bars_for(self.config.stats.signal_timeframe),
        )
        if len(bars) < self.config.stats.min_bars:
            return None

        opens, highs, lows, closes, volumes = bars_to_arrays(bars)
        snap = latest_snapshot(
            opens,
            highs,
            lows,
            closes,
            volumes,
            macd_fast=self.indicators.macd_fast,
            macd_slow=self.indicators.macd_slow,
            macd_signal=self.indicators.macd_signal,
            macd_histogram_double=self.indicators.macd_histogram_double,
            rsi_period=self.indicators.rsi_period,
            bb_period=self.indicators.bb_period,
            bb_std=self.indicators.bb_std,
            atr_period=self.indicators.atr_period,
        )
        if snap is None or snap.atr <= 0:
            return None
        return snap.atr * self.indicators.atr_stop_multiplier
