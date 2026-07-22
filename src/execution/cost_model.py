from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.config import AppConfig, CostConfig, TradingConfig
from src.core.types import SignalSide
from src.market.symbol_info import MarketSymbolInfo


@dataclass
class TradeCostBreakdown:
    notional_jpy: float
    spread_cost_jpy: float
    slippage_cost_jpy: float
    commission_cost_jpy: float
    market_impact_cost_jpy: float
    total_cost_jpy: float
    total_cost_pct_of_notional: float

    @property
    def total_cost_pct_of_capital(self) -> float:
        return self.total_cost_pct_of_notional


@dataclass
class StrategyCostProjection:
    backtest_return_pct: float
    annual_cost_on_notional_pct: float
    annual_cost_on_capital_pct: float
    live_return_pct: float
    trades_per_day: int
    trading_days: int
    capital_jpy: float
    trade_notional_jpy: float


class CostModel:
    """Lesson 02: spread, slippage, commission, market impact estimation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.costs = config.costs
        self.trading = config.trading

    def estimate_trade_cost(
        self,
        symbol_info: MarketSymbolInfo,
        lots: float,
        side: SignalSide,
        daily_volatility: float | None = None,
        slippage_rate: float | None = None,
    ) -> TradeCostBreakdown:
        price = symbol_info.ask if side == SignalSide.BUY else symbol_info.bid
        notional_jpy = self._notional_jpy(symbol_info, lots, price)

        spread_cost_jpy = self._spread_cost_jpy(symbol_info, lots)
        slippage_cost_jpy = self._slippage_cost_jpy(
            notional_jpy,
            symbol_info,
            lots,
            slippage_rate,
        )
        commission_cost_jpy = self._commission_cost_jpy(symbol_info, lots)
        market_impact_cost_jpy = self._market_impact_cost_jpy(
            notional_jpy,
            daily_volatility,
        )

        total = spread_cost_jpy + slippage_cost_jpy + commission_cost_jpy + market_impact_cost_jpy
        pct = (total / notional_jpy * 100.0) if notional_jpy else 0.0

        return TradeCostBreakdown(
            notional_jpy=notional_jpy,
            spread_cost_jpy=spread_cost_jpy,
            slippage_cost_jpy=slippage_cost_jpy,
            commission_cost_jpy=commission_cost_jpy,
            market_impact_cost_jpy=market_impact_cost_jpy,
            total_cost_jpy=total,
            total_cost_pct_of_notional=pct,
        )

    def project_strategy_costs(
        self,
        backtest_return_pct: float,
        capital_jpy: float,
        trade_notional_jpy: float,
        cost_per_trade_pct: float,
        trades_per_day: int | None = None,
        trading_days: int | None = None,
    ) -> StrategyCostProjection:
        trades_per_day = trades_per_day or self.trading.trades_per_day
        trading_days = trading_days or self.trading.trading_days_per_year

        annual_notional = trade_notional_jpy * trades_per_day * trading_days
        annual_cost_jpy = annual_notional * (cost_per_trade_pct / 100.0)

        annual_cost_on_notional_pct = (annual_cost_jpy / annual_notional * 100.0) if annual_notional else 0.0
        annual_cost_on_capital_pct = (annual_cost_jpy / capital_jpy * 100.0) if capital_jpy else 0.0
        live_return_pct = backtest_return_pct - annual_cost_on_capital_pct

        return StrategyCostProjection(
            backtest_return_pct=backtest_return_pct,
            annual_cost_on_notional_pct=annual_cost_on_notional_pct,
            annual_cost_on_capital_pct=annual_cost_on_capital_pct,
            live_return_pct=live_return_pct,
            trades_per_day=trades_per_day,
            trading_days=trading_days,
            capital_jpy=capital_jpy,
            trade_notional_jpy=trade_notional_jpy,
        )

    def estimate_daily_volatility(self, closes: list[float]) -> float:
        if len(closes) < 2:
            return 0.02

        returns = []
        for prev, curr in zip(closes[:-1], closes[1:]):
            if prev:
                returns.append((curr - prev) / prev)

        if not returns:
            return 0.02

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return max(math.sqrt(variance), 0.0001)

    def _notional_jpy(self, symbol_info: MarketSymbolInfo, lots: float, price: float) -> float:
        # contract_size * lots * price = exposure in profit currency (USD for most symbols)
        return abs(symbol_info.contract_size * lots * price)

    def _spread_cost_jpy(self, symbol_info: MarketSymbolInfo, lots: float) -> float:
        return abs(symbol_info.spread_price * symbol_info.contract_size * lots)

    def _slippage_cost_jpy(
        self,
        notional_jpy: float,
        symbol_info: MarketSymbolInfo,
        lots: float,
        slippage_rate: float | None,
    ) -> float:
        rate = slippage_rate if slippage_rate is not None else self.costs.slippage_rate
        if rate > 0:
            return notional_jpy * rate
        if self.costs.use_spread_when_slippage_unknown:
            return self._spread_cost_jpy(symbol_info, lots)
        return 0.0

    def _commission_cost_jpy(self, symbol_info: MarketSymbolInfo, lots: float) -> float:
        if symbol_info.commission_is_zero:
            return 0.0
        # commission field semantics vary by broker; treat as per-lot charge in profit currency
        return abs(symbol_info.commission * lots)

    def _market_impact_cost_jpy(
        self,
        notional_jpy: float,
        daily_volatility: float | None,
    ) -> float:
        sigma = daily_volatility if daily_volatility is not None else 0.02
        # Lesson 02 sqrt law with participation ~ notional fraction proxy
        participation = min(max(notional_jpy / self.costs.default_adv_notional, 1e-6), 1.0)
        impact_rate = self.costs.market_impact_Y * sigma * math.sqrt(participation)
        return notional_jpy * impact_rate
