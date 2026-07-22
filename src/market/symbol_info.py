from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import MarketType


@dataclass
class MarketSymbolInfo:
    symbol: str
    market_type: MarketType
    bid: float
    ask: float
    spread_points: int
    spread_price: float
    point: float
    digits: int
    contract_size: float
    tick_size: float
    tick_value: float
    tick_value_profit: float
    volume_min: float
    volume_max: float
    volume_step: float
    commission: float
    commission_mode: int
    swap_long: float
    swap_short: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    trade_mode: int
    volume_real: float

    @property
    def commission_is_zero(self) -> bool:
        return self.commission == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market_type": self.market_type.value,
            "bid": self.bid,
            "ask": self.ask,
            "spread_points": self.spread_points,
            "spread_price": self.spread_price,
            "point": self.point,
            "digits": self.digits,
            "contract_size": self.contract_size,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "tick_value_profit": self.tick_value_profit,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "commission": self.commission,
            "commission_mode": self.commission_mode,
            "swap_long": self.swap_long,
            "swap_short": self.swap_short,
            "currency_base": self.currency_base,
            "currency_profit": self.currency_profit,
            "currency_margin": self.currency_margin,
            "trade_mode": self.trade_mode,
            "volume_real": self.volume_real,
            "commission_is_zero": self.commission_is_zero,
        }


def resolve_market_type(symbol: str, config: AppConfig) -> MarketType:
    for name in config.symbol_groups.get("indices", []):
        if name == symbol:
            return MarketType.CFD_INDEX
    for name in config.symbol_groups.get("commodities", []):
        if name == symbol:
            return MarketType.COMMODITY
    return MarketType.FOREX


def fetch_market_symbol_info(
    connector: MT5Connector,
    config: AppConfig,
    symbol: str,
) -> MarketSymbolInfo:
    import MetaTrader5 as mt5

    resolved = connector.ensure_symbol(symbol)
    info = mt5.symbol_info(resolved)
    tick = mt5.symbol_info_tick(resolved)

    if info is None or tick is None:
        raise RuntimeError(f"symbol_info unavailable for {resolved}: {mt5.last_error()}")

    spread_price = (tick.ask - tick.bid) if tick.ask and tick.bid else info.spread * info.point

    return MarketSymbolInfo(
        symbol=resolved,
        market_type=resolve_market_type(symbol, config),
        bid=float(tick.bid),
        ask=float(tick.ask),
        spread_points=int(info.spread),
        spread_price=float(spread_price),
        point=float(info.point),
        digits=int(info.digits),
        contract_size=float(info.trade_contract_size),
        tick_size=float(info.trade_tick_size),
        tick_value=float(info.trade_tick_value),
        tick_value_profit=float(info.trade_tick_value_profit),
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        commission=float(getattr(info, "commission", 0.0) or 0.0),
        commission_mode=int(getattr(info, "commission_mode", 0) or 0),
        swap_long=float(info.swap_long),
        swap_short=float(info.swap_short),
        currency_base=str(info.currency_base),
        currency_profit=str(info.currency_profit),
        currency_margin=str(info.currency_margin),
        trade_mode=int(info.trade_mode),
        volume_real=float(getattr(info, "volume_real", 0.0) or 0.0),
    )
