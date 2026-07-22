from __future__ import annotations

import math
import time
import uuid

from src.core.config import AppConfig
from src.core.types import SignalSide
from src.execution.costs.slippage import sqrt_slippage
from src.execution.execution_types import ExecutionFill, SimulatedExecution
from src.market.symbol_info import MarketSymbolInfo


class ConservativeExecutionSimulator:
    """
    Conservative execution simulator — avoids close-price fantasy (Lesson 19.8 Stage 1).
    Uses bid/ask, slippage, latency drift, and partial fills.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.execution_cfg = config.execution
        self.costs = config.costs

    def simulate(
        self,
        *,
        symbol_info: MarketSymbolInfo,
        side: SignalSide,
        lots: float,
        child_lots: list[float],
        daily_volatility: float,
        fantasy_price: float | None = None,
    ) -> SimulatedExecution:
        if lots <= 0:
            return SimulatedExecution(
                expected_price=0.0,
                average_fill_price=0.0,
                filled_lots=0.0,
                requested_lots=0.0,
                latency_ms=0.0,
                slippage_pct=0.0,
                fill_ratio=0.0,
                fills=[],
                status="skipped",
                reason="zero lots",
            )

        expected = fantasy_price or self._reference_price(symbol_info, side)
        conservative = self.execution_cfg.simulator_mode == "conservative"
        multiplier = self.execution_cfg.slippage_conservative_multiplier if conservative else 1.0

        fills: list[ExecutionFill] = []
        total_filled = 0.0
        weighted_price = 0.0
        base_latency = self.execution_cfg.latency_ms

        for idx, child_lot in enumerate(child_lots or [lots]):
            child_latency = base_latency * (1.0 + idx * 0.25)
            fill = self._simulate_child_fill(
                symbol_info=symbol_info,
                side=side,
                lots=child_lot,
                expected_price=expected,
                daily_volatility=daily_volatility,
                latency_ms=child_latency,
                multiplier=multiplier,
                fill_index=idx,
            )
            if fill.lots > 0:
                fills.append(fill)
                total_filled += fill.lots
                weighted_price += fill.price * fill.lots

        avg_price = weighted_price / total_filled if total_filled else expected
        slippage_pct = abs(avg_price - expected) / expected * 100.0 if expected else 0.0
        fill_ratio = total_filled / lots if lots else 0.0

        if fill_ratio >= 0.999:
            status = "filled"
        elif fill_ratio > 0:
            status = "partial"
        else:
            status = "rejected"

        return SimulatedExecution(
            expected_price=expected,
            average_fill_price=avg_price,
            filled_lots=total_filled,
            requested_lots=lots,
            latency_ms=base_latency * len(child_lots or [lots]),
            slippage_pct=slippage_pct,
            fill_ratio=fill_ratio,
            fills=fills,
            status=status,
            reason=f"conservative sim: {status} ({fill_ratio:.0%})",
        )

    def compare_close_vs_realistic(
        self,
        close_price: float,
        symbol_info: MarketSymbolInfo,
        side: SignalSide,
        lots: float,
        daily_volatility: float,
    ) -> dict[str, float]:
        """Demonstrate close-price fantasy vs bid/ask execution (Lesson 19.3.2)."""
        fantasy = close_price
        realistic = self.simulate(
            symbol_info=symbol_info,
            side=side,
            lots=lots,
            child_lots=[lots],
            daily_volatility=daily_volatility,
            fantasy_price=close_price,
        )
        fantasy_slippage = 0.0
        realistic_slippage = realistic.slippage_pct
        return {
            "fantasy_price": fantasy,
            "realistic_price": realistic.average_fill_price,
            "fantasy_slippage_pct": fantasy_slippage,
            "realistic_slippage_pct": realistic_slippage,
            "bias_pct": abs(realistic.average_fill_price - fantasy) / fantasy * 100.0 if fantasy else 0.0,
        }

    def _reference_price(self, symbol_info: MarketSymbolInfo, side: SignalSide) -> float:
        if self.execution_cfg.use_bid_ask_not_close:
            return symbol_info.ask if side == SignalSide.BUY else symbol_info.bid
        return (symbol_info.bid + symbol_info.ask) / 2.0

    def _simulate_child_fill(
        self,
        *,
        symbol_info: MarketSymbolInfo,
        side: SignalSide,
        lots: float,
        expected_price: float,
        daily_volatility: float,
        latency_ms: float,
        multiplier: float,
        fill_index: int,
    ) -> ExecutionFill:
        base = symbol_info.ask if side == SignalSide.BUY else symbol_info.bid
        notional = abs(symbol_info.contract_size * lots * base)
        adv = max(self.costs.default_adv_notional, 1.0)

        slip_rate = sqrt_slippage(notional, adv, daily_volatility, self.costs.slippage_k_sqrt)
        slip_rate = slip_rate * multiplier
        if side == SignalSide.BUY:
            slip_rate += symbol_info.spread_price / base if base else 0.0

        latency_days = latency_ms / (24.0 * 3600.0 * 1000.0)
        latency_drift = daily_volatility * math.sqrt(max(latency_days, 0.0)) * multiplier
        adverse = 1.0 if side == SignalSide.BUY else -1.0
        fill_price = base * (1.0 + adverse * (slip_rate + latency_drift))

        fill_ratio = 1.0
        if self.execution_cfg.partial_fill_enabled:
            liquidity = symbol_info.volume_max * self.execution_cfg.child_order_adv_fraction
            fill_ratio = min(1.0, liquidity / lots) if lots > liquidity else 1.0
            if self.execution_cfg.simulator_mode == "conservative" and fill_ratio > 0.5:
                fill_ratio = min(fill_ratio, 0.95)

        filled_lots = round(lots * fill_ratio, 2)
        slippage_pct = abs(fill_price - expected_price) / expected_price * 100.0 if expected_price else 0.0

        return ExecutionFill(
            fill_id=str(uuid.uuid4())[:8],
            lots=filled_lots,
            price=fill_price,
            timestamp_ms=int(time.time() * 1000) + fill_index * int(latency_ms),
            slippage_pct=slippage_pct,
        )
