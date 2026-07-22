from __future__ import annotations

import time
import uuid

from src.core.config import AppConfig
from src.core.types import TradeSignal
from src.execution.execution_types import SimulatedExecution
from src.execution.telemetry import new_record_id
from src.market.symbol_info import MarketSymbolInfo
from src.ops.logging import new_trace_id
from src.trading_log.store import LiveTradeLogStore
from src.trading_log.types import (
    AgentDecisionMeta,
    FillLogRecord,
    MarketSnapshot,
    OrderLogRecord,
    TradeLogBundle,
    TradeMetrics,
)


class TradeLogRecorder:
    """Build and persist Appendix A compliant trade logs."""

    def __init__(self, config: AppConfig, store: LiveTradeLogStore | None = None) -> None:
        self.config = config
        self.cfg = config.trade_log
        db_path = self.cfg.db_path if store is None else store.db_path
        self.store = store or LiveTradeLogStore(db_path)

    def record_execution(
        self,
        *,
        plan_symbol: str,
        plan_side: str,
        plan_order_type: str,
        plan_lots: float,
        plan_dry_run: bool,
        expected_price: float | None,
        simulated: SimulatedExecution,
        symbol_info: MarketSymbolInfo,
        commission_jpy: float = 0.0,
        signal: TradeSignal | None = None,
        trace_id: str | None = None,
        recent_bars: list[dict] | None = None,
        atr: float = 0.0,
    ) -> str:
        order_id = new_record_id()
        trace = trace_id or new_trace_id("ord")
        submit_ms = int(time.time() * 1000)

        order_price = expected_price or symbol_info.ask
        order = OrderLogRecord(
            order_id=order_id,
            trace_id=trace,
            symbol=plan_symbol,
            side=plan_side,
            order_type=plan_order_type,
            order_price=order_price,
            order_qty=plan_lots,
            submit_ts_ms=submit_ms,
            status=simulated.status,
            dry_run=plan_dry_run,
        )

        fills = [
            FillLogRecord(
                fill_id=fill.fill_id,
                order_id=order_id,
                fill_price=fill.price,
                fill_qty=fill.lots,
                fill_ts_ms=fill.timestamp_ms,
                slippage_pct=fill.slippage_pct,
            )
            for fill in simulated.fills
        ]
        if not fills and simulated.filled_lots > 0:
            fills.append(
                FillLogRecord(
                    fill_id=str(uuid.uuid4())[:8],
                    order_id=order_id,
                    fill_price=simulated.average_fill_price,
                    fill_qty=simulated.filled_lots,
                    fill_ts_ms=submit_ms + int(simulated.latency_ms),
                    slippage_pct=simulated.slippage_pct,
                )
            )

        realized = 0.0
        if simulated.filled_lots > 0 and simulated.expected_price:
            direction = 1.0 if plan_side.lower() == "buy" else -1.0
            realized = (
                direction
                * (simulated.average_fill_price - simulated.expected_price)
                * symbol_info.contract_size
                * simulated.filled_lots
                - commission_jpy
            )

        metrics = TradeMetrics(
            order_id=order_id,
            expected_price=simulated.expected_price,
            average_fill_price=simulated.average_fill_price,
            slippage_pct=simulated.slippage_pct,
            latency_ms=simulated.latency_ms,
            fill_ratio=simulated.fill_ratio,
            commission=commission_jpy,
            tax=0.0,
            realized_pnl=realized,
        )

        market = self._build_market_snapshot(order_id, symbol_info, recent_bars, atr)
        agent = self._build_agent_meta(order_id, trace, plan_lots, signal)

        self.store.record_bundle(
            TradeLogBundle(
                order=order,
                fills=fills,
                metrics=metrics,
                market=market,
                agent=agent,
            )
        )
        return order_id

    def _build_market_snapshot(
        self,
        order_id: str,
        symbol_info: MarketSymbolInfo,
        recent_bars: list[dict] | None,
        atr: float,
    ) -> MarketSnapshot:
        if recent_bars:
            bar = recent_bars[-1]
            o = float(bar["open"])
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])
            vwap = (o + h + l + c) / 4.0
            bar_ts = int(bar["time"])
        else:
            o = h = l = c = symbol_info.bid
            vwap = (symbol_info.bid + symbol_info.ask) / 2.0
            bar_ts = int(time.time())

        return MarketSnapshot(
            order_id=order_id,
            bar_ts=bar_ts,
            bar_open=o,
            bar_high=h,
            bar_low=l,
            bar_close=c,
            bar_vwap=vwap,
            atr_5min=atr,
            bid1=symbol_info.bid,
            ask1=symbol_info.ask,
        )

    def _build_agent_meta(
        self,
        order_id: str,
        trace_id: str,
        target_lots: float,
        signal: TradeSignal | None,
    ) -> AgentDecisionMeta | None:
        if signal is None:
            return None

        action = signal.side.value
        confidence = signal.confidence if signal.confidence is not None else signal.strength
        predicted = signal.predicted_return

        return AgentDecisionMeta(
            order_id=order_id,
            trace_id=trace_id,
            agent_id="PortfolioAgent",
            agent_version=self.cfg.agent_version,
            action=action,
            target_position=target_lots,
            confidence=confidence,
            signal_strength=signal.strength,
            predicted_return=predicted,
            strategy=signal.strategy.value,
        )
