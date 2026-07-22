from __future__ import annotations

import time

from src.core.config import load_config
from src.core.types import SignalSide, StrategyKind, TradeSignal
from src.execution.simulator import ConservativeExecutionSimulator
from src.market.symbol_info import MarketSymbolInfo, MarketType
from src.features.indicators import latest_atr_from_bars
from src.trading_log.recorder import TradeLogRecorder
from src.trading_log.store import LiveTradeLogStore
from src.trading_log.types import (
    AgentDecisionMeta,
    FillLogRecord,
    MarketSnapshot,
    OrderLogRecord,
    TradeLogBundle,
    TradeMetrics,
)


def _symbol_info() -> MarketSymbolInfo:
    return MarketSymbolInfo(
        symbol="EURUSD",
        market_type=MarketType.FOREX,
        bid=1.08500,
        ask=1.08510,
        spread_points=10,
        spread_price=0.00010,
        point=0.00001,
        digits=5,
        contract_size=100_000,
        tick_size=0.00001,
        tick_value=1.0,
        tick_value_profit=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        commission=0.0,
        commission_mode=0,
        swap_long=0.0,
        swap_short=0.0,
        currency_base="EUR",
        currency_profit="USD",
        currency_margin="EUR",
        trade_mode=4,
        volume_real=0.0,
    )


def test_live_trade_log_store_roundtrip() -> None:
    store = LiveTradeLogStore(":memory:")
    order_id = "ord-test-1"
    submit_ms = int(time.time() * 1000)

    bundle = TradeLogBundle(
        order=OrderLogRecord(
            order_id=order_id,
            trace_id="trace-1",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            order_price=1.08510,
            order_qty=0.10,
            submit_ts_ms=submit_ms,
            status="filled",
            dry_run=True,
        ),
        fills=[
            FillLogRecord(
                fill_id="fill-1",
                order_id=order_id,
                fill_price=1.08512,
                fill_qty=0.10,
                fill_ts_ms=submit_ms + 200,
                slippage_pct=0.002,
            )
        ],
        metrics=TradeMetrics(
            order_id=order_id,
            expected_price=1.08510,
            average_fill_price=1.08512,
            slippage_pct=0.002,
            latency_ms=200.0,
            fill_ratio=1.0,
            commission=50.0,
            tax=0.0,
            realized_pnl=-50.0,
        ),
        market=MarketSnapshot(
            order_id=order_id,
            bar_ts=submit_ms // 1000,
            bar_open=1.08500,
            bar_high=1.08520,
            bar_low=1.08490,
            bar_close=1.08510,
            bar_vwap=1.08505,
            atr_5min=0.00050,
            bid1=1.08500,
            ask1=1.08510,
        ),
        agent=AgentDecisionMeta(
            order_id=order_id,
            trace_id="trace-1",
            agent_id="PortfolioAgent",
            agent_version="test_v1",
            action="buy",
            target_position=0.10,
            confidence=0.75,
            signal_strength=0.8,
            predicted_return=0.01,
            strategy="trend_following",
        ),
    )

    store.record_bundle(bundle)
    orders = store.recent_orders(5)
    assert len(orders) == 1
    assert orders[0].order_id == order_id

    fills = store.fills_for_order(order_id)
    assert len(fills) == 1
    assert fills[0].fill_qty == 0.10

    summary = store.summarize()
    assert summary.order_count == 1
    assert summary.fill_count == 1
    assert summary.avg_fill_ratio == 1.0


def test_trade_log_recorder_from_simulation() -> None:
    config = load_config()
    store = LiveTradeLogStore(":memory:")
    recorder = TradeLogRecorder(config, store=store)
    symbol_info = _symbol_info()

    sim = ConservativeExecutionSimulator(config)
    simulated = sim.simulate(
        symbol_info=symbol_info,
        side=SignalSide.BUY,
        lots=0.10,
        child_lots=[0.10],
        daily_volatility=0.02,
        fantasy_price=1.08510,
    )

    signal = TradeSignal(
        symbol="EURUSD",
        side=SignalSide.BUY,
        timeframe="M5",
        strength=0.7,
        reason="test signal",
        strategy=StrategyKind.TREND_FOLLOWING,
        confidence=0.65,
        predicted_return=0.008,
        requested_lots=0.10,
    )
    bars = [
        {
            "time": int(time.time()) - 300,
            "open": 1.08490,
            "high": 1.08520,
            "low": 1.08480,
            "close": 1.08500,
            "tick_volume": 100,
        },
        {
            "time": int(time.time()),
            "open": 1.08500,
            "high": 1.08530,
            "low": 1.08495,
            "close": 1.08510,
            "tick_volume": 120,
        },
    ]

    order_id = recorder.record_execution(
        plan_symbol="EURUSD",
        plan_side="buy",
        plan_order_type="market",
        plan_lots=0.10,
        plan_dry_run=True,
        expected_price=1.08510,
        simulated=simulated,
        symbol_info=symbol_info,
        commission_jpy=25.0,
        signal=signal,
        trace_id="trace-rec",
        recent_bars=bars,
        atr=0.00045,
    )

    assert order_id
    orders = store.recent_orders(1)
    assert orders[0].symbol == "EURUSD"
    fills = store.fills_for_order(order_id)
    assert fills
    summary = store.summarize()
    assert summary.order_count >= 1


def test_atr_from_bars() -> None:
    bars = []
    base = 1.08000
    for i in range(30):
        o = base + i * 0.00010
        bars.append(
            {
                "time": 1_700_000_000 + i * 300,
                "open": o,
                "high": o + 0.00020,
                "low": o - 0.00010,
                "close": o + 0.00005,
                "tick_volume": 50 + i,
            }
        )

    atr = latest_atr_from_bars(bars, atr_period=14)
    assert atr > 0.0
