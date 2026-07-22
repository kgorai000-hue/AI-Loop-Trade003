from __future__ import annotations

import math

import pytest

from src.core.config import ExecutionConfig
from src.core.types import SignalSide
from src.execution.order_router import route_order
from src.execution.simulator import ConservativeExecutionSimulator
from src.execution.telemetry import ExecutionLogRecord, ExecutionTelemetryStore, new_record_id
from src.market.symbol_info import MarketSymbolInfo, MarketType
from src.market.symbol_registry import CanonicalSymbol, MT5SymbolAdapter, SymbolRegistry


def _sample_symbol_info(**overrides) -> MarketSymbolInfo:
    defaults = dict(
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
    defaults.update(overrides)
    return MarketSymbolInfo(**defaults)


def _execution_cfg(**overrides) -> ExecutionConfig:
    defaults = dict(
        max_retries=3,
        retry_backoff_seconds=1.0,
        circuit_breaker_threshold=5,
        enabled=True,
        simulator_mode="conservative",
        latency_ms=200.0,
        partial_fill_enabled=True,
        max_child_orders=5,
        child_order_adv_fraction=0.01,
        quote_max_age_seconds=30.0,
        log_executions=True,
        telemetry_db_path=":memory:",
        slippage_conservative_multiplier=1.5,
        use_bid_ask_not_close=True,
        slippage_threshold_pct=0.05,
        latency_warn_ms=500.0,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def test_symbol_canonical_roundtrip() -> None:
    adapter = MT5SymbolAdapter()
    canonical = adapter.to_canonical("#USSPX500")
    assert str(canonical) == "SPX.MT5"
    assert adapter.to_broker(canonical) == "#USSPX500"


def test_symbol_registry_parse() -> None:
    parsed = CanonicalSymbol.parse("AAPL.NASDAQ")
    assert parsed.symbol == "AAPL"
    assert parsed.venue == "NASDAQ"


def test_order_router_splits_large_orders() -> None:
    cfg = _execution_cfg(max_child_orders=3, child_order_adv_fraction=0.05)
    info = _sample_symbol_info(volume_max=10.0)
    routing = route_order(2.0, info, cfg)
    assert routing.algo == "split_twap"
    assert len(routing.child_lots) >= 2
    assert math.isclose(sum(routing.child_lots), 2.0, rel_tol=0.01)


def test_conservative_simulator_has_slippage() -> None:
    from src.core.config import load_config

    sim = ConservativeExecutionSimulator(load_config())
    result = sim.simulate(
        symbol_info=_sample_symbol_info(),
        side=SignalSide.BUY,
        lots=1.0,
        child_lots=[1.0],
        daily_volatility=0.01,
        fantasy_price=1.08505,
    )
    assert result.slippage_pct > 0
    assert result.average_fill_price > 1.08505
    assert result.status in ("filled", "partial")


def test_close_fantasy_has_zero_slippage() -> None:
    from src.core.config import load_config

    sim = ConservativeExecutionSimulator(load_config())
    comparison = sim.compare_close_vs_realistic(
        close_price=1.08505,
        symbol_info=_sample_symbol_info(),
        side=SignalSide.BUY,
        lots=1.0,
        daily_volatility=0.01,
    )
    assert comparison["fantasy_slippage_pct"] == 0.0
    assert comparison["realistic_slippage_pct"] > 0


def test_telemetry_store_records_execution() -> None:
    store = ExecutionTelemetryStore(":memory:")
    record_id = new_record_id()
    store.record(
        ExecutionLogRecord(
            record_id=record_id,
            timestamp=1_700_000_000,
            symbol="EURUSD",
            canonical_symbol="EURUSD.MT5",
            side="buy",
            order_type="market",
            expected_price=1.08510,
            average_fill_price=1.08515,
            requested_lots=1.0,
            filled_lots=1.0,
            slippage_pct=0.005,
            latency_ms=200.0,
            fill_ratio=1.0,
            commission_jpy=0.0,
            dry_run=True,
            status="filled",
            child_orders=1,
            reason="test",
        )
    )
    records = store.recent(1)
    assert len(records) == 1
    assert records[0].record_id == record_id


def test_quote_staleness_validation() -> None:
    import time

    registry = SymbolRegistry()
    assert registry.validate_quote_age(time.time() - 10, 30.0)
    assert not registry.validate_quote_age(time.time() - 60, 30.0)
    assert not registry.validate_quote_age(None, 30.0)
