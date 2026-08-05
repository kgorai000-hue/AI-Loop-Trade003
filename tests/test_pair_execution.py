from __future__ import annotations

import unittest
from dataclasses import replace

from src.agents.pair_execution import (
    PreparedLeg,
    execute_pair_atomic,
    partition_signals,
    plan_is_filled,
)
from src.core.types import (
    ExecutionPlan,
    RiskDecision,
    RiskDecisionType,
    SignalSide,
    TradeSignal,
)
from src.market.symbol_info import MarketSymbolInfo, MarketType


def _signal(symbol: str, *, pair_id: str | None = None, trade_mode: str = "single") -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        side=SignalSide.BUY if symbol == "EURUSD" else SignalSide.SELL,
        timeframe="M30",
        strength=0.7,
        reason="test",
        pair_id=pair_id,
        trade_mode=trade_mode,
    )


def _plan(symbol: str, *, status: str = "planned", filled: float = 0.0, ticket: int | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        symbol=symbol,
        side=SignalSide.BUY,
        lots=0.02,
        order_type="market",
        estimated_cost_jpy=1.0,
        dry_run=True,
        reason="test",
        filled_lots=filled,
        status=status,
        ticket=ticket,
        trade_mode="pair",
        pair_id="pairs/EURUSD__GBPUSD",
    )


def _leg(symbol: str) -> PreparedLeg:
    info = MarketSymbolInfo(
        symbol=symbol,
        market_type=MarketType.FOREX,
        bid=1.0,
        ask=1.0001,
        spread_points=1,
        spread_price=0.0001,
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
        currency_base="USD",
        currency_profit="USD",
        currency_margin="USD",
        trade_mode=4,
        volume_real=0.0,
    )
    return PreparedLeg(
        signal=_signal(symbol, pair_id="pairs/EURUSD__GBPUSD", trade_mode="pair"),
        risk=RiskDecision(RiskDecisionType.APPROVE, 0.02, "ok"),
        plan=_plan(symbol),
        symbol_info=info,
        daily_vol=0.01,
        bars=[],
        atr=0.001,
    )


class PairExecutionTests(unittest.TestCase):
    def test_partition_preserves_pair_and_normalizes_mode(self) -> None:
        signals = [
            _signal("EURUSD", pair_id="pairs/EURUSD__GBPUSD", trade_mode="single"),
            _signal("GBPUSD", pair_id="pairs/EURUSD__GBPUSD", trade_mode="single"),
            _signal("USDJPY"),
        ]
        pairs, singles = partition_signals(signals)
        self.assertEqual(list(pairs.keys()), ["pairs/EURUSD__GBPUSD"])
        self.assertEqual(len(pairs["pairs/EURUSD__GBPUSD"]), 2)
        self.assertTrue(all(leg.trade_mode == "pair" for leg in pairs["pairs/EURUSD__GBPUSD"]))
        self.assertEqual([s.symbol for s in singles], ["USDJPY"])

    def test_both_legs_filled(self) -> None:
        calls: list[str] = []

        def execute_fn(leg: PreparedLeg) -> ExecutionPlan:
            calls.append(leg.signal.symbol)
            return replace(
                leg.plan,
                status="filled",
                filled_lots=0.02,
                ticket=100 + len(calls),
            )

        closes: list[str] = []

        def close_fn(plan: ExecutionPlan, reason: str) -> ExecutionPlan:
            closes.append(plan.symbol)
            return replace(plan, status="orphan_closed", reason=reason)

        plans = execute_pair_atomic(
            [_leg("EURUSD"), _leg("GBPUSD")],
            execute_fn=execute_fn,
            close_fn=close_fn,
            orphan_retries=3,
            retry_sleep_seconds=0,
        )
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(plan_is_filled(p) for p in plans))
        self.assertEqual(closes, [])

    def test_orphan_retry_then_recover(self) -> None:
        attempts = {"GBPUSD": 0}

        def execute_fn(leg: PreparedLeg) -> ExecutionPlan:
            if leg.signal.symbol == "EURUSD":
                return replace(leg.plan, status="filled", filled_lots=0.02, ticket=1)
            attempts["GBPUSD"] += 1
            if attempts["GBPUSD"] < 3:
                return replace(leg.plan, status="failed", filled_lots=0.0)
            return replace(leg.plan, status="filled", filled_lots=0.02, ticket=2)

        closes: list[str] = []

        def close_fn(plan: ExecutionPlan, reason: str) -> ExecutionPlan:
            closes.append(plan.symbol)
            return replace(plan, status="orphan_closed")

        plans = execute_pair_atomic(
            [_leg("EURUSD"), _leg("GBPUSD")],
            execute_fn=execute_fn,
            close_fn=close_fn,
            orphan_retries=3,
            retry_sleep_seconds=0,
        )
        self.assertEqual(attempts["GBPUSD"], 3)  # 1 initial fail + 2 retries then success on 3rd retry path
        # initial exec + retries until success: attempt 1 fail (initial), retry1 fail, retry2 success → 3 calls for GBPUSD
        self.assertTrue(plan_is_filled(plans[0]))
        self.assertTrue(plan_is_filled(plans[1]))
        self.assertEqual(closes, [])

    def test_orphan_close_after_retries(self) -> None:
        def execute_fn(leg: PreparedLeg) -> ExecutionPlan:
            if leg.signal.symbol == "EURUSD":
                return replace(leg.plan, status="filled", filled_lots=0.02, ticket=11)
            return replace(leg.plan, status="failed", filled_lots=0.0)

        closes: list[str] = []

        def close_fn(plan: ExecutionPlan, reason: str) -> ExecutionPlan:
            closes.append(plan.symbol)
            self.assertIn("orphan", reason)
            return replace(plan, status="orphan_closed", reason=reason)

        plans = execute_pair_atomic(
            [_leg("EURUSD"), _leg("GBPUSD")],
            execute_fn=execute_fn,
            close_fn=close_fn,
            orphan_retries=3,
            retry_sleep_seconds=0,
        )
        self.assertEqual(closes, ["EURUSD"])
        self.assertEqual(plans[-1].status, "orphan_closed")


if __name__ == "__main__":
    unittest.main()
