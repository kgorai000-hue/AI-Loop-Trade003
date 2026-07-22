from __future__ import annotations

import logging
import time
from typing import Any

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import ExecutionPlan, RiskDecision, TradeSignal
from src.execution.cost_model import CostModel
from src.execution.exceptions import (
    ApiTimeoutError,
    DataMissingError,
    ExecutionAttempt,
    ExecutionError,
    OrderRejectedError,
)
from src.execution.execution_types import ExecutionLogRecord
from src.execution.order_router import route_order
from src.execution.simulator import ConservativeExecutionSimulator
from src.execution.telemetry import ExecutionTelemetryStore, new_record_id
from src.market.symbol_info import MarketSymbolInfo
from src.market.symbol_registry import SymbolRegistry
from src.trading_log.recorder import TradeLogRecorder

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """Plan and execute orders with conservative simulation and telemetry (Lesson 19)."""

    def __init__(self, config: AppConfig, connector: MT5Connector, cost_model: CostModel) -> None:
        self.config = config
        self.connector = connector
        self.cost_model = cost_model
        self.execution_cfg = config.execution
        self.simulator = ConservativeExecutionSimulator(config)
        self.symbol_registry = SymbolRegistry(config)
        self.telemetry = ExecutionTelemetryStore(self.execution_cfg.telemetry_db_path)
        self.trade_log = TradeLogRecorder(config) if config.trade_log.enabled else None
        self._consecutive_failures = 0
        self._circuit_open = False

    def plan(
        self,
        signal: TradeSignal,
        risk: RiskDecision,
        symbol_info: MarketSymbolInfo,
        daily_volatility: float,
    ) -> ExecutionPlan:
        lots = risk.approved_lots
        routing = route_order(lots, symbol_info, self.execution_cfg)
        cost = self.cost_model.estimate_trade_cost(
            symbol_info=symbol_info,
            lots=lots,
            side=signal.side,
            daily_volatility=daily_volatility,
        )

        expected_price = (
            symbol_info.ask if signal.side.value == "buy" else symbol_info.bid
        )
        canonical = str(self.symbol_registry.to_canonical(signal.symbol))

        return ExecutionPlan(
            symbol=signal.symbol,
            side=signal.side,
            lots=lots,
            order_type=routing.order_type,
            estimated_cost_jpy=cost.total_cost_jpy,
            dry_run=self.config.trading.dry_run,
            expected_price=expected_price,
            child_orders=len(routing.child_lots) or 1,
            algo=routing.algo,
            canonical_symbol=canonical,
            group_id=signal.group_id,
            trade_mode=signal.trade_mode,
            pair_id=signal.pair_id,
            reason=(
                f"{routing.algo} {routing.order_type}; est. cost {cost.total_cost_jpy:.2f} JPY "
                f"({cost.total_cost_pct_of_notional:.4f}% of notional) | {routing.reason}"
                f" | mode={signal.trade_mode} group={signal.group_id or '-'}"
            ),
        )

    def execute(
        self,
        plan: ExecutionPlan,
        *,
        symbol_info: MarketSymbolInfo | None = None,
        daily_volatility: float = 0.02,
        child_lots: list[float] | None = None,
        signal: TradeSignal | None = None,
        recent_bars: list[dict] | None = None,
        atr: float = 0.0,
        trace_id: str | None = None,
    ) -> ExecutionPlan:
        if self._circuit_open:
            raise OrderRejectedError(
                "circuit breaker open after repeated execution failures; fail-fast"
            )

        if plan.lots <= 0:
            plan.status = "skipped"
            return plan

        if symbol_info is not None:
            tick_time = getattr(symbol_info, "tick_time", None)
            if tick_time is not None and not self.symbol_registry.validate_quote_age(
                tick_time, self.execution_cfg.quote_max_age_seconds
            ):
                raise DataMissingError(
                    f"stale quote for {plan.symbol}; exceeds {self.execution_cfg.quote_max_age_seconds}s"
                )

        if self.config.trading.dry_run:
            return self._execute_dry_run(
                plan,
                symbol_info,
                daily_volatility,
                child_lots,
                signal=signal,
                recent_bars=recent_bars,
                atr=atr,
                trace_id=trace_id,
            )

        attempts: list[ExecutionAttempt] = []
        last_error: ExecutionError | None = None

        for attempt in range(1, self.execution_cfg.max_retries + 1):
            try:
                result = self._execute_live(
                    plan,
                    symbol_info=symbol_info,
                    signal=signal,
                    recent_bars=recent_bars,
                    atr=atr,
                    trace_id=trace_id,
                )
                self._consecutive_failures = 0
                return result
            except ExecutionError as exc:
                last_error = exc
                attempts.append(
                    ExecutionAttempt(
                        attempt=attempt,
                        success=False,
                        message=str(exc),
                        exception_kind=exc.kind,
                    )
                )
                logger.warning(
                    "Execution attempt %d/%d failed (%s): %s",
                    attempt,
                    self.execution_cfg.max_retries,
                    exc.kind.value,
                    exc,
                )
                if attempt < self.execution_cfg.max_retries:
                    time.sleep(self.execution_cfg.retry_backoff_seconds * attempt)

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.execution_cfg.circuit_breaker_threshold:
            self._circuit_open = True
            logger.error("Execution circuit breaker opened after %d failures", self._consecutive_failures)

        if last_error is not None:
            raise last_error
        raise ApiTimeoutError("execution failed without specific error")

    def _execute_dry_run(
        self,
        plan: ExecutionPlan,
        symbol_info: MarketSymbolInfo | None,
        daily_volatility: float,
        child_lots: list[float] | None,
        *,
        signal: TradeSignal | None = None,
        recent_bars: list[dict] | None = None,
        atr: float = 0.0,
        trace_id: str | None = None,
    ) -> ExecutionPlan:
        if not self.execution_cfg.enabled or symbol_info is None:
            logger.info("[DRY RUN] %s %s %.2f lots | %s", plan.side.value.upper(), plan.symbol, plan.lots, plan.reason)
            plan.status = "simulated"
            return plan

        routing = route_order(plan.lots, symbol_info, self.execution_cfg)
        lots_list = child_lots or routing.child_lots or [plan.lots]
        simulated = self.simulator.simulate(
            symbol_info=symbol_info,
            side=plan.side,
            lots=plan.lots,
            child_lots=lots_list,
            daily_volatility=daily_volatility,
            fantasy_price=plan.expected_price,
        )

        plan.average_fill_price = simulated.average_fill_price
        plan.filled_lots = simulated.filled_lots
        plan.latency_ms = simulated.latency_ms
        plan.slippage_pct = simulated.slippage_pct
        plan.fill_ratio = simulated.fill_ratio
        plan.status = simulated.status
        plan.reason = f"[DRY RUN] {simulated.reason}; {plan.reason}"

        if self.execution_cfg.log_executions:
            record_id = new_record_id()
            plan.execution_record_id = self.telemetry.record(
                ExecutionLogRecord(
                    record_id=record_id,
                    timestamp=int(time.time()),
                    symbol=plan.symbol,
                    canonical_symbol=plan.canonical_symbol or plan.symbol,
                    side=plan.side.value,
                    order_type=plan.order_type,
                    expected_price=simulated.expected_price,
                    average_fill_price=simulated.average_fill_price,
                    requested_lots=plan.lots,
                    filled_lots=simulated.filled_lots,
                    slippage_pct=simulated.slippage_pct,
                    latency_ms=simulated.latency_ms,
                    fill_ratio=simulated.fill_ratio,
                    commission_jpy=plan.estimated_cost_jpy,
                    dry_run=True,
                    status=simulated.status,
                    child_orders=plan.child_orders,
                    reason=plan.reason,
                )
            )

        if self.trade_log is not None and symbol_info is not None:
            order_id = self.trade_log.record_execution(
                plan_symbol=plan.symbol,
                plan_side=plan.side.value,
                plan_order_type=plan.order_type,
                plan_lots=plan.lots,
                plan_dry_run=plan.dry_run,
                expected_price=plan.expected_price,
                simulated=simulated,
                symbol_info=symbol_info,
                commission_jpy=plan.estimated_cost_jpy,
                signal=signal,
                trace_id=trace_id,
                recent_bars=recent_bars,
                atr=atr,
            )
            plan.execution_record_id = order_id

        logger.info(
            "[DRY RUN] %s %s req=%.2f fill=%.2f @ %.5f slip=%.3f%% latency=%.0fms | %s",
            plan.side.value.upper(),
            plan.symbol,
            plan.lots,
            plan.filled_lots,
            plan.average_fill_price or 0.0,
            plan.slippage_pct,
            plan.latency_ms,
            plan.status,
        )
        return plan

    def _execute_live(
        self,
        plan: ExecutionPlan,
        *,
        symbol_info: MarketSymbolInfo | None = None,
        signal: TradeSignal | None = None,
        recent_bars: list[dict] | None = None,
        atr: float = 0.0,
        trace_id: str | None = None,
    ) -> ExecutionPlan:
        import MetaTrader5 as mt5

        if not self.execution_cfg.enabled:
            raise OrderRejectedError("execution.enabled is false; refusing live order")

        raw_account = mt5.account_info()
        if raw_account is None or not self.connector.is_demo_account(raw_account):
            raise OrderRejectedError("live order blocked: account is not DEMO")

        info = self.connector.symbol_info(plan.symbol)
        tick = self.connector.symbol_info_tick(plan.symbol)
        if info is None or tick is None:
            raise DataMissingError(f"missing tick/info for {plan.symbol}")

        mid = (float(tick.bid) + float(tick.ask)) / 2.0 if tick.bid and tick.ask else float(tick.ask or tick.bid or 0.0)
        spread_pct = 0.0
        if mid > 0:
            spread_pct = abs(float(tick.ask) - float(tick.bid)) / mid * 100.0
        if spread_pct > self.execution_cfg.slippage_threshold_pct:
            plan.lots = 0.0
            plan.status = "skipped"
            plan.reason = (
                f"Execution Guard: spread {spread_pct:.4f}% > "
                f"{self.execution_cfg.slippage_threshold_pct:.4f}%; lots=0"
            )
            logger.warning("%s", plan.reason)
            return plan

        resolved = self.connector.ensure_symbol(plan.symbol)
        order_type = (
            mt5.ORDER_TYPE_BUY if plan.side.value == "buy" else mt5.ORDER_TYPE_SELL
        )
        price = float(tick.ask if plan.side.value == "buy" else tick.bid)
        filling = self._deal_filling_mode(info)

        started = time.time()
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": resolved,
            "volume": float(plan.lots),
            "type": order_type,
            "price": price,
            "deviation": int(self.config.mt5.deviation),
            "magic": int(self.config.mt5.magic),
            "comment": self._order_comment(plan),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = self.connector.order_send(request)
        latency_ms = (time.time() - started) * 1000.0
        if result is None:
            err = self.connector.last_error()
            raise ApiTimeoutError(f"order_send returned None: {err}")

        retcode = int(result.retcode)
        ok_codes = {
            int(mt5.TRADE_RETCODE_DONE),
            int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008)),
        }
        partial_code = int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))
        if retcode == partial_code:
            raise OrderRejectedError(
                f"partial fill treated as incomplete retcode={retcode} comment={getattr(result, 'comment', '')}"
            )
        if retcode not in ok_codes:
            raise OrderRejectedError(
                f"order rejected retcode={retcode} comment={getattr(result, 'comment', '')}"
            )

        fill_price = float(getattr(result, "price", 0.0) or price)
        filled = float(getattr(result, "volume", 0.0) or plan.lots)
        slip_pct = 0.0
        if plan.expected_price and plan.expected_price > 0:
            slip_pct = abs(fill_price - plan.expected_price) / plan.expected_price * 100.0
        if slip_pct > self.execution_cfg.slippage_threshold_pct:
            # Fail-fast on excessive slippage after fill: freeze new entries via circuit.
            self._circuit_open = True
            logger.error(
                "Fail Fast: slippage %.4f%% > %.4f%% after fill ticket=%s; circuit open",
                slip_pct,
                self.execution_cfg.slippage_threshold_pct,
                getattr(result, "order", None),
            )

        plan.average_fill_price = fill_price
        plan.filled_lots = filled
        plan.latency_ms = latency_ms
        plan.slippage_pct = slip_pct
        plan.fill_ratio = filled / plan.lots if plan.lots > 0 else 0.0
        plan.status = "filled"
        plan.ticket = int(getattr(result, "order", 0) or getattr(result, "deal", 0) or 0)
        plan.dry_run = False
        plan.reason = (
            f"[DEMO-LIVE] ticket={plan.ticket} deal={getattr(result, 'deal', None)} "
            f"mode={plan.trade_mode} group={plan.group_id or '-'} pair={plan.pair_id or '-'} | "
            f"{plan.reason}"
        )

        if self.execution_cfg.log_executions:
            record_id = new_record_id()
            plan.execution_record_id = self.telemetry.record(
                ExecutionLogRecord(
                    record_id=record_id,
                    timestamp=int(time.time()),
                    symbol=plan.symbol,
                    canonical_symbol=plan.canonical_symbol or plan.symbol,
                    side=plan.side.value,
                    order_type=plan.order_type,
                    expected_price=plan.expected_price or price,
                    average_fill_price=fill_price,
                    requested_lots=plan.lots,
                    filled_lots=filled,
                    slippage_pct=slip_pct,
                    latency_ms=latency_ms,
                    fill_ratio=plan.fill_ratio,
                    commission_jpy=plan.estimated_cost_jpy,
                    dry_run=False,
                    status=plan.status,
                    child_orders=plan.child_orders,
                    reason=plan.reason,
                )
            )

        logger.info(
            "[DEMO-LIVE] %s %s ticket=%s req=%.2f fill=%.2f @ %.5f slip=%.3f%% "
            "latency=%.0fms group=%s mode=%s",
            plan.side.value.upper(),
            plan.symbol,
            plan.ticket,
            plan.lots,
            plan.filled_lots,
            plan.average_fill_price or 0.0,
            plan.slippage_pct,
            plan.latency_ms,
            plan.group_id or "-",
            plan.trade_mode,
        )
        return plan

    @staticmethod
    def _deal_filling_mode(info: Any) -> int:
        import MetaTrader5 as mt5

        filling = int(getattr(info, "filling_mode", 0) or 0)
        # SYMBOL_FILLING_FOK=1, IOC=2, RETURN=4 (bit flags)
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    @staticmethod
    def _order_comment(plan: ExecutionPlan) -> str:
        mode = (plan.trade_mode or "s")[0]
        group = (plan.group_id or "G")[-2:]
        base = f"T3{mode}{group}"
        return base[:31]

    def summarize_telemetry(self, limit: int = 50):
        return self.telemetry.summarize(limit)

    def summarize_trade_log(self, limit: int = 50):
        if self.trade_log is None:
            return None
        return self.trade_log.store.summarize(limit)
