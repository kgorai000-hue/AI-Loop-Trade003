from __future__ import annotations

import logging
import time

from typing import TYPE_CHECKING

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import ExecutionPlan, MultiAgentReport
from src.data.store import OHLCVStore
from src.ops.alerts import AlertManager
from src.ops.checklist import post_market_checklist, pre_market_checklist, reconcile_positions
from src.ops.logging import StructuredTradeLogger, new_trace_id
from src.ops.monitoring import FourLayerMonitor, build_monitoring_context
from src.ops.schedule import MarketSessionScheduler
from src.ops.types import OpsReport

if TYPE_CHECKING:
    from src.agents.meta_agent import PipelineResult

logger = logging.getLogger(__name__)


class OperationsAgent:
    """Production operations: 4-layer monitoring, alerts, checklists (Lesson 20)."""

    def __init__(self, config: AppConfig, connector: MT5Connector, store: OHLCVStore) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.cfg = config.ops
        self.monitor = FourLayerMonitor(self.cfg)
        self.alerts = AlertManager(self.cfg)
        self.scheduler = MarketSessionScheduler()
        self.trade_logger = StructuredTradeLogger(self.cfg.structured_log_dir)
        self._last_heartbeat = time.time()

    def assess(
        self,
        pipeline_result: "PipelineResult | None" = None,
        multi_agent_report: MultiAgentReport | None = None,
    ) -> OpsReport:
        if not self.cfg.enabled:
            return OpsReport(enabled=False)

        pipeline_result = pipeline_result or PipelineResult()
        phase = self.scheduler.current_phase().value
        trading_allowed = self.scheduler.trading_allowed()

        daily_dd = 0.0
        if pipeline_result.risk_control_report:
            daily_dd = pipeline_result.risk_control_report.drawdown_pct

        max_trade_pct = 0.0
        for report in pipeline_result.decision_reports:
            max_trade_pct = max(max_trade_pct, report.final_position_pct)

        ctx = build_monitoring_context(
            self.connector,
            agent_latency_ms=multi_agent_report.parallel_elapsed_ms if multi_agent_report else 0.0,
            bus_backlog=multi_agent_report.bus_events if multi_agent_report else 0,
            daily_drawdown_pct=daily_dd,
            weekly_drawdown_pct=daily_dd,
            max_trade_position_pct=max_trade_pct,
            trades_today=len(pipeline_result.execution_plans),
            trades_daily_mean=float(self.config.trading.trades_per_day),
        )
        ctx.heartbeat_age_seconds = time.time() - self._last_heartbeat
        self._last_heartbeat = time.time()

        metrics = self.monitor.evaluate(ctx)
        alerts = self.alerts.from_metrics(metrics)
        for alert in alerts:
            if not alert.suppressed:
                self.alerts.send(alert)

        checklist = pre_market_checklist(
            self.connector,
            self.store,
            self.config.symbols,
            self.config.stats.analysis_timeframe,
            self.cfg,
            dry_run=self.config.trading.dry_run,
        )
        checklist.extend(
            post_market_checklist(
                execution_count=len(pipeline_result.execution_plans),
                alert_count=sum(1 for a in alerts if not a.suppressed),
                pnl_available=pipeline_result.risk_control_report is not None,
            )
        )

        broker_positions = self._broker_position_map()
        local_positions = self._local_position_map(pipeline_result)
        reconcile_warnings = reconcile_positions(local_positions, broker_positions)

        critical = sum(1 for m in metrics if not m.healthy and m.severity.value == "critical")
        warnings = [m.message for m in metrics if not m.healthy and m.message]
        warnings.extend(reconcile_warnings)

        if not trading_allowed and self.cfg.enforce_market_hours:
            warnings.append(f"outside active trading window (phase={phase})")
            trading_allowed = False

        trace_id = new_trace_id("ops")
        self.trade_logger.log_event(
            level="INFO",
            service="operations_agent",
            event="ops_assessment",
            trace_id=trace_id,
            data={
                "phase": phase,
                "critical_count": critical,
                "alert_count": len(alerts),
                "healthy_metrics": sum(1 for m in metrics if m.healthy),
            },
            context={"trading_allowed": trading_allowed},
        )

        for plan in pipeline_result.execution_plans[:3]:
            self._log_execution(plan, trace_id)

        return OpsReport(
            enabled=True,
            session_phase=phase,
            trading_allowed=trading_allowed and critical == 0,
            metrics=metrics,
            alerts=alerts,
            checklist=checklist,
            healthy=critical == 0 and all(item.passed for item in checklist[:3]),
            critical_count=critical,
            warnings=warnings,
        )

    def _log_execution(self, plan: ExecutionPlan, trace_id: str) -> None:
        self.trade_logger.log_event(
            level="INFO",
            service="execution_agent",
            event="order_simulated" if plan.dry_run else "order_submitted",
            trace_id=trace_id,
            data={
                "symbol": plan.symbol,
                "side": plan.side.value.upper(),
                "lots": plan.lots,
                "order_type": plan.order_type,
                "fill_price": plan.average_fill_price,
                "slippage_pct": plan.slippage_pct,
            },
            context={"status": plan.status, "algo": plan.algo},
        )

    def _broker_position_map(self) -> dict[str, float]:
        try:
            from src.ops.providers import MT5ExecutionVenue

            venue = MT5ExecutionVenue(self.connector, dry_run=self.config.trading.dry_run)
            return {p.symbol: p.lots for p in venue.get_positions()}
        except Exception:
            return {}

    @staticmethod
    def _local_position_map(result: "PipelineResult") -> dict[str, float]:
        lots: dict[str, float] = {}
        for plan in result.execution_plans:
            if plan.filled_lots > 0:
                lots[plan.symbol] = lots.get(plan.symbol, 0.0) + plan.filled_lots
        return lots
