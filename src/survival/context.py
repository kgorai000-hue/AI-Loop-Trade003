from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config import AppConfig
from src.core.data_manager import DataManager
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.extensions.human_intervention import HumanInterventionLog
from src.extensions.regulatory import RegulatoryMonitor
from src.survival.constants import DATA_VALIDATION_SYMBOL_LIMIT, INTERVENTION_LOOKBACK_DAYS
from src.survival.types import SurvivalContext

if TYPE_CHECKING:
    from src.agents.meta_agent import PipelineResult


def build_survival_context(
    config: AppConfig,
    *,
    connector: MT5Connector | None = None,
    store: OHLCVStore | None = None,
    pipeline: "PipelineResult | None" = None,
) -> SurvivalContext:
    ctx = SurvivalContext(
        slippage_threshold_pct=config.execution.slippage_threshold_pct,
        risk_circuit_configured=config.risk.drawdown_circuit_pct > 0,
        backtest_gate_configured=config.project.min_backtest_sharpe > 0,
        execution_enabled=config.execution.enabled,
    )

    if connector is not None:
        ctx.mt5_connected = connector.is_connected

    if store is not None:
        dm = DataManager(config, store)
        all_valid = True
        for symbol in config.symbols[:DATA_VALIDATION_SYMBOL_LIMIT]:
            valid, errors = dm.validate(symbol)
            if not valid:
                all_valid = False
                ctx.data_warnings.extend(f"{symbol}: {e}" for e in errors[:2])
        ctx.data_valid = all_valid

    intervention_log = HumanInterventionLog(config.ops.structured_log_dir)
    ctx.manual_interventions = intervention_log.recent_count(days=INTERVENTION_LOOKBACK_DAYS)

    regulatory = RegulatoryMonitor()
    ctx.regulatory_flags = len(regulatory.active_flags())

    if pipeline is not None:
        enrich_context_from_pipeline(ctx, pipeline)

    return ctx


def enrich_context_from_pipeline(ctx: SurvivalContext, pipeline: "PipelineResult") -> None:
    if pipeline.resilience_report:
        rep = pipeline.resilience_report
        ctx.resilience_level = rep.level_name
        ctx.resilience_warnings = list(rep.warnings)
        ctx.data_valid = ctx.data_valid and rep.data_quality_ok

    if pipeline.risk_control_report:
        rc = pipeline.risk_control_report
        ctx.drawdown_pct = rc.drawdown_pct
        ctx.circuit_breaker_active = rc.circuit_breaker_active

    slippage, fill_ratio = _execution_metrics_from_pipeline(pipeline)
    if slippage is not None:
        ctx.avg_slippage_pct = slippage
    if fill_ratio is not None:
        ctx.avg_fill_ratio = fill_ratio

    if pipeline.cost_report:
        ctx.cost_blocked_count = pipeline.cost_report.blocked_count

    if pipeline.portfolio_report:
        pr = pipeline.portfolio_report
        ctx.portfolio_leverage = pr.notional_leverage
        ctx.max_correlation = pr.max_correlation

    if pipeline.evolution_report:
        ev = pipeline.evolution_report
        ctx.evolution_drift = ev.drift_detected
        ctx.projected_ic_12m = ev.projected_ic_12m
        ctx.decay_stage_count = sum(
            1 for s in ev.strategy_states if s.stage.value == "decay"
        )

    if pipeline.ops_report:
        op = pipeline.ops_report
        ctx.ops_healthy = op.healthy
        ctx.critical_alerts = op.critical_count


def _execution_metrics_from_pipeline(
    pipeline: "PipelineResult",
) -> tuple[float | None, float | None]:
    if pipeline.trade_log_report and pipeline.trade_log_report.order_count > 0:
        tl = pipeline.trade_log_report
        return tl.avg_slippage_pct, tl.avg_fill_ratio
    if pipeline.execution_report and pipeline.execution_report.records:
        er = pipeline.execution_report
        return er.avg_slippage_pct, er.avg_fill_ratio
    return None, None
