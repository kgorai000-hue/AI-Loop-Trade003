from __future__ import annotations

from src.agents.registry import standard_agent_registry
from src.core.config import AppConfig
from src.core.data_manager import DataManager
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.system.types import AgentStatus, ChecklistResult, GraduationStage, IntegrationReport


def build_pre_live_checklist(
    config: AppConfig,
    *,
    connector_connected: bool,
    data_ready: bool,
    pipeline_ok: bool,
    backtest_passed: bool | None,
    ops_healthy: bool,
    dry_run: bool,
) -> list[ChecklistResult]:
    """Pre-live checklist from Lesson 21.4 (+ Trade003 demo_live)."""
    stage = str(config.project.graduation_stage).lower()
    if stage == "demo_live":
        mode_ok = (not dry_run) and str(config.account_type).lower() == "demo"
        mode_detail = "demo_live: dry_run=false + account_type=demo"
    else:
        mode_ok = dry_run
        mode_detail = "dry_run=true for paper stage"

    items = [
        ChecklistResult("system", "mt5_connected", connector_connected, "MT5 terminal connected"),
        ChecklistResult("system", "data_available", data_ready, "OHLCV data in store"),
        ChecklistResult("system", "pipeline_runs", pipeline_ok, "MetaAgent pipeline completes"),
        ChecklistResult("system", "ops_monitoring", ops_healthy, "OperationsAgent healthy"),
        ChecklistResult("system", "dry_run_enabled", mode_ok, mode_detail),
        ChecklistResult(
            "system",
            "require_demo",
            bool(config.mt5.require_demo) if stage == "demo_live" else True,
            "mt5.require_demo=true" if stage == "demo_live" else "n/a",
        ),
        ChecklistResult(
            "strategy",
            "backtest_quality_gate",
            backtest_passed is True,
            "backtest gate passed" if backtest_passed else "run scripts/run_backtest_validation.py",
        ),
        ChecklistResult(
            "strategy",
            "cost_model_enabled",
            config.costs.enabled,
            f"slippage model={config.costs.slippage_model}",
        ),
        ChecklistResult(
            "strategy",
            "risk_circuit_breaker",
            config.risk.drawdown_circuit_pct > 0,
            f"circuit at {config.risk.drawdown_circuit_pct}%",
        ),
        ChecklistResult(
            "ops",
            "alert_system",
            config.ops.enabled,
            f"suppress={config.ops.alert_suppress_seconds}s",
        ),
        ChecklistResult(
            "ops",
            "structured_logging",
            bool(config.ops.structured_log_dir),
            config.ops.structured_log_dir,
        ),
    ]
    return items


class SystemIntegrator:
    """Assess end-to-end system readiness (Lesson 21)."""

    def __init__(
        self,
        config: AppConfig,
        connector: MT5Connector,
        store: OHLCVStore,
    ) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.data_manager = DataManager(config, store)

    def assess(
        self,
        *,
        pipeline_ok: bool = False,
        backtest_passed: bool | None = None,
        ops_healthy: bool = True,
    ) -> IntegrationReport:
        stage = self.config.project.graduation_stage
        symbols = self.config.symbols[:3]
        data_ready = bool(symbols) and all(
            len(self.store.get_recent_bars(s, self.config.stats.analysis_timeframe, 10)) >= 10
            for s in symbols
        )

        agents = [
            AgentStatus(spec.name, spec.role.value, True, spec.metric)
            for spec in standard_agent_registry()
        ]

        checklist = build_pre_live_checklist(
            self.config,
            connector_connected=self.connector.is_connected,
            data_ready=data_ready,
            pipeline_ok=pipeline_ok,
            backtest_passed=backtest_passed,
            ops_healthy=ops_healthy,
            dry_run=self.config.trading.dry_run,
        )

        warnings: list[str] = []
        for symbol in symbols:
            valid, errors = self.data_manager.validate(symbol)
            if not valid:
                warnings.extend(f"{symbol}: {e}" for e in errors[:2])

        paper_ready = all(c.passed for c in checklist if c.category in ("system", "strategy"))
        live_ready = paper_ready and backtest_passed is True and not self.config.trading.dry_run

        if stage == GraduationStage.BACKTEST.value and not backtest_passed:
            warnings.append("graduation_stage=backtest: run backtest validation before paper")

        return IntegrationReport(
            graduation_stage=stage,
            agents=agents,
            data_ready=data_ready,
            pipeline_ok=pipeline_ok,
            backtest_gate_passed=backtest_passed,
            pre_live_checklist=checklist,
            ready_for_paper=paper_ready and self.config.trading.dry_run,
            ready_for_live=live_ready,
            warnings=warnings,
        )
