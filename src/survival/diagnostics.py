from __future__ import annotations

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.survival.catalog import DEATH_MODE_BY_ID, DIAGNOSTIC_ORDER
from src.survival.constants import (
    CRISIS_CORRELATION_THRESHOLD,
    INTERVENTION_LOOKBACK_DAYS,
    MIN_FILL_RATIO_EXECUTION,
    MIN_FILL_RATIO_LIQUIDITY,
    PROJECTED_IC_FLOOR,
    RESILIENCE_UNHEALTHY_LEVELS,
)
from src.survival.context import build_survival_context
from src.survival.types import DeathModeStatus, SurvivalContext, SurvivalReport, WeeklyCheckItem


class SurvivalDiagnostics:
    """Appendix B: aggregate health from existing Lesson modules (read-only)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def assess(
        self,
        ctx: SurvivalContext | None = None,
        *,
        connector: MT5Connector | None = None,
        store: OHLCVStore | None = None,
        pipeline=None,
    ) -> SurvivalReport:
        ctx = ctx or build_survival_context(
            self.config,
            connector=connector,
            store=store,
            pipeline=pipeline,
        )
        modes = [self._evaluate_mode(mode_id, ctx) for mode_id in DIAGNOSTIC_ORDER]
        weekly = self._weekly_checklist(ctx)
        warnings = list(ctx.data_warnings)
        warnings.extend(ctx.resilience_warnings)
        for mode in modes:
            if not mode.healthy:
                warnings.append(f"#{mode.mode_id} {mode.name}: {mode.detail}")
        return SurvivalReport(modes=modes, weekly_checklist=weekly, warnings=warnings)

    def build_context(self, **kwargs) -> SurvivalContext:
        return build_survival_context(self.config, **kwargs)

    def _evaluate_mode(self, mode_id: int, ctx: SurvivalContext) -> DeathModeStatus:
        info = DEATH_MODE_BY_ID[mode_id]
        healthy, detail = _MODE_CHECKS[mode_id](ctx, self.config)
        return DeathModeStatus(
            mode_id=mode_id,
            name=info.name,
            healthy=healthy,
            detail=detail,
            source_module=info.source_module,
            diagnostic_priority=DIAGNOSTIC_ORDER.index(mode_id) + 1,
        )

    def _weekly_checklist(self, ctx: SurvivalContext) -> list[WeeklyCheckItem]:
        cfg = self.config
        mapping = [
            ("Data quality checks passed", 1, ctx.data_valid, "bar validation"),
            (
                "Regime state normal",
                3,
                ctx.resilience_level.lower() not in RESILIENCE_UNHEALTHY_LEVELS,
                ctx.resilience_level,
            ),
            (
                "Execution slippage within range",
                4,
                ctx.avg_slippage_pct <= ctx.slippage_threshold_pct,
                f"{ctx.avg_slippage_pct:.3f}%",
            ),
            (
                "Risk thresholds not breached",
                5,
                not ctx.circuit_breaker_active and ctx.drawdown_pct < cfg.risk.drawdown_stop_pct,
                f"DD={ctx.drawdown_pct:.1f}%",
            ),
            (
                "Liquidity indicators normal",
                6,
                ctx.avg_fill_ratio >= MIN_FILL_RATIO_LIQUIDITY and ctx.cost_blocked_count == 0,
                f"fill={ctx.avg_fill_ratio:.0%}",
            ),
            (
                "Correlation matrix normal",
                7,
                ctx.max_correlation < CRISIS_CORRELATION_THRESHOLD,
                f"max={ctx.max_correlation:.2f}",
            ),
            (
                "Leverage within limits",
                8,
                ctx.portfolio_leverage <= cfg.portfolio.max_notional_leverage,
                f"{ctx.portfolio_leverage:.2f}x",
            ),
            (
                "No abnormal manual intervention",
                9,
                ctx.manual_interventions == 0,
                f"count={ctx.manual_interventions}",
            ),
            (
                "System health 100%",
                10,
                ctx.mt5_connected and ctx.ops_healthy,
                f"alerts={ctx.critical_alerts}",
            ),
            ("No regulatory changes", 11, ctx.regulatory_flags == 0, "manual monitor"),
            (
                "Alpha decay within range",
                12,
                not ctx.evolution_drift and ctx.decay_stage_count == 0,
                f"drift={ctx.evolution_drift}",
            ),
            (
                "OOS performance matches expectations",
                2,
                ctx.backtest_gate_configured,
                "run backtest validation weekly",
            ),
        ]
        return [
            WeeklyCheckItem(name=name, passed=passed, detail=str(detail), death_mode_id=mode_id)
            for name, mode_id, passed, detail in mapping
        ]


def _check_data_pollution(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    if ctx.data_valid:
        return True, "data quality checks passed"
    return False, "; ".join(ctx.data_warnings[:3]) or "data validation failed"


def _check_overfitting(ctx: SurvivalContext, config: AppConfig) -> tuple[bool, str]:
    if not ctx.backtest_gate_configured:
        return False, "backtest quality gate not configured"
    return True, (
        f"OOS gate configured (min Sharpe {config.project.min_backtest_sharpe:.1f}); "
        "run scripts/run_backtest_validation.py"
    )


def _check_regime_drift(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    if ctx.resilience_level.lower() in RESILIENCE_UNHEALTHY_LEVELS:
        return False, f"resilience level={ctx.resilience_level}"
    if ctx.resilience_warnings:
        return False, ctx.resilience_warnings[0]
    return True, f"regime health OK (level={ctx.resilience_level})"


def _check_execution_distortion(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    if not ctx.execution_enabled:
        return True, "execution disabled (paper mode)"
    if ctx.avg_slippage_pct > ctx.slippage_threshold_pct:
        return (
            False,
            f"avg slippage {ctx.avg_slippage_pct:.3f}% > "
            f"threshold {ctx.slippage_threshold_pct:.3f}%",
        )
    if ctx.avg_fill_ratio < MIN_FILL_RATIO_EXECUTION:
        return False, f"avg fill ratio {ctx.avg_fill_ratio:.0%} below 90%"
    return True, f"slippage={ctx.avg_slippage_pct:.3f}% fill={ctx.avg_fill_ratio:.0%}"


def _check_risk_failure(ctx: SurvivalContext, config: AppConfig) -> tuple[bool, str]:
    if not ctx.risk_circuit_configured:
        return False, "drawdown circuit breaker not configured"
    if ctx.circuit_breaker_active:
        return False, f"circuit breaker active at {ctx.drawdown_pct:.1f}% drawdown"
    if ctx.drawdown_pct >= config.risk.drawdown_stop_pct:
        return False, f"drawdown {ctx.drawdown_pct:.1f}% at stop tier"
    return True, f"risk OK (DD={ctx.drawdown_pct:.1f}%, circuit configured)"


def _check_liquidity(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    if ctx.cost_blocked_count > 0:
        return False, f"{ctx.cost_blocked_count} signals blocked as untradable"
    if ctx.avg_fill_ratio < MIN_FILL_RATIO_LIQUIDITY:
        return False, f"fill ratio {ctx.avg_fill_ratio:.0%} suggests liquidity stress"
    return True, "tradability and fill ratio within range"


def _check_correlation_spike(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    if ctx.max_correlation >= CRISIS_CORRELATION_THRESHOLD:
        return (
            False,
            f"max pairwise correlation {ctx.max_correlation:.2f} >= {CRISIS_CORRELATION_THRESHOLD}",
        )
    return True, f"correlation OK (max={ctx.max_correlation:.2f})"


def _check_leverage(ctx: SurvivalContext, config: AppConfig) -> tuple[bool, str]:
    limit = config.portfolio.max_notional_leverage
    if ctx.portfolio_leverage > limit:
        return False, f"notional leverage {ctx.portfolio_leverage:.2f}x > limit {limit:.2f}x"
    return True, f"leverage {ctx.portfolio_leverage:.2f}x <= limit {limit:.2f}x"


def _check_human_intervention(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    if ctx.manual_interventions > 0:
        return (
            False,
            f"{ctx.manual_interventions} manual interventions in last {INTERVENTION_LOOKBACK_DAYS} days",
        )
    return True, "no recent manual interventions logged"


def _check_system_failure(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    issues: list[str] = []
    if not ctx.mt5_connected:
        issues.append("MT5 disconnected")
    if not ctx.ops_healthy:
        issues.append("ops monitoring unhealthy")
    if ctx.critical_alerts > 0:
        issues.append(f"{ctx.critical_alerts} critical alerts")
    if issues:
        return False, "; ".join(issues)
    return True, "MT5 connected, ops healthy, no critical alerts"


def _check_regulatory(ctx: SurvivalContext, _config: AppConfig) -> tuple[bool, str]:
    if ctx.regulatory_flags > 0:
        return False, f"{ctx.regulatory_flags} regulatory flag(s) active"
    return True, "no regulatory flags (manual review recommended)"


def _check_counterparty_adaptation(ctx: SurvivalContext, config: AppConfig) -> tuple[bool, str]:
    if ctx.evolution_drift:
        return False, "evolution agent detected performance drift"
    if ctx.decay_stage_count > 0:
        return False, f"{ctx.decay_stage_count} strategy(ies) in decay stage"
    if (
        ctx.projected_ic_12m is not None
        and ctx.projected_ic_12m < PROJECTED_IC_FLOOR
        and config.online_learning.enabled
    ):
        return False, f"projected IC 12m={ctx.projected_ic_12m:.3f} below floor"
    return True, "alpha decay within expected range"


_MODE_CHECKS = {
    1: _check_data_pollution,
    2: _check_overfitting,
    3: _check_regime_drift,
    4: _check_execution_distortion,
    5: _check_risk_failure,
    6: _check_liquidity,
    7: _check_correlation_spike,
    8: _check_leverage,
    9: _check_human_intervention,
    10: _check_system_failure,
    11: _check_regulatory,
    12: _check_counterparty_adaptation,
}
