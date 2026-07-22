from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import ExecutionPlan, MonitorReport

logger = logging.getLogger(__name__)


class MonitorAgent:
    """Monitor execution quality: slippage, fill rate, latency (Lesson 19.13)."""

    def __init__(self, config: AppConfig, slippage_threshold_pct: float | None = None) -> None:
        self.config = config
        self.execution_cfg = config.execution
        self.slippage_threshold_pct = (
            slippage_threshold_pct
            if slippage_threshold_pct is not None
            else self.execution_cfg.slippage_threshold_pct
        )

    def review(self, plan: ExecutionPlan) -> MonitorReport:
        if plan.status == "skipped":
            return MonitorReport(symbol=plan.symbol, status="skipped", message="zero-lot plan skipped")

        if plan.dry_run and plan.status in ("simulated", "filled", "partial"):
            issues: list[str] = []
            if plan.slippage_pct > self.slippage_threshold_pct:
                issues.append(
                    f"slippage {plan.slippage_pct:.4f}% > threshold {self.slippage_threshold_pct:.4f}%"
                )
            if plan.fill_ratio < 1.0:
                issues.append(f"partial fill {plan.fill_ratio:.0%}")
            if plan.latency_ms > self.execution_cfg.latency_warn_ms:
                issues.append(
                    f"latency {plan.latency_ms:.0f}ms > warn {self.execution_cfg.latency_warn_ms:.0f}ms"
                )

            if issues:
                return MonitorReport(
                    symbol=plan.symbol,
                    status="pause",
                    message="; ".join(issues),
                )

            return MonitorReport(
                symbol=plan.symbol,
                status="ok",
                message=(
                    f"simulated {plan.status}: slip={plan.slippage_pct:.4f}% "
                    f"fill={plan.fill_ratio:.0%} latency={plan.latency_ms:.0f}ms"
                ),
            )

        if plan.dry_run:
            return MonitorReport(
                symbol=plan.symbol,
                status="simulated",
                message="dry run - no live execution to monitor",
            )

        cost_pct = (plan.estimated_cost_jpy / max(plan.lots, 0.01)) if plan.lots else 0.0
        if cost_pct > self.slippage_threshold_pct:
            return MonitorReport(
                symbol=plan.symbol,
                status="pause",
                message=f"estimated cost {cost_pct:.4f} exceeds threshold {self.slippage_threshold_pct}",
            )

        return MonitorReport(
            symbol=plan.symbol,
            status="ok",
            message="execution quality within threshold",
        )
