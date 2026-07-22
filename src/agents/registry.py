from __future__ import annotations

from src.core.types import AgentRole, AgentSpec


def standard_agent_registry() -> list[AgentSpec]:
    """Standard multi-agent responsibilities (Lesson 11.4)."""
    return [
        AgentSpec(
            role=AgentRole.META,
            name="MetaAgent",
            responsibilities=["orchestration", "arbitration", "scheduling", "health monitoring"],
            not_responsible=["signal generation", "order execution"],
            metric="system health",
        ),
        AgentSpec(
            role=AgentRole.META,
            name="TradingSystem",
            responsibilities=[
                "end-to-end integration",
                "data sync + pipeline orchestration",
                "graduation stage management",
                "pre-live checklist",
            ],
            not_responsible=["individual agent logic"],
            metric="pipeline completion rate",
        ),
        AgentSpec(
            role=AgentRole.DATA,
            name="DataAgent",
            responsibilities=["MT5 fetch", "validation", "lineage"],
            not_responsible=["trading decisions", "risk"],
            metric="latency, completeness",
        ),
        AgentSpec(
            role=AgentRole.RESEARCH,
            name="ResearchAgent",
            responsibilities=["symbol statistics", "feature summaries"],
            not_responsible=["signals", "execution"],
            metric="analysis coverage",
        ),
        AgentSpec(
            role=AgentRole.REGIME,
            name="RegimeAgent",
            responsibilities=["market state detection", "strategy routing"],
            not_responsible=["trade execution", "position sizing"],
            metric="switch accuracy, latency",
        ),
        AgentSpec(
            role=AgentRole.SIGNAL,
            name="PortfolioAgent",
            responsibilities=["signal generation", "return prediction"],
            not_responsible=["final risk approval", "execution"],
            metric="IC, IR, directional accuracy",
        ),
        AgentSpec(
            role=AgentRole.SIGNAL,
            name="PortfolioConstructionAgent",
            responsibilities=[
                "weight allocation",
                "covariance shrinkage",
                "factor exposure monitoring",
                "leverage control",
            ],
            not_responsible=["signal generation", "risk veto", "execution"],
            metric="risk contribution balance, factor limits",
        ),
        AgentSpec(
            role=AgentRole.DECISION,
            name="DecisionAgent",
            responsibilities=["prediction to sized orders", "position sizing"],
            not_responsible=["risk veto", "execution"],
            metric="sizing quality",
        ),
        AgentSpec(
            role=AgentRole.RESEARCH,
            name="EvolutionAgent",
            responsibilities=[
                "concept drift detection",
                "update scheduling",
                "strategy lifecycle",
                "dynamic signal thresholds",
            ],
            not_responsible=["order execution", "risk veto"],
            metric="IC decay, drift alerts",
        ),
        AgentSpec(
            role=AgentRole.RISK,
            name="RiskAgent",
            responsibilities=[
                "order review",
                "veto power",
                "drawdown circuit breaker",
                "position/symbol/sector limits",
                "audit logging",
            ],
            not_responsible=["signal quality"],
            metric="max drawdown, VaR",
        ),
        AgentSpec(
            role=AgentRole.EXECUTION,
            name="CostEstimatorAgent",
            responsibilities=[
                "slippage and impact estimation",
                "gross-to-net alpha assessment",
                "tradability veto before execution",
                "fill probability and order/ADV checks",
            ],
            not_responsible=["signal generation", "risk limits"],
            metric="net alpha after costs, blocked trades",
        ),
        AgentSpec(
            role=AgentRole.EXECUTION,
            name="ExecutionAgent",
            responsibilities=[
                "order routing and child-order splitting",
                "conservative execution simulation",
                "bid/ask execution (not close-price fantasy)",
                "execution telemetry logging",
            ],
            not_responsible=["signals", "risk limits"],
            metric="slippage, fill rate, latency",
        ),
        AgentSpec(
            role=AgentRole.MONITOR,
            name="OperationsAgent",
            responsibilities=[
                "4-layer health monitoring",
                "structured logging and alerts",
                "pre/post market checklists",
                "disaster recovery and state reconciliation",
                "canary deployment and auto rollback",
            ],
            not_responsible=["signal generation", "order execution"],
            metric="system uptime, alert response time",
        ),
        AgentSpec(
            role=AgentRole.MONITOR,
            name="MonitorAgent",
            responsibilities=[
                "execution quality monitoring",
                "slippage/latency/partial-fill alerts",
                "closed-loop telemetry review",
            ],
            not_responsible=["signal generation"],
            metric="anomaly detection rate",
        ),
        AgentSpec(
            role=AgentRole.POSITION,
            name="PositionAgent",
            responsibilities=["open position review", "exit triggers"],
            not_responsible=["new signal generation"],
            metric="turnover, holding cost",
        ),
        AgentSpec(
            role=AgentRole.HEDGING,
            name="HedgingAgent",
            responsibilities=["beta exposure", "hedge recommendations"],
            not_responsible=["primary signal generation"],
            metric="net beta vs target",
        ),
        AgentSpec(
            role=AgentRole.RESEARCH,
            name="LLMResearchAgent",
            responsibilities=["news sentiment", "strategy diagnostics", "audit logging"],
            not_responsible=["trade signals", "execution", "risk override"],
            metric="sentiment coverage, audit completeness",
        ),
    ]


def evolution_stages() -> dict[int, str]:
    return {
        1: "Single agent - strategy viability",
        2: "Signal + Risk separation (veto)",
        3: "+ Execution agent",
        4: "+ Regime agent",
        5: "Full architecture with Meta + Data + Position",
    }
