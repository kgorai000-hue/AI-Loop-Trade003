from __future__ import annotations

"""Multi-agent system architecture map (Lesson 21.1)."""

AGENT_PIPELINE = [
    "DataAgent",
    "ResearchAgent",
    "RegimeAgent",
    "ResilienceAgent",
    "PortfolioAgent",
    "PortfolioConstructionAgent",
    "EvolutionAgent",
    "LLMResearchAgent",
    "DecisionAgent",
    "CostEstimatorAgent",
    "RiskAgent",
    "ExecutionAgent",
    "MonitorAgent",
    "PositionAgent",
    "HedgingAgent",
    "OperationsAgent",
]

DATA_FLOW = """
Market Data (MT5) --> DataManager --> OHLCVStore
                         |
                         v
Research + Regime --> PortfolioAgent --> PortfolioConstruction
                         |
                         v
Evolution + LLM --> DecisionAgent --> CostEstimator --> RiskAgent
                         |
                         v
ExecutionAgent --> MonitorAgent --> PositionAgent --> OperationsAgent
                         |
                         v
Structured Logs / Alerts / Telemetry
"""

MODULAR_MONOLITH_NOTE = (
    "All agents run in a single process with clean dataclass contracts. "
    "Extract to separate services only when scaling or latency requires it."
)


def pipeline_stages() -> list[tuple[str, str]]:
    return [
        ("1. Data", "DataAgent syncs MT5 OHLCV, validates quality"),
        ("2. Research", "ResearchAgent + RegimeAgent parallel analysis"),
        ("3. Signals", "PortfolioAgent generates regime-weighted signals"),
        ("4. Portfolio", "PortfolioConstructionAgent allocates weights"),
        ("5. Evolution", "EvolutionAgent monitors drift and thresholds"),
        ("6. Decision", "DecisionAgent sizes positions (Kelly/Van Tharp)"),
        ("7. Costs", "CostEstimatorAgent gross-to-net alpha check"),
        ("8. Risk", "RiskAgent 3-tier drawdown + veto power"),
        ("9. Execution", "ExecutionAgent conservative simulation + telemetry"),
        ("10. Ops", "OperationsAgent 4-layer monitoring + alerts"),
    ]
