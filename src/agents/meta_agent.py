from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from src.agents.arbitration import resolve_signal_conflicts, vote_on_proposal
from src.agents.backtest_agent import BacktestAgent
from src.agents.cost_estimator_agent import CostEstimatorAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.hedging_agent import HedgeReport, HedgingAgent
from src.agents.llm_research_agent import LLMResearchAgent
from src.agents.monitor_agent import MonitorAgent
from src.agents.evolution_agent import EvolutionAgent
from src.agents.operations_agent import OperationsAgent
from src.agents.portfolio_construction_agent import PortfolioConstructionAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.position_agent import PositionAgent
from src.agents.regime_agent import RegimeAgent
from src.agents.research_agent import ResearchAgent
from src.agents.resilience_agent import ResilienceAgent
from src.agents.risk_agent import RiskAgent
from src.core.agent_bus import AgentBus
from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import (
    AgentHealth,
    ArbitrationMode,
    ArbitrationResult,
    DecisionReport,
    ExecutionPlan,
    MessagePattern,
    MonitorReport,
    MultiAgentReport,
    PipelineStage,
    PositionAction,
    RegimeAssessment,
    ResilienceReport,
    RiskDecision,
    RiskDecisionType,
    SymbolStatsReport,
    TradeSignal,
)
from src.llm_research.types import LLMResearchReport
from src.online.types import EvolutionReport
from src.ops.types import OpsReport
from src.portfolio.types import PortfolioReport
from src.portfolio.asset_rotation import resolve_scan_symbols
from src.risk.types import RiskControlReport
from src.data.store import OHLCVStore
from src.execution.costs.types import CostPipelineReport
from src.execution.execution_types import ExecutionPipelineReport
from src.execution.cost_model import CostModel
from src.execution.exceptions import ExecutionError
from src.backtest.gate_registry import GateFilterReport, GateRegistry, load_or_build_gate_registry
from src.market.symbol_info import fetch_market_symbol_info
from src.ops.logging import new_trace_id
from src.stats.returns import log_returns
from src.stats.risk import volatility
from src.features.indicators import latest_atr_from_bars
from src.trading_log.types import TradeLogSummary
from src.survival.diagnostics import SurvivalDiagnostics
from src.survival.types import SurvivalReport

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PipelineResult:
    research_reports: list[SymbolStatsReport] = field(default_factory=list)
    market_regime: RegimeAssessment | None = None
    regime_map: dict[str, RegimeAssessment] = field(default_factory=dict)
    raw_signals: list[TradeSignal] = field(default_factory=list)
    signals: list[TradeSignal] = field(default_factory=list)
    decision_reports: list[DecisionReport] = field(default_factory=list)
    risk_decisions: list[RiskDecision] = field(default_factory=list)
    execution_plans: list[ExecutionPlan] = field(default_factory=list)
    monitor_reports: list[MonitorReport] = field(default_factory=list)
    position_actions: list[PositionAction] = field(default_factory=list)
    hedge_report: HedgeReport | None = None
    multi_agent_report: MultiAgentReport | None = None
    resilience_report: ResilienceReport | None = None
    llm_research_report: LLMResearchReport | None = None
    risk_control_report: RiskControlReport | None = None
    portfolio_report: PortfolioReport | None = None
    evolution_report: EvolutionReport | None = None
    cost_report: CostPipelineReport | None = None
    execution_report: ExecutionPipelineReport | None = None
    trade_log_report: TradeLogSummary | None = None
    gate_filter_report: GateFilterReport | None = None
    survival_report: SurvivalReport | None = None
    ops_report: OpsReport | None = None


class MetaAgent:
    """Multi-agent orchestrator: parallel stages, arbitration, failure isolation (Lesson 11)."""

    def __init__(
        self,
        config: AppConfig,
        connector: MT5Connector,
        store: OHLCVStore,
    ) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.bus = AgentBus()
        self.ma = config.multi_agent

        self.cost_model = CostModel(config)
        self.research_agent = ResearchAgent(config, store)
        self.regime_agent = RegimeAgent(config, store)
        self.backtest_agent = BacktestAgent(config, store)
        self.resilience_agent = ResilienceAgent(config)
        self.portfolio_agent = PortfolioAgent(config, store)
        self.portfolio_construction_agent = PortfolioConstructionAgent(config, store)
        self.evolution_agent = EvolutionAgent(config)
        self.cost_estimator_agent = CostEstimatorAgent(config, self.cost_model)
        self.risk_agent = RiskAgent(config, connector, store)
        self.decision_agent = DecisionAgent(config, connector, store, self.risk_agent)
        self.execution_agent = ExecutionAgent(config, connector, self.cost_model)
        self.monitor_agent = MonitorAgent(config)
        self.position_agent = PositionAgent(config, connector, store)
        self.hedging_agent = HedgingAgent(config, connector, store)
        self.llm_research_agent = LLMResearchAgent(config)
        self.operations_agent = OperationsAgent(config, connector, store)

        self._failures: dict[str, int] = {}
        self._circuit_open: set[str] = set()
        self._health: list[AgentHealth] = []
        self._gate_from_cache = False

    def _resolve_gate_registry(self, symbols: list[str]) -> GateRegistry | None:
        bt = self.config.backtest
        if not bt.gate_filter_enabled:
            return None

        timeframe = self.config.stats.signal_timeframe

        def _build() -> GateRegistry:
            registry, from_cache = load_or_build_gate_registry(
                self.backtest_agent,
                self.regime_agent,
                symbols,
                timeframe,
                bt.gate_cache_path,
                bt.gate_cache_max_age_hours,
            )
            self._gate_from_cache = from_cache
            return registry

        return self._safe_call("BacktestAgent", _build, default=None)

    def run(self, symbols: list[str] | None = None) -> PipelineResult:
        # Expand to all Asset Groups when multi-asset rotation room is enabled,
        # so other groups can receive signals and migrate in concurrently.
        symbols = resolve_scan_symbols(self.config, symbols)
        result = PipelineResult()
        arbitration_notes: list[ArbitrationResult] = []
        parallel_start = time.perf_counter()

        if self.ma.parallel_analysis:
            research_reports, market_regime, regime_map, analysis_ms = self._parallel_analysis(symbols)
        else:
            research_reports, market_regime, regime_map, analysis_ms = self._serial_analysis(symbols)

        result.research_reports = research_reports
        result.market_regime = market_regime
        result.regime_map = regime_map

        self.bus.shared.regime_map = regime_map
        self.bus.publish(PipelineStage.REGIME.value, "MetaAgent", regime_map)

        result.resilience_report = self.resilience_agent.build_report(result.market_regime, regime_map)
        self.resilience_agent.apply_degradation(regime_map, result.resilience_report)

        result.risk_control_report = self._safe_call(
            "RiskAgent",
            self.risk_agent.assess_account,
            default=None,
        )
        if result.risk_control_report:
            rc = result.risk_control_report
            if rc.drawdown_level != "normal":
                logger.warning(
                    "RiskAgent drawdown %s: %.1f%% action=%s scale=%.0f%%",
                    rc.drawdown_level,
                    rc.drawdown_pct,
                    rc.drawdown_action,
                    rc.position_scale * 100,
                )
            if rc.circuit_breaker_active:
                logger.critical("RiskAgent circuit breaker active")

        if result.resilience_report.degradation_level > 0:
            logger.info(
                "ResilienceAgent: level %s scale=%.0f%% warnings=%d",
                result.resilience_report.level_name,
                result.resilience_report.position_scale_multiplier * 100,
                len(result.resilience_report.warnings),
            )

        if market_regime:
            logger.info(
                "RegimeAgent market proxy %s: %s strategy=%s scale=%.0f%% (%s)",
                market_regime.symbol,
                market_regime.regime.value,
                market_regime.selected_strategy.value,
                market_regime.position_scale * 100,
                market_regime.reason,
            )

        result.llm_research_report = self._safe_call(
            "LLMResearchAgent",
            lambda: self.llm_research_agent.analyze(symbols, result.research_reports),
            default=None,
        )
        if result.llm_research_report and result.llm_research_report.analyzed_count:
            logger.info(
                "LLMResearchAgent: %d/%d news analyzed, %d feature(s), provider=%s",
                result.llm_research_report.analyzed_count,
                result.llm_research_report.filtered_news_count,
                len(result.llm_research_report.sentiment_features),
                result.llm_research_report.provider,
            )

        sentiment_map: dict[str, float] | None = None
        if self.config.llm_research.use_as_feature and result.llm_research_report:
            sentiment_map = {
                feat.symbol: feat.sentiment_score
                for feat in result.llm_research_report.sentiment_features
            }

        gate_registry = self._resolve_gate_registry(symbols)
        if gate_registry is not None:
            result.gate_filter_report = gate_registry.summary(
                enabled=True,
                from_cache=self._gate_from_cache,
            )
            logger.info(
                "GateRegistry: %d entries, %d passed, %d failed (cache=%s)",
                result.gate_filter_report.total_entries,
                result.gate_filter_report.passed_count,
                result.gate_filter_report.failed_count,
                self._gate_from_cache,
            )

        raw_signals = self._safe_call(
            "PortfolioAgent",
            lambda: self.portfolio_agent.scan(symbols, regime_map, sentiment_map, gate_registry),
            default=[],
        )
        result.raw_signals = raw_signals or []
        if gate_registry is not None and result.gate_filter_report is not None:
            result.gate_filter_report.blocked_generators = gate_registry.blocked_count
        self.bus.publish(PipelineStage.SIGNAL.value, "PortfolioAgent", result.raw_signals)
        logger.info("PortfolioAgent: %d raw signal(s)", len(result.raw_signals))

        resolved_signals, conflict_notes = resolve_signal_conflicts(
            result.raw_signals,
            ArbitrationMode(self.ma.arbitration_mode),
        )
        arbitration_notes.extend(conflict_notes)

        state = self.decision_agent.build_state()
        benchmark_vol = 0.15
        for report in result.research_reports:
            if report.symbol == self.config.portfolio.benchmark_symbol:
                benchmark_vol = max(report.annualized_volatility, 0.05)
                break

        construction_result = self._safe_call(
            "PortfolioConstructionAgent",
            lambda: self.portfolio_construction_agent.allocate(
                resolved_signals,
                result.research_reports,
                equity=state.equity,
                benchmark_vol=benchmark_vol,
            ),
            default=(resolved_signals, None),
        )
        if construction_result:
            resolved_signals, portfolio_report = construction_result
            result.portfolio_report = portfolio_report
            if portfolio_report and portfolio_report.warnings:
                for warning in portfolio_report.warnings[:3]:
                    logger.warning("PortfolioConstruction: %s", warning)

        result.evolution_report = self._safe_call(
            "EvolutionAgent",
            lambda: self.evolution_agent.assess(
                result.research_reports,
                resolved_signals,
                result.portfolio_report,
            ),
            default=None,
        )
        if result.evolution_report and self.config.online_learning.apply_dynamic_threshold:
            resolved_signals = self.evolution_agent.apply_signal_adaptation(
                resolved_signals,
                result.evolution_report,
            )

        if ArbitrationMode(self.ma.arbitration_mode) == ArbitrationMode.VOTING:
            voted: list[TradeSignal] = []
            self.bus.shared.equity = state.equity
            self.bus.shared.exposure_pct = state.current_exposure_pct
            self.bus.shared.open_position_lots = state.open_position_lots

            for signal in resolved_signals:
                regime = regime_map.get(signal.symbol)
                vote_result = vote_on_proposal(
                    signal,
                    regime,
                    state.current_exposure_pct,
                    self.config.risk.max_total_exposure_pct,
                )
                arbitration_notes.append(vote_result)
                if vote_result.approved and vote_result.selected_signal is not None:
                    voted.append(vote_result.selected_signal)
            resolved_signals = voted

        sized_signals, decision_reports = self.decision_agent.decide(resolved_signals)
        result.signals = sized_signals
        result.decision_reports = decision_reports
        self.bus.publish(PipelineStage.DECISION.value, "DecisionAgent", decision_reports)

        for signal in result.signals:
            regime = regime_map.get(signal.symbol)
            requested = (
                signal.requested_lots
                if signal.requested_lots is not None
                else self.config.trading.default_lots
            )

            risk = self._safe_call(
                "RiskAgent",
                lambda s=signal, r=regime, q=requested: self.risk_agent.review(s, q, r),
                default=None,
            )
            if risk is None:
                risk = RiskDecision(
                    decision=RiskDecisionType.REJECT,
                    approved_lots=0.0,
                    reason="RiskAgent unavailable",
                )
            result.risk_decisions.append(risk)

            if self.ma.arbitration_mode == ArbitrationMode.VETO.value and risk.approved_lots <= 0:
                logger.info("RiskAgent veto %s: %s", signal.symbol, risk.reason)
                arbitration_notes.append(
                    ArbitrationResult(
                        symbol=signal.symbol,
                        approved=False,
                        net_score=-1,
                        votes=[],
                        reason=f"risk veto: {risk.reason}",
                    )
                )
                continue

            if risk.approved_lots <= 0:
                logger.info("RiskAgent rejected %s: %s", signal.symbol, risk.reason)
                continue

            symbol_info = fetch_market_symbol_info(self.connector, self.config, signal.symbol)
            bars = self.store.get_recent_bars(
                signal.symbol,
                self.config.stats.analysis_timeframe,
                self.config.stats.min_bars,
            )
            closes = [float(b["close"]) for b in bars]
            rets = log_returns(closes)
            daily_vol = volatility(rets, annualize=False) if len(rets) else 0.02

            atr = latest_atr_from_bars(bars, self.config.indicators.atr_period)

            if self.config.costs.enabled:
                tradability = self._safe_call(
                    "CostEstimatorAgent",
                    lambda s=signal, r=risk, si=symbol_info, dv=daily_vol: self.cost_estimator_agent.assess_trade(
                        s, r.approved_lots, si, dv
                    ),
                    default=None,
                )
                if tradability is not None:
                    if result.cost_report is None:
                        result.cost_report = CostPipelineReport()
                    result.cost_report.assessments.append(tradability)
                    result.cost_report.total_estimated_cost_jpy += (
                        tradability.notional * tradability.costs.total_pct / 100.0
                    )
                    if not tradability.tradable:
                        result.cost_report.blocked_count += 1
                        result.cost_report.warnings.extend(tradability.warnings)
                        if self.config.costs.block_untradable:
                            logger.info(
                                "CostEstimator blocked %s: net alpha %.4f%%",
                                signal.symbol,
                                tradability.net_alpha_pct,
                            )
                            arbitration_notes.append(
                                ArbitrationResult(
                                    symbol=signal.symbol,
                                    approved=False,
                                    net_score=-1,
                                    votes=[],
                                    reason=f"untradable: net alpha {tradability.net_alpha_pct:.4f}%",
                                )
                            )
                            continue

            plan = self.execution_agent.plan(signal, risk, symbol_info, daily_vol)
            try:
                executed = self.execution_agent.execute(
                    plan,
                    symbol_info=symbol_info,
                    daily_volatility=daily_vol,
                    signal=signal,
                    recent_bars=bars,
                    atr=atr,
                    trace_id=new_trace_id("ord"),
                )
            except ExecutionError as exc:
                logger.error("ExecutionAgent failed for %s: %s", signal.symbol, exc)
                executed = plan
            result.execution_plans.append(executed)
            self.bus.publish(PipelineStage.EXECUTION.value, "ExecutionAgent", executed, MessagePattern.QUEUE)

            monitor = self.monitor_agent.review(executed)
            result.monitor_reports.append(monitor)

        if self.config.execution.enabled and self.config.execution.log_executions:
            result.execution_report = self._safe_call(
                "ExecutionAgent",
                lambda: self.execution_agent.summarize_telemetry(),
                default=None,
            )

        if self.config.trade_log.enabled:
            result.trade_log_report = self._safe_call(
                "ExecutionAgent",
                lambda: self.execution_agent.summarize_trade_log(),
                default=None,
            )

        result.position_actions = self._safe_call(
            "PositionAgent",
            self.position_agent.review_positions,
            default=[],
        ) or []

        if self.config.hedging.enabled:
            result.hedge_report = self._safe_call(
                "HedgingAgent",
                lambda: self.hedging_agent.assess(result.signals),
                default=None,
            )
            if result.hedge_report and result.hedge_report.recommendation:
                logger.info("HedgingAgent: %s", result.hedge_report.recommendation.reason)

        parallel_elapsed = (time.perf_counter() - parallel_start) * 1000
        serial_estimate = analysis_ms.get("serial_estimate_ms", parallel_elapsed)

        result.multi_agent_report = MultiAgentReport(
            evolution_stage=self.ma.evolution_stage,
            arbitration_mode=self.ma.arbitration_mode,
            parallel_analysis=self.ma.parallel_analysis,
            agent_health=list(self._health),
            parallel_elapsed_ms=parallel_elapsed,
            serial_estimate_ms=serial_estimate,
            bus_events=self.bus.event_count,
            arbitration_results=arbitration_notes,
        )

        result.ops_report = self._safe_call(
            "OperationsAgent",
            lambda: self.operations_agent.assess(result, result.multi_agent_report),
            default=None,
        )
        if result.ops_report and not result.ops_report.trading_allowed:
            logger.warning(
                "OperationsAgent: trading not allowed (phase=%s, critical=%d)",
                result.ops_report.session_phase,
                result.ops_report.critical_count,
            )

        result.survival_report = self._safe_call(
            "SurvivalDiagnostics",
            lambda: SurvivalDiagnostics(self.config).assess(
                connector=self.connector,
                store=self.store,
                pipeline=result,
            ),
            default=None,
        )

        return result

    def _parallel_analysis(
        self,
        symbols: list[str],
    ) -> tuple[list[SymbolStatsReport], RegimeAssessment | None, dict[str, RegimeAssessment], dict[str, float]]:
        timings: dict[str, float] = {}
        serial_estimate = 0.0

        with ThreadPoolExecutor(max_workers=min(4, len(symbols) + 2)) as pool:
            research_future = pool.submit(
                self._timed_agent,
                "ResearchAgent",
                lambda: self.research_agent.analyze_all(symbols),
            )
            market_future = pool.submit(
                self._timed_agent,
                "RegimeAgent",
                self.regime_agent.assess_market_proxy,
            )
            regime_futures = {
                pool.submit(self._timed_agent, "RegimeAgent", lambda s=sym: self.regime_agent.assess(s)): sym
                for sym in symbols
            }

            research_reports, research_ms = research_future.result(timeout=self.ma.agent_timeout_seconds)
            market_regime, market_ms = market_future.result(timeout=self.ma.agent_timeout_seconds)
            serial_estimate += research_ms + market_ms

            regime_map: dict[str, RegimeAssessment] = {}
            for future in as_completed(regime_futures):
                sym = regime_futures[future]
                assessment, elapsed = future.result(timeout=self.ma.agent_timeout_seconds)
                serial_estimate += elapsed
                if assessment is not None:
                    regime_map[sym] = assessment

        timings["parallel_ms"] = max(research_ms, market_ms) + serial_estimate * 0.3
        timings["serial_estimate_ms"] = serial_estimate
        self.bus.publish(PipelineStage.RESEARCH.value, "ResearchAgent", len(research_reports or []))
        return research_reports or [], market_regime, regime_map, timings

    def _serial_analysis(
        self,
        symbols: list[str],
    ) -> tuple[list[SymbolStatsReport], RegimeAssessment | None, dict[str, RegimeAssessment], dict[str, float]]:
        start = time.perf_counter()
        research_reports, research_ms = self._timed_agent(
            "ResearchAgent",
            lambda: self.research_agent.analyze_all(symbols),
        )
        market_regime, market_ms = self._timed_agent(
            "RegimeAgent",
            self.regime_agent.assess_market_proxy,
        )
        regime_map: dict[str, RegimeAssessment] = {}
        regime_ms = 0.0
        for symbol in symbols:
            assessment, elapsed = self._timed_agent(
                "RegimeAgent",
                lambda s=symbol: self.regime_agent.assess(s),
            )
            regime_ms += elapsed
            if assessment is not None:
                regime_map[symbol] = assessment

        total = (time.perf_counter() - start) * 1000
        return (
            research_reports or [],
            market_regime,
            regime_map,
            {"parallel_ms": total, "serial_estimate_ms": research_ms + market_ms + regime_ms},
        )

    def _timed_agent(self, name: str, fn: Callable[[], T]) -> tuple[T | None, float]:
        start = time.perf_counter()
        result = self._safe_call(name, fn, default=None)
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed

    def _safe_call(self, name: str, fn: Callable[[], T], default: T) -> T:
        if name in self._circuit_open:
            logger.warning("%s circuit open — skipping (failure isolation)", name)
            self._record_health(name, False, 0.0, "circuit open")
            return default

        start = time.perf_counter()
        try:
            result = fn()
            elapsed = (time.perf_counter() - start) * 1000
            self._failures[name] = 0
            self._record_health(name, True, elapsed)
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            self._failures[name] = self._failures.get(name, 0) + 1
            message = str(exc)
            logger.error("%s failed (%d/%d): %s", name, self._failures[name], self.ma.circuit_breaker_threshold, exc)
            if self._failures[name] >= self.ma.circuit_breaker_threshold:
                self._circuit_open.add(name)
                logger.error("%s circuit breaker opened", name)
            self._record_health(name, False, elapsed, message)
            return default

    def _record_health(
        self,
        agent: str,
        healthy: bool,
        elapsed_ms: float,
        last_error: str | None = None,
    ) -> None:
        entry = AgentHealth(
            agent=agent,
            healthy=healthy,
            failures=self._failures.get(agent, 0),
            circuit_open=agent in self._circuit_open,
            elapsed_ms=elapsed_ms,
            last_error=last_error,
        )
        self._health = [h for h in self._health if h.agent != agent]
        self._health.append(entry)
