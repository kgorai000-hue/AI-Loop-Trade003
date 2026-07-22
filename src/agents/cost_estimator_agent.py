from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import DecisionReport, SignalSide, TradeSignal
from src.execution.cost_model import CostModel
from src.execution.costs.impact import almgren_chriss_total_cost
from src.execution.costs.slippage import linear_slippage, sqrt_slippage
from src.execution.costs.tradability import (
    build_cost_components,
    fill_probability,
    gross_to_net_alpha,
    is_tradable,
    opportunity_cost,
)
from src.execution.costs.types import CostPipelineReport, TradabilityAssessment
from src.market.symbol_info import MarketSymbolInfo

logger = logging.getLogger(__name__)


class CostEstimatorAgent:
    """Estimate trading costs and tradability before execution (Lesson 18.7)."""

    def __init__(self, config: AppConfig, cost_model: CostModel | None = None) -> None:
        self.config = config
        self.cfg = config.costs
        self.cost_model = cost_model or CostModel(config)

    def assess_trade(
        self,
        signal: TradeSignal,
        lots: float,
        symbol_info: MarketSymbolInfo,
        daily_volatility: float,
        *,
        gross_alpha_pct: float | None = None,
    ) -> TradabilityAssessment:
        price = symbol_info.ask if signal.side == SignalSide.BUY else symbol_info.bid
        notional = self.cost_model._notional_jpy(symbol_info, lots, price)
        adv = max(self.cfg.default_adv_notional, 1.0)
        order_adv_ratio = notional / adv

        breakdown = self.cost_model.estimate_trade_cost(
            symbol_info=symbol_info,
            lots=lots,
            side=signal.side,
            daily_volatility=daily_volatility,
            slippage_rate=self._slippage_rate(notional, adv, daily_volatility),
        )

        spread_pct = (breakdown.spread_cost_jpy / notional * 100.0) if notional else 0.0
        slippage_pct = (breakdown.slippage_cost_jpy / notional * 100.0) if notional else 0.0
        commission_pct = (breakdown.commission_cost_jpy / notional * 100.0) if notional else 0.0
        impact_pct = (breakdown.market_impact_cost_jpy / notional * 100.0) if notional else 0.0

        ac = almgren_chriss_total_cost(
            participation=min(order_adv_ratio, 1.0),
            sigma=daily_volatility,
            execution_days=1.0,
        )
        impact_pct = max(impact_pct, ac["total_impact"] * 100.0)

        if gross_alpha_pct is None:
            gross_alpha_pct = self._gross_alpha_from_signal(signal)

        opp_pct = 0.0
        if self.cfg.opportunity_cost_enabled:
            opp_pct = opportunity_cost(
                gross_alpha_pct,
                self.cfg.signal_decay_halflife_minutes,
                self.cfg.execution_delay_minutes,
            )

        costs = build_cost_components(
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            impact_pct=impact_pct,
            opportunity_pct=opp_pct,
            spread_pct=spread_pct,
        )
        net_alpha = gross_to_net_alpha(gross_alpha_pct, costs.total_pct)
        fill_prob = fill_probability(
            limit_distance_pct=spread_pct,
            daily_volatility=daily_volatility,
            wait_hours=self.cfg.execution_delay_minutes / 60.0,
        )

        warnings: list[str] = []
        if order_adv_ratio > self.cfg.max_order_adv_ratio:
            warnings.append(
                f"order/ADV {order_adv_ratio:.2%} exceeds limit {self.cfg.max_order_adv_ratio:.2%}"
            )
        if net_alpha <= 0:
            warnings.append(f"net alpha {net_alpha:.4f}% <= 0 after costs")
        if fill_prob < 0.5:
            warnings.append(f"low fill probability {fill_prob:.0%}")

        tradable = is_tradable(net_alpha, self.cfg.min_net_alpha_pct)
        if order_adv_ratio > self.cfg.max_order_adv_ratio:
            tradable = False

        return TradabilityAssessment(
            symbol=signal.symbol,
            side=signal.side.value,
            lots=lots,
            notional=notional,
            gross_alpha_pct=gross_alpha_pct,
            costs=costs,
            net_alpha_pct=net_alpha,
            tradable=tradable,
            fill_probability=fill_prob,
            order_adv_ratio=order_adv_ratio,
            slippage_model=self.cfg.slippage_model,
            warnings=warnings,
        )

    def assess_pipeline(
        self,
        signals: list[TradeSignal],
        decision_reports: list[DecisionReport],
        symbol_infos: dict[str, MarketSymbolInfo],
        daily_vols: dict[str, float],
        approved_lots: dict[str, float],
    ) -> CostPipelineReport:
        if not self.cfg.enabled:
            return CostPipelineReport()

        report = CostPipelineReport()
        decision_map = {d.symbol: d for d in decision_reports}

        for signal in signals:
            lots = approved_lots.get(signal.symbol, 0.0)
            if lots <= 0:
                continue

            symbol_info = symbol_infos.get(signal.symbol)
            if symbol_info is None:
                continue

            daily_vol = daily_vols.get(signal.symbol, 0.02)
            decision = decision_map.get(signal.symbol)
            gross_alpha = decision.predicted_return * 100.0 if decision else None

            assessment = self.assess_trade(
                signal,
                lots,
                symbol_info,
                daily_vol,
                gross_alpha_pct=gross_alpha,
            )
            report.assessments.append(assessment)
            report.total_estimated_cost_jpy += assessment.notional * assessment.costs.total_pct / 100.0

            if not assessment.tradable:
                report.blocked_count += 1
            report.warnings.extend(assessment.warnings)

        return report

    def _slippage_rate(
        self,
        notional: float,
        adv: float,
        sigma: float,
    ) -> float | None:
        model = self.cfg.slippage_model.lower()
        if model == "flat":
            return self.cfg.slippage_rate
        if model == "linear":
            return linear_slippage(notional, adv, self.cfg.slippage_k_linear)
        if model == "sqrt":
            return sqrt_slippage(notional, adv, sigma, self.cfg.slippage_k_sqrt)
        return self.cfg.slippage_rate

    @staticmethod
    def _gross_alpha_from_signal(signal: TradeSignal) -> float:
        if signal.predicted_return is not None:
            return signal.predicted_return * 100.0
        return signal.strength * 1.0
