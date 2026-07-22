from __future__ import annotations

import math

from src.execution.costs.types import CostComponents, GrossNetStrategyReport


def fill_probability(
    limit_distance_pct: float,
    daily_volatility: float,
    *,
    wait_hours: float = 4.0,
) -> float:
    """
    Simplified fill probability ~ 1 - exp(-lambda * time) (Lesson 18.4.1).
    lambda scales with how often price reaches limit given volatility.
    """
    if daily_volatility <= 0:
        return 0.5
    reach_rate = daily_volatility / max(abs(limit_distance_pct), 1e-6)
    lam = min(reach_rate, 5.0)
    return max(0.0, min(1.0, 1.0 - math.exp(-lam * wait_hours / 6.0)))


def opportunity_cost(
    predicted_return_pct: float,
    signal_decay_halflife_minutes: float,
    execution_delay_minutes: float,
) -> float:
    """Lost alpha due to signal decay during execution delay (Lesson 18.6)."""
    if signal_decay_halflife_minutes <= 0 or execution_delay_minutes <= 0:
        return 0.0
    decay = 0.5 ** (execution_delay_minutes / signal_decay_halflife_minutes)
    remaining = predicted_return_pct * decay
    return max(0.0, predicted_return_pct - remaining)


def gross_to_net_alpha(gross_alpha_pct: float, total_cost_pct: float) -> float:
    return gross_alpha_pct - total_cost_pct


def is_tradable(net_alpha_pct: float, min_net_alpha_pct: float = 0.0) -> bool:
    return net_alpha_pct > min_net_alpha_pct


def turnover_annual_cost(turnover_pct: float, cost_per_trade_pct: float, trading_days: int = 252) -> float:
    """
    Annual cost from turnover (Lesson 18 opening scenario).
    turnover_pct: daily turnover as fraction of capital (e.g. 3.0 = 300%).
    """
    daily_cost = turnover_pct * cost_per_trade_pct
    return daily_cost * trading_days


def evaluate_strategy_tradability(
    strategy_name: str,
    gross_alpha_pct: float,
    turnover_pct: float,
    cost_per_trade_pct: float,
    *,
    trading_days: int = 252,
) -> GrossNetStrategyReport:
    annual_cost = turnover_annual_cost(turnover_pct, cost_per_trade_pct, trading_days)
    net = gross_alpha_pct - annual_cost
    return GrossNetStrategyReport(
        strategy_name=strategy_name,
        gross_alpha_pct=gross_alpha_pct,
        turnover_pct=turnover_pct,
        cost_per_trade_pct=cost_per_trade_pct,
        annual_cost_pct=annual_cost,
        net_alpha_pct=net,
        tradable=net > 0,
    )


def build_cost_components(
    *,
    commission_pct: float,
    slippage_pct: float,
    impact_pct: float,
    opportunity_pct: float,
    spread_pct: float = 0.0,
) -> CostComponents:
    return CostComponents(
        explicit_pct=commission_pct,
        slippage_pct=slippage_pct + spread_pct,
        market_impact_pct=impact_pct,
        opportunity_cost_pct=opportunity_pct,
        spread_pct=spread_pct,
    )
