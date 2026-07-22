from __future__ import annotations

import math


def temporary_impact(trade_rate: float, eta: float = 0.1) -> float:
    """Temporary impact proportional to trading speed (Lesson 18.3)."""
    return eta * max(trade_rate, 0.0)


def permanent_impact(participation: float, gamma: float = 0.1) -> float:
    """Permanent impact proportional to total volume consumed (Lesson 18.3)."""
    return gamma * max(participation, 0.0)


def volatility_risk_cost(sigma: float, execution_days: float) -> float:
    """Volatility risk during execution horizon."""
    return sigma * math.sqrt(max(execution_days, 0.0))


def almgren_chriss_total_cost(
    *,
    participation: float,
    sigma: float,
    execution_days: float = 1.0,
    eta: float = 0.1,
    gamma: float = 0.1,
    urgency: float = 1.0,
) -> dict[str, float]:
    """
    Simplified Almgren-Chriss cost decomposition (Lesson 18.3.2).
    urgency > 1 -> faster trade -> higher temporary impact, lower vol risk.
    """
    trade_rate = participation * urgency
    temp = temporary_impact(trade_rate, eta)
    perm = permanent_impact(participation, gamma)
    vol_risk = volatility_risk_cost(sigma, execution_days / urgency)
    total = temp + perm + vol_risk
    return {
        "temporary_impact": temp,
        "permanent_impact": perm,
        "volatility_risk": vol_risk,
        "total_impact": total,
    }
