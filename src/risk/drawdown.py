from __future__ import annotations

from src.risk.types import DrawdownAction, DrawdownLevel, DrawdownState


def evaluate_drawdown(
    drawdown_pct: float,
    *,
    warning_pct: float = 5.0,
    stop_pct: float = 10.0,
    circuit_pct: float = 15.0,
    warning_scale: float = 0.7,
    circuit_breaker_active: bool = False,
) -> DrawdownState:
    """Three-tier drawdown control (Lesson 15.4)."""
    if circuit_breaker_active or drawdown_pct >= circuit_pct:
        return DrawdownState(
            drawdown_pct=drawdown_pct,
            level=DrawdownLevel.CIRCUIT,
            action=DrawdownAction.CIRCUIT_BREAKER,
            position_scale=0.0,
            new_positions_allowed=False,
            circuit_breaker_active=True,
            message=f"circuit breaker: drawdown {drawdown_pct:.1f}% >= {circuit_pct:.1f}%",
        )

    if drawdown_pct >= stop_pct:
        return DrawdownState(
            drawdown_pct=drawdown_pct,
            level=DrawdownLevel.STOP,
            action=DrawdownAction.STOP_NEW_POSITIONS,
            position_scale=0.0,
            new_positions_allowed=False,
            circuit_breaker_active=False,
            message=f"stop level: drawdown {drawdown_pct:.1f}% >= {stop_pct:.1f}%",
        )

    if drawdown_pct >= warning_pct:
        return DrawdownState(
            drawdown_pct=drawdown_pct,
            level=DrawdownLevel.WARNING,
            action=DrawdownAction.REDUCE_RISK,
            position_scale=warning_scale,
            new_positions_allowed=True,
            circuit_breaker_active=False,
            message=f"warning level: drawdown {drawdown_pct:.1f}% >= {warning_pct:.1f}%",
        )

    return DrawdownState(
        drawdown_pct=drawdown_pct,
        level=DrawdownLevel.NORMAL,
        action=DrawdownAction.NORMAL,
        position_scale=1.0,
        new_positions_allowed=True,
        circuit_breaker_active=False,
        message="normal",
    )
