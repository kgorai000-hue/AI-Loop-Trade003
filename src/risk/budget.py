from __future__ import annotations


def inverse_drawdown_weights(strategy_drawdowns: dict[str, float]) -> dict[str, float]:
    """Equal risk contribution by inverse drawdown (Lesson 15.1)."""
    if not strategy_drawdowns:
        return {}
    inv = {name: 1.0 / max(dd, 1e-6) for name, dd in strategy_drawdowns.items()}
    total = sum(inv.values())
    if total <= 0:
        share = 1.0 / len(strategy_drawdowns)
        return {name: share for name in strategy_drawdowns}
    return {name: weight / total for name, weight in inv.items()}


def scale_weights_to_budget(
    weights: dict[str, float],
    *,
    max_portfolio_drawdown: float,
    strategy_drawdowns: dict[str, float],
) -> dict[str, float]:
    """Scale weights so expected portfolio drawdown stays under budget."""
    if not weights:
        return {}
    expected_dd = sum(weights[name] * strategy_drawdowns.get(name, 0.0) for name in weights)
    if expected_dd <= max_portfolio_drawdown or expected_dd <= 0:
        return weights
    scale = max_portfolio_drawdown / expected_dd
    return {name: weight * scale for name, weight in weights.items()}
