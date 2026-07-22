from __future__ import annotations

from src.core.config import AppConfig


def estimate_round_trip_cost_pct(config: AppConfig) -> float:
    """Conservative round-trip cost in percent (e.g. 0.09 = 0.09%)."""
    if config.backtest.round_trip_cost_pct is not None:
        return config.backtest.round_trip_cost_pct

    slippage_pct = config.costs.slippage_rate * 2 * 100.0
    spread_pct = (
        config.costs.slippage_rate * 100.0
        if config.costs.use_spread_when_slippage_unknown
        else 0.0
    )
    return slippage_pct + spread_pct
