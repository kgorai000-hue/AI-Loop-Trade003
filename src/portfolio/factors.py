from __future__ import annotations

from src.core.types import SymbolStatsReport

# Static factor loadings for MT5 symbol universe (Lesson 16.4)
DEFAULT_FACTOR_LOADINGS: dict[str, dict[str, float]] = {
    "#US30": {"market_beta": 0.95, "indices": 1.0, "commodities": 0.0, "forex": 0.0, "momentum": 0.0},
    "#USSPX500": {"market_beta": 1.0, "indices": 1.0, "commodities": 0.0, "forex": 0.0, "momentum": 0.0},
    "#USNDAQ100": {"market_beta": 1.1, "indices": 1.0, "commodities": 0.0, "forex": 0.0, "momentum": 0.2},
    "#Japan225": {"market_beta": 0.7, "indices": 0.8, "commodities": 0.0, "forex": 0.1, "momentum": 0.0},
    "#Germany40": {"market_beta": 0.85, "indices": 0.8, "commodities": 0.0, "forex": 0.1, "momentum": 0.0},
    "#UK100": {"market_beta": 0.8, "indices": 0.8, "commodities": 0.0, "forex": 0.1, "momentum": 0.0},
    "GOLD": {"market_beta": 0.05, "indices": 0.0, "commodities": 1.0, "forex": 0.0, "momentum": -0.1},
    "SILVER": {"market_beta": 0.1, "indices": 0.0, "commodities": 1.0, "forex": 0.0, "momentum": -0.1},
    "WTI": {"market_beta": 0.2, "indices": 0.0, "commodities": 1.0, "forex": 0.0, "momentum": 0.0},
    "EURUSD": {"market_beta": 0.0, "indices": 0.0, "commodities": 0.0, "forex": 1.0, "momentum": 0.0},
    "GBPUSD": {"market_beta": 0.0, "indices": 0.0, "commodities": 0.0, "forex": 1.0, "momentum": 0.0},
    "USDJPY": {"market_beta": 0.0, "indices": 0.0, "commodities": 0.0, "forex": 1.0, "momentum": 0.0},
}


def symbol_factor_loadings(
    symbol: str,
    research: SymbolStatsReport | None = None,
) -> dict[str, float]:
    loadings = dict(DEFAULT_FACTOR_LOADINGS.get(symbol, {"market_beta": 0.5}))
    if research is not None:
        if research.autocorr_lag1 > 0.05:
            loadings["momentum"] = loadings.get("momentum", 0.0) + min(research.autocorr_lag1, 0.5)
        loadings["volatility"] = min(research.annualized_volatility, 1.0)
    return loadings


def portfolio_factor_exposures(
    weights: dict[str, float],
    loadings_by_symbol: dict[str, dict[str, float]],
) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for symbol, weight in weights.items():
        for factor, loading in loadings_by_symbol.get(symbol, {}).items():
            exposures[factor] = exposures.get(factor, 0.0) + weight * loading
    return exposures


def check_factor_limits(
    exposures: dict[str, float],
    limits: dict[str, float],
) -> list[tuple[str, float, float, bool]]:
    results: list[tuple[str, float, float, bool]] = []
    for factor, limit in limits.items():
        value = exposures.get(factor, 0.0)
        breached = abs(value) > limit
        results.append((factor, value, limit, breached))
    return results
