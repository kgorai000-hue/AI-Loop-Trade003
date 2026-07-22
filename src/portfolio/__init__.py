"""Portfolio construction and exposure management (Lesson 16)."""

from src.portfolio.covariance import correlation_matrix, shrunk_covariance
from src.portfolio.factors import check_factor_limits, portfolio_factor_exposures, symbol_factor_loadings
from src.portfolio.leverage import notional_leverage, portfolio_volatility, risk_leverage
from src.portfolio.types import FactorExposure, PortfolioReport, SymbolAllocation
from src.portfolio.weights import (
    apply_correlation_penalty,
    cap_weights,
    equal_risk_contribution_weights,
    equal_weights,
    inverse_volatility_weights,
)

__all__ = [
    "FactorExposure",
    "PortfolioReport",
    "SymbolAllocation",
    "apply_correlation_penalty",
    "cap_weights",
    "check_factor_limits",
    "correlation_matrix",
    "equal_risk_contribution_weights",
    "equal_weights",
    "inverse_volatility_weights",
    "notional_leverage",
    "portfolio_factor_exposures",
    "portfolio_volatility",
    "risk_leverage",
    "shrunk_covariance",
    "symbol_factor_loadings",
]
