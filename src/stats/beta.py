from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.stats.returns import log_returns


@dataclass
class BetaDecomposition:
    beta: float
    alpha_annual: float
    r_squared: float
    total_return: float
    benchmark_total_return: float
    beta_contribution: float
    alpha_contribution: float
    beta_pct: float
    observations: int


@dataclass
class HedgeRatioResult:
    long_notional: float
    long_beta: float
    hedge_instrument_beta: float
    hedge_notional: float
    dollar_neutral_hedge_notional: float
    net_beta_dollar_neutral: float
    net_beta_beta_neutral: float


def align_returns(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(strategy_returns), len(benchmark_returns))
    if length < 2:
        return np.array([]), np.array([])
    return strategy_returns[-length:], benchmark_returns[-length:]


def estimate_beta(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    rf_rate: float = 0.04,
    periods_per_year: int = 252,
) -> tuple[float, float, float]:
    """CAPM regression: R_asset - rf = alpha + beta * (R_m - rf)."""
    asset, bench = align_returns(asset_returns, benchmark_returns)
    if len(asset) < 2:
        return 0.0, 0.0, 0.0

    rf_daily = rf_rate / periods_per_year
    excess_asset = asset - rf_daily
    excess_bench = bench - rf_daily

    slope, intercept, r_value, _, _ = stats.linregress(excess_bench, excess_asset)
    alpha_annual = float(intercept * periods_per_year)
    return float(slope), alpha_annual, float(r_value**2)


def decompose_returns(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    rf_rate: float = 0.04,
    periods_per_year: int = 252,
) -> BetaDecomposition:
    """Decompose strategy returns into alpha and beta contributions (Lesson 8.1)."""
    strat, bench = align_returns(strategy_returns, benchmark_returns)
    if len(strat) < 2:
        return BetaDecomposition(0, 0, 0, 0, 0, 0, 0, 0, 0)

    beta, alpha_annual, r_sq = estimate_beta(strat, bench, rf_rate, periods_per_year)

    strat_simple = np.expm1(strat)
    bench_simple = np.expm1(bench)
    total_return = float(np.prod(1.0 + strat_simple) - 1.0)
    benchmark_total_return = float(np.prod(1.0 + bench_simple) - 1.0)
    beta_contribution = beta * benchmark_total_return
    alpha_contribution = total_return - beta_contribution
    beta_pct = (beta_contribution / total_return * 100.0) if abs(total_return) > 1e-12 else 0.0

    return BetaDecomposition(
        beta=beta,
        alpha_annual=alpha_annual,
        r_squared=r_sq,
        total_return=total_return,
        benchmark_total_return=benchmark_total_return,
        beta_contribution=beta_contribution,
        alpha_contribution=alpha_contribution,
        beta_pct=beta_pct,
        observations=len(strat),
    )


def compute_hedge_ratio(
    long_notional: float,
    long_beta: float,
    hedge_instrument_beta: float = 1.0,
) -> HedgeRatioResult:
    """Beta-neutral vs dollar-neutral hedge sizing (Lesson 8.2)."""
    if hedge_instrument_beta == 0:
        hedge_instrument_beta = 1.0

    hedge_notional = long_notional * long_beta / hedge_instrument_beta
    dollar_neutral = long_notional

    net_beta_dollar = long_notional * long_beta - dollar_neutral * hedge_instrument_beta
    net_beta_beta = long_notional * long_beta - hedge_notional * hedge_instrument_beta

    return HedgeRatioResult(
        long_notional=long_notional,
        long_beta=long_beta,
        hedge_instrument_beta=hedge_instrument_beta,
        hedge_notional=hedge_notional,
        dollar_neutral_hedge_notional=dollar_neutral,
        net_beta_dollar_neutral=net_beta_dollar,
        net_beta_beta_neutral=net_beta_beta,
    )


def estimate_hedge_cost_annual(
    hedge_notional: float,
    capital: float,
    borrow_rate_annual: float,
    trading_cost_pct: float = 0.01,
) -> dict[str, float]:
    """Retail hedge cost estimate (Lesson 8.3)."""
    borrow_cost = hedge_notional * borrow_rate_annual
    trading_cost = hedge_notional * trading_cost_pct
    total = borrow_cost + trading_cost
    return {
        "borrow_cost": borrow_cost,
        "trading_cost": trading_cost,
        "total_cost": total,
        "cost_pct_of_capital": total / capital if capital > 0 else 0.0,
        "breakeven_alpha_pct": total / capital if capital > 0 else 0.0,
    }


def symbol_beta_from_prices(
    closes: list[float],
    benchmark_closes: list[float],
    rf_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    asset_rets = log_returns(closes)
    bench_rets = log_returns(benchmark_closes)
    beta, _, _ = estimate_beta(
        np.array(asset_rets),
        np.array(bench_rets),
        rf_rate,
        periods_per_year,
    )
    return beta
