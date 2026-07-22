from __future__ import annotations

import unittest

import numpy as np

from src.portfolio.covariance import correlation_matrix, shrunk_covariance
from src.portfolio.factors import portfolio_factor_exposures, symbol_factor_loadings
from src.portfolio.leverage import notional_leverage, portfolio_volatility, risk_leverage
from src.portfolio.weights import (
    apply_correlation_penalty,
    equal_risk_contribution_weights,
    equal_weights,
    inverse_volatility_weights,
)


class PortfolioTests(unittest.TestCase):
    def test_equal_vs_inverse_vol_lesson_example(self) -> None:
        vols = {"A": 0.10, "B": 0.30}
        eq = equal_weights(["A", "B"])
        inv = inverse_volatility_weights(vols)
        self.assertAlmostEqual(eq["A"], 0.5)
        self.assertAlmostEqual(inv["A"], 0.75, places=2)
        self.assertAlmostEqual(inv["B"], 0.25, places=2)

    def test_inverse_vol_lesson_16_2_portfolio_vol(self) -> None:
        vols = {"A": 0.10, "B": 0.30}
        inv = inverse_volatility_weights(vols)
        corr = np.eye(2)
        port_vol = portfolio_volatility(inv, vols, corr)
        self.assertAlmostEqual(port_vol, 0.106, places=2)

    def test_erc_balances_risk_contribution(self) -> None:
        vols = np.array([0.10, 0.30])
        corr = np.eye(2)
        cov = np.outer(vols, vols) * corr
        weights = equal_risk_contribution_weights(cov, ["A", "B"])
        w = np.array([weights["A"], weights["B"]])
        port_vol = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / port_vol
        self.assertAlmostEqual(rc[0], rc[1], places=2)

    def test_correlation_penalty_reduces_high_corr(self) -> None:
        weights = {"A": 0.5, "B": 0.5}
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        adjusted = apply_correlation_penalty(weights, corr, ["A", "B"], penalty_strength=0.5)
        self.assertLess(adjusted["A"] + adjusted["B"], 1.0 + 1e-9)
        self.assertAlmostEqual(adjusted["A"] + adjusted["B"], 1.0, places=6)

    def test_shrunk_covariance_improves_condition(self) -> None:
        np.random.seed(0)
        returns = np.random.normal(0, 0.02, (80, 6))
        result = shrunk_covariance(returns)
        sample_cond = np.linalg.cond(result["sample_cov"])
        shrunk_cond = np.linalg.cond(result["covariance"])
        self.assertLess(shrunk_cond, sample_cond)

    def test_factor_exposure_weighted_sum(self) -> None:
        weights = {"#USSPX500": 0.5, "GOLD": 0.5}
        loadings = {sym: symbol_factor_loadings(sym) for sym in weights}
        exposures = portfolio_factor_exposures(weights, loadings)
        self.assertAlmostEqual(exposures["market_beta"], 0.525, places=2)
        self.assertAlmostEqual(exposures["commodities"], 0.5, places=2)

    def test_leverage_metrics(self) -> None:
        self.assertAlmostEqual(notional_leverage(2_000_000, 1_000_000), 2.0)
        self.assertAlmostEqual(risk_leverage(0.30, 0.15), 2.0)

    def test_correlation_matrix_diagonal_one(self) -> None:
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        corr = correlation_matrix(cov)
        self.assertAlmostEqual(corr[0, 0], 1.0)
        self.assertAlmostEqual(corr[1, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
