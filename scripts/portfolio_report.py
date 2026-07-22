from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.portfolio.covariance import shrunk_covariance
from src.portfolio.factors import portfolio_factor_exposures, symbol_factor_loadings
from src.portfolio.leverage import notional_leverage, risk_leverage
from src.portfolio.weights import equal_risk_contribution_weights, equal_weights, inverse_volatility_weights


def print_lesson_content() -> None:
    print("\n=== Lesson 16: Portfolio Construction ===")
    print("  Flow: Signals -> Portfolio Optimization -> Risk -> Execution")
    print("  Methods: equal / inverse_vol / erc + Ledoit-Wolf shrinkage")

    print("\n=== Equal Weight vs Equal Volatility (Lesson 16.2) ===")
    vols = {"A": 0.10, "B": 0.30}
    eq = equal_weights(list(vols))
    inv = inverse_volatility_weights(vols)
    port_vol_eq = np.sqrt((0.5 * 0.10) ** 2 + (0.5 * 0.30) ** 2)
    port_vol_inv = np.sqrt((0.75 * 0.10) ** 2 + (0.25 * 0.30) ** 2)
    print(f"  Equal weight      : A={eq['A']:.0%} B={eq['B']:.0%} -> port vol {port_vol_eq:.1%}")
    print(f"  Inverse vol       : A={inv['A']:.0%} B={inv['B']:.0%} -> port vol {port_vol_inv:.1%}")

    print("\n=== High Correlation Trap (Lesson 16.1) ===")
    corr = np.array([[1.0, 0.85, 0.78], [0.85, 1.0, 0.88], [0.78, 0.88, 1.0]])
    print(f"  3 strategies avg correlation: {corr[np.triu_indices(3, 1)].mean():.2f}")
    print("  -> diversification illusion when all bet same factor")

    print("\n=== Factor Exposure Example ===")
    weights = {"#USSPX500": 0.4, "GOLD": 0.3, "EURUSD": 0.3}
    loadings = {sym: symbol_factor_loadings(sym) for sym in weights}
    exposures = portfolio_factor_exposures(weights, loadings)
    for factor, value in sorted(exposures.items()):
        print(f"  {factor}: {value:.2f}")

    print("\n=== Hidden Leverage ===")
    equity = 1_000_000.0
    notional = 6_500_000.0
    print(f"  notional leverage : {notional_leverage(notional, equity):.1f}x")
    print(f"  risk leverage     : {risk_leverage(0.30, 0.15):.1f}x (30% port / 15% SPX)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio construction report (Lesson 16)")
    parser.add_argument("--paper-only", action="store_true")
    args = parser.parse_args()

    print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    pf = config.portfolio
    print("\n=== Configured Portfolio Layer ===")
    print(f"  enabled           : {pf.enabled}")
    print(f"  weight method     : {pf.weight_method}")
    print(f"  max single weight : {pf.max_single_weight:.0%}")
    print(f"  max risk leverage : {pf.max_risk_leverage:.1f}x")
    print(f"  factor limits     : {pf.factor_limits}")

    np.random.seed(42)
    returns = np.random.normal(0, 0.02, (120, 4))
    shrunk = shrunk_covariance(returns)
    print(f"\n=== Shrinkage Demo (synthetic 4 assets, 120 bars) ===")
    print(f"  shrinkage delta   : {shrunk['shrinkage']:.3f}")
    print(f"  sample cond       : {np.linalg.cond(shrunk['sample_cov']):.0f}")
    print(f"  shrunk cond       : {np.linalg.cond(shrunk['covariance']):.0f}")

    symbols = ["A", "B", "C", "D"]
    erc = equal_risk_contribution_weights(shrunk["covariance"], symbols)
    print("\n=== ERC Weights (synthetic) ===")
    for sym, weight in erc.items():
        print(f"  {sym}: {weight:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
