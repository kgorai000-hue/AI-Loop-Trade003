from __future__ import annotations

import numpy as np

from src.backtest.types import MonteCarloResult


def monte_carlo_simulation(
    returns: np.ndarray,
    n_simulations: int = 1000,
    cost_perturbation: float = 0.20,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Shuffle trade order + perturb costs (Lesson 7.5)."""
    if len(returns) == 0:
        return MonteCarloResult(0, 0, 0, 0, 0, 0, 0)

    rng = np.random.default_rng(seed)
    simulated: list[float] = []
    base = returns.copy()

    for _ in range(n_simulations):
        shuffled = rng.permutation(base)
        noise = rng.uniform(1 - cost_perturbation, 1 + cost_perturbation, len(shuffled))
        adjusted = shuffled * noise
        total = float(np.prod(1.0 + adjusted) - 1.0)
        simulated.append(total)

    arr = np.array(simulated)
    return MonteCarloResult(
        simulations=n_simulations,
        mean_return=float(arr.mean()),
        std_return=float(arr.std()),
        percentile_5=float(np.percentile(arr, 5)),
        percentile_50=float(np.percentile(arr, 50)),
        percentile_95=float(np.percentile(arr, 95)),
        prob_positive=float(np.mean(arr > 0)),
    )
