from __future__ import annotations

from src.core.types import SignalSide
from src.strategies.trend_following import StrategySignal


def evaluate_grid(
    close: float,
    reference_price: float,
    grid_step_pct: float,
    num_grids: int,
    max_loss_pct: float,
    dry_run_only: bool = True,
) -> StrategySignal | None:
    """Grid trading — dry-run only with mandatory stop-loss (Lesson 5.3)."""
    if not dry_run_only or reference_price <= 0:
        return None

    level = int((reference_price - close) / (reference_price * grid_step_pct))
    level = max(-num_grids, min(num_grids, level))
    if level == 0:
        return None

    stop_price = reference_price * (1 - max_loss_pct / 100)
    side = SignalSide.BUY if level > 0 else SignalSide.SELL
    strength = min(abs(level) / num_grids, 1.0)

    return StrategySignal(
        side=side,
        strength=max(strength, 0.2),
        reason=(
            f"[DRY RUN GRID] level={level} ref={reference_price:.5f} "
            f"stop={stop_price:.5f} (max loss {max_loss_pct}%)"
        ),
    )
