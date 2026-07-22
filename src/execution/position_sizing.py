from __future__ import annotations

import math


from src.risk.kelly import bayesian_kelly, kelly_sample_discount


def half_kelly(win_rate: float, reward_risk_ratio: float) -> float:
    """Half-Kelly position cap as fraction of equity (Lesson 10.3)."""
    if reward_risk_ratio <= 0:
        return 0.0
    full_kelly = (win_rate * (reward_risk_ratio + 1) - 1) / reward_risk_ratio
    return max(0.0, full_kelly / 2)


def van_tharp_cap_pct(
    equity: float,
    risk_pct: float,
    stop_loss_distance: float,
    price: float,
) -> float:
    """Van Tharp R-Multiple cap as fraction of equity."""
    if equity <= 0 or stop_loss_distance <= 0 or price <= 0:
        return 0.0
    max_loss = equity * risk_pct
    units = max_loss / stop_loss_distance
    position_value = units * price
    return position_value / equity


def risk_parity_weights(volatilities: dict[str, float]) -> dict[str, float]:
    """Inverse-volatility weights normalized to sum to 1."""
    if not volatilities:
        return {}
    inv_vols = {symbol: 1.0 / max(vol, 1e-6) for symbol, vol in volatilities.items()}
    total = sum(inv_vols.values())
    if total <= 0:
        share = 1.0 / len(volatilities)
        return {symbol: share for symbol in volatilities}
    return {symbol: weight / total for symbol, weight in inv_vols.items()}


def conservative_kelly_cap(
    *,
    win_rate: float,
    reward_risk_ratio: float,
    trade_wins: int | None = None,
    trade_losses: int | None = None,
    avg_win_pct: float | None = None,
    avg_loss_pct: float | None = None,
) -> tuple[float, str]:
    """Return position cap fraction and sizing method label (Lesson 15.2)."""
    if (
        trade_wins is not None
        and trade_losses is not None
        and avg_win_pct is not None
        and avg_loss_pct is not None
        and (trade_wins + trade_losses) >= 30
    ):
        result = bayesian_kelly(
            trade_wins,
            trade_losses,
            avg_win_pct,
            avg_loss_pct,
            apply_half_kelly=True,
        )
        discount = kelly_sample_discount(int(result["sample_size"]))
        cap = float(result["recommendation"]) * discount
        return max(0.0, cap), "bayesian_kelly"

    return half_kelly(win_rate, reward_risk_ratio), "half_kelly"


def hybrid_position_pct(
    *,
    win_rate: float,
    reward_risk_ratio: float,
    equity: float,
    risk_pct: float,
    stop_loss_distance: float,
    price: float,
    max_single_pct: float,
    remaining_exposure_pct: float,
    portfolio_weight: float = 1.0,
    trade_wins: int | None = None,
    trade_losses: int | None = None,
    avg_win_pct: float | None = None,
    avg_loss_pct: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Return final position fraction and sizing breakdown."""
    kelly_cap, kelly_method = conservative_kelly_cap(
        win_rate=win_rate,
        reward_risk_ratio=reward_risk_ratio,
        trade_wins=trade_wins,
        trade_losses=trade_losses,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
    )
    van_tharp = van_tharp_cap_pct(equity, risk_pct, stop_loss_distance, price)
    hard_cap = max(0.0, max_single_pct / 100.0)
    remaining = max(0.0, remaining_exposure_pct / 100.0)

    base_cap = min(kelly_cap, van_tharp, hard_cap, remaining)
    final = base_cap * max(0.0, min(portfolio_weight, 1.0))

    breakdown = {
        "half_kelly_cap_pct": kelly_cap * 100,
        "kelly_method": kelly_method,
        "van_tharp_cap_pct": van_tharp * 100,
        "hard_cap_pct": hard_cap * 100,
        "remaining_exposure_pct": remaining * 100,
        "portfolio_weight_pct": portfolio_weight * 100,
        "final_pct": final * 100,
    }
    return final, breakdown


def notional_per_lot(contract_size: float, price: float) -> float:
    if contract_size <= 0 or price <= 0:
        return 0.0
    return contract_size * price


def lots_from_equity_pct(
    equity: float,
    position_pct: float,
    contract_size: float,
    price: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> float:
    """Convert equity fraction to MT5 lots."""
    if equity <= 0 or position_pct <= 0:
        return 0.0

    target_value = equity * position_pct
    notional = notional_per_lot(contract_size, price)
    if notional <= 0:
        return 0.0

    raw_lots = target_value / notional
    step = volume_step if volume_step > 0 else 0.01
    steps = math.floor(raw_lots / step + 1e-9)
    lots = steps * step
    if lots < volume_min:
        return 0.0
    return min(lots, volume_max)
