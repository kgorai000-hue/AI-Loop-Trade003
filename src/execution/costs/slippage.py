from __future__ import annotations

import math

from src.execution.costs.types import OrderBookLevel


def linear_slippage(order_size: float, adv: float, k: float = 0.3) -> float:
    """Slippage = k * OrderSize / ADV (Lesson 18.2.1)."""
    if adv <= 0:
        return 0.0
    return k * (order_size / adv)


def sqrt_slippage(
    order_size: float,
    adv: float,
    sigma: float,
    k: float = 1.0,
) -> float:
    """Slippage = k * sigma * sqrt(OrderSize / ADV) (Lesson 18.2.2)."""
    if adv <= 0 or order_size <= 0:
        return 0.0
    return k * sigma * math.sqrt(order_size / adv)


def estimate_slippage_from_orderbook(
    order_size: float,
    bids: list[OrderBookLevel],
    asks: list[OrderBookLevel],
    *,
    side: str = "buy",
) -> float:
    """Simulate market order walk through book (Lesson 18.2.3)."""
    if not bids or not asks or order_size <= 0:
        return float("inf")

    mid = (bids[0].price + asks[0].price) / 2.0
    levels = asks if side.lower() == "buy" else bids

    filled = 0.0
    cost = 0.0
    for level in levels:
        if filled >= order_size:
            break
        fill = min(level.size, order_size - filled)
        cost += fill * level.price
        filled += fill

    if filled < order_size:
        return float("inf")

    avg_price = cost / order_size
    slippage = (avg_price - mid) / mid if side.lower() == "buy" else (mid - avg_price) / mid
    return max(0.0, slippage)
