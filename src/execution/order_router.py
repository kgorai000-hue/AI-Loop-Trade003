from __future__ import annotations

import math

from src.core.config import ExecutionConfig
from src.execution.execution_types import OrderRoutingPlan
from src.market.symbol_info import MarketSymbolInfo


def route_order(
    lots: float,
    symbol_info: MarketSymbolInfo,
    execution_cfg: ExecutionConfig,
) -> OrderRoutingPlan:
    """Convert sized signal into executable child orders (Lesson 19.13)."""
    if lots <= 0:
        return OrderRoutingPlan(0.0, [], "none", "market", "zero lots")

    order_type = "limit" if symbol_info.spread_points > 50 else "market"
    chunk = max(symbol_info.volume_step, symbol_info.volume_min)
    max_lots_per_child = max(
        chunk,
        symbol_info.volume_max * execution_cfg.child_order_adv_fraction,
    )

    if lots <= max_lots_per_child or execution_cfg.max_child_orders <= 1:
        return OrderRoutingPlan(
            total_lots=lots,
            child_lots=[lots],
            algo="single",
            order_type=order_type,
            reason=f"single {order_type} order",
        )

    n_children = min(
        execution_cfg.max_child_orders,
        max(2, math.ceil(lots / max_lots_per_child)),
    )
    child_lots: list[float] = []
    remaining = lots
    for idx in range(n_children):
        if idx == n_children - 1:
            child_lots.append(round(remaining, 2))
        else:
            portion = round(lots / n_children / chunk) * chunk
            portion = min(portion, remaining)
            child_lots.append(portion)
            remaining -= portion

    child_lots = [lot for lot in child_lots if lot > 0]
    return OrderRoutingPlan(
        total_lots=lots,
        child_lots=child_lots,
        algo="split_twap",
        order_type=order_type,
        reason=f"split into {len(child_lots)} child {order_type} orders",
    )
