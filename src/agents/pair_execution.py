"""Atomic pair-leg execution: both-or-neither gate + orphan recovery."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from src.core.types import ExecutionPlan, RiskDecision, TradeSignal
from src.market.symbol_info import MarketSymbolInfo

logger = logging.getLogger(__name__)

PAIR_ORPHAN_RETRIES = 3


@dataclass
class PreparedLeg:
    signal: TradeSignal
    risk: RiskDecision
    plan: ExecutionPlan
    symbol_info: MarketSymbolInfo
    daily_vol: float
    bars: list
    atr: float


def is_pair_leg(signal: TradeSignal) -> bool:
    return bool(signal.pair_id)


def partition_signals(
    signals: list[TradeSignal],
) -> tuple[dict[str, list[TradeSignal]], list[TradeSignal]]:
    """Split into pair_id → legs and unpaired singles."""
    from dataclasses import replace

    pairs: dict[str, list[TradeSignal]] = {}
    singles: list[TradeSignal] = []
    for signal in signals:
        if is_pair_leg(signal):
            leg = (
                signal
                if signal.trade_mode == "pair"
                else replace(signal, trade_mode="pair")
            )
            pairs.setdefault(str(leg.pair_id), []).append(leg)
        else:
            singles.append(signal)
    return pairs, singles


def plan_is_filled(plan: ExecutionPlan) -> bool:
    if plan.filled_lots <= 0:
        return False
    status = (plan.status or "").lower()
    if status in {"skipped", "planned", "rejected", "failed", "error"}:
        return False
    return status in {"filled", "simulated", "partial"} or plan.ticket is not None


def execute_pair_atomic(
    legs: list[PreparedLeg],
    *,
    execute_fn: Callable[[PreparedLeg], ExecutionPlan],
    close_fn: Callable[[ExecutionPlan, str], ExecutionPlan],
    orphan_retries: int = PAIR_ORPHAN_RETRIES,
    retry_sleep_seconds: float = 0.5,
) -> list[ExecutionPlan]:
    """
    Execute both pair legs. If exactly one fills, retry the missing leg
    ``orphan_retries`` times; on persistent failure close the filled leg.
    """
    if len(legs) != 2:
        raise ValueError(f"pair execution requires exactly 2 legs, got {len(legs)}")

    pair_id = legs[0].signal.pair_id or "pair"
    results: list[ExecutionPlan] = []
    for leg in legs:
        try:
            results.append(execute_fn(leg))
        except Exception as exc:  # noqa: BLE001
            failed = leg.plan
            failed.status = "failed"
            failed.reason = f"pair exec error: {exc}; {failed.reason}"
            logger.error("Pair %s leg %s failed: %s", pair_id, leg.signal.symbol, exc)
            results.append(failed)

    filled_flags = [plan_is_filled(p) for p in results]
    if all(filled_flags) or not any(filled_flags):
        if all(filled_flags):
            logger.info(
                "Pair %s both legs filled: %s",
                pair_id,
                [f"{p.symbol}:{p.ticket or p.status}" for p in results],
            )
        return results

    # Exactly one leg filled — retry the missing leg.
    missing_idx = 0 if not filled_flags[0] else 1
    filled_idx = 1 - missing_idx
    missing_leg = legs[missing_idx]
    orphan = results[filled_idx]
    logger.warning(
        "Pair %s one-legged after entry: filled=%s missing=%s — retrying missing up to %d times",
        pair_id,
        orphan.symbol,
        missing_leg.signal.symbol,
        orphan_retries,
    )

    for attempt in range(1, orphan_retries + 1):
        try:
            retried = execute_fn(missing_leg)
            results[missing_idx] = retried
            if plan_is_filled(retried):
                logger.info(
                    "Pair %s orphan recovered on attempt %d (%s ticket=%s)",
                    pair_id,
                    attempt,
                    retried.symbol,
                    retried.ticket,
                )
                return results
            logger.warning(
                "Pair %s orphan retry %d/%d for %s status=%s",
                pair_id,
                attempt,
                orphan_retries,
                missing_leg.signal.symbol,
                retried.status,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Pair %s orphan retry %d/%d for %s failed: %s",
                pair_id,
                attempt,
                orphan_retries,
                missing_leg.signal.symbol,
                exc,
            )
        if attempt < orphan_retries and retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)

    close_reason = (
        f"pair {pair_id} orphan close: missing leg {missing_leg.signal.symbol} "
        f"failed after {orphan_retries} retries"
    )
    logger.error(
        "Pair %s unresolved one-leg — closing filled %s ticket=%s",
        pair_id,
        orphan.symbol,
        orphan.ticket,
    )
    closed = close_fn(orphan, close_reason)
    results.append(closed)
    return results
