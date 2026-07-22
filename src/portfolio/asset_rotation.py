"""Cross Asset-Group rotation: concurrent multi-asset application + gradual migration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import AppConfig, AssetRotationConfig
from src.core.types import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class AssetRotationPlan:
    group_weights: dict[str, float] = field(default_factory=dict)
    active_groups: list[str] = field(default_factory=list)
    selected_symbols: list[str] = field(default_factory=list)
    migrated_from: list[str] = field(default_factory=list)
    migrated_to: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    previous_weights: dict[str, float] = field(default_factory=dict)


def _copy_signal(signal: TradeSignal, *, portfolio_weight: float | None = None) -> TradeSignal:
    return TradeSignal(
        symbol=signal.symbol,
        side=signal.side,
        timeframe=signal.timeframe,
        strength=signal.strength,
        reason=signal.reason,
        mode=signal.mode,
        strategy=signal.strategy,
        predicted_return=signal.predicted_return,
        confidence=signal.confidence,
        requested_lots=signal.requested_lots,
        portfolio_weight=portfolio_weight
        if portfolio_weight is not None
        else signal.portfolio_weight,
        group_id=signal.group_id,
        pair_id=signal.pair_id,
        trade_mode=signal.trade_mode,
    )


def _group_id(config: AppConfig, signal: TradeSignal) -> str:
    if signal.group_id:
        return signal.group_id
    group = config.group_for_symbol(signal.symbol)
    return group.name if group else "ungrouped"


def _state_path(config: AppConfig, cfg: AssetRotationConfig) -> Path:
    root = Path(config.intelligence.state_dir)
    if not root.is_absolute():
        from src.core.config import PROJECT_ROOT

        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{cfg.state_key}.json"


def load_previous_weights(config: AppConfig, cfg: AssetRotationConfig) -> dict[str, float]:
    path = _state_path(config, cfg)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("group_weights") or {}
        return {str(k): float(v) for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load asset rotation state: %s", exc)
        return {}


def save_rotation_state(config: AppConfig, cfg: AssetRotationConfig, plan: AssetRotationPlan) -> None:
    path = _state_path(config, cfg)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "group_weights": plan.group_weights,
        "active_groups": plan.active_groups,
        "selected_symbols": plan.selected_symbols,
        "migrated_from": plan.migrated_from,
        "migrated_to": plan.migrated_to,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _score_groups(
    config: AppConfig,
    signals: list[TradeSignal],
    power: float,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for signal in signals:
        gid = _group_id(config, signal)
        strength = max(0.0, float(signal.strength))
        scores[gid] = scores.get(gid, 0.0) + (strength**power)

    # Keep a floor opportunity for tradeable groups with no current signal
    # so migration back into them remains possible next cycle.
    for group in config.asset_groups.values():
        if group.tradeable and group.name not in scores:
            scores[group.name] = 0.0
    return scores


def _normalize_group_weights(
    scores: dict[str, float],
    *,
    previous: dict[str, float],
    cfg: AssetRotationConfig,
) -> dict[str, float]:
    groups = sorted(scores.keys())
    if not groups:
        return {}

    raw = {g: max(0.0, scores[g]) for g in groups}
    total = sum(raw.values())
    if total <= 1e-12:
        # No signal edge: equal opportunity across tradeable groups.
        eq = 1.0 / len(groups)
        base = {g: eq for g in groups}
    else:
        base = {g: raw[g] / total for g in groups}

    # Blend with previous weights for gradual migration (not abrupt switches).
    if cfg.migration_enabled and previous:
        blended: dict[str, float] = {}
        for g in groups:
            prev = float(previous.get(g, base[g]))
            blended[g] = 0.6 * base[g] + 0.4 * prev
        base = blended
        total = sum(base.values()) or 1.0
        base = {g: v / total for g, v in base.items()}

    # Clamp then renormalize — preserves multi-group concurrency.
    clipped = {
        g: min(cfg.max_group_weight, max(cfg.min_group_weight, w)) for g, w in base.items()
    }
    total = sum(clipped.values()) or 1.0
    return {g: w / total for g, w in clipped.items()}


def _select_active_groups(
    weights: dict[str, float],
    cfg: AssetRotationConfig,
) -> list[str]:
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    if not cfg.multi_group_enabled:
        return [ranked[0][0]] if ranked else []
    limit = max(1, int(cfg.max_active_groups))
    return [name for name, _ in ranked[:limit]]


def apply_asset_rotation(
    config: AppConfig,
    signals: list[TradeSignal],
) -> tuple[list[TradeSignal], AssetRotationPlan]:
    """
    Re-weight signals so multiple Asset Groups can apply concurrently,
    and capital can gradually migrate toward stronger groups.
    """
    cfg = config.asset_rotation
    plan = AssetRotationPlan()
    if not cfg.enabled or not signals:
        return signals, plan

    previous = load_previous_weights(config, cfg)
    plan.previous_weights = previous
    scores = _score_groups(config, signals, cfg.strength_power)
    weights = _normalize_group_weights(scores, previous=previous, cfg=cfg)
    active = _select_active_groups(weights, cfg)
    plan.group_weights = weights
    plan.active_groups = active

    # Detect migration (weight flow between groups).
    if cfg.migration_enabled and previous:
        for g, w in weights.items():
            prev = float(previous.get(g, 0.0))
            delta = w - prev
            if delta >= cfg.migration_shift_threshold:
                plan.migrated_to.append(g)
            elif delta <= -cfg.migration_shift_threshold:
                plan.migrated_from.append(g)
        if plan.migrated_from or plan.migrated_to:
            note = (
                f"asset migration: from={plan.migrated_from} to={plan.migrated_to} "
                f"weights={ {k: round(v, 3) for k, v in weights.items()} }"
            )
            plan.notes.append(note)
            logger.info(note)

    # Within each active group, rank symbols and keep concurrent slots.
    by_group: dict[str, list[TradeSignal]] = {}
    for signal in signals:
        gid = _group_id(config, signal)
        if gid not in active:
            continue
        by_group.setdefault(gid, []).append(signal)

    ranked_candidates: list[tuple[float, TradeSignal, str]] = []
    for gid, group_signals in by_group.items():
        gw = weights.get(gid, 0.0)
        for signal in group_signals:
            score = gw * max(0.0, float(signal.strength))
            ranked_candidates.append((score, signal, gid))
    ranked_candidates.sort(key=lambda row: row[0], reverse=True)

    limit = max(1, int(cfg.max_symbols_concurrent))
    selected = ranked_candidates[:limit]
    plan.selected_symbols = [sig.symbol for _, sig, _ in selected]

    # Allocate within-group share among selected symbols of that group.
    selected_by_group: dict[str, list[TradeSignal]] = {}
    for _, sig, gid in selected:
        selected_by_group.setdefault(gid, []).append(sig)

    out: list[TradeSignal] = []
    for gid, group_sigs in selected_by_group.items():
        gw = weights.get(gid, 0.0)
        strength_sum = sum(max(0.0, s.strength) for s in group_sigs) or float(len(group_sigs))
        for sig in group_sigs:
            share = max(0.0, sig.strength) / strength_sum
            # Combine with any existing optimizer weight.
            base = sig.portfolio_weight if sig.portfolio_weight is not None else 1.0
            new_w = float(base) * gw * share
            reason = (
                f"{sig.reason} | rotation group={gid} w={gw:.2f} "
                f"multi={'on' if cfg.multi_group_enabled else 'off'}"
            )
            updated = _copy_signal(sig, portfolio_weight=new_w)
            updated.reason = reason
            out.append(updated)

    plan.notes.append(
        f"active_groups={active} concurrent={plan.selected_symbols} "
        f"weights={ {k: round(v, 3) for k, v in weights.items()} }"
    )
    logger.info(
        "AssetRotation: active=%s selected=%s weights=%s",
        active,
        plan.selected_symbols,
        {k: round(v, 3) for k, v in weights.items()},
    )
    try:
        save_rotation_state(config, cfg, plan)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist asset rotation state: %s", exc)
    return out, plan


def resolve_scan_symbols(config: AppConfig, symbols: list[str] | None) -> list[str]:
    """Expand scan universe to all tradeable groups when multi-asset room is enabled."""
    cfg = config.asset_rotation
    if (
        cfg.enabled
        and cfg.multi_group_enabled
        and cfg.expand_scan_to_all_groups
    ):
        universe = config.tradeable_symbols_all_groups()
        if universe:
            return list(universe)
    return list(symbols or config.symbols)
