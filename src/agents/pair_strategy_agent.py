from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import RegimeAssessment, SignalMode, StrategyKind, TradeSignal
from src.data.store import OHLCVStore
from src.features.indicators import bars_to_arrays
from src.pairs.health import PairHealthThresholds, classify_pair_health
from src.pairs.spread import build_spread_snapshot
from src.pairs.states import ENTRY_ALLOWED
from src.strategies.pairs import evaluate_pair

logger = logging.getLogger(__name__)


class PairStrategyAgent:
    """Group-constrained pairs with rolling-β spread and R1–R5 health gate.

    Single-symbol regimes are reference annotations / soft R5 hints only —
    relationship health decides whether to trade.
    """

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.strategies = config.strategies

    def scan(
        self,
        regime_map: dict[str, RegimeAssessment] | None = None,
    ) -> list[TradeSignal]:
        regime_map = regime_map or {}
        signals: list[TradeSignal] = []
        timeframe = self.config.stats.signal_timeframe
        cfg = self.strategies
        beta_window = int(getattr(cfg, "pair_beta_window", 60))
        beta_short = int(getattr(cfg, "pair_beta_short_window", 40))
        beta_long = int(getattr(cfg, "pair_beta_long_window", 120))
        health_th = PairHealthThresholds(
            max_half_life_bars=float(getattr(cfg, "pair_max_half_life_bars", 48.0)),
            weaken_half_life_mult=float(getattr(cfg, "pair_weaken_half_life_mult", 2.0)),
            max_beta_drift=float(getattr(cfg, "pair_max_beta_drift", 0.35)),
            break_beta_drift=float(getattr(cfg, "pair_break_beta_drift", 0.60)),
            max_abs_trend_slope=float(getattr(cfg, "pair_max_abs_trend_slope", 0.002)),
            break_abs_trend_slope=float(getattr(cfg, "pair_break_abs_trend_slope", 0.005)),
            min_zero_cross_rate=float(getattr(cfg, "pair_min_zero_cross_rate", 0.05)),
            vol_high_mult=float(getattr(cfg, "pair_vol_high_mult", 1.5)),
        )

        hist = max(
            self.config.history_bars_for(timeframe),
            cfg.pair_lookback + beta_long + 20,
        )

        for pair in cfg.pairs:
            if len(pair) != 2:
                continue
            sym_a, sym_b = pair[0], pair[1]
            group_a = self.config.group_for_symbol(sym_a)
            group_b = self.config.group_for_symbol(sym_b)
            if group_a is None or group_b is None:
                logger.warning("Skipping pair outside asset_groups: %s/%s", sym_a, sym_b)
                continue
            if group_a.name != group_b.name:
                logger.warning(
                    "Skipping cross-group pair %s/%s (%s vs %s)",
                    sym_a,
                    sym_b,
                    group_a.name,
                    group_b.name,
                )
                continue
            if not group_a.tradeable:
                continue

            bars_a = self.store.get_recent_bars(sym_a, timeframe, hist)
            bars_b = self.store.get_recent_bars(sym_b, timeframe, hist)
            if len(bars_a) < cfg.pair_lookback + 5 or len(bars_b) < cfg.pair_lookback + 5:
                continue

            _, _, _, closes_a, _ = bars_to_arrays(bars_a)
            _, _, _, closes_b, _ = bars_to_arrays(bars_b)

            snap = build_spread_snapshot(
                closes_a,
                closes_b,
                z_lookback=cfg.pair_lookback,
                beta_window=beta_window,
            )
            if snap is None:
                continue

            health = classify_pair_health(
                snap,
                closes_a=closes_a,
                closes_b=closes_b,
                thresholds=health_th,
                beta_short_window=beta_short,
                beta_long_window=beta_long,
                leg_a_regime=regime_map.get(sym_a),
                leg_b_regime=regime_map.get(sym_b),
            )

            pair_id = self.config.pair_state_id(sym_a, sym_b)
            if health.regime not in ENTRY_ALLOWED or not health.allow_entry:
                logger.info(
                    "Pair gate skip %s regime=%s reasons=%s (%s)",
                    pair_id,
                    health.regime,
                    "; ".join(health.reasons[:3]),
                    health.single_regime_note,
                )
                continue

            leg_a, leg_b = evaluate_pair(
                closes_a,
                closes_b,
                cfg.pair_lookback,
                cfg.pair_zscore_entry,
                cfg.pair_zscore_exit,
                sym_a,
                sym_b,
                beta_window=beta_window,
                z_entry_mult=health.z_entry_mult,
                size_scale=health.size_scale,
                beta=health.beta,
            )
            if leg_a is None and leg_b is None:
                continue

            if health.half_life is not None:
                reason_suffix = (
                    f" | {health.regime} β={health.beta:.3f} HL={health.half_life:.1f}"
                )
            else:
                reason_suffix = f" | {health.regime} β={health.beta:.3f}"
            reason_suffix += f" | {health.single_regime_note}"

            if leg_a is not None:
                signals.append(
                    TradeSignal(
                        symbol=sym_a,
                        side=leg_a.side,
                        timeframe=timeframe,
                        strength=leg_a.strength,
                        mode=SignalMode.MEAN_REVERSION,
                        strategy=StrategyKind.PAIRS,
                        predicted_return=leg_a.strength * 0.01,
                        confidence=leg_a.strength,
                        reason=leg_a.reason + reason_suffix,
                        group_id=group_a.name,
                        pair_id=pair_id,
                        trade_mode="pair",
                    )
                )
            if leg_b is not None:
                signals.append(
                    TradeSignal(
                        symbol=sym_b,
                        side=leg_b.side,
                        timeframe=timeframe,
                        strength=leg_b.strength,
                        mode=SignalMode.MEAN_REVERSION,
                        strategy=StrategyKind.PAIRS,
                        predicted_return=leg_b.strength * 0.01,
                        confidence=leg_b.strength,
                        reason=leg_b.reason + reason_suffix,
                        group_id=group_a.name,
                        pair_id=pair_id,
                        trade_mode="pair",
                    )
                )

        return signals
