from __future__ import annotations

import logging

from src.core.config import AppConfig
from src.core.types import RegimeAssessment, SignalMode, StrategyKind, TradeSignal
from src.data.store import OHLCVStore
from src.features.indicators import bars_to_arrays
from src.strategies.pairs import evaluate_pair

logger = logging.getLogger(__name__)


class PairStrategyAgent:
    """Z-score pairs trading constrained to within-group pairs (Lesson 5.3)."""

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

            regime = regime_map.get(sym_a) or regime_map.get(sym_b)
            if regime is not None and regime.selected_strategy == StrategyKind.CRISIS_HALT:
                continue
            if regime is not None and regime.selected_strategy == StrategyKind.TREND_FOLLOWING:
                continue

            bars_a = self.store.get_recent_bars(sym_a, timeframe, self.config.history_bars_for(timeframe))
            bars_b = self.store.get_recent_bars(sym_b, timeframe, self.config.history_bars_for(timeframe))
            if len(bars_a) < cfg.pair_lookback + 5 or len(bars_b) < cfg.pair_lookback + 5:
                continue

            _, _, _, closes_a, _ = bars_to_arrays(bars_a)
            _, _, _, closes_b, _ = bars_to_arrays(bars_b)
            leg_a, leg_b = evaluate_pair(
                closes_a,
                closes_b,
                cfg.pair_lookback,
                cfg.pair_zscore_entry,
                cfg.pair_zscore_exit,
                sym_a,
                sym_b,
            )
            pair_id = self.config.pair_state_id(sym_a, sym_b)
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
                        reason=leg_a.reason,
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
                        reason=leg_b.reason,
                        group_id=group_a.name,
                        pair_id=pair_id,
                        trade_mode="pair",
                    )
                )

        return signals
