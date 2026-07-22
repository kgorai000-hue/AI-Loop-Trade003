from __future__ import annotations

import logging

import numpy as np

from src.core.config import AppConfig, AssetGroupConfig
from src.core.types import RegimeAssessment, StrategyKind, TradeSignal
from src.data.store import OHLCVStore
from src.backtest.gate_registry import BACKTEST_STRATEGY_NAMES, GateRegistry
from src.agents.mean_reversion_strategy_agent import MeanReversionStrategyAgent
from src.agents.ml_signal_agent import MLSignalAgent
from src.agents.pair_strategy_agent import PairStrategyAgent
from src.agents.trend_strategy_agent import TrendStrategyAgent
from src.features.indicators import bars_to_arrays
from src.strategies.grid import evaluate_grid

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """Regime-driven singles; asset_groups kept for pair universe / tagging."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.trend_agent = TrendStrategyAgent(config, store)
        self.mr_agent = MeanReversionStrategyAgent(config, store)
        self.pair_agent = PairStrategyAgent(config, store)
        self.ml_agent = MLSignalAgent(config, store)

    def scan(
        self,
        symbols: list[str] | None = None,
        regime_map: dict[str, RegimeAssessment] | None = None,
        sentiment_map: dict[str, float] | None = None,
        gate_registry: GateRegistry | None = None,
    ) -> list[TradeSignal]:
        symbols = symbols or self.config.symbols
        regime_map = regime_map or {}
        single_signals: list[TradeSignal] = []

        for symbol in symbols:
            group = self.config.group_for_symbol(symbol)
            if group is not None and not group.tradeable:
                continue

            regime = regime_map.get(symbol)
            if regime is not None and regime.selected_strategy == StrategyKind.CRISIS_HALT:
                continue
            if regime is not None and regime.position_scale <= 0:
                continue

            for sig in self._regime_single_signals(symbol, group, regime, gate_registry):
                if sig.strength > 0.05 and not self._duplicate(single_signals, sig):
                    single_signals.append(sig)

            grid_sig = self._grid_signal(symbol, regime)
            if grid_sig is not None and not self._duplicate(single_signals, grid_sig):
                single_signals.append(grid_sig)

            # ML / feature_score only when regime is uncertain-ish not used; skip on halt.
            # Keep optional ML off regime path only if explicitly enabled and no regime halt.
            if (
                self.config.ml.enabled
                and regime is not None
                and regime.selected_strategy
                in (StrategyKind.TREND_FOLLOWING, StrategyKind.MEAN_REVERSION)
                and self._gate_allows(gate_registry, symbol, "feature_score")
            ):
                ml_sig = self.ml_agent.generate(symbol)
                if ml_sig is not None and not self._duplicate(single_signals, ml_sig):
                    ml_sig = self._tag_single(ml_sig, group)
                    single_signals.append(ml_sig)

        pair_signals = self.pair_agent.scan(regime_map)
        signals = self._arbitrate_single_vs_pair(single_signals, pair_signals)

        if sentiment_map and self.config.llm_research.use_as_feature:
            signals = [self._apply_sentiment_nudge(sig, sentiment_map) for sig in signals]

        return signals

    def _regime_single_signals(
        self,
        symbol: str,
        group: AssetGroupConfig | None,
        regime: RegimeAssessment | None,
        gate_registry: GateRegistry | None,
    ) -> list[TradeSignal]:
        """Singles follow regime only — Group.strategy is ignored for routing."""
        out: list[TradeSignal] = []
        selected = regime.selected_strategy if regime else StrategyKind.CRISIS_HALT
        use_trend = selected == StrategyKind.TREND_FOLLOWING
        use_mr = selected == StrategyKind.MEAN_REVERSION

        if use_trend and self._gate_allows(gate_registry, symbol, "trend_following"):
            sig = self.trend_agent.generate(symbol, regime)
            if sig is not None:
                sig = self._apply_regime_weight(sig, regime, "trend")
                out.append(self._tag_single(sig, group))

        if use_mr and self._gate_allows(gate_registry, symbol, "mean_reversion"):
            sig = self.mr_agent.generate(symbol, regime)
            if sig is not None:
                sig = self._apply_regime_weight(sig, regime, "mean_reversion")
                out.append(self._tag_single(sig, group))

        return out

    def _arbitrate_single_vs_pair(
        self,
        singles: list[TradeSignal],
        pairs: list[TradeSignal],
    ) -> list[TradeSignal]:
        """Prefer pair when strength >= threshold; never hold both for same symbol."""
        threshold = float(self.config.strategies.pair_priority_strength)
        pair_by_id: dict[str, list[TradeSignal]] = {}
        for sig in pairs:
            key = sig.pair_id or sig.symbol
            pair_by_id.setdefault(key, []).append(sig)

        winning_pairs: list[TradeSignal] = []
        blocked_symbols: set[str] = set()
        for pair_id, legs in pair_by_id.items():
            strength = max(leg.strength for leg in legs)
            if strength >= threshold:
                winning_pairs.extend(legs)
                for leg in legs:
                    blocked_symbols.add(leg.symbol)
                logger.info(
                    "Arbitration: prefer pair %s strength=%.3f (blocks %s)",
                    pair_id,
                    strength,
                    sorted({leg.symbol for leg in legs}),
                )

        kept_singles = [
            sig for sig in singles if sig.symbol not in blocked_symbols
        ]
        # Drop weaker pair legs when single wins (pair below threshold).
        return kept_singles + winning_pairs

    @staticmethod
    def _tag_single(signal: TradeSignal, group: AssetGroupConfig | None) -> TradeSignal:
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
            portfolio_weight=signal.portfolio_weight,
            group_id=group.name if group else signal.group_id,
            pair_id=None,
            trade_mode="single",
        )

    @staticmethod
    def _gate_allows(gate_registry: GateRegistry | None, symbol: str, strategy: str) -> bool:
        if gate_registry is None:
            return True
        if strategy not in BACKTEST_STRATEGY_NAMES:
            return True
        return gate_registry.check(symbol, strategy)

    def _grid_signal(
        self,
        symbol: str,
        regime: RegimeAssessment | None,
    ) -> TradeSignal | None:
        cfg = self.config.strategies
        if not cfg.grid_dry_run_only:
            return None
        if regime is not None and regime.selected_strategy != StrategyKind.MEAN_REVERSION:
            return None

        timeframe = self.config.stats.signal_timeframe
        bars = self.store.get_recent_bars(symbol, timeframe, self.config.history_bars_for(timeframe))
        if len(bars) < cfg.trend_ma_long:
            return None

        _, _, _, closes, _ = bars_to_arrays(bars)
        ref = float(np.mean(closes[-cfg.trend_ma_long:]))
        close = float(closes[-1])
        result = evaluate_grid(
            close,
            ref,
            cfg.grid_step_pct,
            cfg.grid_num_grids,
            cfg.grid_max_loss_pct,
            dry_run_only=cfg.grid_dry_run_only,
        )
        if result is None:
            return None

        from src.core.types import SignalMode

        group = self.config.group_for_symbol(symbol)
        return TradeSignal(
            symbol=symbol,
            side=result.side,
            timeframe=timeframe,
            strength=result.strength,
            mode=SignalMode.MEAN_REVERSION,
            strategy=StrategyKind.GRID_DRY_RUN,
            predicted_return=result.strength * 0.005,
            confidence=result.strength * 0.5,
            reason=result.reason,
            group_id=group.name if group else None,
            trade_mode="single",
        )

    @staticmethod
    def _duplicate(signals: list[TradeSignal], candidate: TradeSignal) -> bool:
        for sig in signals:
            if sig.symbol == candidate.symbol and sig.side == candidate.side:
                return True
        return False

    def _apply_sentiment_nudge(
        self,
        signal: TradeSignal,
        sentiment_map: dict[str, float],
    ) -> TradeSignal:
        score = sentiment_map.get(signal.symbol)
        if score is None:
            return signal

        cfg = self.config.llm_research
        nudge = score * cfg.sentiment_nudge_strength
        if signal.side.value == "sell":
            nudge = -nudge

        new_strength = max(0.0, min(1.0, signal.strength + nudge))
        return TradeSignal(
            symbol=signal.symbol,
            side=signal.side,
            timeframe=signal.timeframe,
            strength=round(new_strength, 4),
            reason=f"{signal.reason} | sentiment nudge {score:+.2f}",
            mode=signal.mode,
            strategy=signal.strategy,
            predicted_return=signal.predicted_return,
            confidence=signal.confidence,
            requested_lots=signal.requested_lots,
            portfolio_weight=signal.portfolio_weight,
            group_id=signal.group_id,
            pair_id=signal.pair_id,
            trade_mode=signal.trade_mode,
        )

    @staticmethod
    def _apply_regime_weight(
        signal: TradeSignal,
        regime: RegimeAssessment | None,
        weight_key: str,
    ) -> TradeSignal:
        if regime is None or not regime.strategy_weights:
            return signal
        weight = regime.strategy_weights.get(weight_key, 1.0)
        scale = regime.position_scale if regime.position_scale > 0 else 1.0
        factor = min(1.0, weight * scale)
        return TradeSignal(
            symbol=signal.symbol,
            side=signal.side,
            timeframe=signal.timeframe,
            strength=round(signal.strength * factor, 4),
            reason=f"{signal.reason} | regime weight {weight_key}={weight:.0%}",
            mode=signal.mode,
            strategy=signal.strategy,
            predicted_return=signal.predicted_return,
            confidence=(signal.confidence or signal.strength) * factor,
            requested_lots=signal.requested_lots,
            portfolio_weight=signal.portfolio_weight,
            group_id=signal.group_id,
            pair_id=signal.pair_id,
            trade_mode=signal.trade_mode,
        )
