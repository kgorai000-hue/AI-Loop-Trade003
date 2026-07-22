from __future__ import annotations

import logging

from src.agents.risk_agent import RiskAgent
from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import AgentState, DecisionReport, SignalSide, TradeSignal
from src.data.store import OHLCVStore
from src.execution.position_sizing import hybrid_position_pct, lots_from_equity_pct, risk_parity_weights
from src.market.symbol_info import fetch_market_symbol_info
from src.stats.returns import log_returns
from src.stats.risk import volatility

logger = logging.getLogger(__name__)


class DecisionAgent:
    """Convert predictions to sized trade decisions (Lesson 10)."""

    def __init__(
        self,
        config: AppConfig,
        connector: MT5Connector,
        store: OHLCVStore,
        risk_agent: RiskAgent | None = None,
    ) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.decision = config.decision
        self.risk = config.risk
        self.risk_agent = risk_agent or RiskAgent(config, connector, store)

    def build_state(self) -> AgentState:
        account = self.connector.get_account_info()
        equity = float(account.get("equity", 0.0))
        free_margin = float(account.get("margin_free", account.get("free_margin", equity)))

        import MetaTrader5 as mt5

        open_lots: dict[str, float] = {}
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                sym = str(pos.symbol)
                open_lots[sym] = open_lots.get(sym, 0.0) + float(pos.volume)

        exposure_pct = self.risk_agent._current_exposure_pct()  # noqa: SLF001
        return AgentState(
            equity=equity,
            free_margin=free_margin,
            current_exposure_pct=exposure_pct,
            open_position_lots=open_lots,
        )

    def decide(self, signals: list[TradeSignal]) -> tuple[list[TradeSignal], list[DecisionReport]]:
        if not self.decision.enabled or not signals:
            return self._apply_fallback(signals)

        state = self.build_state()
        if state.equity <= 0:
            logger.warning("DecisionAgent: invalid equity; using fallback lots")
            return self._apply_fallback(signals)

        remaining_exposure = max(
            0.0,
            self.risk.max_total_exposure_pct - state.current_exposure_pct,
        )
        if remaining_exposure <= 0:
            logger.info("DecisionAgent: max total exposure reached; zeroing requested lots")
            return self._zero_signals(signals, "total exposure limit reached")

        vol_map = {signal.symbol: self._annualized_vol(signal.symbol) for signal in signals}
        portfolio_weights = {
            signal.symbol: signal.portfolio_weight
            for signal in signals
            if signal.portfolio_weight is not None
        }
        if portfolio_weights:
            weight_values = portfolio_weights
        elif self.decision.use_risk_parity:
            weight_values = risk_parity_weights(vol_map)
        else:
            weight_values = {signal.symbol: 1.0 / len(signals) for signal in signals}

        decided: list[TradeSignal] = []
        reports: list[DecisionReport] = []
        allocated_exposure_pct = 0.0

        for signal in signals:
            confidence = signal.confidence if signal.confidence is not None else signal.strength
            if confidence < self.decision.min_confidence:
                reports.append(
                    DecisionReport(
                        symbol=signal.symbol,
                        side=signal.side,
                        predicted_return=signal.predicted_return or 0.0,
                        confidence=confidence,
                        annualized_volatility=vol_map.get(signal.symbol, 0.2),
                        half_kelly_cap_pct=0.0,
                        van_tharp_cap_pct=0.0,
                        hard_cap_pct=self.risk.max_single_position_pct,
                        portfolio_weight_pct=0.0,
                        final_position_pct=0.0,
                        requested_lots=0.0,
                        reason=f"confidence {confidence:.3f} below min {self.decision.min_confidence}",
                    )
                )
                decided.append(self._copy_signal(signal, requested_lots=0.0))
                continue

            predicted = signal.predicted_return
            if predicted is None:
                predicted = signal.strength * 0.01

            ann_vol = vol_map.get(signal.symbol, 0.2)
            stop_distance = self._stop_distance(signal.symbol, signal.side)
            price = self._latest_price(signal.symbol, signal.side)
            if price <= 0:
                reports.append(
                    DecisionReport(
                        symbol=signal.symbol,
                        side=signal.side,
                        predicted_return=predicted,
                        confidence=confidence,
                        annualized_volatility=ann_vol,
                        half_kelly_cap_pct=0.0,
                        van_tharp_cap_pct=0.0,
                        hard_cap_pct=self.risk.max_single_position_pct,
                        portfolio_weight_pct=weight_values.get(signal.symbol, 0.0) * 100,
                        final_position_pct=0.0,
                        requested_lots=0.0,
                        reason="missing price data",
                    )
                )
                decided.append(self._copy_signal(signal, requested_lots=0.0))
                continue

            remaining_for_trade = max(0.0, remaining_exposure - allocated_exposure_pct)
            position_pct, breakdown = hybrid_position_pct(
                win_rate=self.decision.win_rate,
                reward_risk_ratio=self.decision.reward_risk_ratio,
                equity=state.equity,
                risk_pct=self.decision.risk_pct_per_trade,
                stop_loss_distance=stop_distance,
                price=price,
                max_single_pct=self.risk.max_single_position_pct,
                remaining_exposure_pct=remaining_for_trade,
                portfolio_weight=weight_values.get(signal.symbol, 1.0 / len(signals)),
                trade_wins=self.decision.trade_wins,
                trade_losses=self.decision.trade_losses,
                avg_win_pct=self.decision.avg_win_pct,
                avg_loss_pct=self.decision.avg_loss_pct,
            )

            position_pct *= confidence
            try:
                info = fetch_market_symbol_info(self.connector, self.config, signal.symbol)
                lots = lots_from_equity_pct(
                    equity=state.equity,
                    position_pct=position_pct,
                    contract_size=info.contract_size,
                    price=price,
                    volume_min=info.volume_min,
                    volume_max=info.volume_max,
                    volume_step=info.volume_step,
                )
                lots = round(lots * self.config.lot_multiplier_for(signal.symbol), 2)
                if lots < info.volume_min:
                    lots = 0.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("DecisionAgent lot conversion failed for %s: %s", signal.symbol, exc)
                lots = self._fallback_lots(signal)

            reason = (
                f"hybrid sizing ({breakdown.get('kelly_method', 'half_kelly')}): "
                f"kelly={breakdown['half_kelly_cap_pct']:.1f}% "
                f"van_tharp={breakdown['van_tharp_cap_pct']:.1f}% "
                f"hard={breakdown['hard_cap_pct']:.1f}% "
                f"rp_weight={breakdown['portfolio_weight_pct']:.1f}% "
                f"final={position_pct * 100:.2f}% -> {lots:.2f} lots"
            )
            reports.append(
                DecisionReport(
                    symbol=signal.symbol,
                    side=signal.side,
                    predicted_return=predicted,
                    confidence=confidence,
                    annualized_volatility=ann_vol,
                    half_kelly_cap_pct=breakdown["half_kelly_cap_pct"],
                    van_tharp_cap_pct=breakdown["van_tharp_cap_pct"],
                    hard_cap_pct=breakdown["hard_cap_pct"],
                    portfolio_weight_pct=breakdown["portfolio_weight_pct"],
                    final_position_pct=position_pct * 100,
                    requested_lots=lots,
                    reason=reason,
                )
            )
            allocated_exposure_pct += position_pct * 100
            decided.append(
                self._copy_signal(
                    signal,
                    predicted_return=predicted,
                    confidence=confidence,
                    requested_lots=lots,
                )
            )

        return decided, reports

    def _annualized_vol(self, symbol: str) -> float:
        bars = self.store.get_recent_bars(
            symbol,
            self.config.stats.analysis_timeframe,
            self.config.history_bars_for(self.config.stats.analysis_timeframe),
        )
        if len(bars) < self.config.stats.vol_window:
            return 0.2
        closes = [float(bar["close"]) for bar in bars]
        returns = log_returns(closes)
        return volatility(
            returns[-self.config.stats.vol_window :],
            annualize=True,
            trading_days=self.config.trading.trading_days_per_year,
        )

    def _stop_distance(self, symbol: str, side: SignalSide) -> float:
        distance = self.risk_agent.get_stop_distance(symbol)
        if distance > 0:
            return distance
        return 1.0

    def _latest_price(self, symbol: str, side: SignalSide) -> float:
        try:
            info = fetch_market_symbol_info(self.connector, self.config, symbol)
            return info.ask if side == SignalSide.BUY else info.bid
        except Exception:  # noqa: BLE001
            bars = self.store.get_recent_bars(symbol, self.config.stats.signal_timeframe, 1)
            if bars:
                return float(bars[-1]["close"])
            return 0.0

    def _fallback_lots(self, signal: TradeSignal) -> float:
        if self.decision.fallback_lots is not None:
            base = self.decision.fallback_lots
        else:
            base = round(self.config.trading.default_lots * signal.strength, 2)
        return round(base * self.config.lot_multiplier_for(signal.symbol), 2)

    def _apply_fallback(
        self,
        signals: list[TradeSignal],
    ) -> tuple[list[TradeSignal], list[DecisionReport]]:
        decided: list[TradeSignal] = []
        reports: list[DecisionReport] = []
        for signal in signals:
            lots = self._fallback_lots(signal)
            decided.append(self._copy_signal(signal, requested_lots=lots))
            reports.append(
                DecisionReport(
                    symbol=signal.symbol,
                    side=signal.side,
                    predicted_return=signal.predicted_return or signal.strength * 0.01,
                    confidence=signal.confidence or signal.strength,
                    annualized_volatility=0.0,
                    half_kelly_cap_pct=0.0,
                    van_tharp_cap_pct=0.0,
                    hard_cap_pct=self.risk.max_single_position_pct,
                    portfolio_weight_pct=0.0,
                    final_position_pct=0.0,
                    requested_lots=lots,
                    reason="decision agent disabled; fallback lots",
                )
            )
        return decided, reports

    def _zero_signals(
        self,
        signals: list[TradeSignal],
        reason: str,
    ) -> tuple[list[TradeSignal], list[DecisionReport]]:
        decided = [self._copy_signal(signal, requested_lots=0.0) for signal in signals]
        reports = [
            DecisionReport(
                symbol=signal.symbol,
                side=signal.side,
                predicted_return=signal.predicted_return or 0.0,
                confidence=signal.confidence or signal.strength,
                annualized_volatility=0.0,
                half_kelly_cap_pct=0.0,
                van_tharp_cap_pct=0.0,
                hard_cap_pct=self.risk.max_single_position_pct,
                portfolio_weight_pct=0.0,
                final_position_pct=0.0,
                requested_lots=0.0,
                reason=reason,
            )
            for signal in signals
        ]
        return decided, reports

    @staticmethod
    def _copy_signal(
        signal: TradeSignal,
        *,
        predicted_return: float | None = None,
        confidence: float | None = None,
        requested_lots: float | None = None,
    ) -> TradeSignal:
        return TradeSignal(
            symbol=signal.symbol,
            side=signal.side,
            timeframe=signal.timeframe,
            strength=signal.strength,
            reason=signal.reason,
            mode=signal.mode,
            strategy=signal.strategy,
            predicted_return=predicted_return if predicted_return is not None else signal.predicted_return,
            confidence=confidence if confidence is not None else signal.confidence,
            requested_lots=requested_lots if requested_lots is not None else signal.requested_lots,
            portfolio_weight=signal.portfolio_weight,
            group_id=signal.group_id,
            pair_id=signal.pair_id,
            trade_mode=signal.trade_mode,
        )
