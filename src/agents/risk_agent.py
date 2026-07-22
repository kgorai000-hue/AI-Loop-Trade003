from __future__ import annotations

import logging
import time

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import RegimeAssessment, RiskDecision, RiskDecisionType, TradeSignal
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.market.symbol_info import fetch_market_symbol_info
from src.risk.audit import RiskAuditStore
from src.risk.drawdown import evaluate_drawdown
from src.risk.stops import atr_stop_distance, fixed_stop_distance, vol_stop_distance
from src.risk.types import DrawdownLevel, RiskControlReport
from src.stats.returns import log_returns
from src.stats.risk import volatility

logger = logging.getLogger(__name__)


class RiskAgent:
    """Three-layer risk control with veto power (Lessons 04, 15)."""

    def __init__(self, config: AppConfig, connector: MT5Connector, store: OHLCVStore) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.risk = config.risk
        self.stats = config.stats
        self.engine = FeatureEngine(config)
        self.audit = RiskAuditStore(config.storage.path)
        self._safe_mode = False

        peak, circuit_active, circuit_until = self.audit.load_state()
        self._peak_equity = peak
        self._circuit_breaker_active = circuit_active
        self._circuit_breaker_until = circuit_until

    def assess_account(self) -> RiskControlReport:
        account = self.connector.get_account_info()
        equity = float(account.get("equity", 0.0))
        if equity <= 0:
            return RiskControlReport(
                peak_equity=0.0,
                current_equity=0.0,
                drawdown_pct=0.0,
                drawdown_level=DrawdownLevel.NORMAL.value,
                drawdown_action="normal",
                position_scale=0.0,
                new_positions_allowed=False,
                circuit_breaker_active=self._circuit_breaker_active,
                total_exposure_pct=100.0,
                warnings=["invalid equity"],
            )

        self._refresh_peak_equity(equity)
        drawdown_pct = self._drawdown_pct(equity)
        self._update_circuit_breaker(drawdown_pct)
        state = evaluate_drawdown(
            drawdown_pct,
            warning_pct=self.risk.drawdown_warning_pct,
            stop_pct=self.risk.drawdown_stop_pct,
            circuit_pct=self.risk.drawdown_circuit_pct,
            warning_scale=self.risk.drawdown_warning_scale,
            circuit_breaker_active=self._circuit_breaker_active,
        )
        exposure = self._current_exposure_pct()
        warnings: list[str] = []
        if state.level != DrawdownLevel.NORMAL:
            warnings.append(state.message)
        if exposure >= self.risk.max_total_exposure_pct * 0.9:
            warnings.append(f"exposure near limit: {exposure:.1f}%")

        return RiskControlReport(
            peak_equity=self._peak_equity,
            current_equity=equity,
            drawdown_pct=drawdown_pct,
            drawdown_level=state.level.value,
            drawdown_action=state.action.value,
            position_scale=state.position_scale,
            new_positions_allowed=state.new_positions_allowed,
            circuit_breaker_active=state.circuit_breaker_active,
            total_exposure_pct=exposure,
            warnings=warnings,
        )

    def review(
        self,
        signal: TradeSignal,
        requested_lots: float,
        regime: RegimeAssessment | None = None,
    ) -> RiskDecision:
        if self._safe_mode:
            return self._finalize(
                signal,
                requested_lots,
                RiskDecisionType.REJECT,
                0.0,
                "safe mode: new positions blocked",
            )

        account = self.connector.get_account_info()
        equity = float(account.get("equity", 0.0))
        if equity <= 0:
            return self._finalize(
                signal,
                requested_lots,
                RiskDecisionType.REJECT,
                0.0,
                "invalid account equity",
            )

        control = self.assess_account()
        if control.circuit_breaker_active:
            return self._finalize(
                signal,
                requested_lots,
                RiskDecisionType.REJECT,
                0.0,
                "circuit breaker active - no new positions",
                drawdown_level=control.drawdown_level,
            )

        if not control.new_positions_allowed:
            return self._finalize(
                signal,
                requested_lots,
                RiskDecisionType.REJECT,
                0.0,
                f"drawdown {control.drawdown_pct:.1f}% blocks new positions",
                drawdown_level=control.drawdown_level,
            )

        requested_pct = self._lots_to_exposure_pct(signal.symbol, requested_lots, equity)
        if requested_pct > self.risk.max_single_position_pct:
            capped_lots = self._exposure_pct_to_lots(
                signal.symbol,
                self.risk.max_single_position_pct,
                equity,
            )
            if capped_lots <= 0:
                return self._finalize(
                    signal,
                    requested_lots,
                    RiskDecisionType.REJECT,
                    0.0,
                    f"single position {requested_pct:.1f}% exceeds max {self.risk.max_single_position_pct:.1f}%",
                    drawdown_level=control.drawdown_level,
                )
            return self._apply_scaling(
                signal,
                requested_lots,
                capped_lots,
                regime,
                control,
                reason=(
                    f"single position cap: {requested_pct:.1f}% -> "
                    f"{self.risk.max_single_position_pct:.1f}%"
                ),
            )

        symbol_exposure = self._symbol_exposure_pct(signal.symbol, equity)
        if symbol_exposure + requested_pct > self.risk.max_symbol_exposure_pct:
            allowed_pct = max(0.0, self.risk.max_symbol_exposure_pct - symbol_exposure)
            allowed_lots = self._exposure_pct_to_lots(signal.symbol, allowed_pct, equity)
            if allowed_lots <= 0:
                return self._finalize(
                    signal,
                    requested_lots,
                    RiskDecisionType.REJECT,
                    0.0,
                    f"symbol {signal.symbol} at max exposure {symbol_exposure:.1f}%",
                    drawdown_level=control.drawdown_level,
                )
            return self._apply_scaling(
                signal,
                requested_lots,
                allowed_lots,
                regime,
                control,
                reason=f"symbol exposure cap for {signal.symbol}",
            )

        sector = self._symbol_sector(signal.symbol)
        sector_exposure = self._sector_exposure_pct(sector, equity)
        if sector_exposure + requested_pct > self.risk.max_sector_exposure_pct:
            allowed_pct = max(0.0, self.risk.max_sector_exposure_pct - sector_exposure)
            allowed_lots = self._exposure_pct_to_lots(signal.symbol, allowed_pct, equity)
            if allowed_lots <= 0:
                return self._finalize(
                    signal,
                    requested_lots,
                    RiskDecisionType.REJECT,
                    0.0,
                    f"sector {sector} at max exposure {sector_exposure:.1f}%",
                    drawdown_level=control.drawdown_level,
                )
            return self._apply_scaling(
                signal,
                requested_lots,
                allowed_lots,
                regime,
                control,
                reason=f"sector {sector} exposure cap",
            )

        total_exposure = control.total_exposure_pct
        if total_exposure + requested_pct > self.risk.max_total_exposure_pct:
            return self._finalize(
                signal,
                requested_lots,
                RiskDecisionType.REJECT,
                0.0,
                f"total exposure {total_exposure:.1f}% + {requested_pct:.1f}% exceeds max",
                drawdown_level=control.drawdown_level,
            )

        return self._apply_scaling(
            signal,
            requested_lots,
            requested_lots,
            regime,
            control,
            reason="within risk limits",
        )

    def enter_safe_mode(self, reason: str) -> None:
        self._safe_mode = True
        logger.critical("RiskAgent safe mode: %s", reason)

    def get_stop_distance(self, symbol: str) -> float:
        bars = self.store.get_recent_bars(symbol, self.stats.signal_timeframe, self.config.history_bars_for(self.stats.signal_timeframe))
        if not bars:
            return 1.0
        price = float(bars[-1]["close"])

        if self.risk.use_atr_stops:
            features = self.engine.build_from_bars(symbol, bars, self.stats.signal_timeframe)
            if features is not None and features.snapshot.atr > 0:
                return atr_stop_distance(
                    features.snapshot.atr,
                    self.config.indicators.atr_stop_multiplier,
                )

        analysis_bars = self.store.get_recent_bars(
            symbol,
            self.stats.analysis_timeframe,
            self.config.history_bars_for(self.stats.analysis_timeframe),
        )
        if len(analysis_bars) >= self.stats.vol_window:
            closes = [float(bar["close"]) for bar in analysis_bars]
            returns = log_returns(closes)
            daily_vol = volatility(
                returns[-self.stats.vol_window :],
                annualize=False,
            )
            vol_dist = vol_stop_distance(price, daily_vol, self.risk.vol_stop_multiplier)
            if vol_dist > 0:
                return vol_dist

        return fixed_stop_distance(price, self.risk.stop_loss_pct)

    def get_atr_stop_distance(self, symbol: str) -> float | None:
        return self.get_stop_distance(symbol)

    def _apply_scaling(
        self,
        signal: TradeSignal,
        requested_lots: float,
        base_lots: float,
        regime: RegimeAssessment | None,
        control: RiskControlReport,
        reason: str,
    ) -> RiskDecision:
        vol_factor = self._volatility_scale(signal.symbol)
        atr_factor = self._atr_scale(signal.symbol)
        regime_scale = regime.position_scale if regime else 1.0
        drawdown_scale = control.position_scale if control.position_scale > 0 else 1.0
        scale = min(vol_factor, atr_factor, regime_scale, drawdown_scale)
        adjusted_lots = round(base_lots * scale, 2)
        stop_distance = self.get_stop_distance(signal.symbol)

        if adjusted_lots <= 0:
            return self._finalize(
                signal,
                requested_lots,
                RiskDecisionType.REJECT,
                0.0,
                "position size reduced to zero by risk scaling",
                drawdown_level=control.drawdown_level,
                stop_distance=stop_distance,
            )

        decision_type = RiskDecisionType.APPROVE
        final_reason = reason
        if adjusted_lots < requested_lots:
            decision_type = RiskDecisionType.REDUCE
            final_reason = (
                f"{reason}; scaled {requested_lots} -> {adjusted_lots} lots "
                f"(vol={vol_factor:.2f}, atr={atr_factor:.2f}, "
                f"regime={regime_scale:.2f}, dd={drawdown_scale:.2f})"
            )

        return self._finalize(
            signal,
            requested_lots,
            decision_type,
            adjusted_lots,
            final_reason,
            drawdown_level=control.drawdown_level,
            stop_distance=stop_distance,
        )

    def _finalize(
        self,
        signal: TradeSignal,
        requested_lots: float,
        decision: RiskDecisionType,
        approved_lots: float,
        reason: str,
        *,
        drawdown_level: str = "normal",
        stop_distance: float | None = None,
    ) -> RiskDecision:
        exposure = self._current_exposure_pct()
        audit_id = None
        if self.risk.audit_enabled:
            audit_id = self.audit.record(
                symbol=signal.symbol,
                requested_lots=requested_lots,
                approved_lots=approved_lots,
                decision=decision.value,
                reason=reason,
                drawdown_pct=self._drawdown_pct(float(self.connector.get_account_info().get("equity", 0.0))),
                drawdown_level=drawdown_level,
                total_exposure_pct=exposure,
            )
        return RiskDecision(
            decision=decision,
            approved_lots=approved_lots,
            reason=reason,
            drawdown_level=drawdown_level,
            stop_distance=stop_distance,
            audit_id=audit_id,
        )

    def _refresh_peak_equity(self, equity: float) -> None:
        if self._peak_equity <= 0 or equity > self._peak_equity:
            self._peak_equity = equity
            self.audit.save_state(
                peak_equity=self._peak_equity,
                circuit_breaker_active=self._circuit_breaker_active,
                circuit_breaker_until=self._circuit_breaker_until,
            )

    def _drawdown_pct(self, equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - equity) / self._peak_equity * 100.0)

    def _update_circuit_breaker(self, drawdown_pct: float) -> None:
        now = int(time.time())
        if self._circuit_breaker_active and self._circuit_breaker_until > 0 and now >= self._circuit_breaker_until:
            self._circuit_breaker_active = False
            self._circuit_breaker_until = 0
            logger.info("RiskAgent circuit breaker cooldown ended")

        if drawdown_pct >= self.risk.drawdown_circuit_pct and not self._circuit_breaker_active:
            self._circuit_breaker_active = True
            cooldown_sec = self.risk.circuit_breaker_cooldown_days * 86400
            self._circuit_breaker_until = now + cooldown_sec
            logger.critical(
                "RiskAgent circuit breaker triggered: drawdown %.1f%% >= %.1f%%",
                drawdown_pct,
                self.risk.drawdown_circuit_pct,
            )
            self.audit.save_state(
                peak_equity=self._peak_equity,
                circuit_breaker_active=True,
                circuit_breaker_until=self._circuit_breaker_until,
            )

    def _symbol_sector(self, symbol: str) -> str:
        for group, symbols in self.config.symbol_groups.items():
            if symbol in symbols:
                return group
        return "other"

    def _lots_to_exposure_pct(self, symbol: str, lots: float, equity: float) -> float:
        if lots <= 0 or equity <= 0:
            return 0.0
        try:
            info = fetch_market_symbol_info(self.connector, self.config, symbol)
            notional = lots * info.contract_size * max(info.bid, info.ask, 1.0)
            return min(notional / equity * 100.0, 100.0)
        except Exception:  # noqa: BLE001
            return min(lots * 10.0, 100.0)

    def _exposure_pct_to_lots(self, symbol: str, exposure_pct: float, equity: float) -> float:
        if exposure_pct <= 0 or equity <= 0:
            return 0.0
        try:
            info = fetch_market_symbol_info(self.connector, self.config, symbol)
            notional = equity * exposure_pct / 100.0
            price = max(info.bid, info.ask, 1.0)
            raw = notional / max(info.contract_size * price, 1e-9)
            step = info.volume_step if info.volume_step > 0 else 0.01
            lots = (int(raw / step)) * step
            if lots < info.volume_min:
                return 0.0
            return min(lots, info.volume_max)
        except Exception:  # noqa: BLE001
            return round(exposure_pct / 10.0, 2)

    def _symbol_exposure_pct(self, symbol: str, equity: float) -> float:
        import MetaTrader5 as mt5

        positions = mt5.positions_get(symbol=symbol)
        if not positions or equity <= 0:
            return 0.0
        total_lots = sum(abs(float(p.volume)) for p in positions)
        return self._lots_to_exposure_pct(symbol, total_lots, equity)

    def _sector_exposure_pct(self, sector: str, equity: float) -> float:
        import MetaTrader5 as mt5

        symbols = set(self.config.symbol_groups.get(sector, []))
        if not symbols:
            return 0.0
        positions = mt5.positions_get()
        if not positions or equity <= 0:
            return 0.0
        total_pct = 0.0
        for pos in positions:
            sym = str(pos.symbol)
            if sym in symbols:
                total_pct += self._lots_to_exposure_pct(sym, abs(float(pos.volume)), equity)
        return total_pct

    def _atr_scale(self, symbol: str) -> float:
        bars = self.store.get_recent_bars(symbol, self.stats.signal_timeframe, self.config.history_bars_for(self.stats.signal_timeframe))
        features = self.engine.build_from_bars(symbol, bars, self.stats.signal_timeframe)
        if features is None:
            return 1.0

        atr_pct = features.snapshot.atr_pct
        if atr_pct >= 0.03:
            return 0.25
        if atr_pct >= 0.015:
            return 0.5
        return 1.0

    def _volatility_scale(self, symbol: str) -> float:
        bars = self.store.get_recent_bars(
            symbol,
            self.stats.analysis_timeframe,
            self.config.history_bars_for(self.stats.analysis_timeframe),
        )
        if len(bars) < self.stats.vol_window:
            return 1.0

        closes = [float(bar["close"]) for bar in bars]
        returns = log_returns(closes)
        ann_vol = volatility(
            returns[-self.stats.vol_window :],
            annualize=True,
            trading_days=self.config.trading.trading_days_per_year,
        )

        if ann_vol >= self.stats.regime_vol_crisis:
            return 0.25
        if ann_vol >= self.stats.regime_vol_bull_max:
            return 0.5
        return 1.0

    def _current_exposure_pct(self) -> float:
        import MetaTrader5 as mt5

        account = self.connector.get_account_info()
        equity = float(account.get("equity", 0.0))
        margin = float(account.get("margin", 0.0))
        if equity <= 0:
            return 100.0
        if margin > 0:
            return margin / equity * 100.0

        positions = mt5.positions_get()
        if not positions:
            return 0.0

        total_pct = 0.0
        for pos in positions:
            sym = str(pos.symbol)
            total_pct += self._lots_to_exposure_pct(sym, abs(float(pos.volume)), equity)
        return min(total_pct, 100.0)
