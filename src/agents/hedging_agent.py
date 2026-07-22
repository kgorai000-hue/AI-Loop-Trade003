from __future__ import annotations

import logging
from dataclasses import dataclass, field

import MetaTrader5 as mt5

from src.core.config import AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import SignalSide, TradeSignal
from src.data.store import OHLCVStore
from src.market.symbol_info import fetch_market_symbol_info
from src.stats.beta import (
    compute_hedge_ratio,
    estimate_hedge_cost_annual,
    symbol_beta_from_prices,
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolBetaExposure:
    symbol: str
    beta: float
    notional_jpy: float
    beta_exposure: float
    side: str


@dataclass
class HedgeRecommendation:
    hedge_symbol: str
    hedge_side: SignalSide
    hedge_lots: float
    hedge_notional_jpy: float
    net_beta_before: float
    net_beta_after: float
    hedge_cost_annual_pct: float
    breakeven_alpha_pct: float
    dry_run: bool
    reason: str


@dataclass
class HedgeReport:
    benchmark: str
    portfolio_beta: float
    dollar_neutral_net_beta: float
    beta_neutral_hedge_notional: float
    exposures: list[SymbolBetaExposure] = field(default_factory=list)
    recommendation: HedgeRecommendation | None = None
    retail_viable: bool = False
    notes: list[str] = field(default_factory=list)


class HedgingAgent:
    """Beta exposure and hedge recommendations (Lesson 08, dry-run for retail)."""

    def __init__(
        self,
        config: AppConfig,
        connector: MT5Connector,
        store: OHLCVStore,
    ) -> None:
        self.config = config
        self.connector = connector
        self.store = store
        self.hedging = config.hedging

    def assess(
        self,
        signals: list[TradeSignal] | None = None,
        capital_jpy: float | None = None,
    ) -> HedgeReport:
        if not self.hedging.enabled:
            return HedgeReport(
                benchmark=self.hedging.market_benchmark,
                portfolio_beta=0.0,
                dollar_neutral_net_beta=0.0,
                beta_neutral_hedge_notional=0.0,
                notes=["Hedging disabled in config"],
            )

        benchmark = self.hedging.market_benchmark
        hedge_symbol = self.hedging.hedge_instrument
        exposures = self._collect_exposures(signals or [])
        portfolio_beta_exp = sum(e.beta_exposure for e in exposures)
        long_notional = sum(e.notional_jpy for e in exposures if e.side == "long")

        if long_notional <= 0 and portfolio_beta_exp == 0:
            long_notional = self._estimate_capital_notional(capital_jpy)

        avg_beta = portfolio_beta_exp / long_notional if long_notional > 0 else 1.0
        ratio = compute_hedge_ratio(
            long_notional,
            avg_beta if long_notional > 0 else portfolio_beta_exp / max(len(exposures), 1),
            self.hedging.hedge_instrument_beta,
        )

        account = self.connector.get_account_info()
        capital = capital_jpy or float(account.get("equity", 0.0))
        cost = estimate_hedge_cost_annual(
            ratio.hedge_notional,
            capital,
            self.hedging.retail_borrow_rate_annual,
            self.hedging.trading_cost_annual_pct,
        )

        recommendation = self._build_recommendation(
            hedge_symbol,
            ratio.hedge_notional,
            portfolio_beta_exp,
            ratio.net_beta_beta_neutral,
            cost["cost_pct_of_capital"],
            cost["breakeven_alpha_pct"],
        )

        notes = [
            "Dollar-neutral != beta-neutral (Lesson 8.4)",
            f"Retail borrow ~{self.hedging.retail_borrow_rate_annual:.0%}/yr on short notional",
        ]
        if cost["breakeven_alpha_pct"] > 0.05:
            notes.append(
                f"Breakeven gross alpha ~{cost['breakeven_alpha_pct']:.1%} to cover hedge costs"
            )
        if not self.hedging.retail_viable:
            notes.append("Market-neutral not viable for retail; recommendations are dry-run only")

        return HedgeReport(
            benchmark=benchmark,
            portfolio_beta=avg_beta,
            dollar_neutral_net_beta=ratio.net_beta_dollar_neutral,
            beta_neutral_hedge_notional=ratio.hedge_notional,
            exposures=exposures,
            recommendation=recommendation,
            retail_viable=self.hedging.retail_viable,
            notes=notes,
        )

    def _collect_exposures(self, signals: list[TradeSignal]) -> list[SymbolBetaExposure]:
        exposures: list[SymbolBetaExposure] = []
        benchmark_bars = self.store.get_recent_bars(
            self.hedging.market_benchmark,
            self.config.stats.analysis_timeframe,
            self.config.history_bars_for(self.config.stats.analysis_timeframe),
        )
        bench_closes = [float(b["close"]) for b in benchmark_bars]

        for signal in signals:
            beta = self._symbol_beta(signal.symbol, bench_closes)
            notional = self._signal_notional(signal)
            side = "long" if signal.side == SignalSide.BUY else "short"
            signed_notional = notional if side == "long" else -notional
            exposures.append(
                SymbolBetaExposure(
                    symbol=signal.symbol,
                    beta=beta,
                    notional_jpy=abs(notional),
                    beta_exposure=signed_notional * beta,
                    side=side,
                )
            )

        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                if any(e.symbol == pos.symbol for e in exposures):
                    continue
                beta = self._symbol_beta(pos.symbol, bench_closes)
                symbol_info = fetch_market_symbol_info(self.connector, self.config, pos.symbol)
                notional = abs(symbol_info.contract_size * float(pos.volume) * symbol_info.bid)
                side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                signed = notional if side == "long" else -notional
                exposures.append(
                    SymbolBetaExposure(
                        symbol=pos.symbol,
                        beta=beta,
                        notional_jpy=notional,
                        beta_exposure=signed * beta,
                        side=side,
                    )
                )

        return exposures

    def _symbol_beta(self, symbol: str, bench_closes: list[float]) -> float:
        bars = self.store.get_recent_bars(
            symbol,
            self.config.stats.analysis_timeframe,
            self.config.history_bars_for(self.config.stats.analysis_timeframe),
        )
        if len(bars) < self.hedging.min_beta_observations or len(bench_closes) < self.hedging.min_beta_observations:
            return self.hedging.default_symbol_beta

        closes = [float(b["close"]) for b in bars]
        return symbol_beta_from_prices(
            closes,
            bench_closes,
            self.config.indicators.risk_free_rate,
            self.config.trading.trading_days_per_year,
        )

    def _signal_notional(self, signal: TradeSignal) -> float:
        try:
            info = fetch_market_symbol_info(self.connector, self.config, signal.symbol)
            lots = (
                signal.requested_lots
                if signal.requested_lots is not None
                else self.config.trading.default_lots * signal.strength
            )
            price = info.ask if signal.side == SignalSide.BUY else info.bid
            return abs(info.contract_size * lots * price)
        except Exception:  # noqa: BLE001
            account = self.connector.get_account_info()
            equity = float(account.get("equity", 1_000_000))
            return equity * self.config.risk.max_single_position_pct / 100.0

    def _estimate_capital_notional(self, capital_jpy: float | None) -> float:
        account = self.connector.get_account_info()
        equity = capital_jpy or float(account.get("equity", 0.0))
        return equity * self.config.risk.max_total_exposure_pct / 100.0

    def _build_recommendation(
        self,
        hedge_symbol: str,
        hedge_notional: float,
        net_beta_before: float,
        net_beta_after: float,
        cost_pct: float,
        breakeven_alpha: float,
    ) -> HedgeRecommendation | None:
        if abs(net_beta_before) < self.hedging.beta_neutral_tolerance:
            return None

        try:
            info = fetch_market_symbol_info(self.connector, self.config, hedge_symbol)
            price = info.bid if net_beta_before > 0 else info.ask
            lots = round(hedge_notional / max(info.contract_size * price, 1e-8), 2)
            lots = max(lots, 0.01)
        except Exception:  # noqa: BLE001
            lots = 0.0

        side = SignalSide.SELL if net_beta_before > 0 else SignalSide.BUY
        prefix = "[DRY RUN HEDGE] " if self.hedging.dry_run_only else ""

        return HedgeRecommendation(
            hedge_symbol=hedge_symbol,
            hedge_side=side,
            hedge_lots=lots,
            hedge_notional_jpy=hedge_notional,
            net_beta_before=net_beta_before,
            net_beta_after=net_beta_after,
            hedge_cost_annual_pct=cost_pct,
            breakeven_alpha_pct=breakeven_alpha,
            dry_run=self.hedging.dry_run_only,
            reason=(
                f"{prefix}Beta-neutral hedge {side.value} {lots:.2f} lots {hedge_symbol} "
                f"(notional ~{hedge_notional:,.0f} JPY)"
            ),
        )
