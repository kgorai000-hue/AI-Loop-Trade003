from __future__ import annotations

import logging

import numpy as np

from src.core.config import AppConfig
from src.core.types import SymbolStatsReport, TradeSignal
from src.data.store import OHLCVStore
from src.portfolio.covariance import correlation_matrix, shrunk_covariance
from src.portfolio.factors import check_factor_limits, portfolio_factor_exposures, symbol_factor_loadings
from src.portfolio.leverage import portfolio_volatility, risk_leverage
from src.portfolio.types import FactorExposure, PortfolioReport, SymbolAllocation
from src.portfolio.weights import (
    apply_correlation_penalty,
    cap_weights,
    equal_risk_contribution_weights,
    equal_weights,
    inverse_volatility_weights,
)
from src.stats.returns import log_returns
from src.stats.risk import volatility

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Signals -> optimal weights -> factor/leverage checks (Lesson 16)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.cfg = config.portfolio

    def allocate(
        self,
        signals: list[TradeSignal],
        research_reports: list[SymbolStatsReport] | None = None,
        *,
        equity: float = 0.0,
        benchmark_vol: float = 0.15,
    ) -> tuple[list[TradeSignal], PortfolioReport]:
        if not self.cfg.enabled or not signals:
            return signals, PortfolioReport(weight_method=self.cfg.weight_method)

        research_map = {report.symbol: report for report in (research_reports or [])}
        symbols = list(dict.fromkeys(signal.symbol for signal in signals))
        vol_map = {symbol: self._annualized_vol(symbol, research_map.get(symbol)) for symbol in symbols}
        returns_matrix, valid_symbols = self._returns_matrix(symbols)

        if self.cfg.weight_method == "equal":
            weights = equal_weights(symbols)
            shrinkage = None
            corr = np.eye(len(symbols))
        elif len(valid_symbols) >= 2 and returns_matrix.shape[0] >= self.cfg.min_bars:
            shrunk = shrunk_covariance(returns_matrix, method=self.cfg.shrinkage_method)
            cov = shrunk["covariance"]
            shrinkage = float(shrunk["shrinkage"])
            sub_symbols = [symbol for symbol in symbols if symbol in valid_symbols]
            idx = [valid_symbols.index(symbol) for symbol in sub_symbols]
            sub_cov = cov[np.ix_(idx, idx)]
            sub_corr = correlation_matrix(sub_cov)

            if self.cfg.weight_method == "erc":
                sub_weights = equal_risk_contribution_weights(sub_cov, sub_symbols)
            else:
                sub_weights = inverse_volatility_weights({s: vol_map[s] for s in sub_symbols})

            weights = {symbol: sub_weights.get(symbol, 0.0) for symbol in symbols}
            total = sum(weights.values())
            if total > 0:
                weights = {symbol: weight / total for symbol, weight in weights.items()}
            else:
                weights = equal_weights(symbols)

            corr = sub_corr if len(sub_symbols) == len(symbols) else np.eye(len(symbols))
        else:
            weights = inverse_volatility_weights(vol_map)
            shrinkage = None
            corr = np.eye(len(symbols))

        if self.cfg.correlation_penalty > 0 and len(symbols) >= 2:
            weights = apply_correlation_penalty(
                weights,
                corr if corr.shape[0] == len(symbols) else np.eye(len(symbols)),
                symbols,
                penalty_strength=self.cfg.correlation_penalty,
                threshold=self.cfg.correlation_high_threshold,
            )

        weights = cap_weights(weights, self.cfg.max_single_weight)

        loadings = {symbol: symbol_factor_loadings(symbol, research_map.get(symbol)) for symbol in symbols}
        factor_values = portfolio_factor_exposures(weights, loadings)
        factor_checks = check_factor_limits(factor_values, self.cfg.factor_limits)

        port_vol = portfolio_volatility(weights, vol_map, corr if corr.shape[0] == len(symbols) else np.eye(len(symbols)))
        risk_lev = risk_leverage(port_vol, benchmark_vol)

        warnings: list[str] = []
        for factor, value, limit, breached in factor_checks:
            if breached:
                warnings.append(f"factor {factor} exposure {value:.2f} exceeds limit {limit:.2f}")

        if risk_lev > self.cfg.max_risk_leverage:
            scale = self.cfg.max_risk_leverage / max(risk_lev, 1e-6)
            weights = {symbol: weight * scale for symbol, weight in weights.items()}
            warnings.append(f"risk leverage capped: {risk_lev:.2f} -> {self.cfg.max_risk_leverage:.2f}")
            port_vol *= scale
            risk_lev = risk_leverage(port_vol, benchmark_vol)
            factor_values = portfolio_factor_exposures(weights, loadings)

        avg_corr = 0.0
        max_corr = 0.0
        if len(symbols) >= 2 and corr.shape[0] == len(symbols):
            off_diag = corr[np.triu_indices(len(symbols), k=1)]
            if off_diag.size:
                avg_corr = float(np.mean(off_diag))
                max_corr = float(np.max(off_diag))

        allocations: list[SymbolAllocation] = []
        allocated_signals: list[TradeSignal] = []
        for signal in signals:
            weight = weights.get(signal.symbol, 0.0)
            if weight <= 0:
                continue
            strength_scale = weight * len(symbols)
            new_strength = max(0.0, min(1.0, signal.strength * strength_scale))
            allocated_signals.append(
                TradeSignal(
                    symbol=signal.symbol,
                    side=signal.side,
                    timeframe=signal.timeframe,
                    strength=round(new_strength, 4),
                    reason=f"{signal.reason} | portfolio weight {weight:.1%}",
                    mode=signal.mode,
                    strategy=signal.strategy,
                    predicted_return=signal.predicted_return,
                    confidence=(signal.confidence or signal.strength) * strength_scale,
                    requested_lots=signal.requested_lots,
                    portfolio_weight=weight,
                    group_id=signal.group_id,
                    pair_id=signal.pair_id,
                    trade_mode=signal.trade_mode,
                )
            )
            allocations.append(
                SymbolAllocation(
                    symbol=signal.symbol,
                    raw_weight=weight,
                    adjusted_weight=weight,
                    signal_strength=new_strength,
                    annualized_volatility=vol_map.get(signal.symbol, 0.2),
                )
            )

        report = PortfolioReport(
            weight_method=self.cfg.weight_method,
            allocations=allocations,
            factor_exposures=[
                FactorExposure(factor=f, exposure=v, limit=self.cfg.factor_limits.get(f), breached=b)
                for f, v, _, b in factor_checks
            ],
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            notional_leverage=0.0,
            risk_leverage=risk_lev,
            portfolio_volatility=port_vol,
            shrinkage=shrinkage,
            warnings=warnings,
        )
        logger.info(
            "PortfolioOptimizer: %d signals -> %d allocated, method=%s vol=%.1f%% risk_lev=%.2f",
            len(signals),
            len(allocated_signals),
            self.cfg.weight_method,
            port_vol * 100,
            risk_lev,
        )
        return allocated_signals, report

    def _annualized_vol(self, symbol: str, research: SymbolStatsReport | None) -> float:
        if research is not None and research.annualized_volatility > 0:
            return research.annualized_volatility
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

    def _returns_matrix(self, symbols: list[str]) -> tuple[np.ndarray, list[str]]:
        series: list[np.ndarray] = []
        valid: list[str] = []
        min_len = self.cfg.min_bars
        for symbol in symbols:
            bars = self.store.get_recent_bars(
                symbol,
                self.config.stats.analysis_timeframe,
                self.config.history_bars_for(self.config.stats.analysis_timeframe),
            )
            if len(bars) < min_len:
                continue
            closes = np.array([float(bar["close"]) for bar in bars], dtype=float)
            rets = log_returns(closes.tolist())
            if len(rets) < min_len - 1:
                continue
            series.append(np.array(rets[-(min_len - 1) :], dtype=float))
            valid.append(symbol)

        if len(valid) < 2:
            return np.zeros((0, 0)), valid
        min_rows = min(len(s) for s in series)
        aligned = np.column_stack([s[-min_rows:] for s in series])
        return aligned, valid
