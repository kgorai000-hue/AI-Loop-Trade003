from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.backtest.costs import estimate_round_trip_cost_pct
from src.backtest.monte_carlo import monte_carlo_simulation
from src.backtest.overfitting import trend_parameter_sensitivity
from src.backtest.quality_gate import evaluate_quality_gates, estimate_bar_span_years
from src.backtest.splits import evaluate_oos_split
from src.backtest.strategies import (
    build_feature_score_signals,
    build_mean_reversion_signals,
    build_trend_signals,
)
from src.backtest.types import (
    BacktestResult,
    MonteCarloResult,
    OOSSplitResult,
    ParameterSensitivityResult,
    QualityGateReport,
    WalkForwardRound,
)
from src.backtest.walk_forward import summarize_walk_forward, walk_forward_validation
from src.core.config import AppConfig
from src.core.history import periods_per_year_for_timeframe
from src.core.types import MarketRegime
from src.data.store import OHLCVStore
from src.features.feature_vector import FeatureEngine
from src.features.indicators import bars_to_arrays

logger = logging.getLogger(__name__)


@dataclass
class StrategyValidationResult:
    strategy_name: str
    backtest: BacktestResult
    oos: OOSSplitResult
    walk_forward: list[WalkForwardRound]
    walk_forward_summary: dict[str, float]
    monte_carlo: MonteCarloResult
    param_sensitivity: list[ParameterSensitivityResult]
    quality_gate: QualityGateReport


@dataclass
class ValidationReport:
    symbol: str
    timeframe: str
    cost_per_trade_pct: float
    strategies: list[StrategyValidationResult] = field(default_factory=list)


class BacktestAgent:
    """Independent validation agent (Lesson 7.8)."""

    def __init__(self, config: AppConfig, store: OHLCVStore) -> None:
        self.config = config
        self.store = store
        self.engine = FeatureEngine(config)

    def validate_symbol(
        self,
        symbol: str,
        timeframe: str | None = None,
        regime: MarketRegime | None = None,
        bars: list | None = None,
    ) -> ValidationReport:
        timeframe = timeframe or self.config.stats.signal_timeframe
        bars = bars or self.store.get_recent_bars(
            symbol, timeframe, self.config.history_bars_for(timeframe)
        )
        if len(bars) < self.config.stats.min_bars:
            raise ValueError(f"Insufficient bars for {symbol} {timeframe}")

        _, _, _, closes, _ = bars_to_arrays(bars)
        closes_arr = np.array(closes, dtype=float)
        cost_pct = estimate_round_trip_cost_pct(self.config)
        span_years = estimate_bar_span_years(
            len(bars), timeframe, self.config.trading.trading_days_per_year
        )

        series = self.engine.compute_series(bars)
        adx = np.array(series["adx"]) if series else np.zeros(len(closes))

        report = ValidationReport(symbol=symbol, timeframe=timeframe, cost_per_trade_pct=cost_pct)
        for name in ("trend_following", "mean_reversion", "feature_score"):
            signals, adx_for = self._build_strategy_signals(name, closes_arr, bars, adx, regime)
            report.strategies.append(
                self._validate_strategy(
                    name,
                    closes_arr,
                    signals,
                    cost_pct,
                    adx_for,
                    len(bars),
                    span_years,
                    timeframe,
                )
            )
        return report

    def validate_strategy(
        self,
        symbol: str,
        strategy_name: str,
        timeframe: str | None = None,
        regime: MarketRegime | None = None,
        bars: list | None = None,
    ) -> StrategyValidationResult:
        timeframe = timeframe or self.config.stats.signal_timeframe
        bars = bars or self.store.get_recent_bars(
            symbol, timeframe, self.config.history_bars_for(timeframe)
        )
        if len(bars) < self.config.stats.min_bars:
            raise ValueError(f"Insufficient bars for {symbol} {timeframe}")

        _, _, _, closes, _ = bars_to_arrays(bars)
        closes_arr = np.array(closes, dtype=float)
        cost_pct = estimate_round_trip_cost_pct(self.config)
        span_years = estimate_bar_span_years(
            len(bars), timeframe, self.config.trading.trading_days_per_year
        )

        series = self.engine.compute_series(bars)
        adx = np.array(series["adx"]) if series else np.zeros(len(closes))
        signals, adx_for = self._build_strategy_signals(
            strategy_name, closes_arr, bars, adx, regime
        )
        return self._validate_strategy(
            strategy_name,
            closes_arr,
            signals,
            cost_pct,
            adx_for,
            len(bars),
            span_years,
            timeframe,
        )

    def _build_strategy_signals(
        self,
        strategy_name: str,
        closes_arr: np.ndarray,
        bars: list,
        adx: np.ndarray,
        regime: MarketRegime | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if strategy_name == "trend_following":
            return build_trend_signals(closes_arr, adx, self.config), adx
        if strategy_name == "mean_reversion":
            return build_mean_reversion_signals(bars, self.config), None
        if strategy_name == "feature_score":
            return build_feature_score_signals(bars, self.config, regime), None
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def _validate_strategy(
        self,
        name: str,
        closes: np.ndarray,
        signals: np.ndarray,
        cost_pct: float,
        adx: np.ndarray | None,
        n_bars: int,
        span_years: float,
        timeframe: str,
    ) -> StrategyValidationResult:
        cfg = self.config
        bt_cfg = cfg.backtest
        rf = cfg.indicators.risk_free_rate
        periods = periods_per_year_for_timeframe(
            timeframe, cfg.trading.trading_days_per_year
        )
        zero_means_flat = name == "trend_following"

        oos, backtest = evaluate_oos_split(
            closes,
            signals,
            name,
            cost_pct,
            bt_cfg.oos_train_ratio,
            bt_cfg.oos_val_ratio,
            rf,
            periods,
            bt_cfg.min_oos_ratio,
            zero_means_flat=zero_means_flat,
        )

        wf = walk_forward_validation(
            closes,
            signals,
            name,
            bt_cfg.train_window,
            bt_cfg.test_window,
            bt_cfg.walk_forward_step,
            cost_pct,
            rf,
            periods,
            zero_means_flat=zero_means_flat,
        )

        mc = monte_carlo_simulation(
            np.array(backtest.returns),
            n_simulations=bt_cfg.monte_carlo_simulations,
            cost_perturbation=bt_cfg.cost_perturbation_pct,
        )

        param_sens: list[ParameterSensitivityResult] = []
        if name == "trend_following" and adx is not None:
            param_sens = trend_parameter_sensitivity(
                closes,
                adx,
                cfg,
                cost_pct,
                bt_cfg.param_sensitivity_pct,
                bt_cfg.param_sensitivity_max_return_change,
                timeframe=timeframe,
            )

        gate = evaluate_quality_gates(
            name,
            cfg,
            backtest,
            oos,
            wf,
            mc,
            param_sens,
            n_bars,
            bt_cfg.strategies_tested,
            span_years,
        )

        return StrategyValidationResult(
            strategy_name=name,
            backtest=backtest,
            oos=oos,
            walk_forward=wf,
            walk_forward_summary=summarize_walk_forward(wf),
            monte_carlo=mc,
            param_sensitivity=param_sens,
            quality_gate=gate,
        )
