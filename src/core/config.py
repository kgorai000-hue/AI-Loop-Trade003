from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.history import history_bars_for_timeframe, max_history_bars_for_timeframes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
LOCAL_SETTINGS_PATH = CONFIG_DIR / "settings.local.yaml"

TRADING_PROFILES: dict[str, dict[str, int | str]] = {
    "low": {"trades_per_day": 2, "primary_timeframe": "D1"},
    "medium": {"trades_per_day": 8, "primary_timeframe": "M30"},
    "high": {"trades_per_day": 20, "primary_timeframe": "M15"},
}


@dataclass
class MT5Config:
    path: str | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None
    timeout_ms: int = 60000
    require_demo: bool = True
    magic: int = 260703
    deviation: int = 50


@dataclass
class AssetGroupConfig:
    name: str
    symbols: list[str]
    strategy: str
    description: str = ""
    tradeable: bool = True
    lot_multiplier: float = 1.0


@dataclass
class DataStorageConfig:
    type: str = "sqlite"
    path: str = "data/mt5_loop.db"


@dataclass
class DataFetchConfig:
    max_retries: int = 5
    backoff_base: float = 2.0
    min_completeness_ratio: float = 0.9
    chunk_size: int = 5000
    range_chunk_days: int = 90


@dataclass
class DataQualityConfig:
    price_jump_threshold: float = 0.20
    missing_rate_warn_pct: float = 5.0
    gap_multiplier: float = 1.5
    freshness_warn_seconds: int = 3600
    anomaly_rate_warn_pct: float = 0.1
    api_error_rate_warn_pct: float = 5.0


@dataclass
class DataConfig:
    default_timeframe: str
    history_bars: int
    timeframes: list[str]
    storage: DataStorageConfig
    fetch: DataFetchConfig
    quality: DataQualityConfig
    source: str = "mt5"
    history_years: float | None = None
    history_bars_by_timeframe: dict[str, int] | None = None


@dataclass
class TradingConfig:
    profile: str
    trades_per_day: int
    trading_days_per_year: int
    default_lots: float
    primary_timeframe: str
    dry_run: bool


@dataclass
class CostConfig:
    slippage_rate: float
    market_impact_Y: float
    use_spread_when_slippage_unknown: bool
    enabled: bool
    slippage_model: str
    slippage_k_linear: float
    slippage_k_sqrt: float
    default_adv_notional: float
    signal_decay_halflife_minutes: float
    execution_delay_minutes: float
    min_net_alpha_pct: float
    max_order_adv_ratio: float
    block_untradable: bool
    opportunity_cost_enabled: bool


@dataclass
class RiskConfig:
    max_total_exposure_pct: float
    max_single_position_pct: float
    max_symbol_exposure_pct: float
    max_sector_exposure_pct: float
    max_risk_per_trade_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    use_atr_stops: bool
    vol_stop_multiplier: float
    drawdown_warning_pct: float
    drawdown_stop_pct: float
    drawdown_circuit_pct: float
    drawdown_warning_scale: float
    circuit_breaker_cooldown_days: int
    circuit_breaker_target_exposure_pct: float
    audit_enabled: bool
    safe_mode_on_failure: bool


@dataclass
class StatsConfig:
    analysis_timeframe: str
    signal_timeframe: str
    min_bars: int
    vol_window: int
    regime_vol_crisis: float
    regime_vol_bull_max: float
    autocorr_lags: list[int]
    momentum_autocorr_threshold: float
    mean_reversion_autocorr_threshold: float


@dataclass
class IndicatorsConfig:
    macd_fast: int
    macd_slow: int
    macd_signal: int
    macd_histogram_double: bool
    rsi_period: int
    bb_period: int
    bb_std: float
    atr_period: int
    atr_stop_multiplier: float
    divergence_lookback: int
    signal_score_threshold: float
    risk_free_rate: float


@dataclass
class StrategiesConfig:
    allocation: str
    uncertain_position_scale: float
    pairs: list[list[str]]
    pair_priority_strength: float
    trend_ma_short: int
    trend_ma_long: int
    adx_period: int
    adx_trend_threshold: float
    adx_sideways_threshold: float
    adx_confirm_days: int
    ma_side_days: int
    mr_rsi_oversold: float
    mr_rsi_overbought: float
    mr_bb_entry_low: float
    mr_bb_entry_high: float
    grid_dry_run_only: bool
    grid_step_pct: float
    grid_num_grids: int
    grid_max_loss_pct: float
    pair_zscore_entry: float
    pair_zscore_exit: float
    pair_lookback: int
    pair_beta_window: int = 60
    pair_beta_short_window: int = 40
    pair_beta_long_window: int = 120
    pair_max_half_life_bars: float = 48.0
    pair_weaken_half_life_mult: float = 2.0
    pair_max_beta_drift: float = 0.35
    pair_break_beta_drift: float = 0.60
    pair_max_abs_trend_slope: float = 0.002
    pair_break_abs_trend_slope: float = 0.005
    pair_min_zero_cross_rate: float = 0.05
    pair_vol_high_mult: float = 1.5


@dataclass
class DecisionConfig:
    enabled: bool
    win_rate: float
    reward_risk_ratio: float
    risk_pct_per_trade: float
    use_risk_parity: bool
    min_confidence: float
    fallback_lots: float | None
    trade_wins: int | None
    trade_losses: int | None
    avg_win_pct: float | None
    avg_loss_pct: float | None


@dataclass
class ExecutionConfig:
    max_retries: int
    retry_backoff_seconds: float
    circuit_breaker_threshold: int
    enabled: bool
    simulator_mode: str
    latency_ms: float
    partial_fill_enabled: bool
    max_child_orders: int
    child_order_adv_fraction: float
    quote_max_age_seconds: float
    log_executions: bool
    telemetry_db_path: str
    slippage_conservative_multiplier: float
    use_bid_ask_not_close: bool
    slippage_threshold_pct: float
    latency_warn_ms: float


@dataclass
class OpsConfig:
    enabled: bool
    heartbeat_timeout_seconds: float
    network_latency_warn_ms: float
    cpu_warn_pct: float
    memory_critical_pct: float
    disk_free_warn_gb: float
    data_disconnect_seconds: float
    api_success_warn_pct: float
    agent_latency_warn_ms: float
    task_queue_warn_count: int
    daily_drawdown_warn_pct: float
    weekly_drawdown_critical_pct: float
    abnormal_trade_position_pct: float
    trade_frequency_deviation_x: float
    alert_suppress_seconds: float
    quiet_hours_enabled: bool
    quiet_hours_start: int
    quiet_hours_end: int
    structured_log_dir: str
    model_registry_path: str
    enforce_market_hours: bool
    allow_trading_in_dry_run: bool
    canary_initial_weight: float
    rollback_max_error_rate: float
    rollback_min_sharpe: float


@dataclass
class TradeLogConfig:
    enabled: bool
    db_path: str
    agent_version: str


@dataclass
class ProjectConfig:
    name: str
    max_symbols: int
    primary_timeframe: str
    graduation_stage: str
    min_backtest_sharpe: float
    max_backtest_drawdown_pct: float
    sync_before_run: bool


@dataclass
class ResidentLoopConfig:
    poll_seconds: int
    review_weekday: int
    review_hour_utc: int
    sharpe_degrade_trigger: float
    run_pipeline_on_bar: bool
    sync_on_poll: bool
    pretrade_optimize: bool = True
    require_adopted_params: bool = True
    optimize_pairs: bool = True
    check_all_asset_groups: bool = True
    # When no valid Anthropic key: seed settings.yaml defaults (skip multi-hour grid).
    no_api_seed_baseline: bool = True
    # When grid finds no improvement: still seed defaults so demo loop can trade.
    seed_baseline_if_no_adopt: bool = True


@dataclass
class IntelligenceConfig:
    enabled: bool
    state_dir: str
    maker_model: str
    checker_model: str
    maker_candidates: int
    max_retries: int
    enable_prompt_cache: bool
    default_strategy: str
    loop: ResidentLoopConfig


@dataclass
class MultiAgentConfig:
    enabled: bool
    evolution_stage: int
    arbitration_mode: str
    parallel_analysis: bool
    agent_timeout_seconds: float
    circuit_breaker_threshold: int


@dataclass
class RegimeConfig:
    method: str
    crisis_vol_threshold: float
    crisis_correlation_threshold: float
    ranging_vol_max: float
    trending_adx_min: float
    trending_return_min: float
    ranging_adx_max: float
    confirm_days: int
    switch_cost_pct: float
    crisis_weight_amplify: float
    transition_strategy: str
    correlation_benchmark: str
    correlation_lookback: int
    gmm_min_samples: int
    probability_threshold: float
    adx_boundary_low: float
    adx_boundary_high: float
    post_switch_uncertain_days: int
    # Five-state hard cutover
    confirm_bars: int = 3
    high_vol_trend_scale: float = 0.5
    trend_lookback: int = 40
    er_lookback: int = 20
    vol_hist_window: int = 120
    enter_trend_score: float = 0.55
    enter_er_trend: float = 0.45
    enter_vol_max_for_stable: float = 0.70
    exit_trend_score: float = 0.35
    exit_er_trend: float = 0.30
    high_vol_enter: float = 0.70
    stress_vol: float = 0.90
    enter_range_abs_trend: float = 0.30
    enter_er_range_max: float = 0.35
    exit_range_abs_trend: float = 0.45
    stress_corr: float = 0.85
    stress_spread: float = 0.70
    regime_conditioned_bt: bool = True


@dataclass
class OnlineLearningConfig:
    enabled: bool
    decay_factor: float
    learning_rate: float
    min_update_interval_days: int
    max_update_interval_days: int
    psi_threshold: float
    performance_drop_threshold: float
    ic_viability_threshold: float
    default_monthly_ic_decay: float
    dynamic_threshold_k: float
    signal_threshold_default: float
    apply_dynamic_threshold: bool
    accuracy_window: int
    accuracy_warning_threshold: float
    min_total_bars: int
    lifecycle_incubation_sharpe_min: float
    lifecycle_maturity_sharpe_min: float
    lifecycle_decay_sharpe_max: float
    correlation_crisis_threshold: float
    sliding_window_days: int
    retrain_interval_days: int


@dataclass
class PortfolioConfig:
    enabled: bool
    weight_method: str
    shrinkage_method: str
    min_bars: int
    max_single_weight: float
    rebalance_threshold_pct: float
    max_notional_leverage: float
    max_risk_leverage: float
    correlation_penalty: float
    correlation_high_threshold: float
    factor_limits: dict[str, float]
    benchmark_symbol: str


@dataclass
class AssetRotationConfig:
    """Room for multi-group concurrency and gradual migration across Asset Groups."""

    enabled: bool = True
    multi_group_enabled: bool = True
    max_active_groups: int = 5
    max_symbols_concurrent: int = 8
    migration_enabled: bool = True
    min_group_weight: float = 0.05
    max_group_weight: float = 0.50
    migration_shift_threshold: float = 0.08
    strength_power: float = 1.0
    expand_scan_to_all_groups: bool = True
    state_key: str = "asset_rotation"


@dataclass
class ResilienceConfig:
    enabled: bool
    uncertain_prob_threshold: float
    clear_prob_threshold: float
    cautious_position_scale: float
    defensive_position_scale: float
    uncertain_position_scale: float
    uncertain_strategy: str
    max_switches_per_week: float
    min_regime_duration_days: float
    oscillation_lookback: int


@dataclass
class LLMResearchConfig:
    enabled: bool
    provider: str
    model: str
    temperature: float
    openai_api_key: str | None
    news_path: str
    news_file: str
    keyword_filter: bool
    max_news_per_run: int
    audit_enabled: bool
    generate_strategy_report: bool
    use_as_feature: bool
    sentiment_nudge_strength: float
    min_confidence: float


@dataclass
class MLConfig:
    enabled: bool
    model_type: str
    label_horizon_bars: int
    label_threshold: float
    lookback: int
    n_splits: int
    min_train_samples: int
    ic_threshold: float
    ir_threshold: float
    ic_degrade_ratio: float
    rolling_ic_window: int
    min_feature_ic: float
    max_features: int
    signal_probability_threshold: float
    random_forest_estimators: int
    random_forest_max_depth: int
    random_state: int


@dataclass
class HedgingConfig:
    enabled: bool
    dry_run_only: bool
    market_benchmark: str
    hedge_instrument: str
    hedge_instrument_beta: float
    retail_borrow_rate_annual: float
    trading_cost_annual_pct: float
    target_net_beta: float
    beta_neutral_tolerance: float
    min_beta_observations: int
    default_symbol_beta: float
    retail_viable: bool


@dataclass
class BacktestConfig:
    train_window: int
    test_window: int
    walk_forward_step: int
    min_walk_forward_rounds: int
    monte_carlo_simulations: int
    cost_perturbation_pct: float
    oos_train_ratio: float
    oos_val_ratio: float
    min_oos_ratio: float
    live_decay_factor: float
    hidden_cost_pct: float
    param_sensitivity_pct: float
    param_sensitivity_max_return_change: float
    strategies_tested: int
    min_mc_prob_positive: float
    round_trip_cost_pct: float | None
    gate_filter_enabled: bool
    gate_cache_path: str
    gate_cache_max_age_hours: float
    # When cache is missing/stale: True = rebuild (hours), False = allow-all (demo-friendly).
    gate_build_on_miss: bool = False


@dataclass
class LoopEngineeringConfig:
    enabled: bool
    min_trades_h1: int
    min_trades_m15: int
    min_wf_test_sharpe: float
    wf_sharpe_improvement: float
    max_mdd_pct: float
    mdd_worsen_tolerance_pct: float
    hard_stop_mdd_pct: float
    hard_stop_min_trades: int
    hard_stop_expected_live_pct: float
    hard_stop_mc_p5_pct: float
    hard_stop_oos_ratio: float
    overfit_is_sharpe: float
    overfit_wf_sharpe: float
    consecutive_hard_stop_limit: int
    tier_b_min_gate_passes: int
    tier_b_mc_prob_positive: float
    tier_b_mc_p5_pct: float
    tier_b_wf_positive_pct: float
    output_dir: str
    baseline_wf_sharpe_stop_delta: float
    stop_on_all_unstable: bool
    stop_on_all_strategies_gate_fail: bool


@dataclass
class AppConfig:
    broker_name: str
    account_type: str
    mt5: MT5Config
    symbols: list[str]
    symbol_groups: dict[str, list[str]]
    asset_groups: dict[str, AssetGroupConfig]
    default_timeframe: str
    history_bars: int
    timeframes: list[str]
    storage: DataStorageConfig
    data: DataConfig
    backtest: BacktestConfig
    hedging: HedgingConfig
    decision: DecisionConfig
    execution: ExecutionConfig
    multi_agent: MultiAgentConfig
    regime: RegimeConfig
    resilience: ResilienceConfig
    portfolio: PortfolioConfig
    asset_rotation: AssetRotationConfig
    online_learning: OnlineLearningConfig
    llm_research: LLMResearchConfig
    ml: MLConfig
    trading: TradingConfig
    costs: CostConfig
    risk: RiskConfig
    ops: OpsConfig
    project: ProjectConfig
    trade_log: TradeLogConfig
    stats: StatsConfig
    indicators: IndicatorsConfig
    strategies: StrategiesConfig
    loop_engineering: LoopEngineeringConfig
    intelligence: IntelligenceConfig
    log_level: str
    log_file: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def history_bars_for(self, timeframe: str | None = None) -> int:
        tf = (timeframe or self.default_timeframe).upper()
        return history_bars_for_timeframe(
            tf,
            history_years=self.data.history_years,
            history_bars=self.history_bars,
            history_bars_by_timeframe=self.data.history_bars_by_timeframe,
            trading_days_per_year=self.trading.trading_days_per_year,
        )

    def group_for_symbol(self, symbol: str) -> AssetGroupConfig | None:
        for group in self.asset_groups.values():
            if symbol in group.symbols:
                return group
        return None

    def lot_multiplier_for(self, symbol: str) -> float:
        group = self.group_for_symbol(symbol)
        if group is None:
            return 1.0
        return float(group.lot_multiplier)

    def pair_state_id(self, symbol_a: str, symbol_b: str) -> str:
        return f"pairs/{symbol_a}__{symbol_b}"

    def tradeable_symbols_all_groups(self) -> list[str]:
        """All tradeable symbols across every Asset Group (engineering universe)."""
        if not self.asset_groups:
            return list(self.symbols)
        ordered: list[str] = []
        seen: set[str] = set()
        for group in self.asset_groups.values():
            if not group.tradeable:
                continue
            for symbol in group.symbols:
                if symbol not in seen:
                    ordered.append(symbol)
                    seen.add(symbol)
        return ordered

    def pairs_all_groups(self) -> list[list[str]]:
        """Within-group pairs whose both legs are tradeable."""
        tradeable = set(self.tradeable_symbols_all_groups())
        out: list[list[str]] = []
        for pair in self.strategies.pairs:
            if len(pair) != 2:
                continue
            if pair[0] in tradeable and pair[1] in tradeable:
                out.append(list(pair))
        return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def _flatten_symbols(symbols_cfg: dict[str, Any] | list[str]) -> list[str]:
    if isinstance(symbols_cfg, list):
        return symbols_cfg

    ordered: list[str] = []
    for group in ("indices", "commodities", "forex"):
        ordered.extend(symbols_cfg.get(group, []))
    return ordered


def _symbol_groups(symbols_cfg: dict[str, Any] | list[str]) -> dict[str, list[str]]:
    if isinstance(symbols_cfg, list):
        return {"all": symbols_cfg}
    return {
        "indices": list(symbols_cfg.get("indices", [])),
        "commodities": list(symbols_cfg.get("commodities", [])),
        "forex": list(symbols_cfg.get("forex", [])),
    }


def _load_asset_groups(raw: dict[str, Any] | None) -> dict[str, AssetGroupConfig]:
    if not raw:
        return {}
    groups: dict[str, AssetGroupConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"asset_groups.{name} must be a mapping")
        symbols = [str(s) for s in cfg.get("symbols", [])]
        groups[str(name)] = AssetGroupConfig(
            name=str(name),
            symbols=symbols,
            strategy=str(cfg.get("strategy", "hybrid")),
            description=str(cfg.get("description", "")),
            tradeable=bool(cfg.get("tradeable", True)),
            lot_multiplier=float(cfg.get("lot_multiplier", 1.0)),
        )
    return groups


def _symbols_from_asset_groups(groups: dict[str, AssetGroupConfig]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups.values():
        for symbol in group.symbols:
            if symbol not in seen:
                ordered.append(symbol)
                seen.add(symbol)
    return ordered


def _validate_pairs_within_groups(
    pairs: list[list[str]],
    groups: dict[str, AssetGroupConfig],
) -> None:
    if not groups:
        return
    symbol_to_group = {
        symbol: group.name for group in groups.values() for symbol in group.symbols
    }
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Pair must have exactly 2 symbols: {pair}")
        left, right = pair[0], pair[1]
        g_left = symbol_to_group.get(left)
        g_right = symbol_to_group.get(right)
        if g_left is None or g_right is None:
            raise ValueError(f"Pair symbols must belong to asset_groups: {pair}")
        if g_left != g_right:
            raise ValueError(
                f"Cross-group pair forbidden: {pair} ({g_left} vs {g_right})"
            )


def _resolve_trading_config(trading_cfg: dict[str, Any]) -> TradingConfig:
    profile = str(trading_cfg.get("profile", "medium"))
    profile_defaults = TRADING_PROFILES.get(profile, TRADING_PROFILES["medium"])

    trades_override = trading_cfg.get("trades_per_day")
    trades_per_day = int(trades_override) if trades_override is not None else int(profile_defaults["trades_per_day"])

    timeframe_override = trading_cfg.get("primary_timeframe")
    primary_timeframe = (
        str(timeframe_override) if timeframe_override else str(profile_defaults["primary_timeframe"])
    )

    return TradingConfig(
        profile=profile,
        trades_per_day=trades_per_day,
        trading_days_per_year=int(trading_cfg.get("trading_days_per_year", 252)),
        default_lots=float(trading_cfg.get("default_lots", 0.1)),
        primary_timeframe=primary_timeframe,
        dry_run=bool(trading_cfg.get("dry_run", True)),
    )


def load_config(
    settings_path: Path | None = None,
    local_settings_path: Path | None = None,
) -> AppConfig:
    settings_path = settings_path or DEFAULT_SETTINGS_PATH
    local_settings_path = local_settings_path or LOCAL_SETTINGS_PATH

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    config_data = _load_yaml(settings_path)
    if local_settings_path.exists():
        config_data = _deep_merge(config_data, _load_yaml(local_settings_path))

    broker_cfg = config_data.get("broker", {})
    mt5_cfg = config_data.get("mt5", {})
    data_cfg = config_data.get("data", {})
    storage_cfg = data_cfg.get("storage", {})
    logging_cfg = config_data.get("logging", {})
    trading_cfg = config_data.get("trading", {})
    costs_cfg = config_data.get("costs", {})
    risk_cfg = config_data.get("risk", {})
    stats_cfg = config_data.get("stats", {})
    indicators_cfg = config_data.get("indicators", {})
    strategies_cfg = config_data.get("strategies", {})
    symbols_cfg = config_data.get("symbols", {})
    asset_groups = _load_asset_groups(config_data.get("asset_groups"))
    project_cfg = config_data.get("project", {})
    max_symbols = int(project_cfg.get("max_symbols", 12))
    if asset_groups:
        symbols = _symbols_from_asset_groups(asset_groups)
    else:
        symbols = _flatten_symbols(symbols_cfg)
    if len(symbols) > max_symbols:
        raise ValueError(
            f"Configured {len(symbols)} symbols exceeds project.max_symbols={max_symbols}"
        )
    pairs_raw = [list(p) for p in strategies_cfg.get("pairs", [])]
    _validate_pairs_within_groups(pairs_raw, asset_groups)
    trend_cfg = strategies_cfg.get("trend", {})
    mr_cfg = strategies_cfg.get("mean_reversion", {})
    grid_cfg = strategies_cfg.get("grid", {})
    pair_cfg = strategies_cfg.get("pairs_trading", {})
    backtest_cfg = config_data.get("backtest", {})
    hedging_cfg = config_data.get("hedging", {})
    decision_cfg = config_data.get("decision", {})
    execution_cfg = config_data.get("execution", {})
    ops_cfg = config_data.get("ops", {})
    trade_log_cfg = config_data.get("trade_log", {})
    multi_agent_cfg = config_data.get("multi_agent", {})
    regime_cfg = config_data.get("regime", {})
    resilience_cfg = config_data.get("resilience", {})
    portfolio_cfg = config_data.get("portfolio", {})
    rotation_cfg = config_data.get("asset_rotation", {})
    online_cfg = config_data.get("online_learning", {})
    llm_cfg = config_data.get("llm_research", {})
    ml_cfg = config_data.get("ml", {})
    loop_cfg = config_data.get("loop_engineering", {})
    intel_cfg = config_data.get("intelligence", {})
    intel_loop_cfg = intel_cfg.get("loop", {}) if isinstance(intel_cfg, dict) else {}

    default_timeframe = str(data_cfg.get("default_timeframe", "H1"))
    timeframes = data_cfg.get("timeframes") or [default_timeframe]
    fetch_cfg = data_cfg.get("fetch", {})
    quality_cfg = data_cfg.get("quality", {})

    data_config = DataConfig(
        default_timeframe=default_timeframe,
        history_bars=int(data_cfg.get("history_bars", 500)),
        timeframes=[str(tf) for tf in timeframes],
        storage=DataStorageConfig(
            type=str(storage_cfg.get("type", "sqlite")),
            path=str(storage_cfg.get("path", "data/mt5_loop.db")),
        ),
        fetch=DataFetchConfig(
            max_retries=int(fetch_cfg.get("max_retries", 5)),
            backoff_base=float(fetch_cfg.get("backoff_base", 2.0)),
            min_completeness_ratio=float(fetch_cfg.get("min_completeness_ratio", 0.9)),
            chunk_size=int(fetch_cfg.get("chunk_size", 5000)),
            range_chunk_days=int(fetch_cfg.get("range_chunk_days", 90)),
        ),
        quality=DataQualityConfig(
            price_jump_threshold=float(quality_cfg.get("price_jump_threshold", 0.20)),
            missing_rate_warn_pct=float(quality_cfg.get("missing_rate_warn_pct", 5.0)),
            gap_multiplier=float(quality_cfg.get("gap_multiplier", 1.5)),
            freshness_warn_seconds=int(quality_cfg.get("freshness_warn_seconds", 3600)),
            anomaly_rate_warn_pct=float(quality_cfg.get("anomaly_rate_warn_pct", 0.1)),
            api_error_rate_warn_pct=float(quality_cfg.get("api_error_rate_warn_pct", 5.0)),
        ),
        source=str(data_cfg.get("source", "mt5")),
        history_years=float(data_cfg["history_years"]) if data_cfg.get("history_years") else None,
        history_bars_by_timeframe={
            str(k).upper(): int(v)
            for k, v in (data_cfg.get("history_bars_by_timeframe") or {}).items()
        }
        or None,
    )

    trading_days_per_year = int(trading_cfg.get("trading_days_per_year", 252))
    resolved_history_bars = max_history_bars_for_timeframes(
        [str(tf) for tf in timeframes],
        history_years=data_config.history_years,
        history_bars=data_config.history_bars,
        history_bars_by_timeframe=data_config.history_bars_by_timeframe,
        trading_days_per_year=trading_days_per_year,
    )

    return AppConfig(
        broker_name=broker_cfg.get("name", "unknown"),
        account_type=broker_cfg.get("account_type", "demo"),
        mt5=MT5Config(
            path=mt5_cfg.get("path"),
            login=mt5_cfg.get("login"),
            password=mt5_cfg.get("password"),
            server=mt5_cfg.get("server"),
            timeout_ms=int(mt5_cfg.get("timeout_ms", 60000)),
            require_demo=bool(mt5_cfg.get("require_demo", True)),
            magic=int(mt5_cfg.get("magic", 260703)),
            deviation=int(mt5_cfg.get("deviation", 50)),
        ),
        symbols=symbols,
        symbol_groups=_symbol_groups(symbols_cfg),
        asset_groups=asset_groups,
        default_timeframe=default_timeframe,
        history_bars=resolved_history_bars,
        timeframes=[str(tf) for tf in timeframes],
        storage=DataStorageConfig(
            type=str(storage_cfg.get("type", "sqlite")),
            path=str(storage_cfg.get("path", "data/mt5_loop.db")),
        ),
        data=data_config,
        backtest=BacktestConfig(
            train_window=int(backtest_cfg.get("train_window", 200)),
            test_window=int(backtest_cfg.get("test_window", 50)),
            walk_forward_step=int(backtest_cfg.get("walk_forward_step", 25)),
            min_walk_forward_rounds=int(backtest_cfg.get("min_walk_forward_rounds", 3)),
            monte_carlo_simulations=int(backtest_cfg.get("monte_carlo_simulations", 1000)),
            cost_perturbation_pct=float(backtest_cfg.get("cost_perturbation_pct", 0.20)),
            oos_train_ratio=float(backtest_cfg.get("oos_train_ratio", 0.70)),
            oos_val_ratio=float(backtest_cfg.get("oos_val_ratio", 0.15)),
            min_oos_ratio=float(backtest_cfg.get("min_oos_ratio", 0.50)),
            live_decay_factor=float(backtest_cfg.get("live_decay_factor", 0.5)),
            hidden_cost_pct=float(backtest_cfg.get("hidden_cost_pct", 0.05)),
            param_sensitivity_pct=float(backtest_cfg.get("param_sensitivity_pct", 0.20)),
            param_sensitivity_max_return_change=float(
                backtest_cfg.get("param_sensitivity_max_return_change", 0.30)
            ),
            strategies_tested=int(backtest_cfg.get("strategies_tested", 3)),
            min_mc_prob_positive=float(backtest_cfg.get("min_mc_prob_positive", 0.50)),
            round_trip_cost_pct=(
                float(backtest_cfg["round_trip_cost_pct"])
                if backtest_cfg.get("round_trip_cost_pct") is not None
                else None
            ),
            gate_filter_enabled=bool(backtest_cfg.get("gate_filter_enabled", True)),
            gate_cache_path=str(backtest_cfg.get("gate_cache_path", "data/gate_registry.json")),
            gate_cache_max_age_hours=float(backtest_cfg.get("gate_cache_max_age_hours", 24.0)),
            gate_build_on_miss=bool(backtest_cfg.get("gate_build_on_miss", False)),
        ),
        hedging=HedgingConfig(
            enabled=bool(hedging_cfg.get("enabled", True)),
            dry_run_only=bool(hedging_cfg.get("dry_run_only", True)),
            market_benchmark=str(hedging_cfg.get("market_benchmark", "#USSPX500")),
            hedge_instrument=str(hedging_cfg.get("hedge_instrument", "#USSPX500")),
            hedge_instrument_beta=float(hedging_cfg.get("hedge_instrument_beta", 1.0)),
            retail_borrow_rate_annual=float(hedging_cfg.get("retail_borrow_rate_annual", 0.05)),
            trading_cost_annual_pct=float(hedging_cfg.get("trading_cost_annual_pct", 0.01)),
            target_net_beta=float(hedging_cfg.get("target_net_beta", 0.0)),
            beta_neutral_tolerance=float(hedging_cfg.get("beta_neutral_tolerance", 0.15)),
            min_beta_observations=int(hedging_cfg.get("min_beta_observations", 60)),
            default_symbol_beta=float(hedging_cfg.get("default_symbol_beta", 1.0)),
            retail_viable=bool(hedging_cfg.get("retail_viable", False)),
        ),
        decision=DecisionConfig(
            enabled=bool(decision_cfg.get("enabled", True)),
            win_rate=float(decision_cfg.get("win_rate", 0.55)),
            reward_risk_ratio=float(decision_cfg.get("reward_risk_ratio", 1.5)),
            risk_pct_per_trade=float(decision_cfg.get("risk_pct_per_trade", 0.01)),
            use_risk_parity=bool(decision_cfg.get("use_risk_parity", True)),
            min_confidence=float(decision_cfg.get("min_confidence", 0.1)),
            fallback_lots=(
                float(decision_cfg["fallback_lots"])
                if decision_cfg.get("fallback_lots") is not None
                else None
            ),
            trade_wins=int(decision_cfg["trade_wins"]) if decision_cfg.get("trade_wins") is not None else None,
            trade_losses=(
                int(decision_cfg["trade_losses"]) if decision_cfg.get("trade_losses") is not None else None
            ),
            avg_win_pct=(
                float(decision_cfg["avg_win_pct"]) if decision_cfg.get("avg_win_pct") is not None else None
            ),
            avg_loss_pct=(
                float(decision_cfg["avg_loss_pct"]) if decision_cfg.get("avg_loss_pct") is not None else None
            ),
        ),
        execution=ExecutionConfig(
            max_retries=int(execution_cfg.get("max_retries", 3)),
            retry_backoff_seconds=float(execution_cfg.get("retry_backoff_seconds", 1.0)),
            circuit_breaker_threshold=int(execution_cfg.get("circuit_breaker_threshold", 5)),
            enabled=bool(execution_cfg.get("enabled", True)),
            simulator_mode=str(execution_cfg.get("simulator_mode", "conservative")),
            latency_ms=float(execution_cfg.get("latency_ms", 200.0)),
            partial_fill_enabled=bool(execution_cfg.get("partial_fill_enabled", True)),
            max_child_orders=int(execution_cfg.get("max_child_orders", 5)),
            child_order_adv_fraction=float(execution_cfg.get("child_order_adv_fraction", 0.01)),
            quote_max_age_seconds=float(execution_cfg.get("quote_max_age_seconds", 30.0)),
            log_executions=bool(execution_cfg.get("log_executions", True)),
            telemetry_db_path=str(execution_cfg.get("telemetry_db_path", "data/execution_telemetry.sqlite")),
            slippage_conservative_multiplier=float(
                execution_cfg.get("slippage_conservative_multiplier", 1.5)
            ),
            use_bid_ask_not_close=bool(execution_cfg.get("use_bid_ask_not_close", True)),
            slippage_threshold_pct=float(execution_cfg.get("slippage_threshold_pct", 0.05)),
            latency_warn_ms=float(execution_cfg.get("latency_warn_ms", 500.0)),
        ),
        multi_agent=MultiAgentConfig(
            enabled=bool(multi_agent_cfg.get("enabled", True)),
            evolution_stage=int(multi_agent_cfg.get("evolution_stage", 5)),
            arbitration_mode=str(multi_agent_cfg.get("arbitration_mode", "veto")),
            parallel_analysis=bool(multi_agent_cfg.get("parallel_analysis", True)),
            agent_timeout_seconds=float(multi_agent_cfg.get("agent_timeout_seconds", 30)),
            circuit_breaker_threshold=int(multi_agent_cfg.get("circuit_breaker_threshold", 3)),
        ),
        regime=RegimeConfig(
            method=str(regime_cfg.get("method", "five_state")),
            crisis_vol_threshold=float(regime_cfg.get("crisis_vol_threshold", 0.30)),
            crisis_correlation_threshold=float(regime_cfg.get("crisis_correlation_threshold", 0.80)),
            ranging_vol_max=float(regime_cfg.get("ranging_vol_max", 0.15)),
            trending_adx_min=float(regime_cfg.get("trending_adx_min", 25.0)),
            trending_return_min=float(regime_cfg.get("trending_return_min", 0.05)),
            ranging_adx_max=float(regime_cfg.get("ranging_adx_max", 20.0)),
            confirm_days=int(regime_cfg.get("confirm_days", 3)),
            switch_cost_pct=float(regime_cfg.get("switch_cost_pct", 0.005)),
            crisis_weight_amplify=float(regime_cfg.get("crisis_weight_amplify", 1.5)),
            transition_strategy=str(regime_cfg.get("transition_strategy", "risk_first")),
            correlation_benchmark=str(regime_cfg.get("correlation_benchmark", "#USSPX500")),
            correlation_lookback=int(regime_cfg.get("correlation_lookback", 20)),
            gmm_min_samples=int(regime_cfg.get("gmm_min_samples", 60)),
            probability_threshold=float(regime_cfg.get("probability_threshold", 0.5)),
            adx_boundary_low=float(regime_cfg.get("adx_boundary_low", 22.0)),
            adx_boundary_high=float(regime_cfg.get("adx_boundary_high", 28.0)),
            post_switch_uncertain_days=int(regime_cfg.get("post_switch_uncertain_days", 2)),
            confirm_bars=int(regime_cfg.get("confirm_bars", regime_cfg.get("confirm_days", 3))),
            high_vol_trend_scale=float(regime_cfg.get("high_vol_trend_scale", 0.5)),
            trend_lookback=int(regime_cfg.get("trend_lookback", 40)),
            er_lookback=int(regime_cfg.get("er_lookback", 20)),
            vol_hist_window=int(regime_cfg.get("vol_hist_window", 120)),
            enter_trend_score=float(regime_cfg.get("enter_trend_score", 0.55)),
            enter_er_trend=float(regime_cfg.get("enter_er_trend", 0.45)),
            enter_vol_max_for_stable=float(regime_cfg.get("enter_vol_max_for_stable", 0.70)),
            exit_trend_score=float(regime_cfg.get("exit_trend_score", 0.35)),
            exit_er_trend=float(regime_cfg.get("exit_er_trend", 0.30)),
            high_vol_enter=float(regime_cfg.get("high_vol_enter", 0.70)),
            stress_vol=float(regime_cfg.get("stress_vol", 0.90)),
            enter_range_abs_trend=float(regime_cfg.get("enter_range_abs_trend", 0.30)),
            enter_er_range_max=float(regime_cfg.get("enter_er_range_max", 0.35)),
            exit_range_abs_trend=float(regime_cfg.get("exit_range_abs_trend", 0.45)),
            stress_corr=float(regime_cfg.get("stress_corr", 0.85)),
            stress_spread=float(regime_cfg.get("stress_spread", 0.70)),
            regime_conditioned_bt=bool(regime_cfg.get("regime_conditioned_bt", True)),
        ),
        resilience=ResilienceConfig(
            enabled=bool(resilience_cfg.get("enabled", True)),
            uncertain_prob_threshold=float(resilience_cfg.get("uncertain_prob_threshold", 0.50)),
            clear_prob_threshold=float(resilience_cfg.get("clear_prob_threshold", 0.70)),
            cautious_position_scale=float(resilience_cfg.get("cautious_position_scale", 0.70)),
            defensive_position_scale=float(resilience_cfg.get("defensive_position_scale", 0.50)),
            uncertain_position_scale=float(resilience_cfg.get("uncertain_position_scale", 0.50)),
            uncertain_strategy=str(resilience_cfg.get("uncertain_strategy", "worst_case")),
            max_switches_per_week=float(resilience_cfg.get("max_switches_per_week", 3.0)),
            min_regime_duration_days=float(resilience_cfg.get("min_regime_duration_days", 5.0)),
            oscillation_lookback=int(resilience_cfg.get("oscillation_lookback", 6)),
        ),
        portfolio=PortfolioConfig(
            enabled=bool(portfolio_cfg.get("enabled", True)),
            weight_method=str(portfolio_cfg.get("weight_method", "inverse_vol")),
            shrinkage_method=str(portfolio_cfg.get("shrinkage_method", "ledoit_wolf")),
            min_bars=int(portfolio_cfg.get("min_bars", 60)),
            max_single_weight=float(portfolio_cfg.get("max_single_weight", 0.25)),
            rebalance_threshold_pct=float(portfolio_cfg.get("rebalance_threshold_pct", 5.0)),
            max_notional_leverage=float(portfolio_cfg.get("max_notional_leverage", 3.0)),
            max_risk_leverage=float(portfolio_cfg.get("max_risk_leverage", 2.0)),
            correlation_penalty=float(portfolio_cfg.get("correlation_penalty", 0.5)),
            correlation_high_threshold=float(portfolio_cfg.get("correlation_high_threshold", 0.75)),
            factor_limits={
                str(k): float(v)
                for k, v in (portfolio_cfg.get("factor_limits") or {}).items()
            },
            benchmark_symbol=str(portfolio_cfg.get("benchmark_symbol", "#USSPX500")),
        ),
        asset_rotation=AssetRotationConfig(
            enabled=bool(rotation_cfg.get("enabled", True)),
            multi_group_enabled=bool(rotation_cfg.get("multi_group_enabled", True)),
            max_active_groups=int(rotation_cfg.get("max_active_groups", 5)),
            max_symbols_concurrent=int(rotation_cfg.get("max_symbols_concurrent", 8)),
            migration_enabled=bool(rotation_cfg.get("migration_enabled", True)),
            min_group_weight=float(rotation_cfg.get("min_group_weight", 0.05)),
            max_group_weight=float(rotation_cfg.get("max_group_weight", 0.50)),
            migration_shift_threshold=float(
                rotation_cfg.get("migration_shift_threshold", 0.08)
            ),
            strength_power=float(rotation_cfg.get("strength_power", 1.0)),
            expand_scan_to_all_groups=bool(
                rotation_cfg.get("expand_scan_to_all_groups", True)
            ),
            state_key=str(rotation_cfg.get("state_key", "asset_rotation")),
        ),
        online_learning=OnlineLearningConfig(
            enabled=bool(online_cfg.get("enabled", True)),
            decay_factor=float(online_cfg.get("decay_factor", 0.95)),
            learning_rate=float(online_cfg.get("learning_rate", 0.01)),
            min_update_interval_days=int(online_cfg.get("min_update_interval_days", 5)),
            max_update_interval_days=int(online_cfg.get("max_update_interval_days", 20)),
            psi_threshold=float(online_cfg.get("psi_threshold", 0.2)),
            performance_drop_threshold=float(online_cfg.get("performance_drop_threshold", 0.3)),
            ic_viability_threshold=float(online_cfg.get("ic_viability_threshold", 0.03)),
            default_monthly_ic_decay=float(online_cfg.get("default_monthly_ic_decay", 0.05)),
            dynamic_threshold_k=float(online_cfg.get("dynamic_threshold_k", 1.5)),
            signal_threshold_default=float(online_cfg.get("signal_threshold_default", 0.5)),
            apply_dynamic_threshold=bool(online_cfg.get("apply_dynamic_threshold", False)),
            accuracy_window=int(online_cfg.get("accuracy_window", 5)),
            accuracy_warning_threshold=float(online_cfg.get("accuracy_warning_threshold", 0.5)),
            min_total_bars=int(online_cfg.get("min_total_bars", 120)),
            lifecycle_incubation_sharpe_min=float(online_cfg.get("lifecycle_incubation_sharpe_min", 1.0)),
            lifecycle_maturity_sharpe_min=float(online_cfg.get("lifecycle_maturity_sharpe_min", 1.5)),
            lifecycle_decay_sharpe_max=float(online_cfg.get("lifecycle_decay_sharpe_max", 0.8)),
            correlation_crisis_threshold=float(online_cfg.get("correlation_crisis_threshold", 0.75)),
            sliding_window_days=int(online_cfg.get("sliding_window_days", 252)),
            retrain_interval_days=int(online_cfg.get("retrain_interval_days", 20)),
        ),
        llm_research=LLMResearchConfig(
            enabled=bool(llm_cfg.get("enabled", False)),
            provider=str(llm_cfg.get("provider", "mock")),
            model=str(llm_cfg.get("model", "gpt-4o-mini")),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            openai_api_key=llm_cfg.get("openai_api_key"),
            news_path=str(llm_cfg.get("news_path", "data/news")),
            news_file=str(llm_cfg.get("news_file", "sample_news.json")),
            keyword_filter=bool(llm_cfg.get("keyword_filter", True)),
            max_news_per_run=int(llm_cfg.get("max_news_per_run", 20)),
            audit_enabled=bool(llm_cfg.get("audit_enabled", True)),
            generate_strategy_report=bool(llm_cfg.get("generate_strategy_report", True)),
            use_as_feature=bool(llm_cfg.get("use_as_feature", False)),
            sentiment_nudge_strength=float(llm_cfg.get("sentiment_nudge_strength", 0.05)),
            min_confidence=float(llm_cfg.get("min_confidence", 0.5)),
        ),
        ml=MLConfig(
            enabled=bool(ml_cfg.get("enabled", False)),
            model_type=str(ml_cfg.get("model_type", "auto")),
            label_horizon_bars=int(ml_cfg.get("label_horizon_bars", 5)),
            label_threshold=float(ml_cfg.get("label_threshold", 0.001)),
            lookback=int(ml_cfg.get("lookback", 20)),
            n_splits=int(ml_cfg.get("n_splits", 5)),
            min_train_samples=int(ml_cfg.get("min_train_samples", 100)),
            ic_threshold=float(ml_cfg.get("ic_threshold", 0.03)),
            ir_threshold=float(ml_cfg.get("ir_threshold", 0.5)),
            ic_degrade_ratio=float(ml_cfg.get("ic_degrade_ratio", 0.5)),
            rolling_ic_window=int(ml_cfg.get("rolling_ic_window", 20)),
            min_feature_ic=float(ml_cfg.get("min_feature_ic", 0.01)),
            max_features=int(ml_cfg.get("max_features", 12)),
            signal_probability_threshold=float(ml_cfg.get("signal_probability_threshold", 0.55)),
            random_forest_estimators=int(ml_cfg.get("random_forest_estimators", 100)),
            random_forest_max_depth=int(ml_cfg.get("random_forest_max_depth", 5)),
            random_state=int(ml_cfg.get("random_state", 42)),
        ),
        trading=_resolve_trading_config(trading_cfg),
        costs=CostConfig(
            slippage_rate=float(costs_cfg.get("slippage_rate", 0.0003)),
            market_impact_Y=float(costs_cfg.get("market_impact_Y", 0.5)),
            use_spread_when_slippage_unknown=bool(
                costs_cfg.get("use_spread_when_slippage_unknown", True)
            ),
            enabled=bool(costs_cfg.get("enabled", True)),
            slippage_model=str(costs_cfg.get("slippage_model", "sqrt")),
            slippage_k_linear=float(costs_cfg.get("slippage_k_linear", 0.3)),
            slippage_k_sqrt=float(costs_cfg.get("slippage_k_sqrt", 1.0)),
            default_adv_notional=float(costs_cfg.get("default_adv_notional", 100_000_000.0)),
            signal_decay_halflife_minutes=float(
                costs_cfg.get("signal_decay_halflife_minutes", 120.0)
            ),
            execution_delay_minutes=float(costs_cfg.get("execution_delay_minutes", 5.0)),
            min_net_alpha_pct=float(costs_cfg.get("min_net_alpha_pct", 0.0)),
            max_order_adv_ratio=float(costs_cfg.get("max_order_adv_ratio", 0.05)),
            block_untradable=bool(costs_cfg.get("block_untradable", True)),
            opportunity_cost_enabled=bool(costs_cfg.get("opportunity_cost_enabled", True)),
        ),
        risk=RiskConfig(
            max_total_exposure_pct=float(risk_cfg.get("max_total_exposure_pct", 60.0)),
            max_single_position_pct=float(risk_cfg.get("max_single_position_pct", 10.0)),
            max_symbol_exposure_pct=float(risk_cfg.get("max_symbol_exposure_pct", 20.0)),
            max_sector_exposure_pct=float(risk_cfg.get("max_sector_exposure_pct", 30.0)),
            max_risk_per_trade_pct=float(risk_cfg.get("max_risk_per_trade_pct", 1.0)),
            stop_loss_pct=float(risk_cfg.get("stop_loss_pct", 2.0)),
            take_profit_pct=float(risk_cfg.get("take_profit_pct", 5.0)),
            trailing_stop_pct=float(risk_cfg.get("trailing_stop_pct", 3.0)),
            use_atr_stops=bool(risk_cfg.get("use_atr_stops", True)),
            vol_stop_multiplier=float(risk_cfg.get("vol_stop_multiplier", 2.5)),
            drawdown_warning_pct=float(risk_cfg.get("drawdown_warning_pct", 5.0)),
            drawdown_stop_pct=float(risk_cfg.get("drawdown_stop_pct", 10.0)),
            drawdown_circuit_pct=float(risk_cfg.get("drawdown_circuit_pct", 15.0)),
            drawdown_warning_scale=float(risk_cfg.get("drawdown_warning_scale", 0.7)),
            circuit_breaker_cooldown_days=int(risk_cfg.get("circuit_breaker_cooldown_days", 5)),
            circuit_breaker_target_exposure_pct=float(
                risk_cfg.get("circuit_breaker_target_exposure_pct", 30.0)
            ),
            audit_enabled=bool(risk_cfg.get("audit_enabled", True)),
            safe_mode_on_failure=bool(risk_cfg.get("safe_mode_on_failure", True)),
        ),
        ops=OpsConfig(
            enabled=bool(ops_cfg.get("enabled", True)),
            heartbeat_timeout_seconds=float(ops_cfg.get("heartbeat_timeout_seconds", 30.0)),
            network_latency_warn_ms=float(ops_cfg.get("network_latency_warn_ms", 200.0)),
            cpu_warn_pct=float(ops_cfg.get("cpu_warn_pct", 80.0)),
            memory_critical_pct=float(ops_cfg.get("memory_critical_pct", 90.0)),
            disk_free_warn_gb=float(ops_cfg.get("disk_free_warn_gb", 10.0)),
            data_disconnect_seconds=float(ops_cfg.get("data_disconnect_seconds", 60.0)),
            api_success_warn_pct=float(ops_cfg.get("api_success_warn_pct", 95.0)),
            agent_latency_warn_ms=float(ops_cfg.get("agent_latency_warn_ms", 5000.0)),
            task_queue_warn_count=int(ops_cfg.get("task_queue_warn_count", 100)),
            daily_drawdown_warn_pct=float(ops_cfg.get("daily_drawdown_warn_pct", 3.0)),
            weekly_drawdown_critical_pct=float(ops_cfg.get("weekly_drawdown_critical_pct", 5.0)),
            abnormal_trade_position_pct=float(ops_cfg.get("abnormal_trade_position_pct", 10.0)),
            trade_frequency_deviation_x=float(ops_cfg.get("trade_frequency_deviation_x", 3.0)),
            alert_suppress_seconds=float(ops_cfg.get("alert_suppress_seconds", 300.0)),
            quiet_hours_enabled=bool(ops_cfg.get("quiet_hours_enabled", True)),
            quiet_hours_start=int(ops_cfg.get("quiet_hours_start", 22)),
            quiet_hours_end=int(ops_cfg.get("quiet_hours_end", 7)),
            structured_log_dir=str(ops_cfg.get("structured_log_dir", "logs/structured")),
            model_registry_path=str(ops_cfg.get("model_registry_path", "data/model_registry.sqlite")),
            enforce_market_hours=bool(ops_cfg.get("enforce_market_hours", False)),
            allow_trading_in_dry_run=bool(ops_cfg.get("allow_trading_in_dry_run", True)),
            canary_initial_weight=float(ops_cfg.get("canary_initial_weight", 0.05)),
            rollback_max_error_rate=float(ops_cfg.get("rollback_max_error_rate", 0.05)),
            rollback_min_sharpe=float(ops_cfg.get("rollback_min_sharpe", 0.5)),
        ),
        project=ProjectConfig(
            name=str(project_cfg.get("name", "AI-Loop-Trade002")),
            max_symbols=int(project_cfg.get("max_symbols", 12)),
            primary_timeframe=str(project_cfg.get("primary_timeframe", "M30")),
            graduation_stage=str(project_cfg.get("graduation_stage", "paper")),
            min_backtest_sharpe=float(project_cfg.get("min_backtest_sharpe", 1.0)),
            max_backtest_drawdown_pct=float(project_cfg.get("max_backtest_drawdown_pct", 20.0)),
            sync_before_run=bool(project_cfg.get("sync_before_run", False)),
        ),
        trade_log=TradeLogConfig(
            enabled=bool(trade_log_cfg.get("enabled", True)),
            db_path=str(trade_log_cfg.get("db_path", "data/trade_log.sqlite")),
            agent_version=str(trade_log_cfg.get("agent_version", "mt5_loop_v1")),
        ),
        stats=StatsConfig(
            analysis_timeframe=str(stats_cfg.get("analysis_timeframe", "D1")),
            signal_timeframe=str(stats_cfg.get("signal_timeframe", "M30")),
            min_bars=int(stats_cfg.get("min_bars", 60)),
            vol_window=int(stats_cfg.get("vol_window", 20)),
            regime_vol_crisis=float(stats_cfg.get("regime_vol_crisis", 0.40)),
            regime_vol_bull_max=float(stats_cfg.get("regime_vol_bull_max", 0.20)),
            autocorr_lags=[int(x) for x in stats_cfg.get("autocorr_lags", [1, 5])],
            momentum_autocorr_threshold=float(stats_cfg.get("momentum_autocorr_threshold", 0.05)),
            mean_reversion_autocorr_threshold=float(
                stats_cfg.get("mean_reversion_autocorr_threshold", -0.05)
            ),
        ),
        indicators=IndicatorsConfig(
            macd_fast=int(indicators_cfg.get("macd_fast", 12)),
            macd_slow=int(indicators_cfg.get("macd_slow", 26)),
            macd_signal=int(indicators_cfg.get("macd_signal", 9)),
            macd_histogram_double=bool(indicators_cfg.get("macd_histogram_double", False)),
            rsi_period=int(indicators_cfg.get("rsi_period", 14)),
            bb_period=int(indicators_cfg.get("bb_period", 20)),
            bb_std=float(indicators_cfg.get("bb_std", 2.0)),
            atr_period=int(indicators_cfg.get("atr_period", 14)),
            atr_stop_multiplier=float(indicators_cfg.get("atr_stop_multiplier", 2.0)),
            divergence_lookback=int(indicators_cfg.get("divergence_lookback", 30)),
            signal_score_threshold=float(indicators_cfg.get("signal_score_threshold", 0.15)),
            risk_free_rate=float(indicators_cfg.get("risk_free_rate", 0.04)),
        ),
        strategies=StrategiesConfig(
            allocation=str(strategies_cfg.get("allocation", "dynamic")),
            uncertain_position_scale=float(strategies_cfg.get("uncertain_position_scale", 0.5)),
            pairs=pairs_raw,
            pair_priority_strength=float(strategies_cfg.get("pair_priority_strength", 0.55)),
            trend_ma_short=int(trend_cfg.get("ma_short", 5)),
            trend_ma_long=int(trend_cfg.get("ma_long", 20)),
            adx_period=int(trend_cfg.get("adx_period", 14)),
            adx_trend_threshold=float(trend_cfg.get("adx_trend_threshold", 25)),
            adx_sideways_threshold=float(trend_cfg.get("adx_sideways_threshold", 20)),
            adx_confirm_days=int(trend_cfg.get("adx_confirm_days", 5)),
            ma_side_days=int(trend_cfg.get("ma_side_days", 10)),
            mr_rsi_oversold=float(mr_cfg.get("rsi_oversold", 30)),
            mr_rsi_overbought=float(mr_cfg.get("rsi_overbought", 70)),
            mr_bb_entry_low=float(mr_cfg.get("bb_entry_low", 0.15)),
            mr_bb_entry_high=float(mr_cfg.get("bb_entry_high", 0.85)),
            grid_dry_run_only=bool(grid_cfg.get("dry_run_only", True)),
            grid_step_pct=float(grid_cfg.get("grid_step_pct", 0.04)),
            grid_num_grids=int(grid_cfg.get("num_grids", 5)),
            grid_max_loss_pct=float(grid_cfg.get("max_loss_pct", 5.0)),
            pair_zscore_entry=float(pair_cfg.get("zscore_entry", 2.0)),
            pair_zscore_exit=float(pair_cfg.get("zscore_exit", 0.5)),
            pair_lookback=int(pair_cfg.get("lookback", 20)),
            pair_beta_window=int(pair_cfg.get("beta_window", 60)),
            pair_beta_short_window=int(pair_cfg.get("beta_short_window", 40)),
            pair_beta_long_window=int(pair_cfg.get("beta_long_window", 120)),
            pair_max_half_life_bars=float(pair_cfg.get("max_half_life_bars", 48.0)),
            pair_weaken_half_life_mult=float(pair_cfg.get("weaken_half_life_mult", 2.0)),
            pair_max_beta_drift=float(pair_cfg.get("max_beta_drift", 0.35)),
            pair_break_beta_drift=float(pair_cfg.get("break_beta_drift", 0.60)),
            pair_max_abs_trend_slope=float(pair_cfg.get("max_abs_trend_slope", 0.002)),
            pair_break_abs_trend_slope=float(pair_cfg.get("break_abs_trend_slope", 0.005)),
            pair_min_zero_cross_rate=float(pair_cfg.get("min_zero_cross_rate", 0.05)),
            pair_vol_high_mult=float(pair_cfg.get("vol_high_mult", 1.5)),
        ),
        loop_engineering=LoopEngineeringConfig(
            enabled=bool(loop_cfg.get("enabled", True)),
            min_trades_h1=int(loop_cfg.get("min_trades_h1", 30)),
            min_trades_m15=int(loop_cfg.get("min_trades_m15", 50)),
            min_wf_test_sharpe=float(loop_cfg.get("min_wf_test_sharpe", 0.5)),
            wf_sharpe_improvement=float(loop_cfg.get("wf_sharpe_improvement", 0.1)),
            max_mdd_pct=float(loop_cfg.get("max_mdd_pct", 20.0)),
            mdd_worsen_tolerance_pct=float(loop_cfg.get("mdd_worsen_tolerance_pct", 2.0)),
            hard_stop_mdd_pct=float(loop_cfg.get("hard_stop_mdd_pct", 30.0)),
            hard_stop_min_trades=int(loop_cfg.get("hard_stop_min_trades", 20)),
            hard_stop_expected_live_pct=float(loop_cfg.get("hard_stop_expected_live_pct", -5.0)),
            hard_stop_mc_p5_pct=float(loop_cfg.get("hard_stop_mc_p5_pct", -20.0)),
            hard_stop_oos_ratio=float(loop_cfg.get("hard_stop_oos_ratio", 0.30)),
            overfit_is_sharpe=float(loop_cfg.get("overfit_is_sharpe", 3.0)),
            overfit_wf_sharpe=float(loop_cfg.get("overfit_wf_sharpe", 0.3)),
            consecutive_hard_stop_limit=int(loop_cfg.get("consecutive_hard_stop_limit", 5)),
            tier_b_min_gate_passes=int(loop_cfg.get("tier_b_min_gate_passes", 10)),
            tier_b_mc_prob_positive=float(loop_cfg.get("tier_b_mc_prob_positive", 0.50)),
            tier_b_mc_p5_pct=float(loop_cfg.get("tier_b_mc_p5_pct", -10.0)),
            tier_b_wf_positive_pct=float(loop_cfg.get("tier_b_wf_positive_pct", 0.50)),
            output_dir=str(loop_cfg.get("output_dir", "data/loop_results")),
            baseline_wf_sharpe_stop_delta=float(loop_cfg.get("baseline_wf_sharpe_stop_delta", 0.2)),
            stop_on_all_unstable=bool(loop_cfg.get("stop_on_all_unstable", True)),
            stop_on_all_strategies_gate_fail=bool(
                loop_cfg.get("stop_on_all_strategies_gate_fail", True)
            ),
        ),
        intelligence=IntelligenceConfig(
            enabled=bool(intel_cfg.get("enabled", True)),
            state_dir=str(intel_cfg.get("state_dir", "state")),
            maker_model=str(intel_cfg.get("maker_model", "claude-sonnet-4-5")),
            checker_model=str(intel_cfg.get("checker_model", "claude-opus-4-8")),
            maker_candidates=int(intel_cfg.get("maker_candidates", 6)),
            max_retries=int(intel_cfg.get("max_retries", 5)),
            enable_prompt_cache=bool(intel_cfg.get("enable_prompt_cache", True)),
            default_strategy=str(intel_cfg.get("default_strategy", "feature_score")),
            loop=ResidentLoopConfig(
                poll_seconds=int(intel_loop_cfg.get("poll_seconds", 30)),
                review_weekday=int(intel_loop_cfg.get("review_weekday", 5)),
                review_hour_utc=int(intel_loop_cfg.get("review_hour_utc", 6)),
                sharpe_degrade_trigger=float(
                    intel_loop_cfg.get("sharpe_degrade_trigger", 0.20)
                ),
                run_pipeline_on_bar=bool(intel_loop_cfg.get("run_pipeline_on_bar", True)),
                sync_on_poll=bool(intel_loop_cfg.get("sync_on_poll", True)),
                pretrade_optimize=bool(intel_loop_cfg.get("pretrade_optimize", True)),
                require_adopted_params=bool(
                    intel_loop_cfg.get("require_adopted_params", True)
                ),
                optimize_pairs=bool(intel_loop_cfg.get("optimize_pairs", True)),
                check_all_asset_groups=bool(
                    intel_loop_cfg.get("check_all_asset_groups", True)
                ),
                no_api_seed_baseline=bool(
                    intel_loop_cfg.get("no_api_seed_baseline", True)
                ),
                seed_baseline_if_no_adopt=bool(
                    intel_loop_cfg.get("seed_baseline_if_no_adopt", True)
                ),
            ),
        ),
        log_level=str(logging_cfg.get("level", "INFO")),
        log_file=str(logging_cfg.get("file", "logs/mt5_loop.log")),
        raw=config_data,
    )
