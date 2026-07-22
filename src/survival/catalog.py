from __future__ import annotations

from src.survival.types import DeathModeInfo

# Diagnostic order from Appendix B comprehensive table (priority 1-12).
DIAGNOSTIC_ORDER: tuple[int, ...] = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2)

DEATH_MODES: tuple[DeathModeInfo, ...] = (
    DeathModeInfo(
        mode_id=1,
        name="data_pollution",
        name_ja="データ汚染死",
        definition="System uses wrong, missing, or contaminated data.",
        symptoms=(
            "Great backtest but sudden live losses",
            "Strategy behaves abnormally from a specific date",
            "Signals invert vs market",
        ),
        prevention=(
            "Data quality pipeline (outliers, gaps, jumps)",
            "Multi-source cross-validation",
            "Realtime vs historical consistency checks",
            "Data change alerts",
        ),
        source_module="src/data/quality.py",
    ),
    DeathModeInfo(
        mode_id=2,
        name="overfitting",
        name_ja="過学習死",
        definition="Strategy memorizes noise instead of signal.",
        symptoms=(
            "Backtest Sharpe > 3, live Sharpe < 0.5",
            "Very different results across past periods",
            "Small parameter tweaks change returns dramatically",
        ),
        prevention=(
            "Strict out-of-sample testing",
            "Limit features/parameters",
            "Cross-validation + walk-forward",
            "Skepticism toward perfect backtests",
        ),
        source_module="src/backtest/overfitting.py",
    ),
    DeathModeInfo(
        mode_id=3,
        name="regime_drift",
        name_ja="レジームドリフト死",
        definition="Market regime shifts but strategy still assumes the old regime.",
        symptoms=(
            "Previously working strategy suddenly fails",
            "Loss streak exceeds historical max drawdown",
            "Signal direction consistently wrong vs trend",
        ),
        prevention=(
            "Regime detection module",
            "Rolling correlation vs benchmark",
            "Multi-strategy diversification",
            "Periodic strategy assumption review",
        ),
        source_module="src/regime/detection.py",
    ),
    DeathModeInfo(
        mode_id=4,
        name="execution_distortion",
        name_ja="執行歪曲死",
        definition="Systematic gap between backtest assumptions and live execution.",
        symptoms=(
            "Live return far below backtest",
            "Slippage exceeds expectations",
            "Orders frequently unfilled",
        ),
        prevention=(
            "Conservative cost assumptions",
            "Tick-level backtests when possible",
            "Calibrate with small live samples",
            "Separate predictable vs capturable alpha",
        ),
        source_module="src/execution/simulator.py",
    ),
    DeathModeInfo(
        mode_id=5,
        name="risk_management_failure",
        name_ja="リスク管理失敗死",
        definition="Risk rules have loopholes or fail to execute.",
        symptoms=(
            "Single trade loss exceeds threshold",
            "Drawdown triggers circuit breaker but system ignores it",
            "Stop orders not executed",
        ),
        prevention=(
            "Risk independent from strategy",
            "Multi-layer risk (position, portfolio, system)",
            "No bypass for anyone",
            "Periodic circuit-breaker drills",
        ),
        source_module="src/risk/drawdown.py",
    ),
    DeathModeInfo(
        mode_id=6,
        name="liquidity_evaporation",
        name_ja="流動性蒸発死",
        definition="Cannot exit because market liquidity vanished.",
        symptoms=(
            "Stop orders cannot fill",
            "Slippage far above normal",
            "Forced liquidation at bad prices",
        ),
        prevention=(
            "Avoid concentrated single-asset positions",
            "Monitor order book depth not just price",
            "Liquidity stress tests",
            "Maintain cash buffer for margin calls",
        ),
        source_module="src/execution/costs/tradability.py",
    ),
    DeathModeInfo(
        mode_id=7,
        name="correlation_spike",
        name_ja="相関スパイク死",
        definition="Crisis correlations approach 1 and diversification fails.",
        symptoms=(
            "Diversified portfolio drops together",
            "All strategies lose simultaneously",
            "Hedges fail",
        ),
        prevention=(
            "Stress test with crisis correlations",
            "Hold truly uncorrelated assets",
            "Cut exposure on crisis warnings",
            "Size leverage for crisis scenarios",
        ),
        source_module="src/portfolio/covariance.py",
    ),
    DeathModeInfo(
        mode_id=8,
        name="leverage_collapse",
        name_ja="レバレッジ破綻死",
        definition="Excessive leverage wipes capital on one adverse move.",
        symptoms=(
            "Cannot meet margin call",
            "Forced liquidation",
            "Principal loss exceeds 50%",
        ),
        prevention=(
            "Leverage limits (recommended < 2x)",
            "Volatility-adjusted leverage",
            "50% margin buffer",
            "Stress test 2x volatility",
        ),
        source_module="src/portfolio/leverage.py",
    ),
    DeathModeInfo(
        mode_id=9,
        name="human_intervention",
        name_ja="人間介入死",
        definition="Manual overrides of system decisions cause larger losses.",
        symptoms=(
            "Manually cancel stop orders",
            "Add to losing positions to average down",
            "Override signals on intuition",
        ),
        prevention=(
            "Strict operating procedures",
            "Written justification for manual actions",
            "Post-review all manual interventions",
            "Ban intervention if win rate < 50%",
        ),
        source_module="src/extensions/human_intervention.py",
    ),
    DeathModeInfo(
        mode_id=10,
        name="system_failure",
        name_ja="システム障害死",
        definition="Technical failures prevent normal operation.",
        symptoms=(
            "Order submission fails",
            "Market data feed interrupted",
            "System crash or latency spikes",
        ),
        prevention=(
            "High-availability design",
            "Realtime health monitoring and alerts",
            "Safe mode on failure",
            "Disaster recovery drills",
        ),
        source_module="src/ops/monitoring.py",
    ),
    DeathModeInfo(
        mode_id=11,
        name="regulatory_change",
        name_ja="規制変更死",
        definition="Legal changes make strategy impossible or illegal.",
        symptoms=(
            "Trade type banned",
            "Tax policy changes profit math",
            "Margin requirements increase",
        ),
        prevention=(
            "Diversify strategy types and regions",
            "Monitor regulatory trends",
            "Avoid single-policy dependency",
            "Buffer period for regulatory change",
        ),
        source_module="src/extensions/regulatory.py",
    ),
    DeathModeInfo(
        mode_id=12,
        name="counterparty_adaptation",
        name_ja="カウンターパーティ適応死",
        definition="Market participants learn and trade against your strategy.",
        symptoms=(
            "Returns gradually decay",
            "Someone always trades ahead of you",
            "Previously valid signals stop working",
        ),
        prevention=(
            "Diversify signal sources and timing",
            "Avoid fixed-pattern trading",
            "Monitor capacity and marginal returns",
            "Continuously develop replacement strategies",
            "Protect strategy details",
        ),
        source_module="src/online/alpha_decay.py",
    ),
)

DEATH_MODE_BY_ID: dict[int, DeathModeInfo] = {m.mode_id: m for m in DEATH_MODES}
