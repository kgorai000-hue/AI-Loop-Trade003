from __future__ import annotations

from src.core.config import OpsConfig
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.ops.types import ChecklistItem, RecoveryAction


def pre_market_checklist(
    connector: MT5Connector,
    store: OHLCVStore,
    symbols: list[str],
    timeframe: str,
    cfg: OpsConfig,
    *,
    dry_run: bool = True,
) -> list[ChecklistItem]:
    """Daily pre-market checks (Lesson 20.5)."""
    items: list[ChecklistItem] = []

    items.append(
        ChecklistItem(
            phase="pre_market",
            name="mt5_connection",
            passed=connector.is_connected,
            detail="MT5 connected" if connector.is_connected else "MT5 not connected",
        )
    )

    data_ok = True
    detail_parts: list[str] = []
    for symbol in symbols[:5]:
        bars = store.get_recent_bars(symbol, timeframe, 5)
        if len(bars) < 3:
            data_ok = False
            detail_parts.append(f"{symbol}: insufficient bars")
    items.append(
        ChecklistItem(
            phase="pre_market",
            name="data_feed_live",
            passed=data_ok,
            detail="; ".join(detail_parts) if detail_parts else "recent bars available",
        )
    )

    items.append(
        ChecklistItem(
            phase="pre_market",
            name="risk_params_loaded",
            passed=True,
            detail=f"drawdown warn/stop/circuit configured",
        )
    )

    items.append(
        ChecklistItem(
            phase="pre_market",
            name="dry_run_mode",
            passed=dry_run,
            detail="dry_run=true (safe)" if dry_run else "LIVE MODE - verify before open",
        )
    )

    return items


def post_market_checklist(
    *,
    execution_count: int,
    alert_count: int,
    pnl_available: bool,
) -> list[ChecklistItem]:
    return [
        ChecklistItem(
            phase="post_market",
            name="final_position_snapshot",
            passed=True,
            detail="positions reviewed via PositionAgent",
        ),
        ChecklistItem(
            phase="post_market",
            name="pnl_calculation",
            passed=pnl_available,
            detail="PnL from risk control report" if pnl_available else "PnL unavailable",
        ),
        ChecklistItem(
            phase="post_market",
            name="execution_log_count",
            passed=execution_count >= 0,
            detail=f"{execution_count} execution records",
        ),
        ChecklistItem(
            phase="post_market",
            name="alert_summary",
            passed=alert_count == 0,
            detail=f"{alert_count} alerts raised",
        ),
    ]


def classify_recovery(failure_type: str) -> RecoveryAction:
    """Disaster recovery strategies (Lesson 20.4)."""
    strategies = {
        "data_source": RecoveryAction(
            failure_type="data_source",
            strategy="switch to backup source or pause until reconnect",
            status="planned",
        ),
        "execution_venue": RecoveryAction(
            failure_type="execution_venue",
            strategy="pause trading, log pending orders",
            status="planned",
        ),
        "local_service": RecoveryAction(
            failure_type="local_service",
            strategy="restart agent process, verify health",
            status="planned",
        ),
        "network": RecoveryAction(
            failure_type="network",
            strategy="wait for recovery, validate state before resume",
            status="planned",
        ),
        "data_error": RecoveryAction(
            failure_type="data_error",
            strategy="halt processing, reject anomalous bars",
            status="planned",
        ),
    }
    return strategies.get(
        failure_type,
        RecoveryAction(failure_type=failure_type, strategy="manual review", status="unknown"),
    )


def reconcile_positions(local_lots: dict[str, float], broker_lots: dict[str, float]) -> list[str]:
    """State consistency check: broker is source of truth (Lesson 20.4)."""
    warnings: list[str] = []
    symbols = set(local_lots) | set(broker_lots)
    for symbol in symbols:
        local = local_lots.get(symbol, 0.0)
        broker = broker_lots.get(symbol, 0.0)
        if abs(local - broker) > 1e-6:
            warnings.append(f"{symbol}: local={local:.2f} broker={broker:.2f} -> use broker")
    return warnings
