from __future__ import annotations

import sqlite3
from pathlib import Path

from src.core.config import PROJECT_ROOT
from src.trading_log.types import (
    AgentDecisionMeta,
    FillLogRecord,
    MarketSnapshot,
    OrderLogRecord,
    TradeLogBundle,
    TradeLogSummary,
    TradeMetrics,
)


class LiveTradeLogStore:
    """Appendix A: order -> fill -> metrics closed-loop persistence."""

    def __init__(self, db_path: str | Path) -> None:
        raw = str(db_path)
        self._memory_mode = raw == ":memory:"
        if self._memory_mode:
            self.db_path = raw
            self._conn = sqlite3.connect(raw, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        else:
            path = Path(db_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)
            self._conn = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                order_price REAL NOT NULL,
                order_qty REAL NOT NULL,
                submit_ts_ms INTEGER NOT NULL,
                cancel_ts_ms INTEGER,
                status TEXT NOT NULL,
                dry_run INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                fill_price REAL NOT NULL,
                fill_qty REAL NOT NULL,
                fill_ts_ms INTEGER NOT NULL,
                slippage_pct REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            );
            CREATE TABLE IF NOT EXISTS trade_metrics (
                order_id TEXT PRIMARY KEY,
                expected_price REAL NOT NULL,
                average_fill_price REAL NOT NULL,
                slippage_pct REAL NOT NULL,
                latency_ms REAL NOT NULL,
                fill_ratio REAL NOT NULL,
                commission REAL NOT NULL,
                tax REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            );
            CREATE TABLE IF NOT EXISTS market_snapshots (
                order_id TEXT PRIMARY KEY,
                bar_ts INTEGER NOT NULL,
                bar_open REAL NOT NULL,
                bar_high REAL NOT NULL,
                bar_low REAL NOT NULL,
                bar_close REAL NOT NULL,
                bar_vwap REAL NOT NULL,
                atr_5min REAL NOT NULL,
                bid1 REAL NOT NULL,
                ask1 REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            );
            CREATE TABLE IF NOT EXISTS agent_decisions (
                order_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                agent_version TEXT NOT NULL,
                action TEXT NOT NULL,
                target_position REAL NOT NULL,
                confidence REAL NOT NULL,
                signal_strength REAL NOT NULL,
                predicted_return REAL,
                strategy TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            );
            CREATE INDEX IF NOT EXISTS idx_orders_submit ON orders(submit_ts_ms);
            CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
            """
        )
        conn.commit()

    def record_bundle(self, bundle: TradeLogBundle) -> str:
        conn = self._connect()
        order = bundle.order
        conn.execute(
            """
            INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.trace_id,
                order.symbol,
                order.side,
                order.order_type,
                order.order_price,
                order.order_qty,
                order.submit_ts_ms,
                order.cancel_ts_ms,
                order.status,
                1 if order.dry_run else 0,
            ),
        )
        for fill in bundle.fills:
            conn.execute(
                """
                INSERT OR REPLACE INTO fills VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.fill_id,
                    fill.order_id,
                    fill.fill_price,
                    fill.fill_qty,
                    fill.fill_ts_ms,
                    fill.slippage_pct,
                ),
            )
        if bundle.metrics:
            m = bundle.metrics
            conn.execute(
                """
                INSERT OR REPLACE INTO trade_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m.order_id,
                    m.expected_price,
                    m.average_fill_price,
                    m.slippage_pct,
                    m.latency_ms,
                    m.fill_ratio,
                    m.commission,
                    m.tax,
                    m.realized_pnl,
                ),
            )
        if bundle.market:
            s = bundle.market
            conn.execute(
                """
                INSERT OR REPLACE INTO market_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s.order_id,
                    s.bar_ts,
                    s.bar_open,
                    s.bar_high,
                    s.bar_low,
                    s.bar_close,
                    s.bar_vwap,
                    s.atr_5min,
                    s.bid1,
                    s.ask1,
                ),
            )
        if bundle.agent:
            a = bundle.agent
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.order_id,
                    a.trace_id,
                    a.agent_id,
                    a.agent_version,
                    a.action,
                    a.target_position,
                    a.confidence,
                    a.signal_strength,
                    a.predicted_return,
                    a.strategy,
                ),
            )
        conn.commit()
        return order.order_id

    def recent_orders(self, limit: int = 50) -> list[OrderLogRecord]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM orders ORDER BY submit_ts_ms DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            OrderLogRecord(
                order_id=row["order_id"],
                trace_id=row["trace_id"],
                symbol=row["symbol"],
                side=row["side"],
                order_type=row["order_type"],
                order_price=row["order_price"],
                order_qty=row["order_qty"],
                submit_ts_ms=row["submit_ts_ms"],
                cancel_ts_ms=row["cancel_ts_ms"],
                status=row["status"],
                dry_run=bool(row["dry_run"]),
            )
            for row in rows
        ]

    def fills_for_order(self, order_id: str) -> list[FillLogRecord]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM fills WHERE order_id = ? ORDER BY fill_ts_ms",
            (order_id,),
        ).fetchall()
        return [
            FillLogRecord(
                fill_id=row["fill_id"],
                order_id=row["order_id"],
                fill_price=row["fill_price"],
                fill_qty=row["fill_qty"],
                fill_ts_ms=row["fill_ts_ms"],
                slippage_pct=row["slippage_pct"],
            )
            for row in rows
        ]

    def summarize(self, limit: int = 100) -> TradeLogSummary:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT m.* FROM trade_metrics m
            JOIN orders o ON o.order_id = m.order_id
            ORDER BY o.submit_ts_ms DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        if not rows:
            return TradeLogSummary()

        fill_count = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        return TradeLogSummary(
            order_count=len(rows),
            fill_count=int(fill_count),
            avg_slippage_pct=sum(r["slippage_pct"] for r in rows) / len(rows),
            avg_fill_ratio=sum(r["fill_ratio"] for r in rows) / len(rows),
            avg_latency_ms=sum(r["latency_ms"] for r in rows) / len(rows),
            total_commission=sum(r["commission"] for r in rows),
        )
