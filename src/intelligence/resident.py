"""Resident M30 loop: pretrade optimize, demo-live pipeline, weekend review."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.core.config import PROJECT_ROOT, AppConfig
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.intelligence.loop import IntelligenceLoop, apply_state_overrides
from src.intelligence.params import LoopParams, params_from_config
from src.intelligence.persistence import StateStore
from src.intelligence.process_lock import ProcessLock
from src.intelligence.validator import ParamValidator
from src.system.runner import TradingSystem

logger = logging.getLogger(__name__)


class ResidentLoopEngine:
    """
    Multi-symbol (up to 12) resident loop for Trade003 demo-live.

    Startup:
      - optional mandatory optimize for each symbol (+ within-group pairs)
      - refuse pipeline until adopted params exist when require_adopted_params
    Each poll cycle:
      1) ensure MT5 connection (demo-only)
      2) optional sync of M30 bars
      3) for each symbol with a new closed bar → demo-live pipeline
      4) on configured weekday/hour → review (degraded metrics → optimize)
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        symbols: list[str] | None = None,
        strategy: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.config = config
        self.symbols = list(symbols or config.symbols)
        if len(self.symbols) > config.project.max_symbols:
            raise ValueError(
                f"{len(self.symbols)} symbols exceeds max_symbols={config.project.max_symbols}"
            )
        self.strategy = strategy or config.intelligence.default_strategy
        self.timeframe = (
            timeframe or config.trading.primary_timeframe or "M30"
        ).upper()

        loop_cfg = config.intelligence.loop
        self.poll_seconds = int(loop_cfg.poll_seconds)
        self.review_weekday = int(loop_cfg.review_weekday)
        self.review_hour_utc = int(loop_cfg.review_hour_utc)
        self.sharpe_degrade_trigger = float(loop_cfg.sharpe_degrade_trigger)
        self.run_pipeline_on_bar = bool(loop_cfg.run_pipeline_on_bar)
        self.sync_on_poll = bool(loop_cfg.sync_on_poll)
        self.pretrade_optimize = bool(loop_cfg.pretrade_optimize)
        self.require_adopted_params = bool(loop_cfg.require_adopted_params)
        self.optimize_pairs = bool(loop_cfg.optimize_pairs)

        self.state_dir = config.intelligence.state_dir
        self.stores = {sym: StateStore(self.state_dir, sym) for sym in self.symbols}
        self.pair_ids = [
            config.pair_state_id(pair[0], pair[1])
            for pair in config.strategies.pairs
            if len(pair) == 2
        ]
        self.pair_stores = {
            pair_id: StateStore(self.state_dir, pair_id) for pair_id in self.pair_ids
        }
        self._last_review_date: Optional[str] = self._load_last_review_date()
        self._pretrade_ready = False

        self.connector = MT5Connector(config)
        self.ohlcv = OHLCVStore(config.storage.path)
        self._lock = ProcessLock.for_project(PROJECT_ROOT, self.state_dir)

    def start(self) -> None:
        self._lock.acquire_or_exit()
        self.connector.connect()
        logger.info(
            "ResidentLoopEngine started symbols=%s pairs=%s tf=%s poll=%ss dry_run=%s stage=%s",
            self.symbols,
            self.pair_ids,
            self.timeframe,
            self.poll_seconds,
            self.config.trading.dry_run,
            self.config.project.graduation_stage,
        )
        if self.pretrade_optimize:
            self.run_pretrade_optimize()
        self._pretrade_ready = self._has_adopted_params()
        if self.require_adopted_params and not self._pretrade_ready:
            logger.warning(
                "No adopted params yet; pipeline will wait until optimize succeeds"
            )

    def stop(self) -> None:
        try:
            self.connector.disconnect()
        finally:
            self._lock.release()

    def _load_last_review_date(self) -> Optional[str]:
        dates: list[str] = []
        for store in list(self.stores.values()) + list(self.pair_stores.values()):
            raw = store.read_state().get("last_review_date")
            if isinstance(raw, str) and raw:
                dates.append(raw)
        return max(dates) if dates else None

    def _persist_last_review_date(self, day_key: str) -> None:
        self._last_review_date = day_key
        for store in list(self.stores.values()) + list(self.pair_stores.values()):
            store.update_state(last_review_date=day_key)

    def _has_adopted_params(self) -> bool:
        for store in self.stores.values():
            params = store.get_params().overrides
            if params:
                return True
        return False

    def run_pretrade_optimize(self) -> list[dict[str, Any]]:
        """Mandatory optimize before first demo-live trade cycle."""
        outcomes: list[dict[str, Any]] = []
        logger.info("Pretrade optimize starting for %d symbols", len(self.symbols))
        for symbol in self.symbols:
            try:
                loop = IntelligenceLoop(
                    self.config,
                    self.ohlcv,
                    symbol=symbol,
                    strategy=self.strategy,
                    timeframe=self.timeframe,
                )
                outcome = loop.run()
                outcomes.append(
                    {
                        "target": symbol,
                        "kind": "single",
                        "accepted": outcome.accepted,
                        "path": outcome.path,
                        "message": outcome.message,
                    }
                )
                logger.info(
                    "Pretrade optimize %s accepted=%s path=%s",
                    symbol,
                    outcome.accepted,
                    outcome.path,
                )
            except Exception as exc:
                logger.exception("Pretrade optimize failed for %s", symbol)
                outcomes.append(
                    {"target": symbol, "kind": "single", "accepted": False, "error": str(exc)}
                )

        if self.optimize_pairs:
            for pair in self.config.strategies.pairs:
                if len(pair) != 2:
                    continue
                pair_id = self.config.pair_state_id(pair[0], pair[1])
                try:
                    loop = IntelligenceLoop(
                        self.config,
                        self.ohlcv,
                        symbol=pair[0],
                        strategy=self.strategy,
                        timeframe=self.timeframe,
                        state_key=pair_id,
                    )
                    outcome = loop.run()
                    outcomes.append(
                        {
                            "target": pair_id,
                            "kind": "pair",
                            "accepted": outcome.accepted,
                            "path": outcome.path,
                            "message": outcome.message,
                        }
                    )
                    logger.info(
                        "Pretrade pair optimize %s accepted=%s path=%s",
                        pair_id,
                        outcome.accepted,
                        outcome.path,
                    )
                except Exception as exc:
                    logger.exception("Pair optimize failed for %s", pair_id)
                    outcomes.append(
                        {"target": pair_id, "kind": "pair", "accepted": False, "error": str(exc)}
                    )

        self._pretrade_ready = self._has_adopted_params()
        return outcomes

    def should_review(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if now.weekday() != self.review_weekday:
            return False
        if now.hour < self.review_hour_utc:
            return False
        day_key = now.strftime("%Y-%m-%d")
        persisted = self._load_last_review_date()
        if persisted == day_key or self._last_review_date == day_key:
            return False
        return True

    def ensure_connection(self) -> bool:
        try:
            if not self.connector.is_connected:
                self.connector.connect()
            return True
        except Exception:
            logger.exception("MT5 reconnect failed")
            try:
                self.connector.disconnect()
            except Exception:
                pass
            try:
                self.connector.connect()
                return True
            except Exception:
                logger.exception("MT5 reconnect retry failed")
                return False

    def _latest_bar_time(self, symbol: str) -> Optional[str]:
        bars = self.ohlcv.get_recent_bars(symbol, self.timeframe, 1)
        if not bars:
            return None
        return str(bars[-1].get("time") or "")

    def _process_symbol_bar(self, symbol: str) -> dict[str, Any] | None:
        if self.require_adopted_params and not self._pretrade_ready:
            return {
                "symbol": symbol,
                "skipped": "waiting_for_adopted_params",
                "pipeline_ok": False,
            }

        store = self.stores[symbol]
        latest = self._latest_bar_time(symbol)
        if not latest:
            return None
        last = store.read_state().get("last_processed_bar")
        if last and str(last) >= latest:
            return None

        result: dict[str, Any] = {
            "symbol": symbol,
            "bar_time": latest,
            "pipeline_ok": False,
        }
        if not self.run_pipeline_on_bar:
            store.update_state(last_processed_bar=latest)
            result["skipped"] = "pipeline_disabled"
            return result

        try:
            cfg = apply_state_overrides(self.config, symbol)
            system = TradingSystem(cfg, self.connector, self.ohlcv)
            run = system.run(symbols=[symbol], sync_first=False)
            result["pipeline_ok"] = bool(run.integration.pipeline_ok)
            result["signals"] = len(run.pipeline.signals)
            tickets = [
                getattr(plan, "ticket", None)
                for plan in run.pipeline.execution_plans
                if getattr(plan, "ticket", None)
            ]
            result["tickets"] = tickets
            store.update_state(last_processed_bar=latest)
            logger.info(
                "%s new bar=%s pipeline_ok=%s signals=%s tickets=%s dry_run=%s",
                symbol,
                latest,
                result["pipeline_ok"],
                result["signals"],
                tickets,
                cfg.trading.dry_run,
            )
        except Exception as exc:
            logger.exception("Pipeline failed for %s on bar %s", symbol, latest)
            result["error"] = str(exc)
        return result

    def poll_once(self) -> list[dict[str, Any]]:
        if not self.ensure_connection():
            return []

        if self.require_adopted_params and not self._pretrade_ready:
            logger.info("Adopted params missing — re-running pretrade optimize")
            self.run_pretrade_optimize()
            if not self._pretrade_ready:
                return []

        if self.sync_on_poll:
            try:
                system = TradingSystem(self.config, self.connector, self.ohlcv)
                synced = system.sync_data(self.symbols)
                logger.debug("Synced %s bars", synced)
            except Exception:
                logger.exception("Sync failed during poll")

        events: list[dict[str, Any]] = []
        for symbol in self.symbols:
            try:
                event = self._process_symbol_bar(symbol)
                if event:
                    events.append(event)
            except Exception:
                logger.exception("poll error for %s", symbol)
        return events

    def metrics_degraded(self, symbol: str) -> bool:
        store = self.stores[symbol]
        last = store.read_state().get("last_metrics") or {}
        prev = last.get("wf_avg_test_sharpe")
        if prev is None:
            prev = last.get("sharpe")
        if prev is None:
            return True

        defaults = params_from_config(self.config)
        working = LoopParams(overrides={**defaults.overrides, **store.get_params().overrides})
        validator = ParamValidator(
            self.config,
            self.ohlcv,
            symbol=symbol,
            strategy=self.strategy,
            timeframe=self.timeframe,
        )
        try:
            fresh = validator.baseline(working)
        except Exception as exc:
            logger.warning("%s fresh baseline failed: %s", symbol, exc)
            return False

        cur = float(fresh.walk_forward_summary.get("avg_test_sharpe", 0.0))
        if abs(float(prev)) > 1e-9:
            deg = max(0.0, (float(prev) - cur) / abs(float(prev)))
            if deg >= self.sharpe_degrade_trigger:
                logger.info(
                    "%s degraded sharpe prev=%.3f cur=%.3f deg=%.2f",
                    symbol,
                    float(prev),
                    cur,
                    deg,
                )
                return True
        if not fresh.quality_gate.passed:
            logger.info("%s quality gate failed on review baseline", symbol)
            return True
        return False

    def review_subloop(self) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        for symbol in self.symbols:
            logger.info("Review sub-loop starting for %s", symbol)
            try:
                if self.metrics_degraded(symbol):
                    logger.info("%s metrics degraded -> optimizing", symbol)
                    loop = IntelligenceLoop(
                        self.config,
                        self.ohlcv,
                        symbol=symbol,
                        strategy=self.strategy,
                        timeframe=self.timeframe,
                    )
                    outcome = loop.run()
                    outcomes.append(
                        {
                            "symbol": symbol,
                            "ok": True,
                            "optimized": True,
                            "accepted": outcome.accepted,
                            "path": outcome.path,
                            "message": outcome.message,
                            "params": outcome.params,
                            "metrics": outcome.metrics,
                        }
                    )
                else:
                    logger.info("%s metrics stable -> skip optimize", symbol)
                    outcomes.append(
                        {
                            "symbol": symbol,
                            "ok": True,
                            "optimized": False,
                            "skipped": True,
                            "reason": "metrics_stable",
                        }
                    )
            except Exception as exc:
                logger.exception("Review failed for %s", symbol)
                outcomes.append({"symbol": symbol, "ok": False, "error": str(exc)})

        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._persist_last_review_date(day_key)
        self._pretrade_ready = self._has_adopted_params()
        return outcomes

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                self.poll_once()
                if self.should_review():
                    self.review_subloop()
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            logger.info("ResidentLoopEngine interrupted by user")
        finally:
            self.stop()
