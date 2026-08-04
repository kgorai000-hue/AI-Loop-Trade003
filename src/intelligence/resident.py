"""Resident M30 loop: pretrade optimize, demo-live pipeline, weekend review."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.agents.regime_agent import RegimeAgent
from src.core.config import PROJECT_ROOT, AppConfig
from src.core.mt5_connector import MT5Connector
from src.core.types import StrategyKind
from src.data.store import OHLCVStore
from src.intelligence.loop import IntelligenceLoop, apply_state_overrides
from src.intelligence.params import (
    OPTIMIZE_STRATEGIES,
    LoopParams,
    params_from_config,
    strategy_state_key,
)
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
        self.check_all_asset_groups = bool(
            getattr(loop_cfg, "check_all_asset_groups", True)
        )

        # Trading subset (CLI --symbol) vs full engineering universe across Asset Groups.
        self.trade_symbols = list(self.symbols)
        if self.check_all_asset_groups:
            self.engineering_symbols = list(config.tradeable_symbols_all_groups())
        else:
            self.engineering_symbols = list(self.trade_symbols)
        if not self.engineering_symbols:
            self.engineering_symbols = list(self.trade_symbols)

        self.state_dir = config.intelligence.state_dir
        self.stores = {
            strategy_state_key(sym, strat): StateStore(
                self.state_dir, strategy_state_key(sym, strat)
            )
            for sym in self.engineering_symbols
            for strat in OPTIMIZE_STRATEGIES
        }
        from src.intelligence.params import symbol_to_state_key

        self.bar_stores = {
            sym: StateStore(self.state_dir, f"{symbol_to_state_key(sym)}__runtime")
            for sym in self.engineering_symbols
        }
        pair_source = (
            config.pairs_all_groups()
            if self.check_all_asset_groups
            else [
                pair
                for pair in config.strategies.pairs
                if len(pair) == 2
                and pair[0] in self.trade_symbols
                and pair[1] in self.trade_symbols
            ]
        )
        self.pair_ids = [
            config.pair_state_id(pair[0], pair[1])
            for pair in pair_source
            if len(pair) == 2
        ]
        self._pair_legs = {
            config.pair_state_id(pair[0], pair[1]): (pair[0], pair[1])
            for pair in pair_source
            if len(pair) == 2
        }
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
            "ResidentLoopEngine started trade=%s engineering=%s pairs=%s "
            "check_all_groups=%s tf=%s poll=%ss dry_run=%s stage=%s",
            self.trade_symbols,
            self.engineering_symbols,
            self.pair_ids,
            self.check_all_asset_groups,
            self.timeframe,
            self.poll_seconds,
            self.config.trading.dry_run,
            self.config.project.graduation_stage,
        )
        self._pretrade_ready = self._has_adopted_params()
        if self.pretrade_optimize:
            if self._pretrade_ready:
                logger.info(
                    "Adopted/seeded params already present — skipping pretrade optimize"
                )
            else:
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
        """True if every engineering symbol has at least one adopted strategy."""
        if not self.engineering_symbols:
            return False
        for symbol in self.engineering_symbols:
            found = False
            for strat in OPTIMIZE_STRATEGIES:
                store = self.stores.get(strategy_state_key(symbol, strat))
                if store is None:
                    continue
                state = store.read_state()
                if state.get("accepted") or store.get_params().overrides:
                    found = True
                    break
            if not found:
                return False
        return True

    def _group_name(self, symbol: str) -> str:
        group = self.config.group_for_symbol(symbol)
        return group.name if group else "ungrouped"

    def _optimize_strategies(self) -> tuple[str, ...]:
        return OPTIMIZE_STRATEGIES

    def _live_strategy_for_symbol(self, symbol: str) -> str | None:
        """Resolve current regime → strategy name, or None if halted."""
        agent = RegimeAgent(self.config, self.ohlcv)
        assessment = agent.assess(symbol, self.timeframe)
        if assessment is None:
            return None
        if assessment.selected_strategy == StrategyKind.TREND_FOLLOWING:
            return "trend_following"
        if assessment.selected_strategy == StrategyKind.MEAN_REVERSION:
            return "mean_reversion"
        return None

    def run_pretrade_optimize(self) -> list[dict[str, Any]]:
        """Optimize every symbol × trend/MR before first demo-live trade."""
        outcomes: list[dict[str, Any]] = []
        symbols = list(self.engineering_symbols)
        logger.info(
            "Pretrade optimize across %d symbols × %s (check_all=%s)",
            len(symbols),
            list(self._optimize_strategies()),
            self.check_all_asset_groups,
        )
        for symbol in symbols:
            group_id = self._group_name(symbol)
            for strategy in self._optimize_strategies():
                try:
                    loop = IntelligenceLoop(
                        self.config,
                        self.ohlcv,
                        symbol=symbol,
                        strategy=strategy,
                        timeframe=self.timeframe,
                    )
                    outcome = loop.run()
                    outcomes.append(
                        {
                            "target": symbol,
                            "group": group_id,
                            "kind": "single",
                            "strategy": strategy,
                            "accepted": outcome.accepted,
                            "path": outcome.path,
                            "message": outcome.message,
                        }
                    )
                    logger.info(
                        "Pretrade optimize %s [%s] %s accepted=%s path=%s",
                        symbol,
                        group_id,
                        strategy,
                        outcome.accepted,
                        outcome.path,
                    )
                except Exception as exc:
                    logger.exception(
                        "Pretrade optimize failed for %s [%s] %s",
                        symbol,
                        group_id,
                        strategy,
                    )
                    outcomes.append(
                        {
                            "target": symbol,
                            "group": group_id,
                            "kind": "single",
                            "strategy": strategy,
                            "accepted": False,
                            "error": str(exc),
                        }
                    )

        if self.optimize_pairs:
            for pair_id, (leg_a, leg_b) in self._pair_legs.items():
                group_id = self._group_name(leg_a)
                try:
                    loop = IntelligenceLoop(
                        self.config,
                        self.ohlcv,
                        symbol=leg_a,
                        strategy="mean_reversion",
                        timeframe=self.timeframe,
                        state_key=pair_id,
                    )
                    outcome = loop.run()
                    outcomes.append(
                        {
                            "target": pair_id,
                            "group": group_id,
                            "kind": "pair",
                            "legs": [leg_a, leg_b],
                            "accepted": outcome.accepted,
                            "path": outcome.path,
                            "message": outcome.message,
                        }
                    )
                    logger.info(
                        "Pretrade pair optimize %s [%s] accepted=%s path=%s",
                        pair_id,
                        group_id,
                        outcome.accepted,
                        outcome.path,
                    )
                except Exception as exc:
                    logger.exception("Pretrade pair optimize failed for %s", pair_id)
                    outcomes.append(
                        {
                            "target": pair_id,
                            "group": group_id,
                            "kind": "pair",
                            "accepted": False,
                            "error": str(exc),
                        }
                    )

        self._log_pretrade_summary(outcomes)
        self._pretrade_ready = self._has_adopted_params()
        return outcomes

    def _log_pretrade_summary(self, outcomes: list[dict[str, Any]]) -> None:
        summary = self._summarize_group_checks(outcomes)
        logger.info("Pretrade Asset Group check summary: %s", summary)
        outcomes.append({"kind": "group_summary", "summary": summary})

    def _summarize_group_checks(self, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        by_group: dict[str, dict[str, int]] = {}
        for row in outcomes:
            if row.get("kind") == "group_summary":
                continue
            group = str(row.get("group") or "ungrouped")
            bucket = by_group.setdefault(
                group, {"checked": 0, "accepted": 0, "failed": 0, "stable": 0, "optimized": 0}
            )
            bucket["checked"] += 1
            if row.get("accepted"):
                bucket["accepted"] += 1
            if row.get("error") or (row.get("accepted") is False and row.get("ok") is False):
                bucket["failed"] += 1
            if row.get("optimized"):
                bucket["optimized"] += 1
            if row.get("skipped") or row.get("reason") == "metrics_stable":
                bucket["stable"] += 1
        return {
            "groups": by_group,
            "groups_checked": sorted(by_group.keys()),
            "total_checked": sum(v["checked"] for v in by_group.values()),
        }

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
        # Store keys use MT5-resolved names; try resolved first then raw config symbol.
        candidates = [symbol]
        try:
            resolved = self.connector.ensure_symbol(symbol)
            if resolved and resolved not in candidates:
                candidates.insert(0, resolved)
        except Exception:
            pass
        for name in candidates:
            bars = self.ohlcv.get_recent_bars(name, self.timeframe, 1)
            if bars:
                return str(bars[-1].get("time") or "")
        return None

    def _process_symbol_bar(self, symbol: str) -> dict[str, Any] | None:
        if self.require_adopted_params and not self._pretrade_ready:
            return {
                "symbol": symbol,
                "skipped": "waiting_for_adopted_params",
                "pipeline_ok": False,
            }

        store = self.bar_stores.get(symbol)
        if store is None:
            from src.intelligence.params import symbol_to_state_key

            store = StateStore(self.state_dir, f"{symbol_to_state_key(symbol)}__runtime")
            self.bar_stores[symbol] = store
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
            logger.info("%s bar=%s skipped=pipeline_disabled", symbol, latest)
            return result

        strategy = self._live_strategy_for_symbol(symbol)
        if strategy is None:
            store.update_state(last_processed_bar=latest)
            result["skipped"] = "regime_halt"
            logger.info("%s bar=%s skipped=regime_halt", symbol, latest)
            return result

        strat_key = strategy_state_key(symbol, strategy)
        strat_store = self.stores.get(strat_key)
        if strat_store is None or (
            not strat_store.read_state().get("accepted")
            and not strat_store.get_params().overrides
        ):
            store.update_state(last_processed_bar=latest)
            result["skipped"] = f"no_adopted_params:{strategy}"
            logger.info(
                "%s bar=%s skipped=no_adopted_params strategy=%s",
                symbol,
                latest,
                strategy,
            )
            return result

        try:
            cfg = apply_state_overrides(self.config, symbol, strategy=strategy)
            system = TradingSystem(cfg, self.connector, self.ohlcv)
            run = system.run(symbols=[symbol], sync_first=False)
            result["pipeline_ok"] = bool(run.integration.pipeline_ok)
            result["signals"] = len(run.pipeline.signals)
            result["strategy"] = strategy
            tickets = [
                getattr(plan, "ticket", None)
                for plan in run.pipeline.execution_plans
                if getattr(plan, "ticket", None)
            ]
            result["tickets"] = tickets
            store.update_state(last_processed_bar=latest)
            logger.info(
                "%s new bar=%s strategy=%s pipeline_ok=%s signals=%s tickets=%s dry_run=%s",
                symbol,
                latest,
                strategy,
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
                # Poll sync: primary TF only (full multi-TF sync floods logs every 30s).
                sync_symbols = list(
                    dict.fromkeys([*self.engineering_symbols, *self.trade_symbols])
                )
                summary = system.data_agent.sync_all(
                    symbols=sync_symbols,
                    timeframes=[self.timeframe],
                )
                logger.debug(
                    "Synced %s bars across %s symbols (%s only)",
                    summary.total_stored,
                    len(sync_symbols),
                    self.timeframe,
                )
            except Exception:
                logger.exception("Sync failed during poll")

        events: list[dict[str, Any]] = []
        for symbol in self.trade_symbols:
            try:
                event = self._process_symbol_bar(symbol)
                if event:
                    events.append(event)
            except Exception:
                logger.exception("poll error for %s", symbol)
        if not events:
            # Heartbeat so operators know the loop is alive between M30 closes.
            sample = self.trade_symbols[0] if self.trade_symbols else None
            latest = self._latest_bar_time(sample) if sample else None
            last = None
            if sample:
                store = self.bar_stores.get(sample)
                if store is None:
                    from src.intelligence.params import symbol_to_state_key

                    store = StateStore(
                        self.state_dir, f"{symbol_to_state_key(sample)}__runtime"
                    )
                    self.bar_stores[sample] = store
                last = store.read_state().get("last_processed_bar")
            logger.info(
                "poll idle: no new %s bars (%d symbols) sample=%s last=%s latest=%s",
                self.timeframe,
                len(self.trade_symbols),
                sample,
                last,
                latest,
            )
        return events

    def metrics_degraded(self, symbol: str, strategy: str) -> bool:
        key = strategy_state_key(symbol, strategy)
        store = self.stores.get(key)
        if store is None:
            store = StateStore(self.state_dir, key)
            self.stores[key] = store
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
            strategy=strategy,
            timeframe=self.timeframe,
        )
        try:
            fresh = validator.baseline(working)
        except Exception as exc:
            logger.warning("%s [%s] %s fresh baseline failed: %s", symbol, self._group_name(symbol), strategy, exc)
            return False

        cur = float(fresh.walk_forward_summary.get("avg_test_sharpe", 0.0))
        if abs(float(prev)) > 1e-9:
            deg = max(0.0, (float(prev) - cur) / abs(float(prev)))
            if deg >= self.sharpe_degrade_trigger:
                logger.info(
                    "%s [%s] %s degraded sharpe prev=%.3f cur=%.3f deg=%.2f",
                    symbol,
                    self._group_name(symbol),
                    strategy,
                    float(prev),
                    cur,
                    deg,
                )
                return True
        if not fresh.quality_gate.passed:
            logger.info(
                "%s [%s] %s quality gate failed on review baseline",
                symbol,
                self._group_name(symbol),
                strategy,
            )
            return True
        return False

    def review_subloop(self) -> list[dict[str, Any]]:
        """Check every Asset Group symbol × strategy (and pairs); optimize when degraded."""
        outcomes: list[dict[str, Any]] = []
        symbols = list(self.engineering_symbols)
        logger.info(
            "Review sub-loop checking %d symbols × %s across Asset Groups %s",
            len(symbols),
            list(self._optimize_strategies()),
            sorted({self._group_name(s) for s in symbols}),
        )
        for symbol in symbols:
            group_id = self._group_name(symbol)
            for strategy in self._optimize_strategies():
                logger.info("Review check %s [%s] %s", symbol, group_id, strategy)
                try:
                    degraded = self.metrics_degraded(symbol, strategy)
                    if degraded:
                        logger.info(
                            "%s [%s] %s metrics degraded -> optimizing",
                            symbol,
                            group_id,
                            strategy,
                        )
                        loop = IntelligenceLoop(
                            self.config,
                            self.ohlcv,
                            symbol=symbol,
                            strategy=strategy,
                            timeframe=self.timeframe,
                        )
                        outcome = loop.run()
                        outcomes.append(
                            {
                                "symbol": symbol,
                                "group": group_id,
                                "kind": "single",
                                "strategy": strategy,
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
                        logger.info(
                            "%s [%s] %s metrics stable -> skip optimize",
                            symbol,
                            group_id,
                            strategy,
                        )
                        outcomes.append(
                            {
                                "symbol": symbol,
                                "group": group_id,
                                "kind": "single",
                                "strategy": strategy,
                                "ok": True,
                                "optimized": False,
                                "skipped": True,
                                "reason": "metrics_stable",
                                "accepted": True,
                            }
                        )
                except Exception as exc:
                    logger.exception("Review failed for %s [%s] %s", symbol, group_id, strategy)
                    outcomes.append(
                        {
                            "symbol": symbol,
                            "group": group_id,
                            "kind": "single",
                            "strategy": strategy,
                            "ok": False,
                            "accepted": False,
                            "error": str(exc),
                        }
                    )

        if self.optimize_pairs:
            for pair_id, (leg_a, leg_b) in self._pair_legs.items():
                group_id = self._group_name(leg_a)
                logger.info("Review pair check %s [%s]", pair_id, group_id)
                try:
                    # Use first leg degradation as proxy for pair review.
                    if self.metrics_degraded(leg_a, "mean_reversion") or self.metrics_degraded(
                        leg_b, "mean_reversion"
                    ):
                        loop = IntelligenceLoop(
                            self.config,
                            self.ohlcv,
                            symbol=leg_a,
                            strategy="mean_reversion",
                            timeframe=self.timeframe,
                            state_key=pair_id,
                        )
                        outcome = loop.run()
                        outcomes.append(
                            {
                                "symbol": pair_id,
                                "group": group_id,
                                "kind": "pair",
                                "legs": [leg_a, leg_b],
                                "ok": True,
                                "optimized": True,
                                "accepted": outcome.accepted,
                                "path": outcome.path,
                                "message": outcome.message,
                            }
                        )
                    else:
                        outcomes.append(
                            {
                                "symbol": pair_id,
                                "group": group_id,
                                "kind": "pair",
                                "legs": [leg_a, leg_b],
                                "ok": True,
                                "optimized": False,
                                "skipped": True,
                                "reason": "metrics_stable",
                                "accepted": True,
                            }
                        )
                except Exception as exc:
                    logger.exception("Review pair failed for %s [%s]", pair_id, group_id)
                    outcomes.append(
                        {
                            "symbol": pair_id,
                            "group": group_id,
                            "kind": "pair",
                            "ok": False,
                            "accepted": False,
                            "error": str(exc),
                        }
                    )

        summary = self._summarize_group_checks(outcomes)
        logger.info("Review Asset Group check summary: %s", summary)
        outcomes.append({"kind": "group_summary", "ok": True, "summary": summary})

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
