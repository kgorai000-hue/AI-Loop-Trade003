from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.config import PROJECT_ROOT

if TYPE_CHECKING:
    from src.agents.backtest_agent import BacktestAgent, ValidationReport
    from src.agents.regime_agent import RegimeAgent

logger = logging.getLogger(__name__)

BACKTEST_STRATEGY_NAMES = ("trend_following", "mean_reversion", "feature_score")


@dataclass
class GateEntry:
    symbol: str
    timeframe: str
    strategy: str
    passed: bool
    sharpe: float = 0.0


@dataclass
class GateFilterReport:
    enabled: bool
    timeframe: str
    total_entries: int = 0
    passed_count: int = 0
    failed_count: int = 0
    blocked_generators: int = 0
    cache_age_hours: float = 0.0
    from_cache: bool = False


@dataclass
class GateRegistry:
    """Backtest quality-gate results for live signal filtering."""

    timeframe: str
    entries: dict[tuple[str, str], GateEntry] = field(default_factory=dict)
    updated_at: int = 0
    blocked_count: int = 0

    def allows(self, symbol: str, strategy: str) -> bool:
        entry = self.entries.get((symbol, strategy))
        if entry is None:
            return True
        return entry.passed

    def check(self, symbol: str, strategy: str) -> bool:
        if self.allows(symbol, strategy):
            return True
        self.blocked_count += 1
        logger.info("Quality gate blocked %s / %s on %s", strategy, symbol, self.timeframe)
        return False

    def summary(self, *, enabled: bool, from_cache: bool = False) -> GateFilterReport:
        passed = sum(1 for e in self.entries.values() if e.passed)
        failed = len(self.entries) - passed
        age_h = (time.time() - self.updated_at) / 3600.0 if self.updated_at else 0.0
        return GateFilterReport(
            enabled=enabled,
            timeframe=self.timeframe,
            total_entries=len(self.entries),
            passed_count=passed,
            failed_count=failed,
            blocked_generators=self.blocked_count,
            cache_age_hours=round(age_h, 2),
            from_cache=from_cache,
        )

    def is_fresh(self, max_age_hours: float) -> bool:
        if self.updated_at <= 0:
            return False
        return (time.time() - self.updated_at) < max_age_hours * 3600.0

    def save(self, path: str | Path) -> None:
        file_path = _resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timeframe": self.timeframe,
            "updated_at": self.updated_at,
            "entries": [asdict(entry) for entry in self.entries.values()],
        }
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, timeframe: str) -> GateRegistry | None:
        file_path = _resolve_path(path)
        if not file_path.exists():
            return None
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load gate registry %s: %s", file_path, exc)
            return None

        if raw.get("timeframe") != timeframe:
            return None

        entries: dict[tuple[str, str], GateEntry] = {}
        for row in raw.get("entries", []):
            entry = GateEntry(
                symbol=str(row["symbol"]),
                timeframe=str(row.get("timeframe", timeframe)),
                strategy=str(row["strategy"]),
                passed=bool(row["passed"]),
                sharpe=float(row.get("sharpe", 0.0)),
            )
            entries[(entry.symbol, entry.strategy)] = entry

        return cls(
            timeframe=timeframe,
            entries=entries,
            updated_at=int(raw.get("updated_at", 0)),
        )

    @classmethod
    def from_validation_report(cls, report: ValidationReport) -> GateRegistry:
        entries: dict[tuple[str, str], GateEntry] = {}
        for result in report.strategies:
            entries[(report.symbol, result.strategy_name)] = GateEntry(
                symbol=report.symbol,
                timeframe=report.timeframe,
                strategy=result.strategy_name,
                passed=result.quality_gate.passed,
                sharpe=result.backtest.performance.sharpe_ratio,
            )
        return cls(
            timeframe=report.timeframe,
            entries=entries,
            updated_at=int(time.time()),
        )

    @classmethod
    def merge_reports(cls, reports: list[ValidationReport], timeframe: str) -> GateRegistry:
        entries: dict[tuple[str, str], GateEntry] = {}
        for report in reports:
            partial = cls.from_validation_report(report)
            entries.update(partial.entries)
        return cls(timeframe=timeframe, entries=entries, updated_at=int(time.time()))


def build_gate_registry(
    backtest_agent: BacktestAgent,
    regime_agent: RegimeAgent,
    symbols: list[str],
    timeframe: str,
) -> GateRegistry:
    reports: list[ValidationReport] = []
    for symbol in symbols:
        try:
            regime = regime_agent.assess(symbol)
            market_regime = regime.regime if regime else None
            reports.append(backtest_agent.validate_symbol(symbol, timeframe, market_regime))
        except ValueError as exc:
            logger.warning("Gate validation skipped for %s %s: %s", symbol, timeframe, exc)
    return GateRegistry.merge_reports(reports, timeframe)


def load_or_build_gate_registry(
    backtest_agent: BacktestAgent,
    regime_agent: RegimeAgent,
    symbols: list[str],
    timeframe: str,
    cache_path: str,
    max_age_hours: float,
) -> tuple[GateRegistry, bool]:
    cached = GateRegistry.load(cache_path, timeframe)
    if cached is not None and cached.is_fresh(max_age_hours) and cached.entries:
        return cached, True

    registry = build_gate_registry(backtest_agent, regime_agent, symbols, timeframe)
    if registry.entries:
        registry.save(cache_path)
    return registry, False


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p
