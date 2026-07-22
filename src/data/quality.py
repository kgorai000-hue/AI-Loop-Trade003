from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.core.config import DataQualityConfig
from src.data.store import BarRecord

TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    start_time: str
    end_time: str
    total_rows: int
    gap_count: int
    missing_rate_pct: float
    null_values: dict[str, int]
    zero_volume: int
    duplicate_timestamps: int
    rejected_bars: int
    anomalies: list[str] = field(default_factory=list)
    is_valid: bool = True

    def add_anomaly(self, message: str) -> None:
        self.anomalies.append(message)
        self.is_valid = False


def bars_to_dataframe(bars: list[BarRecord]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    rows = [
        {
            "time": bar.time,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "tick_volume": bar.tick_volume,
            "spread": bar.spread,
            "real_volume": bar.real_volume,
        }
        for bar in bars
    ]
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def validate_bar(
    bar: BarRecord,
    prev_close: float | None,
    cfg: DataQualityConfig,
) -> tuple[bool, str | None]:
    """Return (ok, rejection_reason)."""
    prices = (bar.open, bar.high, bar.low, bar.close)
    if any(not np.isfinite(p) for p in prices):
        return False, "non-finite OHLC"

    if bar.high < bar.low:
        return False, "high < low"

    if bar.open < bar.low or bar.open > bar.high:
        return False, "open outside high/low"

    if bar.close < bar.low or bar.close > bar.high:
        return False, "close outside high/low"

    if bar.tick_volume < 0:
        return False, "negative tick_volume"

    if prev_close is not None and prev_close > 0:
        jump = abs(bar.close - prev_close) / prev_close
        if jump > cfg.price_jump_threshold:
            return False, f"price jump {jump:.1%}"

    return True, None


def filter_valid_bars(
    bars: list[BarRecord],
    cfg: DataQualityConfig,
) -> tuple[list[BarRecord], list[str]]:
    """Exclude anomalous bars; dedupe timestamps (keep last)."""
    if not bars:
        return [], []

    sorted_bars = sorted(bars, key=lambda b: b.time)
    by_time: dict[int, BarRecord] = {}
    rejections: list[str] = []
    prev_close: float | None = None

    for bar in sorted_bars:
        ok, reason = validate_bar(bar, prev_close, cfg)
        if not ok:
            rejections.append(f"{bar.symbol} {bar.timeframe} t={bar.time}: {reason}")
            continue
        by_time[bar.time] = bar
        prev_close = bar.close

    return list(by_time.values()), rejections


def _is_weekend_gap(diff_seconds: float, interval: int) -> bool:
    """FX/CFD weekend closures are expected, not data quality failures."""
    if diff_seconds <= interval * 1.5:
        return False
    hours = diff_seconds / 3600
    return 36 <= hours <= 96


def check_data_quality(
    bars: list[BarRecord] | list[dict],
    symbol: str,
    timeframe: str,
    cfg: DataQualityConfig,
    rejected_count: int = 0,
) -> DataQualityReport:
    """Quality report for stored or fetched bars (Lesson 06)."""
    if not bars:
        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            start_time="n/a",
            end_time="n/a",
            total_rows=0,
            gap_count=0,
            missing_rate_pct=0.0,
            null_values={},
            zero_volume=0,
            duplicate_timestamps=0,
            rejected_bars=rejected_count,
            is_valid=False,
            anomalies=["no data"],
        )

    if isinstance(bars[0], BarRecord):
        df = bars_to_dataframe(bars)  # type: ignore[arg-type]
    else:
        df = pd.DataFrame(bars)
        df.index = pd.to_datetime(df["time"], unit="s", utc=True)

    interval = TIMEFRAME_SECONDS.get(timeframe.upper(), 3600)
    times = df["time"].to_numpy(dtype=np.int64)
    gap_count = 0
    missing_rows = 0
    if len(times) > 1:
        diffs = np.diff(times)
        for diff in diffs:
            if diff > interval * cfg.gap_multiplier:
                gap_count += 1
                if not _is_weekend_gap(float(diff), interval):
                    missing_rows += max(int(diff // interval) - 1, 0)

    expected_span = len(df) + missing_rows
    missing_rate_pct = missing_rows / expected_span * 100 if expected_span else 0.0

    null_values = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().any()}
    zero_volume = int((df["tick_volume"] == 0).sum()) if "tick_volume" in df.columns else 0
    duplicates = int(df.index.duplicated().sum())

    report = DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        start_time=str(df.index[0]),
        end_time=str(df.index[0]) if len(df) == 1 else str(df.index[-1]),
        total_rows=len(df),
        gap_count=gap_count,
        missing_rate_pct=missing_rate_pct,
        null_values=null_values,
        zero_volume=zero_volume,
        duplicate_timestamps=duplicates,
        rejected_bars=rejected_count,
    )

    if missing_rate_pct > cfg.missing_rate_warn_pct:
        report.add_anomaly(f"High missing rate: {missing_rate_pct:.1f}%")

    if gap_count > 0:
        report.anomalies.append(f"Time gaps detected: {gap_count} (weekends expected for FX/CFD)")

    if zero_volume > 0:
        report.anomalies.append(f"Zero volume bars: {zero_volume}")

    if null_values:
        report.add_anomaly(f"Null values: {null_values}")

    if duplicates > 0:
        report.add_anomaly(f"Duplicate timestamps: {duplicates}")

    if "close" in df.columns and len(df) > 1:
        returns = df["close"].pct_change()
        extreme = int((returns.abs() > cfg.price_jump_threshold).sum())
        if extreme > 0:
            report.anomalies.append(f"Extreme price moves: {extreme}")

    last_ts = int(times[-1])
    age_seconds = datetime.now(tz=timezone.utc).timestamp() - last_ts
    if age_seconds > cfg.freshness_warn_seconds:
        report.anomalies.append(f"Stale data: last bar {age_seconds / 3600:.1f}h ago")

    critical = [
        a for a in report.anomalies
        if any(
            key in a.lower()
            for key in ("null", "duplicate", "high missing", "no data")
        )
    ]
    report.is_valid = len(critical) == 0

    return report
