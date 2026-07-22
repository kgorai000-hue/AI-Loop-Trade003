from __future__ import annotations

from src.core.history import (
    history_bars_for_timeframe,
    max_history_bars_for_timeframes,
    periods_per_year_for_timeframe,
)


def test_history_bars_for_h1_one_and_half_years() -> None:
    bars = history_bars_for_timeframe(
        "H1",
        history_years=1.5,
        history_bars=2000,
        trading_days_per_year=252,
    )
    assert bars == 9072


def test_history_bars_for_d1_one_and_half_years() -> None:
    bars = history_bars_for_timeframe(
        "D1",
        history_years=1.5,
        history_bars=2000,
        trading_days_per_year=252,
    )
    assert bars == 378


def test_history_bars_override_by_timeframe() -> None:
    bars = history_bars_for_timeframe(
        "H1",
        history_years=1.5,
        history_bars=2000,
        history_bars_by_timeframe={"H1": 10000},
        trading_days_per_year=252,
    )
    assert bars == 10000


def test_max_history_bars_for_timeframes() -> None:
    bars = max_history_bars_for_timeframes(
        ["M15", "H1", "D1"],
        history_years=1.5,
        history_bars=2000,
        trading_days_per_year=252,
    )
    assert bars == 36288


def test_periods_per_year_for_timeframe_m30() -> None:
    assert periods_per_year_for_timeframe("M30") == 12096.0
