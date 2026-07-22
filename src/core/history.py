from __future__ import annotations

BARS_PER_TRADING_DAY: dict[str, float] = {
    "M1": 1440,
    "M5": 288,
    "M15": 96,
    "M30": 48,
    "H1": 24,
    "H4": 6,
    "D1": 1,
    "W1": 1 / 5,
    "MN1": 1 / 21,
}


def bars_per_trading_day(timeframe: str) -> float:
    return float(BARS_PER_TRADING_DAY.get(timeframe.upper(), 24))


def periods_per_year_for_timeframe(
    timeframe: str,
    trading_days_per_year: int = 252,
) -> float:
    """Number of bars per year for Sharpe / annualization of bar returns.

    Example: M30 -> 252 * 48 = 12096 periods/year.
    """
    return float(trading_days_per_year) * bars_per_trading_day(timeframe)


def history_bars_for_timeframe(
    timeframe: str,
    *,
    history_years: float | None,
    history_bars: int,
    history_bars_by_timeframe: dict[str, int] | None = None,
    trading_days_per_year: int = 252,
) -> int:
    """Target bar count for a timeframe (e.g. H1 x 1.5y ≈ 9072 bars)."""
    tf = timeframe.upper()
    if history_bars_by_timeframe and tf in history_bars_by_timeframe:
        return int(history_bars_by_timeframe[tf])
    if history_years is not None and history_years > 0:
        return int(history_years * trading_days_per_year * bars_per_trading_day(tf))
    return history_bars


def max_history_bars_for_timeframes(
    timeframes: list[str],
    *,
    history_years: float | None,
    history_bars: int,
    history_bars_by_timeframe: dict[str, int] | None = None,
    trading_days_per_year: int = 252,
) -> int:
    if not timeframes:
        return history_bars
    return max(
        history_bars_for_timeframe(
            tf,
            history_years=history_years,
            history_bars=history_bars,
            history_bars_by_timeframe=history_bars_by_timeframe,
            trading_days_per_year=trading_days_per_year,
        )
        for tf in timeframes
    )
