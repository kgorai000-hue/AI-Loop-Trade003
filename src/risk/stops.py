from __future__ import annotations


def fixed_stop_distance(entry_price: float, stop_pct: float) -> float:
    """Fixed percentage stop distance (Lesson 15.3)."""
    if entry_price <= 0 or stop_pct <= 0:
        return 0.0
    return entry_price * (stop_pct / 100.0)


def atr_stop_distance(atr: float, multiplier: float) -> float:
    """ATR-based stop distance: N x ATR."""
    if atr <= 0 or multiplier <= 0:
        return 0.0
    return atr * multiplier


def vol_stop_distance(entry_price: float, daily_vol: float, multiplier: float) -> float:
    """Volatility-adjusted stop: k x sigma (daily vol as fraction)."""
    if entry_price <= 0 or daily_vol <= 0 or multiplier <= 0:
        return 0.0
    return entry_price * daily_vol * multiplier


def stop_price(entry_price: float, distance: float, *, side: str = "buy") -> float:
    if side.lower() in ("buy", "long"):
        return entry_price - distance
    return entry_price + distance
