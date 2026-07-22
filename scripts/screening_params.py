"""Canonical adopted screening parameter maps (M30, return-max, gates OFF).

Single source of truth for BEST_* maps. Runners and adopted_params import from here.
"""
from __future__ import annotations

# Config symbol order
SYMBOL_ORDER = [
    "#US30",
    "#USSPX500",
    "#USNDAQ100",
    "#Japan225",
    "#Germany40",
    "#UK100",
    "GOLD",
    "SILVER",
    "WTI",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
]

# Best RSI bands from M30 nogate comparison (return-max per symbol)
BEST_MR_RSI_BY_SYMBOL: dict[str, tuple[float, float]] = {
    "#US30": (25.0, 75.0),
    "#USSPX500": (25.0, 75.0),
    "#USNDAQ100": (35.0, 65.0),
    "#Japan225": (30.0, 70.0),
    "#Germany40": (25.0, 75.0),
    "#UK100": (30.0, 70.0),
    "GOLD": (30.0, 70.0),
    "SILVER": (35.0, 65.0),
    "WTI": (35.0, 65.0),
    "EURUSD": (25.0, 75.0),
    "GBPUSD": (25.0, 75.0),
    "USDJPY": (35.0, 65.0),
}

# Best BB entry bands from M30 nogate comparison (return-max; RSI fixed to best map)
BEST_MR_BB_BY_SYMBOL: dict[str, tuple[float, float]] = {
    "#US30": (0.15, 0.85),
    "#USSPX500": (0.15, 0.85),
    "#USNDAQ100": (0.20, 0.80),
    "#Japan225": (0.10, 0.90),
    "#Germany40": (0.10, 0.90),
    "#UK100": (0.20, 0.80),
    "GOLD": (0.10, 0.90),
    "SILVER": (0.15, 0.85),
    "WTI": (0.15, 0.85),
    "EURUSD": (0.15, 0.85),
    "GBPUSD": (0.15, 0.85),
    "USDJPY": (0.15, 0.85),
}

# Best adx_trend_threshold from prior M30 grid (return-max)
BEST_ADX_TREND: dict[str, float] = {
    "#US30": 26.0,
    "SILVER": 18.0,
    "#Germany40": 18.0,
    "#Japan225": 30.0,
    "GOLD": 24.0,
    "USDJPY": 28.0,
    "GBPUSD": 24.0,
    "#UK100": 24.0,
    "EURUSD": 30.0,
    "WTI": 20.0,
    "#USSPX500": 26.0,
    "#USNDAQ100": 30.0,
}

# Best adx_sideways_threshold from M30 nogate grid (return-max; trend ADX fixed above)
BEST_ADX_SIDEWAYS: dict[str, float] = {
    "#US30": 15.0,
    "GOLD": 15.0,
    "#Germany40": 22.0,
    "USDJPY": 15.0,
    "#Japan225": 22.0,
    "GBPUSD": 15.0,
    "EURUSD": 15.0,
    "#USSPX500": 15.0,
    "#UK100": 15.0,
    "SILVER": 24.0,
    "WTI": 15.0,
    "#USNDAQ100": 18.0,
}

# Best MA pair from M30 nogate sequential grid (short then long; ADX maps fixed)
BEST_MA_BY_SYMBOL: dict[str, tuple[int, int]] = {
    "SILVER": (8, 20),
    "WTI": (3, 40),
    "GOLD": (8, 40),
    "#US30": (10, 20),
    "#USNDAQ100": (10, 40),
    "#Germany40": (10, 20),
    "#Japan225": (8, 15),
    "EURUSD": (5, 40),
    "USDJPY": (8, 20),
    "#USSPX500": (3, 40),
    "GBPUSD": (5, 40),
    "#UK100": (3, 30),
}

# Best signal_score_threshold from M30 nogate grid (return-max; prefer trades>0)
BEST_SIGNAL_SCORE_BY_SYMBOL: dict[str, float] = {
    "#USNDAQ100": 0.25,
    "#USSPX500": 0.25,
    "#UK100": 0.20,
    "USDJPY": 0.25,
    "EURUSD": 0.25,
    "GBPUSD": 0.25,
    "#US30": 0.25,
    "#Germany40": 0.20,
    # 0 trades at all thresholds on M30 2000 bars — keep default low
    "#Japan225": 0.05,
    "GOLD": 0.05,
    "SILVER": 0.05,
    "WTI": 0.05,
}


def rsi_band_label(symbol: str) -> str:
    """Format RSI band as '25/75' for Word export scripts."""
    oversold, overbought = BEST_MR_RSI_BY_SYMBOL[symbol]
    return f"{int(oversold)}/{int(overbought)}"
