from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IndicatorSnapshot:
    ema_fast: float
    ema_slow: float
    macd_diff: float
    macd_dea: float
    macd_histogram: float
    macd_histogram_delta: float
    rsi: float
    rsi_delta: float
    bb_middle: float
    bb_upper: float
    bb_lower: float
    bb_position: float
    atr: float
    atr_pct: float
    ma20: float
    adx: float = 0.0


def ema(values: np.ndarray, span: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    alpha = 2.0 / (span + 1)
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def sma(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full(len(values), np.nan)
    kernel = np.ones(window) / window
    rolled = np.convolve(values, kernel, mode="valid")
    padded = np.full(len(values), np.nan)
    padded[window - 1 :] = rolled
    return padded


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)

    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(len(closes), np.nan)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for idx in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[idx - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx - 1]) / period
        if avg_loss == 0:
            rsi[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[idx] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    histogram_double: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    diff = ema_fast - ema_slow
    dea = ema(diff, signal)
    multiplier = 2.0 if histogram_double else 1.0
    histogram = multiplier * (diff - dea)
    return diff, dea, histogram


def compute_bollinger(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    middle = sma(closes, period)
    std = np.full(len(closes), np.nan)
    position = np.full(len(closes), np.nan)

    for idx in range(period - 1, len(closes)):
        window = closes[idx - period + 1 : idx + 1]
        std[idx] = np.std(window, ddof=1)
        upper = middle[idx] + num_std * std[idx]
        lower = middle[idx] - num_std * std[idx]
        if upper > lower:
            position[idx] = (closes[idx] - lower) / (upper - lower)

    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower, position


def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    if len(closes) < 2:
        return np.full(len(closes), np.nan)

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))

    atr = np.full(len(closes), np.nan)
    if len(tr) >= period:
        atr[period - 1] = np.mean(tr[:period])
        for idx in range(period, len(tr)):
            atr[idx] = (atr[idx - 1] * (period - 1) + tr[idx]) / period
    return atr


def compute_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    if len(closes) < period + 2:
        return np.full(len(closes), np.nan)

    up_move = np.diff(highs, prepend=highs[0])
    down_move = -np.diff(lows, prepend=lows[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))

    atr = np.full(len(closes), np.nan)
    plus_di = np.full(len(closes), np.nan)
    minus_di = np.full(len(closes), np.nan)
    dx = np.full(len(closes), np.nan)
    adx = np.full(len(closes), np.nan)

    atr[period - 1] = np.mean(tr[1:period])
    pdm = np.mean(plus_dm[1:period])
    mdm = np.mean(minus_dm[1:period])
    plus_di[period - 1] = 100 * pdm / atr[period - 1] if atr[period - 1] else 0
    minus_di[period - 1] = 100 * mdm / atr[period - 1] if atr[period - 1] else 0

    for idx in range(period, len(closes)):
        atr[idx] = (atr[idx - 1] * (period - 1) + tr[idx]) / period
        pdm = (pdm * (period - 1) + plus_dm[idx]) / period
        mdm = (mdm * (period - 1) + minus_dm[idx]) / period
        plus_di[idx] = 100 * pdm / atr[idx] if atr[idx] else 0
        minus_di[idx] = 100 * mdm / atr[idx] if atr[idx] else 0
        di_sum = plus_di[idx] + minus_di[idx]
        dx[idx] = 100 * abs(plus_di[idx] - minus_di[idx]) / di_sum if di_sum else 0

    first_adx = period + period - 2
    if first_adx < len(closes):
        adx[first_adx] = np.nanmean(dx[period - 1 : first_adx + 1])
        for idx in range(first_adx + 1, len(closes)):
            adx[idx] = (adx[idx - 1] * (period - 1) + dx[idx]) / period

    return adx


def compute_indicators(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_histogram_double: bool = False,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    adx_period: int = 14,
) -> dict[str, np.ndarray]:
    diff, dea, histogram = compute_macd(
        closes,
        fast=macd_fast,
        slow=macd_slow,
        signal=macd_signal,
        histogram_double=macd_histogram_double,
    )
    rsi = compute_rsi(closes, rsi_period)
    bb_middle, bb_upper, bb_lower, bb_position = compute_bollinger(closes, bb_period, bb_std)
    atr = compute_atr(highs, lows, closes, atr_period)
    adx = compute_adx(highs, lows, closes, adx_period)

    hist_delta = np.full(len(closes), np.nan)
    hist_delta[1:] = np.diff(histogram)

    rsi_delta = np.full(len(closes), np.nan)
    rsi_delta[1:] = np.diff(rsi)

    ema_fast = ema(closes, macd_fast)
    ema_slow = ema(closes, macd_slow)

    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "macd_diff": diff,
        "macd_dea": dea,
        "macd_histogram": histogram,
        "macd_histogram_delta": hist_delta,
        "rsi": rsi,
        "rsi_delta": rsi_delta,
        "bb_middle": bb_middle,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_position": bb_position,
        "atr": atr,
        "adx": adx,
        "ma20": bb_middle,
        "volume": volumes.astype(float),
    }


def latest_snapshot(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    **kwargs,
) -> IndicatorSnapshot | None:
    data = compute_indicators(opens, highs, lows, closes, volumes, **kwargs)
    idx = len(closes) - 1
    if idx < 1 or np.isnan(data["rsi"][idx]):
        return None

    atr_val = float(data["atr"][idx]) if not np.isnan(data["atr"][idx]) else 0.0
    close_val = float(closes[idx])
    atr_pct = atr_val / close_val if close_val else 0.0

    hist_delta = data["macd_histogram_delta"][idx]
    rsi_delta = data["rsi_delta"][idx]

    return IndicatorSnapshot(
        ema_fast=float(data["ema_fast"][idx]),
        ema_slow=float(data["ema_slow"][idx]),
        macd_diff=float(data["macd_diff"][idx]),
        macd_dea=float(data["macd_dea"][idx]),
        macd_histogram=float(data["macd_histogram"][idx]),
        macd_histogram_delta=float(hist_delta) if not np.isnan(hist_delta) else 0.0,
        rsi=float(data["rsi"][idx]),
        rsi_delta=float(rsi_delta) if not np.isnan(rsi_delta) else 0.0,
        bb_middle=float(data["bb_middle"][idx]) if not np.isnan(data["bb_middle"][idx]) else close_val,
        bb_upper=float(data["bb_upper"][idx]) if not np.isnan(data["bb_upper"][idx]) else close_val,
        bb_lower=float(data["bb_lower"][idx]) if not np.isnan(data["bb_lower"][idx]) else close_val,
        bb_position=float(data["bb_position"][idx]) if not np.isnan(data["bb_position"][idx]) else 0.5,
        atr=atr_val,
        atr_pct=atr_pct,
        ma20=float(data["ma20"][idx]) if not np.isnan(data["ma20"][idx]) else close_val,
        adx=float(data["adx"][idx]) if not np.isnan(data["adx"][idx]) else 0.0,
    )


def bars_to_arrays(bars: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opens = np.array([float(b["open"]) for b in bars], dtype=float)
    highs = np.array([float(b["high"]) for b in bars], dtype=float)
    lows = np.array([float(b["low"]) for b in bars], dtype=float)
    closes = np.array([float(b["close"]) for b in bars], dtype=float)
    volumes = np.array([float(b.get("tick_volume", b.get("volume", 0))) for b in bars], dtype=float)
    return opens, highs, lows, closes, volumes


def latest_atr_from_bars(bars: list[dict], atr_period: int = 14) -> float:
    """Latest ATR from OHLCV bar dicts (shared by trade log and pipeline)."""
    if len(bars) < atr_period + 2:
        return 0.0
    _, highs, lows, closes, _ = bars_to_arrays(bars)
    atr = compute_atr(highs, lows, closes, atr_period)
    val = atr[-1]
    return float(val) if np.isfinite(val) else 0.0
