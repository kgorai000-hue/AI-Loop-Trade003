from __future__ import annotations

from enum import Enum

import numpy as np


class DivergenceType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


def _local_peaks(values: np.ndarray, order: int = 3) -> list[int]:
    peaks: list[int] = []
    for idx in range(order, len(values) - order):
        window = values[idx - order : idx + order + 1]
        if values[idx] == np.max(window):
            peaks.append(idx)
    return peaks


def _local_troughs(values: np.ndarray, order: int = 3) -> list[int]:
    troughs: list[int] = []
    for idx in range(order, len(values) - order):
        window = values[idx - order : idx + order + 1]
        if values[idx] == np.min(window):
            troughs.append(idx)
    return troughs


def detect_divergence(
    prices: np.ndarray,
    indicator: np.ndarray,
    lookback: int = 30,
) -> DivergenceType | None:
    if len(prices) < lookback or len(indicator) < lookback:
        return None

    prices = prices[-lookback:]
    indicator = indicator[-lookback:]

    peaks_p = _local_peaks(prices)
    peaks_i = _local_peaks(indicator)
    if len(peaks_p) >= 2 and len(peaks_i) >= 2:
        p1, p2 = peaks_p[-2], peaks_p[-1]
        i1, i2 = peaks_i[-2], peaks_i[-1]
        if prices[p2] > prices[p1] and indicator[i2] < indicator[i1]:
            return DivergenceType.BEARISH

    troughs_p = _local_troughs(prices)
    troughs_i = _local_troughs(indicator)
    if len(troughs_p) >= 2 and len(troughs_i) >= 2:
        p1, p2 = troughs_p[-2], troughs_p[-1]
        i1, i2 = troughs_i[-2], troughs_i[-1]
        if prices[p2] < prices[p1] and indicator[i2] > indicator[i1]:
            return DivergenceType.BULLISH

    return None
