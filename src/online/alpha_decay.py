from __future__ import annotations

import math


def projected_ic(initial_ic: float, monthly_decay_rate: float, months: int) -> float:
    """IC after N months with compound monthly decay (Lesson 17.1)."""
    return max(0.0, initial_ic * ((1.0 - monthly_decay_rate) ** months))


def implied_monthly_decay(initial_ic: float, final_ic: float, months: int) -> float:
    if initial_ic <= 0 or months <= 0 or final_ic <= 0:
        return 0.0
    ratio = final_ic / initial_ic
    return 1.0 - (ratio ** (1.0 / months))


def expected_annual_return_from_ic(ic: float, sigma: float = 0.20, trading_days: int = 252) -> float:
    """Approximate ann return = IC * sqrt(252) * sigma (Lesson 17.1)."""
    return ic * math.sqrt(trading_days) * sigma


def months_until_ic_below(initial_ic: float, threshold: float, monthly_decay_rate: float) -> int | None:
    if initial_ic <= threshold or monthly_decay_rate <= 0:
        return 0 if initial_ic <= threshold else None
    ratio = threshold / initial_ic
    if ratio <= 0:
        return None
    months = math.log(ratio) / math.log(1.0 - monthly_decay_rate)
    return max(0, int(math.ceil(months)))
