from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ParameterSpec:
  """One parameter loop target (change only this vs baseline)."""

  name: str
  strategy: str
  priority: int
  values: tuple[Any, ...]
  apply: Callable[[Any], list[tuple[str, Any]]]
  description: str = ""


def _frange(start: float, stop: float, step: float) -> tuple[float, ...]:
  values: list[float] = []
  current = start
  while current <= stop + 1e-9:
    values.append(round(current, 4))
    current += step
  return tuple(values)


def _irange(start: int, stop: int, step: int) -> tuple[int, ...]:
  return tuple(range(start, stop + 1, step))


def default_parameter_specs() -> list[ParameterSpec]:
  """Exploration order from loop criteria PDF."""
  return [
    ParameterSpec(
      name="signal_score_threshold",
      strategy="feature_score",
      priority=1,
      values=_frange(0.05, 0.30, 0.05),
      apply=lambda v: [("indicators.signal_score_threshold", v)],
      description="Feature score entry threshold",
    ),
    ParameterSpec(
      name="adx_trend_threshold",
      strategy="trend_following",
      priority=2,
      values=tuple(float(x) for x in _irange(18, 30, 2)),
      apply=lambda v: [("strategies.adx_trend_threshold", v)],
      description="ADX filter for trend strategy",
    ),
    ParameterSpec(
      name="rsi_oversold_overbought_25_75",
      strategy="mean_reversion",
      priority=3,
      values=((25.0, 75.0),),
      apply=lambda v: [
        ("strategies.mr_rsi_oversold", v[0]),
        ("strategies.mr_rsi_overbought", v[1]),
      ],
      description="RSI mean-reversion band (25/75)",
    ),
    ParameterSpec(
      name="rsi_oversold_overbought_35_65",
      strategy="mean_reversion",
      priority=4,
      values=((35.0, 65.0),),
      apply=lambda v: [
        ("strategies.mr_rsi_oversold", v[0]),
        ("strategies.mr_rsi_overbought", v[1]),
      ],
      description="RSI mean-reversion band (35/65)",
    ),
    ParameterSpec(
      name="bb_entry_20_80",
      strategy="mean_reversion",
      priority=5,
      values=((0.20, 0.80),),
      apply=lambda v: [
        ("strategies.mr_bb_entry_low", v[0]),
        ("strategies.mr_bb_entry_high", v[1]),
      ],
      description="BB entry band (0.20/0.80)",
    ),
    ParameterSpec(
      name="bb_entry_10_90",
      strategy="mean_reversion",
      priority=6,
      values=((0.10, 0.90),),
      apply=lambda v: [
        ("strategies.mr_bb_entry_low", v[0]),
        ("strategies.mr_bb_entry_high", v[1]),
      ],
      description="BB entry band (0.10/0.90)",
    ),
    ParameterSpec(
      name="adx_sideways_threshold",
      strategy="trend_following",
      priority=7,
      values=tuple(float(x) for x in (15, 18, 20, 22, 24)),
      apply=lambda v: [("strategies.adx_sideways_threshold", v)],
      description="ADX sideways flatten threshold",
    ),
    ParameterSpec(
      name="trading_profile",
      strategy="feature_score",
      priority=8,
      values=("low", "medium", "high"),
      apply=lambda v: [("trading.profile", v)],
      description="Trading profile (affects trades/day; backtest reference)",
    ),
  ]


def specs_for_strategy(strategy: str) -> list[ParameterSpec]:
  return sorted(
    [s for s in default_parameter_specs() if s.strategy == strategy],
    key=lambda s: s.priority,
  )
