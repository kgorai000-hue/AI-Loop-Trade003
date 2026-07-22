from __future__ import annotations

import math

import pytest

from src.execution.costs.impact import almgren_chriss_total_cost, permanent_impact, temporary_impact
from src.execution.costs.slippage import estimate_slippage_from_orderbook, linear_slippage, sqrt_slippage
from src.execution.costs.tradability import (
    evaluate_strategy_tradability,
    fill_probability,
    gross_to_net_alpha,
    is_tradable,
    opportunity_cost,
    turnover_annual_cost,
)
from src.execution.costs.types import OrderBookLevel


def test_linear_slippage_scales_with_participation() -> None:
    small = linear_slippage(1_000_000, 100_000_000, k=0.3)
    large = linear_slippage(5_000_000, 100_000_000, k=0.3)
    assert large > small
    assert math.isclose(small, 0.003)


def test_sqrt_slippage_sublinear() -> None:
    linear = linear_slippage(4_000_000, 100_000_000, k=0.3)
    sqrt = sqrt_slippage(4_000_000, 100_000_000, sigma=0.02, k=1.0)
    assert sqrt < linear


def test_orderbook_slippage_increases_with_size() -> None:
    bids = [OrderBookLevel(100.0, 1000), OrderBookLevel(99.9, 2000)]
    asks = [OrderBookLevel(100.1, 1000), OrderBookLevel(100.2, 2000)]
    small = estimate_slippage_from_orderbook(500, bids, asks, side="buy")
    large = estimate_slippage_from_orderbook(1500, bids, asks, side="buy")
    assert large > small
    assert small >= 0


def test_almgren_chriss_faster_trade_higher_temp_impact() -> None:
    slow = almgren_chriss_total_cost(participation=0.02, sigma=0.02, urgency=0.5)
    fast = almgren_chriss_total_cost(participation=0.02, sigma=0.02, urgency=2.0)
    assert fast["temporary_impact"] > slow["temporary_impact"]
    assert slow["volatility_risk"] > fast["volatility_risk"]


def test_turnover_shredder_kills_alpha() -> None:
    report = evaluate_strategy_tradability(
        "high_turnover",
        gross_alpha_pct=15.0,
        turnover_pct=3.0,
        cost_per_trade_pct=0.08,
        trading_days=252,
    )
    assert report.annual_cost_pct > report.gross_alpha_pct
    assert not report.tradable


def test_gross_to_net_alpha() -> None:
    net = gross_to_net_alpha(0.5, 0.15)
    assert math.isclose(net, 0.35)
    assert is_tradable(net)


def test_opportunity_cost_grows_with_delay() -> None:
    short = opportunity_cost(0.5, 120.0, 5.0)
    long = opportunity_cost(0.5, 120.0, 60.0)
    assert long > short > 0


def test_fill_probability_in_valid_range() -> None:
    prob = fill_probability(0.1, 0.02, wait_hours=4.0)
    assert 0.0 <= prob <= 1.0


def test_turnover_annual_cost_formula() -> None:
    annual = turnover_annual_cost(3.0, 0.08, trading_days=252)
    assert math.isclose(annual, 3.0 * 0.08 * 252)


def test_impact_components_non_negative() -> None:
    assert temporary_impact(0.1) >= 0
    assert permanent_impact(0.05) >= 0
