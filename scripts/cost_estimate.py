from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.core.types import SignalSide
from src.data.spread_store import SpreadStore, snapshot_from_market_info
from src.data.store import OHLCVStore
from src.execution.cost_model import CostModel
from src.market.symbol_info import fetch_market_symbol_info


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate trading costs (Lesson 02)")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol for per-trade estimate")
    parser.add_argument("--lots", type=float, help="Lot size")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Account capital in JPY")
    parser.add_argument("--backtest-return", type=float, default=35.0, help="Backtest annual return percent")
    parser.add_argument("--trades-per-day", type=int, help="Override trades per day")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level)
    lots = args.lots or config.trading.default_lots

    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    spread_store = SpreadStore(config.storage.path)
    cost_model = CostModel(config)

    try:
        connector.connect()

        print("\n=== Trading Profile ===")
        print(f"  profile         : {config.trading.profile}")
        print(f"  trades/day      : {args.trades_per_day or config.trading.trades_per_day}")
        print(f"  primary TF      : {config.trading.primary_timeframe}")
        print(f"  slippage rate   : {config.costs.slippage_rate:.4%}")

        print("\n=== Per-Trade Cost Estimate ===")
        info = fetch_market_symbol_info(connector, config, args.symbol)
        spread_store.save_snapshots([snapshot_from_market_info(info)])

        bars = store.get_recent_bars(info.symbol, config.trading.primary_timeframe, 30)
        closes = [float(b["close"]) for b in bars]
        daily_vol = cost_model.estimate_daily_volatility(closes)

        for side in (SignalSide.BUY, SignalSide.SELL):
            cost = cost_model.estimate_trade_cost(
                symbol_info=info,
                lots=lots,
                side=side,
                daily_volatility=daily_vol,
            )
            print(f"\n  [{side.value.upper()}] {info.symbol} x {lots} lots")
            print(f"    notional JPY     : {cost.notional_jpy:,.2f}")
            print(f"    spread cost      : {cost.spread_cost_jpy:,.2f}")
            print(f"    slippage cost    : {cost.slippage_cost_jpy:,.2f}")
            print(f"    commission cost  : {cost.commission_cost_jpy:,.2f}")
            print(f"    market impact    : {cost.market_impact_cost_jpy:,.2f}")
            print(f"    total cost       : {cost.total_cost_jpy:,.2f} ({cost.total_cost_pct_of_notional:.4f}%)")

        buy_cost = cost_model.estimate_trade_cost(info, lots, SignalSide.BUY, daily_vol)
        trade_notional = buy_cost.notional_jpy

        print("\n=== Strategy Cost Projection (Lesson 02 paper exercise style) ===")
        projection = cost_model.project_strategy_costs(
            backtest_return_pct=args.backtest_return,
            capital_jpy=args.capital,
            trade_notional_jpy=trade_notional,
            cost_per_trade_pct=buy_cost.total_cost_pct_of_notional,
            trades_per_day=args.trades_per_day,
        )
        print(f"  capital JPY              : {projection.capital_jpy:,.0f}")
        print(f"  trade notional JPY       : {projection.trade_notional_jpy:,.2f}")
        print(f"  trades/day               : {projection.trades_per_day}")
        print(f"  trading days/year        : {projection.trading_days}")
        print(f"  backtest return          : {projection.backtest_return_pct:.2f}%")
        print(f"  annual cost (on notional): {projection.annual_cost_on_notional_pct:.4f}%")
        print(f"  annual cost (on capital) : {projection.annual_cost_on_capital_pct:.2f}%")
        print(f"  estimated live return    : {projection.live_return_pct:.2f}%")

        print("\n=== Spread Snapshot Saved ===")
        latest = spread_store.latest_by_symbol()
        row = next((r for r in latest if r["symbol"] == info.symbol), None)
        if row:
            print(f"  {row['symbol']}: spread={row['spread_points']}pt bid={row['bid']} ask={row['ask']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
