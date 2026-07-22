from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.decision_agent import DecisionAgent
from src.agents.portfolio_agent import PortfolioAgent
from src.agents.regime_agent import RegimeAgent
from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.execution.position_sizing import half_kelly, risk_parity_weights, van_tharp_cap_pct


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def print_lesson_scenario() -> None:
    """Lesson 10 scenario exercise: risk parity with single-position cap."""
    print("\n=== Lesson 10 Scenario (Risk Parity) ===")
    vols = {"AAPL": 0.25, "TSLA": 0.50, "MSFT": 0.20}
    weights = risk_parity_weights(vols)
    capital = 100_000.0
    max_single = 0.20

    print("  Predictions: AAPL +1.2%, TSLA +0.8%, MSFT +0.5%")
    print("  Volatility : AAPL 25%, TSLA 50%, MSFT 20%")
    print("  Capital    : $100,000 | max single 20%\n")

    total_alloc = 0.0
    for symbol, weight in weights.items():
        raw = capital * weight
        capped = min(raw, capital * max_single)
        total_alloc += capped
        print(f"  {symbol:5} weight={weight:5.1%} raw=${raw:,.0f} capped=${capped:,.0f}")

    print(f"\n  Total exposure: ${total_alloc:,.0f} ({total_alloc / capital:.1%})")

    print("\n=== Half-Kelly + Van Tharp (defaults) ===")
    kelly = half_kelly(0.55, 1.5)
    van_tharp = van_tharp_cap_pct(100_000, 0.01, 10.0, 200.0)
    print(f"  Half-Kelly cap     : {kelly:.2%}")
    print(f"  Van Tharp cap      : {van_tharp:.2%}")
    print(f"  Hard cap (5%)      : 5.00%")
    print(f"  Final (min)        : {min(kelly, van_tharp, 0.05):.2%}  # AAPL $200 example")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prediction-to-decision report (Lesson 10)"
    )
    parser.add_argument("--symbol", action="append", help="Limit to specific symbols")
    parser.add_argument("--scenario-only", action="store_true", help="Print book scenario only")
    args = parser.parse_args()

    print_lesson_scenario()
    if args.scenario_only:
        return 0

    config = load_config()
    setup_logging(config.log_level)

    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    regime_agent = RegimeAgent(config, store)
    portfolio_agent = PortfolioAgent(config, store)
    decision_agent = DecisionAgent(config, connector, store)

    try:
        connector.connect()
        symbols = args.symbol or config.symbols

        regime_map = {sym: regime_agent.assess(sym) for sym in symbols}
        regime_map = {k: v for k, v in regime_map.items() if v is not None}

        raw_signals = portfolio_agent.scan(symbols, regime_map)
        sized_signals, reports = decision_agent.decide(raw_signals)
        state = decision_agent.build_state()

        print(f"\n=== Agent State ===")
        print(f"  equity            : {state.equity:,.2f}")
        print(f"  free margin       : {state.free_margin:,.2f}")
        print(f"  exposure          : {state.current_exposure_pct:.1f}%")
        print(f"  open positions    : {len(state.open_position_lots)}")

        print(f"\n=== Raw Signals ({len(raw_signals)}) ===")
        for signal in raw_signals:
            pred = signal.predicted_return if signal.predicted_return is not None else signal.strength * 0.01
            conf = signal.confidence if signal.confidence is not None else signal.strength
            print(
                f"  {signal.symbol:12} {signal.side.value:4} pred={pred:+.3%} "
                f"conf={conf:.2f} | {signal.strategy.value}"
            )

        print(f"\n=== Decisions ({len(reports)}) ===")
        for report in reports:
            print(
                f"  {report.symbol:12} {report.side.value:4} "
                f"lots={report.requested_lots:.2f} final={report.final_position_pct:.2f}%"
            )
            print(
                f"    kelly={report.half_kelly_cap_pct:.1f}% "
                f"van_tharp={report.van_tharp_cap_pct:.1f}% "
                f"rp={report.portfolio_weight_pct:.1f}%"
            )
            print(f"    {report.reason}")

        print(f"\n=== Pipeline Handoff ===")
        for signal in sized_signals:
            lots = signal.requested_lots if signal.requested_lots is not None else 0.0
            print(f"  -> RiskAgent review: {signal.symbol} requested_lots={lots:.2f}")

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
