#!/usr/bin/env python3
"""AI-Loop-Trade003 entry point — demo-live multi-agent pipeline + intelligence loop."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.intelligence.loop import IntelligenceLoop, apply_state_overrides
from src.system.runner import TradingSystem


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _validate_stage(config) -> str | None:
    stage = str(config.project.graduation_stage).lower()
    dry = bool(config.trading.dry_run)
    account = str(config.account_type).lower()

    if stage == "paper" and not dry:
        return "ERROR: graduation_stage=paper requires trading.dry_run=true"
    if stage == "demo_live":
        if dry:
            return "ERROR: graduation_stage=demo_live requires trading.dry_run=false"
        if account != "demo":
            return "ERROR: graduation_stage=demo_live requires broker.account_type=demo"
        if not config.mt5.require_demo:
            return "ERROR: graduation_stage=demo_live requires mt5.require_demo=true"
    if stage in {"small_live", "scale_up"} and account == "demo":
        return None  # still demo-safe
    return None


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config()
    setup_logging(config.log_level)

    err = _validate_stage(config)
    if err:
        print(err)
        return 1

    # Single-symbol runs apply adopted STATE params.
    if args.symbol and len(args.symbol) == 1:
        config = apply_state_overrides(config, args.symbol[0])

    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    system = TradingSystem(config, connector, store)

    try:
        connector.connect()
        result = system.run(
            symbols=args.symbol,
            sync_first=args.sync or config.project.sync_before_run,
        )

        print("\n=== AI-Loop-Trade003 Demo-Live Pipeline ===")
        print(f"  graduation stage  : {result.integration.graduation_stage}")
        print(f"  dry_run             : {config.trading.dry_run}")
        print(f"  data ready          : {result.integration.data_ready}")
        print(f"  pipeline ok         : {result.integration.pipeline_ok}")
        print(f"  ready for paper     : {result.integration.ready_for_paper}")
        print(f"  signals             : {len(result.pipeline.signals)}")
        print(f"  execution plans     : {len(result.pipeline.execution_plans)}")
        tickets = [
            getattr(p, "ticket", None)
            for p in result.pipeline.execution_plans
            if getattr(p, "ticket", None)
        ]
        if tickets:
            print(f"  MT5 tickets         : {tickets}")
        print(f"  agents registered   : {len(result.integration.agents)}")
        print(f"  asset groups        : {list(config.asset_groups.keys())}")

        if result.integration.warnings:
            print("\n  Warnings:")
            for warning in result.integration.warnings[:5]:
                print(f"    - {warning}")

        failed = [c for c in result.integration.pre_live_checklist if not c.passed]
        if failed:
            print(f"\n  Checklist items pending ({len(failed)}):")
            for item in failed[:5]:
                print(f"    [{item.category}] {item.item}: {item.detail}")

        return 0 if result.integration.pipeline_ok else 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


def cmd_sync(args: argparse.Namespace) -> int:
    config = load_config()
    setup_logging(config.log_level)
    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    system = TradingSystem(config, connector, store)

    try:
        connector.connect()
        stored = system.sync_data(args.symbol)
        print(f"Synced {stored} bars")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        connector.disconnect()


def cmd_status(args: argparse.Namespace) -> int:
    from src.system.report import print_system_status

    return print_system_status(paper_only=args.paper_only)


def _resolve_symbols(config, symbols: list[str] | None) -> list[str]:
    if symbols:
        return symbols
    # Default optimize/review universe: every tradeable Asset Group member.
    return list(config.tradeable_symbols_all_groups() or config.symbols)


def cmd_optimize(args: argparse.Namespace) -> int:
    """Maker->Checker->Validator for one or more symbols (grid fallback without API key)."""
    config = load_config()
    setup_logging(config.log_level)

    if not config.intelligence.enabled:
        print("ERROR: intelligence.enabled is false in settings.yaml")
        return 1

    store = OHLCVStore(config.storage.path)
    strategy = args.strategy or config.intelligence.default_strategy
    timeframe = (args.timeframe or config.trading.primary_timeframe or "M30").upper()
    symbols = _resolve_symbols(config, args.symbol)

    results = []
    exit_code = 0
    for symbol in symbols:
        print(f"\n=== optimize {symbol} {timeframe} ({strategy}) ===")
        loop = IntelligenceLoop(
            config,
            store,
            symbol=symbol,
            strategy=strategy,
            timeframe=timeframe,
        )
        outcome = loop.run()
        payload = {
            "symbol": outcome.symbol,
            "strategy": outcome.strategy,
            "timeframe": outcome.timeframe,
            "path": outcome.path,
            "accepted": outcome.accepted,
            "params": outcome.params,
            "metrics": outcome.metrics,
            "message": outcome.message,
            "trials": len(outcome.trials),
        }
        results.append(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        if not outcome.accepted and outcome.path == "error":
            exit_code = 1

    if args.pairs:
        for pair in config.strategies.pairs:
            if len(pair) != 2:
                continue
            pair_id = config.pair_state_id(pair[0], pair[1])
            print(f"\n=== optimize pair {pair_id} ===")
            loop = IntelligenceLoop(
                config,
                store,
                symbol=pair[0],
                strategy=strategy,
                timeframe=timeframe,
                state_key=pair_id,
            )
            outcome = loop.run()
            payload = {
                "pair_id": pair_id,
                "symbol": outcome.symbol,
                "accepted": outcome.accepted,
                "path": outcome.path,
                "message": outcome.message,
            }
            results.append(payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    if args.json and len(results) > 1:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return exit_code


def cmd_loop(args: argparse.Namespace) -> int:
    """Resident M30 poll + weekend review/optimize loop (demo-live orders)."""
    from src.intelligence.resident import ResidentLoopEngine

    config = load_config()
    setup_logging(config.log_level)

    err = _validate_stage(config)
    if err:
        print(err)
        return 1

    if not config.intelligence.enabled:
        print("ERROR: intelligence.enabled is false in settings.yaml")
        return 1

    engine = ResidentLoopEngine(
        config,
        symbols=args.symbol,
        strategy=args.strategy,
        timeframe=args.timeframe,
    )
    engine.run_forever()
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Force weekend review sub-loop once (degraded -> optimize)."""
    from src.intelligence.resident import ResidentLoopEngine

    config = load_config()
    setup_logging(config.log_level)

    if not config.intelligence.enabled:
        print("ERROR: intelligence.enabled is false in settings.yaml")
        return 1

    engine = ResidentLoopEngine(
        config,
        symbols=args.symbol,
        strategy=args.strategy,
        timeframe=args.timeframe,
    )
    try:
        engine.start()
        outcomes = engine.review_subloop()
        print(json.dumps(outcomes, ensure_ascii=False, indent=2, default=str))
        return 0 if all(o.get("ok") for o in outcomes) else 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        engine.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI-Loop-Trade003 - FxPro demo-live + asset groups + Maker/Checker/Validator"
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run full agent pipeline (demo-live when configured)")
    run_p.add_argument("--symbol", action="append", help="Limit to symbols")
    run_p.add_argument("--sync", action="store_true", help="Sync MT5 data before run")
    run_p.set_defaults(func=cmd_run)

    sync_p = sub.add_parser("sync", help="Sync MT5 OHLCV data")
    sync_p.add_argument("--symbol", action="append")
    sync_p.set_defaults(func=cmd_sync)

    status_p = sub.add_parser("status", help="Show architecture and checklist")
    status_p.add_argument("--paper-only", action="store_true")
    status_p.set_defaults(func=cmd_status)

    opt_p = sub.add_parser(
        "optimize",
        help="Maker->Checker->Validator param update (grid fallback if no API key)",
    )
    opt_p.add_argument(
        "--symbol",
        action="append",
        help="Symbol(s); default=all Asset Groups (tradeable universe)",
    )
    opt_p.add_argument("--strategy", default=None, help="Backtest strategy name")
    opt_p.add_argument("--timeframe", default="M30", help="Timeframe (default M30)")
    opt_p.add_argument("--pairs", action="store_true", help="Also optimize within-group pairs")
    opt_p.add_argument("--json", action="store_true", help="Emit combined JSON for multi-symbol")
    opt_p.set_defaults(func=cmd_optimize)

    loop_p = sub.add_parser(
        "loop",
        help="Resident M30 poll + weekend review/optimize (Ctrl+C to stop)",
    )
    loop_p.add_argument("--symbol", action="append", help="Optional subset of symbols")
    loop_p.add_argument("--strategy", default=None)
    loop_p.add_argument("--timeframe", default="M30")
    loop_p.set_defaults(func=cmd_loop)

    rev_p = sub.add_parser(
        "review",
        help="Force review once (degraded metrics -> optimize)",
    )
    rev_p.add_argument("--symbol", action="append", help="Optional subset of symbols")
    rev_p.add_argument("--strategy", default=None)
    rev_p.add_argument("--timeframe", default="M30")
    rev_p.set_defaults(func=cmd_review)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
