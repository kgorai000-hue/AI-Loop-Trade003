from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.mt5_connector import MT5Connector
from src.data.store import OHLCVStore
from src.survival.catalog import DEATH_MODES, DIAGNOSTIC_ORDER
from src.survival.diagnostics import SurvivalDiagnostics
from src.survival.reporting import print_survival_report


def print_lesson_content() -> None:
    print("\n=== Appendix B: 12 Ways Quant Systems Die ===")
    print("  Advanced course teaches failure modes, not just profits.\n")

    print("=== 12 Death Modes ===")
    for mode in DEATH_MODES:
        print(f"  #{mode.mode_id:2} {mode.name_ja} ({mode.name})")
        print(f"      -> existing: {mode.source_module}")

    print("\n=== Diagnostic Order (troubleshoot in this sequence) ===")
    for idx, mode_id in enumerate(DIAGNOSTIC_ORDER, 1):
        mode = next(m for m in DEATH_MODES if m.mode_id == mode_id)
        print(f"  {idx:2}. {mode.name_ja}")

    print("\n=== Prevention (sample - see catalog for full list) ===")
    for mode in DEATH_MODES[:3]:
        print(f"  #{mode.mode_id} {mode.name_ja}")
        for item in mode.prevention[:2]:
            print(f"      [ ] {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Appendix B death modes diagnostic report")
    parser.add_argument("--paper-only", action="store_true", help="Show death mode catalog only")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Skip lesson catalog (for scheduled log output)",
    )
    args = parser.parse_args()

    if not args.compact:
        print_lesson_content()
    if args.paper_only:
        return 0

    config = load_config()
    connector = MT5Connector(config)
    store = OHLCVStore(config.storage.path)
    connected = False
    try:
        connector.connect()
        connected = connector.is_connected
    except ConnectionError as exc:
        print(f"WARNING: MT5 not available: {exc}")

    diag = SurvivalDiagnostics(config)
    report = diag.assess(connector=connector if connected else None, store=store)

    if connected:
        connector.disconnect()

    print_survival_report(report, compact=args.compact)

    return 1 if report.failed_modes else 0


if __name__ == "__main__":
    raise SystemExit(main())
