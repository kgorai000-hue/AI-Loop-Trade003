from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.system.report import print_system_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Project integration report (Lesson 21)")
    parser.add_argument("--paper-only", action="store_true")
    args = parser.parse_args()
    return print_system_status(paper_only=args.paper_only)


if __name__ == "__main__":
    raise SystemExit(main())
