from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extensions.distributed import EXTRACTION_CRITERIA, ServiceBoundary
from src.extensions.hft import LatencyBudget


def test_course_summary_constants_importable() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "course_summary",
        PROJECT_ROOT / "scripts" / "course_summary.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert len(module.CORE_INSIGHTS) == 5
    assert len(module.MISCONCEPTIONS) >= 8
    assert len(module.KNOWLEDGE_QUIZ) == 5


def test_extension_stubs_exist() -> None:
    assert ServiceBoundary.RISK_RUST.value == "risk_rust"
    assert len(EXTRACTION_CRITERIA) >= 3
    gap = LatencyBudget.gap_report()
    assert "risk_check_ms" in gap
