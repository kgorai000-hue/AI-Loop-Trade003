from __future__ import annotations

from src.survival.catalog import DEATH_MODE_BY_ID
from src.survival.types import SurvivalReport


def print_survival_report(
    report: SurvivalReport,
    *,
    title: str = "Appendix B Survival Diagnostic",
    compact: bool = False,
    warning_limit: int = 8,
) -> None:
    print(f"\n=== {title} ===")
    if compact:
        print(f"  overall healthy   : {report.overall_healthy}")
        print(f"  failed modes      : {len(report.failed_modes)}")
        for mode in report.modes:
            if not mode.healthy:
                print(f"  FAIL #{mode.mode_id} {mode.name}: {mode.detail}")
        failed_weekly = [w for w in report.weekly_checklist if not w.passed]
        if failed_weekly:
            print(f"  weekly failures   : {len(failed_weekly)}")
            for item in failed_weekly[:3]:
                print(f"    - {item.name}: {item.detail}")
        return

    status = "HEALTHY" if report.overall_healthy else "ATTENTION"
    print(f"  overall           : {status}")
    print(f"  failed modes      : {len(report.failed_modes)}")

    print("\n=== Mode Status (diagnostic order) ===")
    for mode in report.modes:
        flag = "OK" if mode.healthy else "FAIL"
        info = DEATH_MODE_BY_ID[mode.mode_id]
        print(f"  [{flag:4}] #{mode.mode_id:2} {info.name_ja:16} | {mode.detail}")

    print("\n=== Weekly Health Checklist ===")
    for item in report.weekly_checklist:
        flag = "OK" if item.passed else "FAIL"
        print(f"  [{flag:4}] {item.name:40} | {item.detail}")

    if report.warnings:
        print("\n=== Warnings ===")
        for warning in report.warnings[:warning_limit]:
            print(f"  - {warning}")
