from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.core.config import PROJECT_ROOT


@dataclass
class InterventionRecord:
    timestamp: int
    operator: str
    action: str
    symbol: str
    justification: str
    outcome: str = "pending"


class HumanInterventionLog:
    """
    Appendix B mode #9 stub: manual override audit trail.
    Not wired into live trading — log interventions for post-review.
    """

    def __init__(self, log_dir: str | Path = "logs/structured") -> None:
        path = Path(log_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        self.log_path = path / "manual_interventions.jsonl"

    def record(self, record: InterventionRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def recent_count(self, days: int = 7) -> int:
        if not self.log_path.exists():
            return 0
        cutoff = int(time.time()) - days * 86400
        count = 0
        with self.log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(row.get("timestamp", 0)) >= cutoff:
                    count += 1
        return count

    def recent(self, days: int = 7, limit: int = 20) -> list[InterventionRecord]:
        if not self.log_path.exists():
            return []
        cutoff = int(time.time()) - days * 86400
        rows: list[InterventionRecord] = []
        with self.log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(row.get("timestamp", 0)) < cutoff:
                    continue
                rows.append(InterventionRecord(**row))
        return rows[-limit:]
