from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT
from src.ops.types import AlertSeverity, MonitorLevel

logger = logging.getLogger(__name__)


def new_trace_id(prefix: str = "tx") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


class StructuredTradeLogger:
    """Structured JSON trade logs with trace_id (Lesson 20.2)."""

    def __init__(self, log_dir: str | Path = "logs/structured") -> None:
        self.log_dir = Path(log_dir)
        if not self.log_dir.is_absolute():
            self.log_dir = PROJECT_ROOT / self.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        *,
        level: str,
        service: str,
        event: str,
        trace_id: str,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "service": service,
            "trace_id": trace_id,
            "event": event,
            "data": data,
            "context": context or {},
        }
        path = self.log_dir / f"{service}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        log_fn = getattr(logger, level.lower(), logger.info)
        log_fn("[%s] %s trace=%s", service, event, trace_id)
        return record
