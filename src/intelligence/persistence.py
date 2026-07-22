"""STATE.md / SKILL.md persistence for per-symbol adopted AppConfig params."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.intelligence.params import LoopParams, symbol_to_state_key

logger = logging.getLogger(__name__)

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
BACKUP_KEEP = 5


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class StateStore:
    """Per-symbol YAML-in-Markdown state for Trade002 intelligence loop."""

    def __init__(self, state_dir: str | Path, symbol: str) -> None:
        self.symbol = symbol
        self.state_key = symbol_to_state_key(symbol)
        self.root = Path(state_dir) / self.state_key
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "STATE.md"
        self.skill_path = self.root / "SKILL.md"
        self._lock = _lock_for(self.root)
        self._ensure_files()

    def _default_state(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": "M30",
            "updated_at": None,
            "params": {},
            "last_metrics": {},
            "accepted": False,
            "strategy": "feature_score",
            "path": None,
            "last_maker_run": None,
            "last_review_date": None,
        }

    def _ensure_files(self) -> None:
        with self._lock:
            if not self.state_path.exists():
                self._write_state_unlocked(self._default_state(), backup=False)
            if not self.skill_path.exists():
                _atomic_write_text(
                    self.skill_path,
                    (
                        f"# SKILL — {self.state_key}\n\n"
                        "Lessons from Checker rejections and Validator failures.\n"
                        "Maker reads this before the next search.\n\n"
                        "## Lessons\n\n- (none yet)\n"
                    ),
                )

    @staticmethod
    def _extract_yaml_block(text: str) -> tuple[dict[str, Any], bool]:
        match = re.search(r"```ya?ml\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        raw = match.group(1) if match else text
        try:
            data = yaml.safe_load(raw)
        except Exception as exc:
            logger.error("Failed to parse STATE YAML: %s", exc)
            return {}, False
        if not isinstance(data, dict):
            return {}, False
        return data, True

    def _rotate_backup_unlocked(self) -> None:
        if not self.state_path.exists():
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        backup = self.root / f"STATE.md.{ts}.bak"
        try:
            shutil.copy2(self.state_path, backup)
        except OSError:
            return
        backups = sorted(self.root.glob("STATE.md.*.bak"), reverse=True)
        for stale in backups[BACKUP_KEEP:]:
            try:
                stale.unlink()
            except OSError:
                pass

    def _read_state_unlocked(self) -> dict[str, Any]:
        try:
            text = self.state_path.read_text(encoding="utf-8")
        except OSError:
            return self._default_state()
        state, ok = self._extract_yaml_block(text)
        if not ok:
            return self._default_state()
        defaults = self._default_state()
        for key, value in defaults.items():
            state.setdefault(key, value)
        return state

    def read_state(self) -> dict[str, Any]:
        with self._lock:
            return self._read_state_unlocked()

    def _write_state_unlocked(self, state: dict[str, Any], *, backup: bool = True) -> None:
        state = dict(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        body = yaml.safe_dump(state, sort_keys=False, allow_unicode=True)
        content = f"# STATE — {self.state_key}\n\n```yaml\n{body}```\n"
        if backup:
            self._rotate_backup_unlocked()
        _atomic_write_text(self.state_path, content)

    def update_state(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read_state_unlocked()
            for key, value in kwargs.items():
                if isinstance(value, dict) and isinstance(state.get(key), dict):
                    merged = dict(state[key])
                    merged.update(value)
                    state[key] = merged
                else:
                    state[key] = value
            self._write_state_unlocked(state, backup=True)
            return state

    def get_params(self) -> LoopParams:
        state = self.read_state()
        raw = state.get("params") or {}
        if not isinstance(raw, dict):
            return LoopParams()
        return LoopParams(overrides={str(k): v for k, v in raw.items()})

    def skills_text(self, max_lessons: int = 40) -> str:
        lessons = self.read_skills()[-max_lessons:]
        if not lessons:
            return "(none yet)"
        return "\n".join(f"- {x}" for x in lessons)

    def read_skills(self) -> list[str]:
        with self._lock:
            if not self.skill_path.exists():
                return []
            text = self.skill_path.read_text(encoding="utf-8")
            lessons: list[str] = []
            in_lessons = False
            for line in text.splitlines():
                if line.strip().lower().startswith("## lessons"):
                    in_lessons = True
                    continue
                if in_lessons:
                    if line.startswith("## "):
                        break
                    match = re.match(r"\s*-\s+(.*)", line)
                    if match:
                        item = match.group(1).strip()
                        if item and item != "(none yet)":
                            lessons.append(item)
            return lessons

    def append_lesson(self, lesson: str) -> None:
        lesson = lesson.strip()
        if not lesson:
            return
        with self._lock:
            existing = self.read_skills()
            bare = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", lesson)
            for item in existing:
                if re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", item) == bare:
                    return
            existing.append(lesson)
            existing = existing[-200:]
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            lines = [
                f"# SKILL — {self.state_key}",
                "",
                "Lessons from Checker rejections and Validator failures.",
                "Maker reads this before the next search.",
                "",
                "## Lessons",
                "",
            ]
            for item in existing:
                if item.startswith("["):
                    lines.append(f"- {item}")
                else:
                    lines.append(f"- [{ts}] {item}")
            lines.append("")
            _atomic_write_text(self.skill_path, "\n".join(lines))
