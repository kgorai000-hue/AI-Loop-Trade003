from __future__ import annotations

from src.online.drift import DDMDetector


class AdaptiveUpdateScheduler:
    """Adaptive update scheduling (Lesson 17.4)."""

    def __init__(
        self,
        min_interval: int = 5,
        max_interval: int = 20,
    ) -> None:
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.last_update = 0
        self.drift_detector = DDMDetector()

    def should_update(self, day: int, recent_errors: list[float]) -> tuple[bool, str]:
        days_since = day - self.last_update
        drift_detected = False
        for err in recent_errors:
            if self.drift_detector.update(err):
                drift_detected = True
                break

        if days_since >= self.max_interval:
            return True, "max interval reached, forced update"
        if drift_detected and days_since >= self.min_interval:
            return True, "drift detected, trigger update"
        return False, "no update needed"

    def record_update(self, day: int) -> None:
        self.last_update = day
        self.drift_detector.reset()
