from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from enum import Enum
from zoneinfo import ZoneInfo


class SessionPhase(str, Enum):
    OFF_HOURS = "off_hours"
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    CLOSE_ONLY = "close_only"
    POST_MARKET = "post_market"
    SHUTDOWN = "shutdown"


@dataclass
class MarketSchedule:
    timezone: str
    pre_market: dt_time
    market_open: dt_time
    close_only: dt_time
    market_close: dt_time
    post_market: dt_time
    shutdown: dt_time
    weekdays_only: bool = True


DEFAULT_FX_SCHEDULE = MarketSchedule(
    timezone="UTC",
    pre_market=dt_time(0, 15),
    market_open=dt_time(0, 30),
    close_only=dt_time(23, 50),
    market_close=dt_time(23, 59),
    post_market=dt_time(23, 59),
    shutdown=dt_time(23, 59),
    weekdays_only=True,
)


class MarketSessionScheduler:
    """Market-hours automation (Lesson 20.5)."""

    def __init__(self, schedule: MarketSchedule | None = None) -> None:
        self.schedule = schedule or DEFAULT_FX_SCHEDULE

    def current_phase(self, now: datetime | None = None) -> SessionPhase:
        now = now or datetime.now(ZoneInfo(self.schedule.timezone))
        if self.schedule.weekdays_only and now.weekday() >= 5:
            return SessionPhase.OFF_HOURS

        t = now.time()
        s = self.schedule
        if t < s.pre_market:
            return SessionPhase.OFF_HOURS
        if t < s.market_open:
            return SessionPhase.PRE_MARKET
        if t < s.close_only:
            return SessionPhase.MARKET_OPEN
        if t < s.market_close:
            return SessionPhase.CLOSE_ONLY
        if t < s.post_market:
            return SessionPhase.POST_MARKET
        if t < s.shutdown:
            return SessionPhase.SHUTDOWN
        return SessionPhase.OFF_HOURS

    def trading_allowed(self, phase: SessionPhase | None = None) -> bool:
        phase = phase or self.current_phase()
        return phase in (SessionPhase.MARKET_OPEN, SessionPhase.CLOSE_ONLY)

    def new_entries_allowed(self, phase: SessionPhase | None = None) -> bool:
        phase = phase or self.current_phase()
        return phase == SessionPhase.MARKET_OPEN
