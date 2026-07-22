from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RegulatoryFlag:
    region: str
    category: str
    summary: str
    severity: str = "info"
    effective_date: str = ""


@dataclass
class RegulatoryMonitor:
    """
    Appendix B mode #11 stub: regulatory change watchlist.
    Populate manually or via future news feed — not auto-wired.
    """

    flags: list[RegulatoryFlag] = field(default_factory=list)

    def add_flag(self, flag: RegulatoryFlag) -> None:
        self.flags.append(flag)

    def active_flags(self) -> list[RegulatoryFlag]:
        today = datetime.now(timezone.utc).date().isoformat()
        return [
            f
            for f in self.flags
            if not f.effective_date or f.effective_date <= today
        ]

    def checklist(self) -> list[str]:
        return [
            "Review broker/regional short-selling rules",
            "Check margin requirement changes",
            "Monitor tax treatment for FX gains",
            "Verify symbol trade_mode still allows intended orders",
        ]
