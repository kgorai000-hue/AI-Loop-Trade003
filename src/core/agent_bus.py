from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.types import MessagePattern


@dataclass
class AgentEvent:
    topic: str
    sender: str
    payload: Any
    pattern: MessagePattern
    timestamp: float = field(default_factory=time.time)


@dataclass
class SharedAgentState:
    """Consistent view of positions and regime for all agents (Lesson 11.3)."""
    regime_map: dict[str, Any] = field(default_factory=dict)
    open_position_lots: dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    exposure_pct: float = 0.0


class AgentBus:
    """In-process pub/sub + shared state (modular monolith, Lesson 11.6)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[AgentEvent], None]]] = defaultdict(list)
        self._events: list[AgentEvent] = []
        self.shared = SharedAgentState()

    def publish(
        self,
        topic: str,
        sender: str,
        payload: Any,
        pattern: MessagePattern = MessagePattern.PUB_SUB,
    ) -> None:
        event = AgentEvent(topic=topic, sender=sender, payload=payload, pattern=pattern)
        self._events.append(event)
        for handler in self._subscribers.get(topic, []):
            handler(event)

    def subscribe(self, topic: str, handler: Callable[[AgentEvent], None]) -> None:
        self._subscribers[topic].append(handler)

    def request(self, topic: str, sender: str, payload: Any) -> None:
        self.publish(topic, sender, payload, MessagePattern.REQUEST_RESPONSE)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def recent_events(self, limit: int = 20) -> list[AgentEvent]:
        return self._events[-limit:]
