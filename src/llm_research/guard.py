from __future__ import annotations

import re
from typing import Any

FORBIDDEN_ACTION_PATTERNS = (
    r"\bfull\s+position\s+buy\b",
    r"\bexecute\s+(buy|sell)\b",
    r"\bplace\s+order\b",
    r"\bmarket\s+order\b",
)

ALLOWED_TRADING_SIGNALS = {"bullish", "bearish", "mixed", "neutral"}


class LLMGuardError(Exception):
    pass


def validate_sentiment_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Hard constraints: schema + no direct trading instructions (Lesson 14.5)."""
    required = ("symbol", "event_type", "sentiment_overall", "confidence")
    for key in required:
        if key not in data:
            raise LLMGuardError(f"missing required field: {key}")

    raw_text = str(data)
    lowered = raw_text.lower()
    for pattern in FORBIDDEN_ACTION_PATTERNS:
        if re.search(pattern, lowered):
            raise LLMGuardError(f"forbidden trading action in LLM output: {pattern}")

    signal = str(data.get("trading_signal", "neutral")).lower()
    if signal not in ALLOWED_TRADING_SIGNALS:
        data["trading_signal"] = "neutral"

    sentiment = float(data["sentiment_overall"])
    if sentiment < -1.0 or sentiment > 1.0:
        raise LLMGuardError("sentiment_overall out of range [-1, 1]")

    confidence = float(data["confidence"])
    if confidence < 0.0 or confidence > 1.0:
        raise LLMGuardError("confidence out of range [0, 1]")

    return data


def assert_not_execution_path(action_taken: str) -> None:
    forbidden = ("execute", "order", "risk_override", "position_size")
    lowered = action_taken.lower()
    for word in forbidden:
        if word in lowered and "not" not in lowered:
            raise LLMGuardError(f"LLM output must not reach execution path: {action_taken}")
