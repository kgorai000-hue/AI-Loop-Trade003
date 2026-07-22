from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...


class MockLLMClient(LLMClient):
    """Rule-based sentiment for offline/demo use (Lesson 14 default)."""

    POSITIVE = ("beat", "surge", "rise", "gain", "strong", "record", "above", "growth")
    NEGATIVE = ("miss", "fall", "drop", "decline", "weak", "below", "loss", "cut", "crisis")

    SYMBOL_MAP = {
        "AAPL": "#USNDAQ100",
        "APPLE": "#USNDAQ100",
        "S&P": "#USSPX500",
        "SPX": "#USSPX500",
        "NASDAQ": "#USNDAQ100",
        "DOW": "#US30",
        "GOLD": "GOLD",
        "EURUSD": "EURUSD",
        "USDJPY": "USDJPY",
    }

    def complete(self, system: str, user: str) -> str:
        text = user.lower()
        sentiment = 0.0
        for word in self.POSITIVE:
            if word in text:
                sentiment += 0.15
        for word in self.NEGATIVE:
            if word in text:
                sentiment -= 0.15
        sentiment = max(-1.0, min(1.0, sentiment))

        symbol = "UNKNOWN"
        for key, mapped in self.SYMBOL_MAP.items():
            if key.lower() in text:
                symbol = mapped
                break

        event_type = "other"
        if "earnings" in text or "revenue" in text or "quarter" in text:
            event_type = "earnings"
        elif "rate" in text or "fed" in text or "policy" in text:
            event_type = "policy"
        elif "oil" in text or "gold" in text:
            event_type = "macro"

        if sentiment > 0.2:
            signal = "bullish"
        elif sentiment < -0.2:
            signal = "bearish"
        elif abs(sentiment) < 0.05:
            signal = "neutral"
        else:
            signal = "mixed"

        payload = {
            "symbol": symbol,
            "event_type": event_type,
            "sentiment_overall": round(sentiment, 3),
            "key_points": [
                {"topic": "headline", "sentiment": round(sentiment, 3), "value": "mock extraction"},
            ],
            "trading_signal": signal,
            "confidence": 0.65,
        }
        return json.dumps(payload)


class OpenAIClient(LLMClient):
    """Optional OpenAI provider (requires API key in settings.local.yaml)."""

    def __init__(self, api_key: str, model: str, temperature: float, timeout_sec: int = 30) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def complete(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        content = data["choices"][0]["message"]["content"]
        return _extract_json(content)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("no JSON object in LLM response")


def build_llm_client(
    provider: str,
    *,
    api_key: str | None,
    model: str,
    temperature: float,
) -> LLMClient:
    if provider == "openai":
        if not api_key:
            raise ValueError("openai provider requires llm_research.openai_api_key")
        return OpenAIClient(api_key=api_key, model=model, temperature=temperature)
    return MockLLMClient()
