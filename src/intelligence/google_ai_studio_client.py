"""Google AI Studio (Gemini) API client with exponential backoff."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GoogleAIClientError(RuntimeError):
    pass


class GoogleAIClient:
    """
    Thin wrapper around the Google Generative AI SDK.
    API key: GEMINI_API_KEY environment variable only.
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 60.0,
    ) -> None:
        self.max_retries = max(1, int(max_retries))
        self.base_delay_sec = float(base_delay_sec)
        self.max_delay_sec = float(max_delay_sec)
        self._client = None

    @staticmethod
    def looks_like_api_key(api_key: str | None) -> bool:
        """True only for a non-placeholder Google AI Studio key shape."""
        key = (api_key or "").strip()
        if not key:
            return False
        if "..." in key:
            return False
        # Google AI Studio keys typically start with "AIza" or similar
        if len(key) < 20:
            return False
        return True

    def available(self) -> bool:
        return self.looks_like_api_key(os.environ.get("GEMINI_API_KEY"))

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not self.looks_like_api_key(api_key):
            raise GoogleAIClientError(
                "GEMINI_API_KEY is missing or looks like a placeholder"
            )
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise GoogleAIClientError(
                "google-generativeai package not installed; pip install google-generativeai"
            ) from exc
        genai.configure(api_key=api_key)
        self._client = genai
        return self._client

    def _is_retryable(self, exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if "rate" in name or "rate_limit" in msg or "429" in msg:
            return True
        if "overloaded" in msg or "timeout" in msg or "timed out" in msg:
            return True
        if "500" in msg or "502" in msg or "503" in msg or "529" in msg:
            return True
        # Google API exception attributes
        status = getattr(exc, "status_code", None)
        if status in (408, 429, 500, 502, 503, 529):
            return True
        return False

    def generate_content(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        client = self._get_client()
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                genai_model = client.GenerativeModel(
                    model_name=model,
                    system_instruction=system,
                )
                
                kwargs: dict[str, Any] = {
                    "contents": user,
                    "generation_config": {
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    },
                }

                response = genai_model.generate_content(**kwargs)
                if response.text:
                    return response.text.strip()
                else:
                    raise GoogleAIClientError("Empty response from Google AI Studio")
                    
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise GoogleAIClientError(str(exc)) from exc
                delay = min(
                    self.max_delay_sec,
                    self.base_delay_sec * (2 ** (attempt - 1)),
                )
                delay *= 0.5 + random.random()  # jitter
                logger.warning(
                    "Google AI Studio call failed (attempt %d/%d): %s; sleep %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise GoogleAIClientError(str(last_exc) if last_exc else "unknown error")

    @staticmethod
    def extract_json(text: str) -> Any:
        """Parse JSON from a model reply; tolerate fenced blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # drop first fence and optional trailing fence
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # find first { or [
            start_obj = text.find("{")
            start_arr = text.find("[")
            starts = [i for i in (start_obj, start_arr) if i >= 0]
            if not starts:
                raise
            start = min(starts)
            snippet = text[start:]
            # balance braces roughly
            for end in range(len(snippet), 0, -1):
                try:
                    return json.loads(snippet[:end])
                except json.JSONDecodeError:
                    continue
            raise
