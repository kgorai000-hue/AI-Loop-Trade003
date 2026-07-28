"""Google Gemini API client (AI Studio + Agent Platform express mode)."""

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
    Thin wrapper around the Google Gen AI SDK (`google-genai`).

    Env candidates: ``GOOGLE_API_KEY``, ``GEMINI_API_KEY``.
    Preference: keys starting with ``AQ.`` (Agent Platform express), then
    ``GOOGLE_API_KEY``, then ``GEMINI_API_KEY``.

    Backend:
      - ``AQ....`` → Agent Platform express (``vertexai=True``)
      - otherwise → Gemini Developer API (AI Studio)
      - override with ``GEMINI_BACKEND=vertex|ai_studio``
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
        self._backend: str | None = None

    @staticmethod
    def resolve_api_key() -> str | None:
        """Prefer AQ. express keys, then GOOGLE_API_KEY, then GEMINI_API_KEY."""
        google = (os.environ.get("GOOGLE_API_KEY") or "").strip()
        gemini = (os.environ.get("GEMINI_API_KEY") or "").strip()
        ordered = [k for k in (google, gemini) if k]
        for key in ordered:
            if GoogleAIClient.looks_like_api_key(key) and key.startswith("AQ."):
                return key
        for key in ordered:
            if GoogleAIClient.looks_like_api_key(key):
                return key
        return None

    @staticmethod
    def looks_like_api_key(api_key: str | None) -> bool:
        """True for AI Studio (AIza...) or Agent Platform express (AQ....) keys."""
        key = (api_key or "").strip()
        if not key:
            return False
        if "..." in key:
            return False
        if len(key) < 20:
            return False
        return True

    @staticmethod
    def resolve_backend(api_key: str) -> str:
        """Return ``vertex`` (Agent Platform) or ``ai_studio`` (Developer API)."""
        override = (os.environ.get("GEMINI_BACKEND") or "").strip().lower()
        if override in {"vertex", "agent", "agent_platform", "express"}:
            return "vertex"
        if override in {"ai_studio", "gemini", "developer"}:
            return "ai_studio"
        if api_key.startswith("AQ."):
            return "vertex"
        return "ai_studio"

    def available(self) -> bool:
        return self.looks_like_api_key(self.resolve_api_key())

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = self.resolve_api_key()
        if not self.looks_like_api_key(api_key):
            raise GoogleAIClientError(
                "GEMINI_API_KEY / GOOGLE_API_KEY is missing or looks like a placeholder"
            )
        assert api_key is not None
        try:
            from google import genai
        except ImportError as exc:
            raise GoogleAIClientError(
                "google-genai package not installed; pip install google-genai"
            ) from exc

        backend = self.resolve_backend(api_key)
        self._backend = backend

        # Avoid SDK ambiguity when both env vars are set.
        prev_google = os.environ.get("GOOGLE_API_KEY")
        prev_gemini = os.environ.get("GEMINI_API_KEY")
        prev_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        prev_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        try:
            if backend == "vertex":
                os.environ["GOOGLE_API_KEY"] = api_key
                os.environ.pop("GEMINI_API_KEY", None)
                # Express mode: API key must not be combined with project/location.
                os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
                os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
                self._client = genai.Client(vertexai=True, api_key=api_key)
                logger.info(
                    "GoogleAIClient using Agent Platform express mode "
                    "(vertexai=True, key_prefix=%s...)",
                    api_key[:5],
                )
            else:
                os.environ["GEMINI_API_KEY"] = api_key
                os.environ.pop("GOOGLE_API_KEY", None)
                self._client = genai.Client(api_key=api_key)
                logger.info(
                    "GoogleAIClient using Gemini Developer API "
                    "(AI Studio, key_prefix=%s...)",
                    api_key[:5],
                )
        finally:
            if prev_google is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = prev_google
            if prev_gemini is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = prev_gemini
            if prev_location is None:
                os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
            else:
                os.environ["GOOGLE_CLOUD_LOCATION"] = prev_location
            if prev_project is None:
                os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            else:
                os.environ["GOOGLE_CLOUD_PROJECT"] = prev_project
        return self._client

    def _is_retryable(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        # Billing / prepaid depletion will not recover with retries.
        if "resource_exhausted" in msg and (
            "prepayment" in msg or "credits are depleted" in msg or "billing" in msg
        ):
            return False
        name = type(exc).__name__.lower()
        if "rate" in name or "rate_limit" in msg or "429" in msg:
            return True
        if "overloaded" in msg or "timeout" in msg or "timed out" in msg:
            return True
        if "500" in msg or "502" in msg or "503" in msg or "529" in msg:
            return True
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
        json_mode: bool = False,
    ) -> str:
        client = self._get_client()
        last_exc: Optional[BaseException] = None

        try:
            from google.genai import types
        except ImportError as exc:
            raise GoogleAIClientError(
                "google-genai package not installed; pip install google-genai"
            ) from exc

        for attempt in range(1, self.max_retries + 1):
            try:
                config_kwargs: dict[str, Any] = {
                    "system_instruction": system,
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
                if json_mode:
                    config_kwargs["response_mime_type"] = "application/json"
                # Newer Flash/Pro may spend output budget on "thinking"; keep room for JSON.
                thinking = getattr(types, "ThinkingConfig", None)
                if thinking is not None and any(
                    tag in model for tag in ("2.5", "3.5", "3.6", "3-")
                ):
                    try:
                        config_kwargs["thinking_config"] = thinking(thinking_budget=0)
                    except Exception:
                        pass

                response = client.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = getattr(response, "text", None)
                if text:
                    return str(text).strip()
                raise GoogleAIClientError("Empty response from Gemini API")

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
                    "Gemini API call failed (attempt %d/%d): %s; sleep %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise GoogleAIClientError(str(last_exc) if last_exc else "unknown error")

    @staticmethod
    def extract_json(text: str) -> Any:
        """Parse JSON from a model reply; tolerate fenced blocks and light truncation."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
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
            start_obj = text.find("{")
            start_arr = text.find("[")
            starts = [i for i in (start_obj, start_arr) if i >= 0]
            if not starts:
                raise
            start = min(starts)
            snippet = text[start:]
            for end in range(len(snippet), 0, -1):
                try:
                    return json.loads(snippet[:end])
                except json.JSONDecodeError:
                    continue
            repaired = GoogleAIClient._repair_truncated_json(snippet)
            if repaired is not None:
                return repaired
            raise

    @staticmethod
    def _repair_truncated_json(snippet: str) -> Any | None:
        """Best-effort close of truncated JSON objects/arrays/strings."""
        in_string = False
        escape = False
        stack: list[str] = []
        for ch in snippet:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack and ch == stack[-1]:
                stack.pop()

        candidate = snippet.rstrip()
        if in_string:
            candidate += '"'
        # Drop trailing comma before closing
        candidate = candidate.rstrip()
        if candidate.endswith(","):
            candidate = candidate[:-1]
        while stack:
            candidate += stack.pop()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
