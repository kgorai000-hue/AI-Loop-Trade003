"""Maker: Google AI Studio (Gemini) proposals for AppConfig strategy parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.intelligence.google_ai_studio_client import GoogleAIClient, GoogleAIClientError
from src.intelligence.params import (
    LoopParams,
    allowed_space_description,
    canonicalize_candidate,
    candidate_key,
)

logger = logging.getLogger(__name__)

MAKER_SYSTEM = """You are Maker, a quant parameter explorer for AI-Loop-Trade002.

The trading pipeline is a FIXED multi-agent system (regime → signals → portfolio → risk).
You propose AppConfig parameter overrides only — do NOT invent new indicators, strategies, or code.

Primary timeframe is fixed at M30. Never propose trading.profile changes.

Output ONLY valid JSON:
{
  "candidates": [
    {
      "overrides": {
        "<AppConfig.dot.path>": <value>
      },
      "rationale": "<short reason>"
    }
  ]
}

Hard constraints:
- Every key/value MUST be chosen from the ALLOWED SPACE provided in context.
- Prefer small, diverse changes (1-3 keys per candidate).
- Respect SKILL lessons (avoid known failure modes).
"""


@dataclass
class MakerCandidate:
    params: LoopParams
    rationale: str = ""


class StrategyMaker:
    def __init__(
        self,
        client: GoogleAIClient,
        model: str = "gemini-1.5-pro",
        n_candidates: int = 6,
    ) -> None:
        self.client = client
        self.model = model
        self.n_candidates = max(1, int(n_candidates))

    def propose(
        self,
        *,
        current_params: LoopParams,
        last_metrics: dict[str, Any],
        skills_text: str,
        symbol: str,
        strategy: str,
    ) -> list[MakerCandidate]:
        context = (
            f"SYMBOL: {symbol}\nSTRATEGY: {strategy}\nTIMEFRAME: M30\n\n"
            f"SKILL LESSONS:\n{skills_text}\n\n"
            f"ALLOWED SPACE:\n{allowed_space_description()}\n"
        )
        user = (
            f"{context}\n\n"
            f"Propose exactly {self.n_candidates} distinct parameter candidates.\n"
            f"Current params: {current_params.as_dict()}\n"
            f"Last metrics: {last_metrics}\n"
            "Favor diversity across the allowed space while avoiding SKILL failure modes."
        )
        try:
            raw = self.client.generate_content(
                model=self.model,
                system=MAKER_SYSTEM,
                user=user,
                max_tokens=2048,
                temperature=0.4,
            )
            data = self.client.extract_json(raw)
        except (GoogleAIClientError, Exception) as exc:
            logger.error("Maker propose failed: %s", exc)
            return []
        return self._parse_candidates(data)

    def _parse_candidates(self, data: Any) -> list[MakerCandidate]:
        if not isinstance(data, dict):
            return []
        items = data.get("candidates")
        if not isinstance(items, list):
            return []

        out: list[MakerCandidate] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_overrides = item.get("overrides")
            if not isinstance(raw_overrides, dict):
                continue
            cleaned = canonicalize_candidate(
                {str(k): v for k, v in raw_overrides.items()}
            )
            if not cleaned:
                logger.info("Maker candidate rejected (invalid space): %s", raw_overrides)
                continue
            key = candidate_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                MakerCandidate(
                    params=LoopParams(overrides=cleaned),
                    rationale=str(item.get("rationale") or ""),
                )
            )
        return out
