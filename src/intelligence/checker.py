"""Checker: adversarial Google AI Studio (Gemini) review of Maker candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.intelligence.google_ai_studio_client import GoogleAIClient, GoogleAIClientError
from src.intelligence.maker import MakerCandidate
from src.intelligence.params import LoopParams, canonicalize_candidate, candidate_key

logger = logging.getLogger(__name__)

CHECKER_SYSTEM = """You are Checker, an adversarial quant auditor for AI-Loop-Trade003.

Reject suspicious AppConfig parameter candidates before they waste a backtest.
You do NOT invent strategies. Review for:
- Values outside the intended economic regime (too aggressive thresholds)
- Likely overfit / data-snooping patterns
- Violations of SKILL lessons
- Thin rationale

Primary timeframe is fixed at M30.

Output ONLY valid JSON:
{
  "reviews": [
    {
      "overrides": { "<path>": <value>, "...": "..." },
      "decision": "approve" | "reject",
      "reason": "<concise>"
    }
  ]
}
"""


@dataclass
class CheckerReview:
    params: LoopParams
    approved: bool
    reason: str = ""


class StrategyChecker:
    def __init__(
        self,
        client: GoogleAIClient,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.client = client
        self.model = model

    def review(
        self,
        candidates: list[MakerCandidate],
        *,
        skills_text: str,
    ) -> list[CheckerReview]:
        if not candidates:
            return []

        context = f"SKILL LESSONS:\n{skills_text}\n"
        payload = [
            {"overrides": c.params.as_dict(), "rationale": c.rationale}
            for c in candidates
        ]
        user = (
            f"{context}\n\n"
            "Adversarially review each candidate. Prefer reject when uncertain.\n"
            f"CANDIDATES:\n{payload}"
        )
        try:
            raw = self.client.generate_content(
                model=self.model,
                system=CHECKER_SYSTEM,
                user=user,
                max_tokens=8192,
                temperature=0.0,
                json_mode=True,
            )
            data = self.client.extract_json(raw)
        except (GoogleAIClientError, Exception) as exc:
            logger.error("Checker review failed: %s", exc)
            return [
                CheckerReview(
                    params=c.params,
                    approved=False,
                    reason=f"checker_error: {exc}",
                )
                for c in candidates
            ]
        return self._parse_reviews(data, candidates)

    def _parse_reviews(
        self,
        data: Any,
        candidates: list[MakerCandidate],
    ) -> list[CheckerReview]:
        by_key = {candidate_key(c.params.as_dict()): c for c in candidates}
        reviews: list[CheckerReview] = []
        items: list[Any] = []
        if isinstance(data, dict):
            items = data.get("reviews") or []
        if not isinstance(items, list):
            items = []

        seen: set[tuple[tuple[str, str], ...]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            raw = item.get("overrides")
            if not isinstance(raw, dict):
                continue
            cleaned = canonicalize_candidate({str(k): v for k, v in raw.items()})
            if not cleaned:
                continue
            key = candidate_key(cleaned)
            if key not in by_key:
                continue
            decision = str(item.get("decision", "reject")).strip().lower()
            approved = decision in ("approve", "pass", "accepted", "accept")
            reviews.append(
                CheckerReview(
                    params=LoopParams(overrides=cleaned),
                    approved=approved,
                    reason=str(item.get("reason") or ""),
                )
            )
            seen.add(key)

        for key, cand in by_key.items():
            if key not in seen:
                reviews.append(
                    CheckerReview(
                        params=cand.params,
                        approved=False,
                        reason="missing from checker response",
                    )
                )
        return reviews
