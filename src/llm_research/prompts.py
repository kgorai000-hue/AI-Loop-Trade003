from __future__ import annotations

SENTIMENT_SYSTEM = (
    "You are a financial analyst. Output JSON only. "
    "Do NOT recommend buy, sell, or execute trades."
)

SENTIMENT_USER_TEMPLATE = """Analyze the following news for trading-related information.

News:
{headline}
{body}

Output JSON with these fields only:
- symbol (ticker or mapped symbol)
- event_type (earnings/product/policy/macro/other)
- sentiment_overall (-1 to 1)
- key_points (array of {{topic, sentiment, value}})
- trading_signal (bullish/bearish/mixed/neutral) — informational only
- confidence (0 to 1)

JSON only, no explanation."""


def build_sentiment_prompt(headline: str, body: str) -> str:
    return SENTIMENT_USER_TEMPLATE.format(headline=headline, body=body)
