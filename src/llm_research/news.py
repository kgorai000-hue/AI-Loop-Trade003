from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NewsItem:
    id: str
    headline: str
    body: str
    symbols: list[str]
    tags: list[str]


def load_news_file(path: Path) -> list[NewsItem]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items: list[NewsItem] = []
    for entry in data:
        items.append(
            NewsItem(
                id=str(entry.get("id", "")),
                headline=str(entry.get("headline", "")),
                body=str(entry.get("body", "")),
                symbols=[str(s) for s in entry.get("symbols", [])],
                tags=[str(t) for t in entry.get("tags", [])],
            )
        )
    return items


def filter_news_for_symbols(
    items: list[NewsItem],
    symbols: list[str],
    *,
    keyword_filter: bool = True,
    max_items: int = 100,
) -> list[NewsItem]:
    symbol_set = set(symbols)
    alias_map = {
        "AAPL": "#USNDAQ100",
        "USNDAQ": "#USNDAQ100",
        "SPX": "#USSPX500",
        "SP500": "#USSPX500",
    }

    matched: list[NewsItem] = []
    for item in items:
        item_symbols = set(item.symbols)
        for sym in list(item_symbols):
            if sym in alias_map:
                item_symbols.add(alias_map[sym])

        if item_symbols & symbol_set:
            matched.append(item)
            continue

        if not keyword_filter:
            continue

        text = f"{item.headline} {item.body}".upper()
        for sym in symbols:
            if sym.strip("#").upper() in text or sym.upper() in text:
                matched.append(item)
                break

    return matched[:max_items]
