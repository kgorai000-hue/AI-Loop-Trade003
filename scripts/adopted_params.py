"""Central adopted-parameter table for M30 screening (return-max, gates OFF).

Maps live in scripts/screening_params.py; this module is the export surface
(Excel + DataFrame + apply_adopted_params).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.screening_params import (
    BEST_ADX_SIDEWAYS,
    BEST_ADX_TREND,
    BEST_MA_BY_SYMBOL,
    BEST_MR_BB_BY_SYMBOL,
    BEST_MR_RSI_BY_SYMBOL,
    BEST_SIGNAL_SCORE_BY_SYMBOL,
    SYMBOL_ORDER,
)

PROTOCOL = {
    "timeframe": "M30",
    "bars": 2000,
    "mc_sims": 100,
    "quality_gates": "OFF",
    "selection": "return-max per symbol (one-at-a-time grids)",
    "atr": "not tuned (unused in backtest signal path)",
}


def adopted_rows() -> list[dict]:
    rows: list[dict] = []
    for symbol in SYMBOL_ORDER:
        rsi = BEST_MR_RSI_BY_SYMBOL[symbol]
        bb = BEST_MR_BB_BY_SYMBOL[symbol]
        ma = BEST_MA_BY_SYMBOL[symbol]
        rows.append(
            {
                "symbol": symbol,
                "mr_rsi_oversold": rsi[0],
                "mr_rsi_overbought": rsi[1],
                "mr_bb_entry_low": bb[0],
                "mr_bb_entry_high": bb[1],
                "adx_trend_threshold": BEST_ADX_TREND[symbol],
                "adx_sideways_threshold": BEST_ADX_SIDEWAYS[symbol],
                "ma_short": ma[0],
                "ma_long": ma[1],
                "signal_score_threshold": BEST_SIGNAL_SCORE_BY_SYMBOL[symbol],
            }
        )
    return rows


def adopted_dataframe() -> pd.DataFrame:
    return pd.DataFrame(adopted_rows())


def apply_adopted_params(config, symbol: str) -> None:
    """Apply per-symbol adopted screening maps onto config (in place)."""
    rsi = BEST_MR_RSI_BY_SYMBOL[symbol]
    bb = BEST_MR_BB_BY_SYMBOL[symbol]
    ma = BEST_MA_BY_SYMBOL[symbol]
    config.strategies.mr_rsi_oversold = float(rsi[0])
    config.strategies.mr_rsi_overbought = float(rsi[1])
    config.strategies.mr_bb_entry_low = float(bb[0])
    config.strategies.mr_bb_entry_high = float(bb[1])
    config.strategies.adx_trend_threshold = float(BEST_ADX_TREND[symbol])
    config.strategies.adx_sideways_threshold = float(BEST_ADX_SIDEWAYS[symbol])
    config.strategies.trend_ma_short = int(ma[0])
    config.strategies.trend_ma_long = int(ma[1])
    config.indicators.signal_score_threshold = float(BEST_SIGNAL_SCORE_BY_SYMBOL[symbol])


def export_excel(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = adopted_dataframe()
    meta = pd.DataFrame([{**PROTOCOL, "generated_at_utc": datetime.now(timezone.utc).isoformat()}])
    sources = pd.DataFrame(
        [
            {"param": "mr_rsi", "map": "BEST_MR_RSI_BY_SYMBOL", "source": "scripts/screening_params.py"},
            {"param": "mr_bb", "map": "BEST_MR_BB_BY_SYMBOL", "source": "scripts/screening_params.py"},
            {"param": "adx_trend", "map": "BEST_ADX_TREND", "source": "scripts/screening_params.py"},
            {"param": "adx_sideways", "map": "BEST_ADX_SIDEWAYS", "source": "scripts/screening_params.py"},
            {"param": "ma_short/long", "map": "BEST_MA_BY_SYMBOL", "source": "scripts/screening_params.py"},
            {
                "param": "signal_score",
                "map": "BEST_SIGNAL_SCORE_BY_SYMBOL",
                "source": "scripts/screening_params.py",
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="AdoptedParams", index=False)
        meta.to_excel(writer, sheet_name="Protocol", index=False)
        sources.to_excel(writer, sheet_name="Sources", index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export adopted parameter table")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output xlsx path (default: data/reports/adopted_params_M30_<stamp>.xlsx)",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.output or (
        PROJECT_ROOT / f"data/reports/adopted_params_M30_{stamp}.xlsx"
    )
    path = export_excel(out)
    df = adopted_dataframe()
    print("=== Adopted parameters (M30 screening) ===")
    print(df.to_string(index=False))
    print(f"\n  report saved : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
