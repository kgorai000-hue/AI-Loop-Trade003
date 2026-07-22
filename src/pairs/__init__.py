"""Within-group pairs: hedge-ratio spread + R1–R5 health gate."""

from src.pairs.health import PairHealthResult, PairHealthThresholds, classify_pair_health
from src.pairs.spread import SpreadSnapshot, build_spread_snapshot, rolling_ols_beta
from src.pairs.states import ENTRY_ALLOWED, SIZE_SCALE, Z_ENTRY_MULT, PairRegime

__all__ = [
    "ENTRY_ALLOWED",
    "PairHealthResult",
    "PairHealthThresholds",
    "PairRegime",
    "SIZE_SCALE",
    "SpreadSnapshot",
    "Z_ENTRY_MULT",
    "build_spread_snapshot",
    "classify_pair_health",
    "rolling_ols_beta",
]
