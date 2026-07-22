"""Pair relationship regimes R1–R5 (spread health, not single-symbol A–E)."""

from __future__ import annotations

from enum import Enum


class PairRegime(str, Enum):
    R1_STABLE_MR = "r1_stable_mr"
    R2_VOLATILE_MR = "r2_volatile_mr"
    R3_WEAKENING = "r3_weakening"
    R4_STRUCTURAL_BREAK = "r4_structural_break"
    R5_EVENT_EXEC = "r5_event_exec"


# Allow new entries
ENTRY_ALLOWED = frozenset(
    {PairRegime.R1_STABLE_MR.value, PairRegime.R2_VOLATILE_MR.value}
)

# Size multipliers by regime
SIZE_SCALE = {
    PairRegime.R1_STABLE_MR.value: 1.0,
    PairRegime.R2_VOLATILE_MR.value: 0.5,
    PairRegime.R3_WEAKENING.value: 0.0,
    PairRegime.R4_STRUCTURAL_BREAK.value: 0.0,
    PairRegime.R5_EVENT_EXEC.value: 0.0,
}

# Z-entry multipliers (R2 widens threshold)
Z_ENTRY_MULT = {
    PairRegime.R1_STABLE_MR.value: 1.0,
    PairRegime.R2_VOLATILE_MR.value: 1.25,
    PairRegime.R3_WEAKENING.value: 1.5,
    PairRegime.R4_STRUCTURAL_BREAK.value: 99.0,
    PairRegime.R5_EVENT_EXEC.value: 99.0,
}
