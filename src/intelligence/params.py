"""AppConfig-path parameter helpers for Maker→Checker→Validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.backtest.parameter_spaces import ParameterSpec, default_parameter_specs
from src.core.config import AppConfig

# M30 is fixed for Trade002 — do not let the loop switch trading.profile (changes TF).
_EXCLUDED_SPEC_NAMES = frozenset({"trading_profile"})


@dataclass(frozen=True)
class LoopParams:
    """Adopted strategy knobs as AppConfig dot-path → value."""

    overrides: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.overrides)

    def merge(self, other: dict[str, Any]) -> "LoopParams":
        merged = dict(self.overrides)
        merged.update(other)
        return LoopParams(overrides=merged)


def symbol_to_state_key(symbol: str) -> str:
    return symbol.lstrip("#").replace("/", "_").replace("\\", "_")


def llm_parameter_specs() -> list[ParameterSpec]:
    """Parameter spaces exposed to Maker (excludes profile / TF switches)."""
    return [s for s in default_parameter_specs() if s.name not in _EXCLUDED_SPEC_NAMES]


def allowed_space_description() -> str:
    lines: list[str] = []
    for spec in llm_parameter_specs():
        values_repr = []
        for value in spec.values:
            values_repr.append(repr(value))
        paths = [path for path, _ in spec.apply(spec.values[0])]
        lines.append(
            f"- {spec.name}: paths={paths}; allowed_values=[{', '.join(values_repr)}]; "
            f"{spec.description}"
        )
    return "\n".join(lines)


def params_from_config(config: AppConfig) -> LoopParams:
    overrides: dict[str, Any] = {}
    for spec in llm_parameter_specs():
        sample = spec.apply(spec.values[0])
        if len(sample) == 1:
            path = sample[0][0]
            overrides[path] = _read_path(config, path)
        else:
            for path, _ in sample:
                overrides[path] = _read_path(config, path)
    return LoopParams(overrides=overrides)


def _read_path(config: AppConfig, path: str) -> Any:
    obj: Any = config
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def overrides_to_tuples(overrides: dict[str, Any]) -> list[tuple[str, Any]]:
    return list(overrides.items())


def canonicalize_candidate(overrides: dict[str, Any]) -> dict[str, Any] | None:
    """
    Map a free-form Maker dict onto allowed discrete values.
    Returns None if any key/value is invalid.
    """
    if not overrides:
        return None

    from collections import defaultdict

    path_to_values: dict[str, set[Any]] = {}
    paired_groups: dict[frozenset[str], list[ParameterSpec]] = defaultdict(list)
    for spec in llm_parameter_specs():
        sample = spec.apply(spec.values[0])
        if len(sample) > 1:
            paired_groups[frozenset(p for p, _ in sample)].append(spec)
            continue
        path = sample[0][0]
        allowed: set[Any] = set()
        for value in spec.values:
            for p, v in spec.apply(value):
                if p == path:
                    allowed.add(v)
        path_to_values[path] = allowed

    result: dict[str, Any] = {}
    used_paths: set[str] = set()

    for path_set, group_specs in paired_groups.items():
        ordered_paths = [p for p, _ in group_specs[0].apply(group_specs[0].values[0])]
        if not all(p in overrides for p in ordered_paths):
            if any(p in overrides for p in ordered_paths):
                return None
            continue
        proposed = tuple(overrides[p] for p in ordered_paths)
        matched = False
        for spec in group_specs:
            for value in spec.values:
                applied = spec.apply(value)
                applied_vals = tuple(v for _, v in applied)
                if _values_close(proposed, applied_vals):
                    for p, v in applied:
                        result[p] = v
                        used_paths.add(p)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            return None

    for path, value in overrides.items():
        if path in used_paths:
            continue
        if path not in path_to_values:
            return None
        matched_val = _match_allowed(value, path_to_values[path])
        if matched_val is None:
            return None
        result[path] = matched_val

    return result


def _values_close(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, float) or isinstance(y, float):
            if abs(float(x) - float(y)) > 1e-9:
                return False
        elif x != y:
            return False
    return True


def _match_allowed(value: Any, allowed: set[Any]) -> Any | None:
    for candidate in allowed:
        if isinstance(value, float) or isinstance(candidate, float):
            if abs(float(value) - float(candidate)) <= 1e-9:
                return candidate
        elif value == candidate:
            return candidate
    return None


def candidate_key(overrides: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, repr(v)) for k, v in overrides.items()))
