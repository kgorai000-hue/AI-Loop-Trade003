from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from src.regime.detection import RegimeDetectionResult, adjusted_strategy_weights


class ProbabilisticRegimeDetector:
    """GMM-based regime probabilities (HMM alternative without hmmlearn, Lesson 12.2)."""

    STATE_NAMES = ("trending", "mean_reverting", "crisis")

    def __init__(self, n_states: int = 3, random_state: int = 42) -> None:
        self.n_states = n_states
        self.model = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            random_state=random_state,
            max_iter=100,
        )
        self._fitted = False
        self._state_map: dict[int, str] = {}

    def fit(self, features: np.ndarray) -> None:
        clean = features[~np.isnan(features).any(axis=1)]
        if len(clean) < self.n_states * 5:
            raise ValueError("insufficient samples for probabilistic regime model")
        self.model.fit(clean)
        self._fitted = True
        self._state_map = self._infer_state_mapping(clean)

    def predict_proba_latest(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        clean = features[~np.isnan(features).any(axis=1)]
        if len(clean) == 0:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        posteriors = self.model.predict_proba(clean)
        return posteriors[-1]

    def to_detection_result(
        self,
        features: np.ndarray,
        *,
        threshold: float = 0.5,
        asset_correlation: float = 0.0,
    ) -> RegimeDetectionResult:
        probs_raw = self.predict_proba_latest(features)
        mapped = {"trending": 0.0, "mean_reverting": 0.0, "crisis": 0.0, "transition": 0.0}
        for idx, p in enumerate(probs_raw):
            name = self._state_map.get(idx, self.STATE_NAMES[idx % 3])
            mapped[name] = mapped.get(name, 0.0) + float(p)

        total = sum(mapped.values())
        if total > 0:
            mapped = {k: v / total for k, v in mapped.items()}

        max_prob = max(mapped.values()) if mapped else 0.0
        top_label = max(mapped, key=mapped.get)  # type: ignore[arg-type]
        is_transition = max_prob < threshold

        from src.core.types import MarketRegime

        regime_map = {
            "trending": MarketRegime.BULL,
            "mean_reverting": MarketRegime.SIDEWAYS,
            "crisis": MarketRegime.CRISIS,
            "transition": MarketRegime.SIDEWAYS,
        }
        label = "transition" if is_transition else top_label
        weights = adjusted_strategy_weights(mapped, transition_risk_first=is_transition)

        return RegimeDetectionResult(
            primary_regime=regime_map.get(label, MarketRegime.SIDEWAYS),
            label=label,
            probabilities=mapped,
            strategy_weights=weights,
            is_transition=is_transition,
            reason=f"gmm: {label} p={max_prob:.2f}",
            asset_correlation=asset_correlation,
            detection_method="gmm",
        )

    def _infer_state_mapping(self, features: np.ndarray) -> dict[int, str]:
        labels = self.model.predict(features)
        mapping: dict[int, str] = {}
        for state in range(self.n_states):
            mask = labels == state
            if not np.any(mask):
                mapping[state] = self.STATE_NAMES[state % 3]
                continue
            subset = features[mask]
            mean_vol = float(np.mean(subset[:, 1])) if subset.shape[1] > 1 else 0.0
            mean_ret = float(np.mean(subset[:, 0]))
            if mean_vol >= 0.30:
                mapping[state] = "crisis"
            elif mean_ret >= 0.02:
                mapping[state] = "trending"
            else:
                mapping[state] = "mean_reverting"
        return mapping
