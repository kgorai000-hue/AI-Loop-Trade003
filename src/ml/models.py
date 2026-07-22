from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import MLConfig


def recommend_model_type(n_samples: int, cfg: MLConfig) -> str:
    """Model selection by data size (Lesson 9.3)."""
    if cfg.model_type != "auto":
        return cfg.model_type
    if n_samples < 5000:
        return "ridge"
    if n_samples < 50000:
        return "random_forest"
    return "random_forest"


def build_classifier(model_type: str, cfg: MLConfig) -> Any:
    if model_type == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
    return RandomForestClassifier(
        n_estimators=cfg.random_forest_estimators,
        max_depth=cfg.random_forest_max_depth,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def build_regressor(model_type: str, cfg: MLConfig) -> Any:
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=cfg.random_forest_estimators,
            max_depth=cfg.random_forest_max_depth,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=cfg.random_state)),
        ]
    )


def predict_scores(model: Any, X) -> np.ndarray:
    arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(arr)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)
    preds = model.predict(arr)
    return np.asarray(preds, dtype=float)


def feature_importance(model: Any, feature_names: list[str]) -> list[tuple[str, float]]:
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("model", model)
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(est.coef_).ravel()
    else:
        return []
    pairs = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in pairs) or 1.0
    return [(name, val / total) for name, val in pairs]
