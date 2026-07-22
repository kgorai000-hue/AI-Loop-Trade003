from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.core.config import AppConfig
from src.ml.features import (
    align_xy,
    build_feature_frame,
    create_classification_labels,
    create_forward_return_labels,
    univariate_ic_screen,
)
from src.ml.ic import calculate_ic, calculate_ir, long_short_spread, rolling_ic
from src.ml.models import build_classifier, feature_importance, predict_scores, recommend_model_type


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    ic: float
    accuracy: float
    long_short: float


@dataclass
class MLTrainReport:
    symbol: str
    timeframe: str
    model_type: str
    n_samples: int
    n_features: int
    selected_features: list[str]
    fold_results: list[FoldResult] = field(default_factory=list)
    mean_ic: float = 0.0
    mean_ir: float = 0.0
    mean_accuracy: float = 0.0
    top_feature_importance: list[tuple[str, float]] = field(default_factory=list)
    ic_decay_warning: bool = False
    viable: bool = False


class MLTrainer:
    """Walk-forward supervised learning with IC evaluation (Lesson 9)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ml = config.ml

    def train_and_evaluate(
        self,
        bars: list[dict],
        symbol: str,
        timeframe: str,
    ) -> MLTrainReport:
        features = build_feature_frame(bars, self.config)
        labels = create_classification_labels(
            bars,
            self.ml.label_horizon_bars,
            self.ml.label_threshold,
        )
        forward_ret = create_forward_return_labels(bars, self.ml.label_horizon_bars)
        X, y = align_xy(features, labels)
        forward_ret = forward_ret.loc[X.index]

        if len(X) < self.ml.min_train_samples:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.ml.min_train_samples}")

        screened = univariate_ic_screen(X, forward_ret, self.ml.min_feature_ic)
        selected = [name for name, _ in screened[: self.ml.max_features]]
        if not selected:
            selected = list(X.columns[: self.ml.max_features])
        X = X[selected]

        model_type = recommend_model_type(len(X), self.ml)
        tscv = TimeSeriesSplit(n_splits=self.ml.n_splits)
        fold_results: list[FoldResult] = []
        all_signals: list[float] = []
        all_returns: list[float] = []
        last_model = None

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = build_classifier(model_type, self.ml)
            model.fit(X_train, y_train)
            last_model = model

            scores = predict_scores(model, X_test)
            test_ret = forward_ret.iloc[test_idx].to_numpy()
            ic = calculate_ic(scores, test_ret)
            acc = float(np.mean((scores >= 0.5).astype(int) == y_test.to_numpy()))
            ls = long_short_spread(scores, test_ret)

            fold_results.append(
                FoldResult(
                    fold=fold,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    ic=ic,
                    accuracy=acc,
                    long_short=ls,
                )
            )
            all_signals.extend(scores.tolist())
            all_returns.extend(test_ret.tolist())

        ic_roll = rolling_ic(np.array(all_signals), np.array(all_returns), self.ml.rolling_ic_window)
        mean_ic = float(np.nanmean([f.ic for f in fold_results]))
        mean_ir = calculate_ir(ic_roll)
        mean_acc = float(np.mean([f.accuracy for f in fold_results]))

        from src.ml.ic import detect_ic_decay

        decayed, _, _ = detect_ic_decay(
            ic_roll,
            self.ml.rolling_ic_window,
            self.ml.rolling_ic_window * 2,
            self.ml.ic_degrade_ratio,
        )

        importance: list[tuple[str, float]] = []
        if last_model is not None:
            importance = feature_importance(last_model, selected)

        viable = mean_ic >= self.ml.ic_threshold and mean_ir >= self.ml.ir_threshold

        return MLTrainReport(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            n_samples=len(X),
            n_features=len(selected),
            selected_features=selected,
            fold_results=fold_results,
            mean_ic=mean_ic,
            mean_ir=mean_ir,
            mean_accuracy=mean_acc,
            top_feature_importance=importance[:5],
            ic_decay_warning=decayed,
            viable=viable,
        )

    def predict_latest(
        self,
        bars: list[dict],
        report: MLTrainReport,
    ) -> tuple[float, float] | None:
        """Fit on all but last row, predict probability for latest bar."""
        features = build_feature_frame(bars, self.config)
        labels = create_classification_labels(
            bars,
            self.ml.label_horizon_bars,
            self.ml.label_threshold,
        )
        X, y = align_xy(features, labels)
        if len(X) < self.ml.min_train_samples:
            return None

        cols = [c for c in report.selected_features if c in X.columns]
        if not cols:
            cols = list(X.columns)
        X = X[cols]

        if len(X) < 2:
            return None

        model = build_classifier(report.model_type, self.ml)
        model.fit(X.iloc[:-1], y.iloc[:-1])
        score = float(predict_scores(model, X.iloc[[-1]])[0])
        confidence = abs(score - 0.5) * 2
        return score, confidence
