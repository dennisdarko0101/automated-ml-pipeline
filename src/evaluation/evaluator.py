"""Model evaluation with comprehensive metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalReport:
    """Evaluation report for a single model."""

    metrics: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    confusion_mat: list[list[int]] | None = None
    plots_data: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else 0.0

    @property
    def cv_std(self) -> float:
        return float(np.std(self.cv_scores)) if self.cv_scores else 0.0


class ModelEvaluator:
    """Evaluate models with classification or regression metrics."""

    def __init__(self, cv_folds: int = 5) -> None:
        self.cv_folds = cv_folds

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = "classification",
    ) -> EvalReport:
        """Produce a full evaluation report."""
        preds = model.predict(X_test)

        if task_type == "classification":
            return self._evaluate_classification(model, X_test, y_test, preds)
        return self._evaluate_regression(model, X_test, y_test, preds)

    def _evaluate_classification(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preds: np.ndarray,
    ) -> EvalReport:
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, preds, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
        }

        # AUC-ROC (only for binary or if predict_proba exists)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    metrics["auc_roc"] = float(roc_auc_score(y, proba[:, 1]))
                else:
                    metrics["auc_roc"] = float(
                        roc_auc_score(y, proba, multi_class="ovr", average="weighted")
                    )
        except (ValueError, TypeError):
            pass

        cm = confusion_matrix(y, preds).tolist()

        cv_scores = cross_val_score(
            model, X, y, cv=min(self.cv_folds, len(y)), scoring="f1_weighted",
        ).tolist()

        recommendations = self._classification_recommendations(metrics)

        return EvalReport(
            metrics=metrics,
            cv_scores=cv_scores,
            confusion_mat=cm,
            plots_data={"confusion_matrix": cm},
            recommendations=recommendations,
        )

    def _evaluate_regression(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preds: np.ndarray,
    ) -> EvalReport:
        mse_val = float(mean_squared_error(y, preds))
        metrics: dict[str, float] = {
            "mae": float(mean_absolute_error(y, preds)),
            "mse": mse_val,
            "rmse": float(np.sqrt(mse_val)),
            "r2": float(r2_score(y, preds)),
        }

        # MAPE — guard against zero values
        try:
            metrics["mape"] = float(mean_absolute_percentage_error(y, preds))
        except (ValueError, ZeroDivisionError):
            pass

        cv_scores = cross_val_score(
            model, X, y, cv=min(self.cv_folds, len(y)), scoring="r2",
        ).tolist()

        residuals = (y.values - preds).tolist()
        recommendations = self._regression_recommendations(metrics)

        return EvalReport(
            metrics=metrics,
            cv_scores=cv_scores,
            plots_data={"residuals": residuals},
            recommendations=recommendations,
        )

    @staticmethod
    def _classification_recommendations(metrics: dict[str, float]) -> list[str]:
        recs: list[str] = []
        f1 = metrics.get("f1", 0)
        if f1 < 0.6:
            recs.append("F1 score is low — consider more feature engineering or data collection")
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        if precision > 0 and recall > 0 and abs(precision - recall) > 0.15:
            recs.append("Large precision-recall gap — consider adjusting classification threshold")
        if metrics.get("auc_roc", 1) < 0.7:
            recs.append("AUC-ROC is low — model has weak discriminative ability")
        return recs

    @staticmethod
    def _regression_recommendations(metrics: dict[str, float]) -> list[str]:
        recs: list[str] = []
        r2 = metrics.get("r2", 0)
        if r2 < 0.5:
            recs.append("R2 is low — model explains little variance; try non-linear models")
        mape = metrics.get("mape", 0)
        if mape > 0.3:
            recs.append("MAPE > 30% — predictions have large relative error")
        return recs
