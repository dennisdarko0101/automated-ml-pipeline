"""AutoTrainer — train multiple models with hyperparameter tuning and MLflow tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV

import lightgbm as lgb
import xgboost as xgb

from src.training.hyperparams import HyperparameterSpace
from src.utils.logging import get_logger

logger = get_logger(__name__)


MODEL_REGISTRY_CLS: dict[str, dict[str, type]] = {
    "classification": {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": xgb.XGBClassifier,
        "lightgbm": lgb.LGBMClassifier,
    },
    "regression": {
        "linear_regression": LinearRegression,
        "random_forest": RandomForestRegressor,
        "xgboost": xgb.XGBRegressor,
        "lightgbm": lgb.LGBMRegressor,
    },
}


@dataclass
class ModelCandidate:
    """A single trained model candidate."""

    model_name: str
    model: Any
    params: dict[str, Any]
    metrics: dict[str, float]
    training_time: float


@dataclass
class TrainResult:
    """Result of an AutoTrainer run."""

    best_model: ModelCandidate
    all_results: list[ModelCandidate]
    training_time: float

    @property
    def ranking(self) -> list[str]:
        return [c.model_name for c in self.all_results]


class AutoTrainer:
    """Train and compare multiple models automatically."""

    def __init__(
        self,
        task_type: str = "classification",
        models: list[str] | None = None,
        n_trials: int = 10,
        cv_folds: int = 5,
        metric: str | None = None,
        random_state: int = 42,
    ) -> None:
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be classification or regression, got {task_type}")
        self.task_type = task_type
        self.models = models or list(MODEL_REGISTRY_CLS[task_type].keys())
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric or ("f1" if task_type == "classification" else "r2")
        self.random_state = random_state
        self.hp_space = HyperparameterSpace(task_type=task_type)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainResult:
        """Train all configured models and return ranked results."""
        start = time.time()
        candidates: list[ModelCandidate] = []

        for model_name in self.models:
            try:
                candidate = self._train_single(model_name, X_train, y_train, X_val, y_val)
                candidates.append(candidate)
                logger.info(
                    "model_trained",
                    model=model_name,
                    metric=self.metric,
                    score=candidate.metrics.get(self.metric, 0),
                )
            except Exception as exc:
                logger.warning("model_training_failed", model=model_name, error=str(exc))

        if not candidates:
            raise RuntimeError("All model trainings failed")

        # Sort by primary metric (descending — higher is better for all default metrics)
        sort_ascending = self.metric in ("mae", "mse", "rmse", "mape")
        candidates.sort(
            key=lambda c: c.metrics.get(self.metric, 0),
            reverse=not sort_ascending,
        )

        total_time = time.time() - start
        return TrainResult(
            best_model=candidates[0],
            all_results=candidates,
            training_time=total_time,
        )

    def _train_single(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> ModelCandidate:
        """Train a single model with hyperparameter tuning."""
        t0 = time.time()
        model_cls = MODEL_REGISTRY_CLS[self.task_type][model_name]
        search_space = self.hp_space.get_space(model_name)

        scoring = self._get_scoring()

        if search_space:
            base_model = model_cls(random_state=self.random_state) if "random_state" in model_cls.__init__.__code__.co_varnames else model_cls()
            search = RandomizedSearchCV(
                base_model,
                search_space,
                n_iter=self.n_trials,
                cv=self.cv_folds,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_model = model_cls()
            best_model.fit(X_train, y_train)
            best_params = {}

        metrics = self._compute_metrics(best_model, X_val, y_val)
        training_time = time.time() - t0

        self._log_to_mlflow(model_name, best_params, metrics, best_model)

        return ModelCandidate(
            model_name=model_name,
            model=best_model,
            params=best_params,
            metrics=metrics,
            training_time=training_time,
        )

    def _get_scoring(self) -> str:
        """Map our metric names to sklearn scoring strings."""
        mapping = {
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
        }
        return mapping.get(self.metric, self.metric)

    def _compute_metrics(
        self, model: Any, X: pd.DataFrame, y: pd.Series,
    ) -> dict[str, float]:
        """Compute evaluation metrics on a dataset."""
        preds = model.predict(X)
        if self.task_type == "classification":
            return {
                "accuracy": float(accuracy_score(y, preds)),
                "f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
            }
        return {
            "mae": float(mean_absolute_error(y, preds)),
            "mse": float(mean_squared_error(y, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "r2": float(r2_score(y, preds)),
        }

    @staticmethod
    def _log_to_mlflow(
        model_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        model: Any,
    ) -> None:
        """Log training run to MLflow (best-effort)."""
        try:
            with mlflow.start_run(nested=True, run_name=model_name):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            logger.debug("mlflow_logging_skipped", model=model_name)
