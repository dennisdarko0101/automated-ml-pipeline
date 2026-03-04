"""Hyperparameter search spaces for supported models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import randint, uniform


def _classification_spaces() -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "C": uniform(0.01, 10),
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [500, 1000],
        },
        "random_forest": {
            "n_estimators": randint(50, 300),
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
        },
        "xgboost": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        },
        "lightgbm": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "num_leaves": randint(20, 100),
            "subsample": uniform(0.6, 0.4),
        },
    }


def _regression_spaces() -> dict[str, dict[str, Any]]:
    return {
        "linear_regression": {},
        "random_forest": {
            "n_estimators": randint(50, 300),
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
        },
        "xgboost": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        },
        "lightgbm": {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "num_leaves": randint(20, 100),
            "subsample": uniform(0.6, 0.4),
        },
    }


@dataclass
class HyperparameterSpace:
    """Hyperparameter search space manager."""

    task_type: str = "classification"
    custom_spaces: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_space(self, model_name: str) -> dict[str, Any]:
        """Get the search space for a specific model."""
        if model_name in self.custom_spaces:
            return self.custom_spaces[model_name]

        defaults = (
            _classification_spaces()
            if self.task_type == "classification"
            else _regression_spaces()
        )
        return defaults.get(model_name, {})

    def get_all_spaces(self) -> dict[str, dict[str, Any]]:
        """Get search spaces for all default models."""
        defaults = (
            _classification_spaces()
            if self.task_type == "classification"
            else _regression_spaces()
        )
        merged = {**defaults, **self.custom_spaces}
        return merged

    def set_custom_space(self, model_name: str, space: dict[str, Any]) -> None:
        """Override the search space for a specific model."""
        self.custom_spaces[model_name] = space
