"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris


# ── Datasets ───────────────────────────────────────────────────────────


@pytest.fixture
def iris_df() -> pd.DataFrame:
    """Iris classification dataset as a DataFrame."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


@pytest.fixture
def diabetes_df() -> pd.DataFrame:
    """Diabetes regression dataset as a DataFrame."""
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic DataFrame for unit tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n) * 2 + 1,
        "num3": np.random.randn(n),
        "cat1": np.random.choice(["a", "b", "c"], n),
        "cat2": np.random.choice(["x", "y"], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    """Synthetic regression DataFrame."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": np.random.randn(n),
        "target": 3 * x1 + 2 * x2 + np.random.randn(n) * 0.5,
    })


@pytest.fixture
def iris_csv(iris_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write iris data to a temporary CSV file."""
    p = tmp_path / "iris.csv"
    iris_df.to_csv(p, index=False)
    return p


@pytest.fixture
def diabetes_csv(diabetes_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write diabetes data to a temporary CSV file."""
    p = tmp_path / "diabetes.csv"
    diabetes_df.to_csv(p, index=False)
    return p


# ── Configs ────────────────────────────────────────────────────────────


@pytest.fixture
def classification_config_path(iris_csv: Path, tmp_path: Path) -> Path:
    """Write a classification pipeline YAML config."""
    config = f"""
name: test-classification
data_source: "{iris_csv.as_posix()}"
target_column: target
test_size: 0.2
val_size: 0.1
random_state: 42
feature_config:
  numeric_strategy: standard
  categorical_strategy: onehot
  missing_strategy: median
  correlation_threshold: 0.95
  variance_threshold: 0.01
training_config:
  task_type: classification
  models:
    - logistic_regression
    - random_forest
  n_trials: 2
  cv_folds: 3
  metric: f1
evaluation_config:
  primary_metric: f1
  significance_level: 0.05
"""
    p = tmp_path / "classification.yaml"
    p.write_text(config)
    return p


@pytest.fixture
def regression_config_path(diabetes_csv: Path, tmp_path: Path) -> Path:
    """Write a regression pipeline YAML config."""
    config = f"""
name: test-regression
data_source: "{diabetes_csv.as_posix()}"
target_column: target
test_size: 0.2
val_size: 0.1
random_state: 42
feature_config:
  numeric_strategy: standard
  missing_strategy: mean
training_config:
  task_type: regression
  models:
    - linear_regression
    - random_forest
  n_trials: 2
  cv_folds: 3
  metric: r2
evaluation_config:
  primary_metric: r2
  significance_level: 0.05
"""
    p = tmp_path / "regression.yaml"
    p.write_text(config)
    return p


# ── MLflow mock ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_mlflow():
    """Disable MLflow tracking in tests."""
    with patch("mlflow.start_run") as mock_run:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=MagicMock(info=MagicMock(run_id="test-run-id")))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_run.return_value = mock_ctx
        with patch("mlflow.log_params"), \
             patch("mlflow.log_metrics"), \
             patch("mlflow.sklearn.log_model"), \
             patch("mlflow.active_run", return_value=MagicMock(info=MagicMock(run_id="test-run-id"))), \
             patch("mlflow.set_tracking_uri"):
            yield
