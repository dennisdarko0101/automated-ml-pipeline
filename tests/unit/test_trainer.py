"""Tests for AutoTrainer — 16 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from src.training.hyperparams import HyperparameterSpace
from src.training.trainer import AutoTrainer, ModelCandidate, TrainResult


def _split_data(df: pd.DataFrame, target_col: str = "target"):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_val, y_val


class TestAutoTrainerClassification:
    def test_train_returns_result(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification",
            models=["logistic_regression", "random_forest"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert isinstance(result, TrainResult)
        assert len(result.all_results) == 2

    def test_best_model_is_first(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification",
            models=["logistic_regression", "random_forest"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        best_score = result.best_model.metrics["f1"]
        for c in result.all_results:
            assert c.metrics["f1"] <= best_score + 1e-9

    def test_model_candidate_fields(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification", models=["logistic_regression"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        cand = result.best_model
        assert cand.model_name == "logistic_regression"
        assert cand.model is not None
        assert "accuracy" in cand.metrics
        assert "f1" in cand.metrics
        assert cand.training_time > 0

    def test_xgboost_model(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification", models=["xgboost"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert result.best_model.model_name == "xgboost"

    def test_lightgbm_model(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification", models=["lightgbm"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert result.best_model.model_name == "lightgbm"

    def test_training_time_tracked(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification", models=["logistic_regression"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert result.training_time > 0

    def test_ranking_property(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification",
            models=["logistic_regression", "random_forest"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert len(result.ranking) == 2


class TestAutoTrainerRegression:
    def test_regression_training(self, diabetes_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(diabetes_df)
        trainer = AutoTrainer(
            task_type="regression",
            models=["linear_regression", "random_forest"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert len(result.all_results) == 2
        assert "r2" in result.best_model.metrics

    def test_regression_xgboost(self, diabetes_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(diabetes_df)
        trainer = AutoTrainer(
            task_type="regression", models=["xgboost"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert "mae" in result.best_model.metrics

    def test_regression_lightgbm(self, diabetes_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(diabetes_df)
        trainer = AutoTrainer(
            task_type="regression", models=["lightgbm"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert "rmse" in result.best_model.metrics


class TestAutoTrainerEdgeCases:
    def test_invalid_task_type(self):
        with pytest.raises(ValueError, match="classification or regression"):
            AutoTrainer(task_type="clustering")

    def test_custom_metric(self, iris_df: pd.DataFrame):
        X_tr, y_tr, X_val, y_val = _split_data(iris_df)
        trainer = AutoTrainer(
            task_type="classification", models=["logistic_regression"],
            n_trials=2, cv_folds=2, metric="accuracy",
        )
        result = trainer.train(X_tr, y_tr, X_val, y_val)
        assert "accuracy" in result.best_model.metrics


class TestHyperparameterSpace:
    def test_classification_spaces(self):
        hp = HyperparameterSpace(task_type="classification")
        space = hp.get_space("random_forest")
        assert "n_estimators" in space
        assert "max_depth" in space

    def test_regression_spaces(self):
        hp = HyperparameterSpace(task_type="regression")
        space = hp.get_space("xgboost")
        assert "learning_rate" in space

    def test_custom_space(self):
        hp = HyperparameterSpace()
        hp.set_custom_space("my_model", {"param1": [1, 2, 3]})
        assert hp.get_space("my_model") == {"param1": [1, 2, 3]}

    def test_get_all_spaces(self):
        hp = HyperparameterSpace(task_type="classification")
        all_spaces = hp.get_all_spaces()
        assert "logistic_regression" in all_spaces
        assert "random_forest" in all_spaces
        assert "xgboost" in all_spaces
        assert "lightgbm" in all_spaces
