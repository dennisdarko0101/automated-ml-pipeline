"""Tests for model evaluator — 16 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluation.evaluator import EvalReport, ModelEvaluator


def _fit_classifier(iris_df: pd.DataFrame):
    X = iris_df.drop(columns=["target"])
    y = iris_df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_tr, y_tr)
    return model, X_te, y_te


def _fit_regressor(diabetes_df: pd.DataFrame):
    X = diabetes_df.drop(columns=["target"])
    y = diabetes_df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_tr, y_tr)
    return model, X_te, y_te


class TestClassificationEvaluation:
    def test_evaluate_returns_report(self, iris_df: pd.DataFrame):
        model, X_te, y_te = _fit_classifier(iris_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="classification")
        assert isinstance(report, EvalReport)

    def test_classification_metrics_present(self, iris_df: pd.DataFrame):
        model, X_te, y_te = _fit_classifier(iris_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="classification")
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in report.metrics

    def test_auc_roc_present(self, iris_df: pd.DataFrame):
        model, X_te, y_te = _fit_classifier(iris_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="classification")
        assert "auc_roc" in report.metrics

    def test_confusion_matrix(self, iris_df: pd.DataFrame):
        model, X_te, y_te = _fit_classifier(iris_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="classification")
        assert report.confusion_mat is not None
        assert len(report.confusion_mat) == 3  # 3 classes

    def test_cv_scores(self, iris_df: pd.DataFrame):
        model, X_te, y_te = _fit_classifier(iris_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="classification")
        assert len(report.cv_scores) == 3
        assert report.cv_mean > 0

    def test_binary_classification(self):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = pd.Series((X["a"] + X["b"] > 0).astype(int))
        model = LogisticRegression()
        model.fit(X, y)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X, y, task_type="classification")
        assert "auc_roc" in report.metrics

    def test_low_f1_recommendation(self):
        report = EvalReport(metrics={"f1": 0.3, "precision": 0.5, "recall": 0.2})
        # Simulate: the evaluator adds recommendations inside _classification_recommendations
        recs = ModelEvaluator._classification_recommendations(report.metrics)
        assert any("F1" in r for r in recs)

    def test_precision_recall_gap_recommendation(self):
        recs = ModelEvaluator._classification_recommendations(
            {"f1": 0.8, "precision": 0.95, "recall": 0.6, "auc_roc": 0.9}
        )
        assert any("precision-recall" in r.lower() for r in recs)


class TestRegressionEvaluation:
    def test_evaluate_returns_report(self, diabetes_df: pd.DataFrame):
        model, X_te, y_te = _fit_regressor(diabetes_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="regression")
        assert isinstance(report, EvalReport)

    def test_regression_metrics_present(self, diabetes_df: pd.DataFrame):
        model, X_te, y_te = _fit_regressor(diabetes_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="regression")
        for key in ("mae", "mse", "rmse", "r2"):
            assert key in report.metrics

    def test_mape_present(self, diabetes_df: pd.DataFrame):
        model, X_te, y_te = _fit_regressor(diabetes_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="regression")
        assert "mape" in report.metrics

    def test_residuals_in_plots_data(self, diabetes_df: pd.DataFrame):
        model, X_te, y_te = _fit_regressor(diabetes_df)
        evaluator = ModelEvaluator(cv_folds=3)
        report = evaluator.evaluate(model, X_te, y_te, task_type="regression")
        assert "residuals" in report.plots_data

    def test_low_r2_recommendation(self):
        recs = ModelEvaluator._regression_recommendations({"r2": 0.2, "mape": 0.1})
        assert any("R2" in r for r in recs)

    def test_high_mape_recommendation(self):
        recs = ModelEvaluator._regression_recommendations({"r2": 0.8, "mape": 0.5})
        assert any("MAPE" in r for r in recs)


class TestEvalReportProperties:
    def test_cv_mean_empty(self):
        report = EvalReport()
        assert report.cv_mean == 0.0

    def test_cv_std_empty(self):
        report = EvalReport()
        assert report.cv_std == 0.0
