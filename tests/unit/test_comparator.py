"""Tests for model comparator — 11 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluation.comparator import ComparisonReport, ModelComparator
from src.training.trainer import ModelCandidate


def _make_candidates_classification(iris_df: pd.DataFrame) -> tuple:
    X = iris_df.drop(columns=["target"])
    y = iris_df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_tr, y_tr)

    candidates = [
        ModelCandidate("logistic_regression", lr, {}, {"f1": 0.9}, 1.0),
        ModelCandidate("random_forest", rf, {}, {"f1": 0.95}, 2.0),
    ]
    return candidates, X_te, y_te


def _make_candidates_regression(diabetes_df: pd.DataFrame) -> tuple:
    X = diabetes_df.drop(columns=["target"])
    y = diabetes_df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_tr, y_tr)

    candidates = [
        ModelCandidate("linear_regression", lr, {}, {"r2": 0.4}, 0.5),
        ModelCandidate("random_forest", rf, {}, {"r2": 0.5}, 1.5),
    ]
    return candidates, X_te, y_te


class TestModelComparatorClassification:
    def test_compare_returns_report(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="classification")
        assert isinstance(report, ComparisonReport)

    def test_rankings_ordered(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="classification")
        assert report.rankings[0]["rank"] == 1
        f1_first = report.rankings[0]["metrics"]["f1"]
        f1_second = report.rankings[1]["metrics"]["f1"]
        assert f1_first >= f1_second

    def test_best_model_name(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="classification")
        assert report.best_model_name in ("logistic_regression", "random_forest")

    def test_significance_results(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="classification")
        assert "test" in report.significance_results

    def test_comparison_table(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="classification")
        table = report.comparison_table
        assert isinstance(table, pd.DataFrame)
        assert "model" in table.columns
        assert len(table) == 2

    def test_recommendation_non_empty(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="classification")
        assert len(report.recommendation) > 0

    def test_evaluation_reports_per_model(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="classification")
        assert "logistic_regression" in report.evaluation_reports
        assert "random_forest" in report.evaluation_reports


class TestModelComparatorRegression:
    def test_regression_compare(self, diabetes_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_regression(diabetes_df)
        comparator = ModelComparator(primary_metric="r2", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="regression")
        assert isinstance(report, ComparisonReport)
        assert "mae" in report.evaluation_reports[report.best_model_name].metrics

    def test_regression_ranking(self, diabetes_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_regression(diabetes_df)
        comparator = ModelComparator(primary_metric="r2", cv_folds=3)
        report = comparator.compare(candidates, X_te, y_te, task_type="regression")
        r2_first = report.rankings[0]["metrics"]["r2"]
        r2_second = report.rankings[1]["metrics"]["r2"]
        assert r2_first >= r2_second


class TestSingleCandidate:
    def test_single_model_no_significance(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare([candidates[0]], X_te, y_te, task_type="classification")
        assert report.best_model_name == "logistic_regression"
        assert report.significance_results == {}

    def test_single_model_recommendation(self, iris_df: pd.DataFrame):
        candidates, X_te, y_te = _make_candidates_classification(iris_df)
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        report = comparator.compare([candidates[0]], X_te, y_te, task_type="classification")
        assert "logistic_regression" in report.recommendation
