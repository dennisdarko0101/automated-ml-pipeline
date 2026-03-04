"""Tests for feature selection — 13 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.selector import FeatureReport, FeatureSelector


class TestCorrelationFilter:
    def test_removes_correlated_features(self):
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({"a": x, "b": x + np.random.randn(100) * 0.01, "c": np.random.randn(100)})
        selector = FeatureSelector()
        report = selector.correlation_filter(df, threshold=0.9)
        assert "b" in report.dropped
        assert "a" in report.selected
        assert "c" in report.selected

    def test_keeps_uncorrelated_features(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        selector = FeatureSelector()
        report = selector.correlation_filter(df, threshold=0.95)
        assert len(report.dropped) == 0

    def test_empty_numeric(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        selector = FeatureSelector()
        report = selector.correlation_filter(df)
        assert len(report.selected) == 1


class TestVarianceFilter:
    def test_removes_low_variance(self):
        df = pd.DataFrame({
            "constant": [1.0] * 100,
            "varied": np.random.randn(100),
        })
        selector = FeatureSelector()
        report = selector.variance_filter(df, threshold=0.01)
        assert "constant" in report.dropped
        assert "varied" in report.selected

    def test_keeps_all_when_variance_ok(self):
        df = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        selector = FeatureSelector()
        report = selector.variance_filter(df, threshold=0.01)
        assert len(report.dropped) == 0

    def test_report_reasons(self):
        df = pd.DataFrame({"constant": [5.0] * 50, "ok": np.random.randn(50)})
        selector = FeatureSelector()
        report = selector.variance_filter(df, threshold=0.01)
        assert "constant" in report.reasons


class TestImportanceFilter:
    def test_selects_top_k(self, iris_df: pd.DataFrame):
        selector = FeatureSelector()
        target = iris_df["target"]
        features = iris_df.drop(columns=["target"])
        report = selector.importance_filter(features, target, top_k=2)
        assert report.n_selected == 2
        assert report.n_dropped == 2

    def test_without_top_k(self, iris_df: pd.DataFrame):
        selector = FeatureSelector()
        target = iris_df["target"]
        features = iris_df.drop(columns=["target"])
        report = selector.importance_filter(features, target)
        assert report.n_selected == 4

    def test_regression_task(self, diabetes_df: pd.DataFrame):
        selector = FeatureSelector()
        target = diabetes_df["target"]
        features = diabetes_df.drop(columns=["target"])
        report = selector.importance_filter(features, target, top_k=5, task_type="regression")
        assert report.n_selected == 5


class TestRecursiveElimination:
    def test_rfe_classification(self, iris_df: pd.DataFrame):
        selector = FeatureSelector()
        target = iris_df["target"]
        features = iris_df.drop(columns=["target"])
        report = selector.recursive_elimination(features, target, min_features=2)
        assert report.n_selected == 2
        assert report.n_dropped == 2

    def test_rfe_regression(self, diabetes_df: pd.DataFrame):
        selector = FeatureSelector()
        target = diabetes_df["target"]
        features = diabetes_df.drop(columns=["target"])
        report = selector.recursive_elimination(
            features, target, min_features=5, task_type="regression",
        )
        assert report.n_selected == 5

    def test_rfe_few_features(self):
        df = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        target = pd.Series(np.random.choice([0, 1], 50))
        selector = FeatureSelector()
        report = selector.recursive_elimination(df, target, min_features=5)
        # Not enough features to eliminate
        assert report.n_selected == 2

    def test_feature_report_properties(self):
        report = FeatureReport(selected=["a", "b"], dropped=["c"], reasons={"c": "low importance"})
        assert report.n_selected == 2
        assert report.n_dropped == 1
