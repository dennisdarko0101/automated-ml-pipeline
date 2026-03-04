"""End-to-end integration tests — 9 tests using real sklearn datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from src.config.pipeline_config import (
    EvaluationConfig,
    FeatureConfig,
    PipelineConfig,
    TrainingConfig,
)
from src.data.ingestion import DataIngester, DataValidator
from src.data.splitter import DataSplitter
from src.evaluation.comparator import ModelComparator
from src.evaluation.evaluator import ModelEvaluator
from src.features.engineer import FeatureEngineer, FeaturePipeline
from src.features.selector import FeatureSelector
from src.training.trainer import AutoTrainer


class TestEndToEndClassification:
    """Full pipeline on the Iris dataset."""

    def test_full_classification_pipeline(self, iris_df: pd.DataFrame):
        # 1. Validate
        validator = DataValidator()
        report = validator.validate(iris_df, expected_columns=["target"])
        assert report.is_valid

        # 2. Feature engineering
        target = iris_df["target"]
        features = iris_df.drop(columns=["target"])
        eng = FeatureEngineer()
        features = eng.handle_missing(features, strategy="median")
        features = eng.scale_numeric(features, strategy="standard")

        # 3. Feature selection
        selector = FeatureSelector()
        sel_report = selector.variance_filter(features)
        features = features[sel_report.selected]

        # 4. Split
        df = pd.concat([features, target.reset_index(drop=True)], axis=1)
        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(df, "target", test_size=0.2, val_size=0.1)

        # 5. Train
        trainer = AutoTrainer(
            task_type="classification",
            models=["logistic_regression", "random_forest"],
            n_trials=2, cv_folds=3,
        )
        result = trainer.train(split.X_train, split.y_train, split.X_val, split.y_val)
        assert len(result.all_results) == 2

        # 6. Evaluate
        comparator = ModelComparator(primary_metric="f1", cv_folds=3)
        comparison = comparator.compare(
            result.all_results, split.X_test, split.y_test, task_type="classification",
        )
        assert comparison.best_model is not None
        assert comparison.evaluation_reports[comparison.best_model_name].metrics["f1"] > 0.5

    def test_classification_with_feature_pipeline(self, iris_df: pd.DataFrame):
        target = iris_df["target"]
        features = iris_df.drop(columns=["target"])

        pipeline = FeaturePipeline()
        pipeline.add_step("missing", "handle_missing", strategy="median")
        pipeline.add_step("scale", "scale_numeric", strategy="standard")
        features = pipeline.fit_transform(features, target)

        df = pd.concat([features, target.reset_index(drop=True)], axis=1)
        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(df, "target", test_size=0.2, val_size=0.1)

        trainer = AutoTrainer(
            task_type="classification", models=["random_forest"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(split.X_train, split.y_train, split.X_val, split.y_val)
        assert result.best_model.metrics["accuracy"] > 0.7

    def test_classification_from_csv(self, iris_csv: Path):
        ingester = DataIngester()
        df = ingester.ingest_csv(iris_csv)

        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(df, "target", test_size=0.2, val_size=0.1)

        trainer = AutoTrainer(
            task_type="classification", models=["logistic_regression"],
            n_trials=2, cv_folds=2,
        )
        result = trainer.train(split.X_train, split.y_train, split.X_val, split.y_val)
        assert result.best_model.metrics["f1"] > 0.5

    def test_cross_validation_folds(self, iris_df: pd.DataFrame):
        splitter = DataSplitter(random_state=42)
        folds = list(splitter.cross_validation_folds(iris_df, "target", n_folds=5))
        assert len(folds) == 5
        for X_tr, y_tr, X_val, y_val in folds:
            assert len(X_tr) > len(X_val)


class TestEndToEndRegression:
    """Full pipeline on the Diabetes dataset."""

    def test_full_regression_pipeline(self, diabetes_df: pd.DataFrame):
        # Validate
        validator = DataValidator()
        report = validator.validate(diabetes_df, expected_columns=["target"])
        assert report.is_valid

        # Feature engineering
        target = diabetes_df["target"]
        features = diabetes_df.drop(columns=["target"])
        eng = FeatureEngineer()
        features = eng.scale_numeric(features, strategy="standard")

        # Split
        df = pd.concat([features, target.reset_index(drop=True)], axis=1)
        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(df, "target", test_size=0.2, val_size=0.1, stratify=False)

        # Train
        trainer = AutoTrainer(
            task_type="regression",
            models=["linear_regression", "random_forest"],
            n_trials=2, cv_folds=3,
        )
        result = trainer.train(split.X_train, split.y_train, split.X_val, split.y_val)

        # Evaluate
        comparator = ModelComparator(primary_metric="r2", cv_folds=3)
        comparison = comparator.compare(
            result.all_results, split.X_test, split.y_test, task_type="regression",
        )
        assert comparison.best_model is not None
        assert "mae" in comparison.evaluation_reports[comparison.best_model_name].metrics

    def test_regression_with_interactions(self, diabetes_df: pd.DataFrame):
        target = diabetes_df["target"]
        features = diabetes_df.drop(columns=["target"])
        eng = FeatureEngineer()
        cols = list(features.columns[:3])
        features = eng.create_interaction_features(features, cols)
        assert features.shape[1] > diabetes_df.shape[1] - 1

    def test_regression_feature_selection(self, diabetes_df: pd.DataFrame):
        target = diabetes_df["target"]
        features = diabetes_df.drop(columns=["target"])
        selector = FeatureSelector()
        report = selector.importance_filter(features, target, top_k=5, task_type="regression")
        assert report.n_selected == 5

    def test_time_based_split(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "feature": np.random.randn(n),
            "target": np.random.randn(n),
        })
        splitter = DataSplitter(random_state=42)
        split = splitter.time_based_split(df, "target", "time", test_size=0.2, val_size=0.1)
        assert len(split.X_train) + len(split.X_val) + len(split.X_test) == n

    def test_data_splitter_shapes(self, iris_df: pd.DataFrame):
        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(iris_df, "target", test_size=0.2, val_size=0.1)
        shapes = split.shapes
        assert shapes["train"][0] > shapes["test"][0]
        assert shapes["train"][0] > shapes["val"][0]
