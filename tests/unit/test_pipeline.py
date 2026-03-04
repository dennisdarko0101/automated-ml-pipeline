"""Tests for orchestration pipeline — 16 tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.pipeline_config import (
    EvaluationConfig,
    FeatureConfig,
    PipelineConfig,
    TrainingConfig,
    load_pipeline_config,
)
from src.data.splitter import DataSplitter, SplitResult
from src.orchestration.pipeline import (
    engineer_features,
    evaluate_models,
    ingest_data,
    register_best_model,
    select_features,
    split_data,
    train_models,
    validate_data,
)
from src.orchestration.scheduler import PipelineScheduler


def _make_config(data_source: str, task_type: str = "classification") -> PipelineConfig:
    return PipelineConfig(
        name="test-pipeline",
        data_source=data_source,
        target_column="target",
        feature_config=FeatureConfig(),
        training_config=TrainingConfig(
            task_type=task_type, models=["logistic_regression", "random_forest"],
            n_trials=2, cv_folds=2,
        ),
        evaluation_config=EvaluationConfig(primary_metric="f1" if task_type == "classification" else "r2"),
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )


class TestPipelineConfig:
    def test_load_from_yaml(self, classification_config_path: Path):
        config = load_pipeline_config(classification_config_path)
        assert config.name == "test-classification"
        assert config.target_column == "target"

    def test_load_regression_config(self, regression_config_path: Path):
        config = load_pipeline_config(regression_config_path)
        assert config.training_config.task_type == "regression"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("/nonexistent.yaml")

    def test_validation_errors(self):
        config = PipelineConfig(name="", data_source="", target_column="")
        errors = config.validate()
        assert len(errors) >= 3

    def test_invalid_task_type(self):
        config = PipelineConfig(
            name="test", data_source="data.csv", target_column="target",
            training_config=TrainingConfig(task_type="unsupervised"),
        )
        errors = config.validate()
        assert any("task_type" in e for e in errors)


class TestIngestTask:
    def test_ingest_csv(self, iris_csv: Path):
        config = _make_config(str(iris_csv))
        df = ingest_data.fn(config)
        assert len(df) == 150

    def test_ingest_parquet(self, iris_df: pd.DataFrame, tmp_path: Path):
        path = tmp_path / "data.parquet"
        iris_df.to_parquet(path, index=False)
        config = _make_config(str(path))
        df = ingest_data.fn(config)
        assert len(df) == 150


class TestValidateTask:
    def test_valid_data(self, iris_df: pd.DataFrame, iris_csv: Path):
        config = _make_config(str(iris_csv))
        report = validate_data.fn(iris_df, config)
        assert report.is_valid


class TestEngineerTask:
    def test_engineer_features(self, iris_df: pd.DataFrame, iris_csv: Path):
        config = _make_config(str(iris_csv))
        result_df, pipeline = engineer_features.fn(iris_df, config)
        assert "target" in result_df.columns
        assert pipeline._is_fitted


class TestSelectTask:
    def test_select_features(self, iris_df: pd.DataFrame, iris_csv: Path):
        config = _make_config(str(iris_csv))
        result_df, report = select_features.fn(iris_df, config)
        assert "target" in result_df.columns
        assert report.n_selected > 0


class TestSplitTask:
    def test_split_data(self, iris_df: pd.DataFrame, iris_csv: Path):
        config = _make_config(str(iris_csv))
        split = split_data.fn(iris_df, config)
        assert isinstance(split, SplitResult)
        assert len(split.X_train) > len(split.X_test)


class TestTrainTask:
    def test_train_models(self, iris_df: pd.DataFrame, iris_csv: Path):
        config = _make_config(str(iris_csv))
        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(iris_df, "target", test_size=0.2, val_size=0.1)
        result = train_models.fn(split, config)
        assert len(result.all_results) == 2


class TestEvaluateTask:
    def test_evaluate_models(self, iris_df: pd.DataFrame, iris_csv: Path):
        config = _make_config(str(iris_csv))
        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(iris_df, "target", test_size=0.2, val_size=0.1)
        train_result = train_models.fn(split, config)
        comparison = evaluate_models.fn(train_result, split, config)
        assert comparison.best_model_name in ("logistic_regression", "random_forest")


class TestRegisterTask:
    def test_register_best_model(self, iris_df: pd.DataFrame, iris_csv: Path):
        config = _make_config(str(iris_csv))
        splitter = DataSplitter(random_state=42)
        split = splitter.train_val_test_split(iris_df, "target", test_size=0.2, val_size=0.1)
        train_result = train_models.fn(split, config)
        comparison = evaluate_models.fn(train_result, split, config)
        run_id = register_best_model.fn(comparison, config)
        assert isinstance(run_id, str)


class TestPipelineScheduler:
    def test_schedule(self):
        scheduler = PipelineScheduler()
        scheduler.schedule("my-pipeline", "0 0 * * *")
        assert scheduler.get_schedule("my-pipeline") == "0 0 * * *"

    def test_list_schedules(self):
        scheduler = PipelineScheduler()
        scheduler.schedule("a", "0 * * * *")
        scheduler.schedule("b", "0 0 * * *")
        schedules = scheduler.list_schedules()
        assert len(schedules) == 2

    def test_get_history_empty(self):
        scheduler = PipelineScheduler()
        assert scheduler.get_history("nonexistent") == []
