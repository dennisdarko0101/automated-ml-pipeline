"""Tests for feature engineering — 22 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer, FeaturePipeline, PipelineStep


class TestAutoDetectTypes:
    def test_numeric_columns(self, iris_df: pd.DataFrame):
        ct = FeatureEngineer.auto_detect_types(iris_df.drop(columns=["target"]))
        assert len(ct.numeric) == 4

    def test_categorical_columns(self, sample_df: pd.DataFrame):
        ct = FeatureEngineer.auto_detect_types(sample_df)
        assert "cat1" in ct.categorical
        assert "cat2" in ct.categorical

    def test_datetime_detection(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01", "2024-01-02"]), "val": [1, 2]})
        ct = FeatureEngineer.auto_detect_types(df)
        assert "dt" in ct.datetime

    def test_text_detection(self):
        df = pd.DataFrame({"text": [f"word_{i}" for i in range(100)]})
        ct = FeatureEngineer.auto_detect_types(df)
        assert "text" in ct.text


class TestEncodeCategorical:
    def test_onehot_encoding(self, sample_df: pd.DataFrame):
        eng = FeatureEngineer()
        result = eng.encode_categorical(sample_df, columns=["cat1"], strategy="onehot")
        assert "cat1" not in result.columns
        assert any("cat1_" in c for c in result.columns)

    def test_label_encoding(self, sample_df: pd.DataFrame):
        eng = FeatureEngineer()
        result = eng.encode_categorical(sample_df, columns=["cat1"], strategy="label")
        assert "cat1" in result.columns
        assert result["cat1"].dtype in (np.int64, np.intp)

    def test_target_encoding(self, sample_df: pd.DataFrame):
        eng = FeatureEngineer()
        target = sample_df["target"]
        result = eng.encode_categorical(
            sample_df.drop(columns=["target"]),
            columns=["cat1"],
            strategy="target",
            target=target,
        )
        assert pd.api.types.is_numeric_dtype(result["cat1"])

    def test_target_encoding_requires_target(self, sample_df: pd.DataFrame):
        eng = FeatureEngineer()
        with pytest.raises(ValueError, match="target series required"):
            eng.encode_categorical(sample_df, columns=["cat1"], strategy="target")

    def test_onehot_transform(self, sample_df: pd.DataFrame):
        eng = FeatureEngineer()
        eng.encode_categorical(sample_df, columns=["cat1"], strategy="onehot", fit=True)
        result = eng.encode_categorical(sample_df, columns=["cat1"], strategy="onehot", fit=False)
        assert "cat1" not in result.columns

    def test_auto_detect_categorical(self, sample_df: pd.DataFrame):
        eng = FeatureEngineer()
        result = eng.encode_categorical(sample_df, strategy="label")
        assert pd.api.types.is_numeric_dtype(result["cat1"])
        assert pd.api.types.is_numeric_dtype(result["cat2"])

    def test_no_categorical_columns(self, iris_df: pd.DataFrame):
        eng = FeatureEngineer()
        features = iris_df.drop(columns=["target"])
        result = eng.encode_categorical(features, strategy="onehot")
        assert result.shape == features.shape


class TestScaleNumeric:
    def test_standard_scaling(self, iris_df: pd.DataFrame):
        eng = FeatureEngineer()
        features = iris_df.drop(columns=["target"])
        result = eng.scale_numeric(features, strategy="standard")
        assert abs(result.iloc[:, 0].mean()) < 0.1

    def test_minmax_scaling(self, iris_df: pd.DataFrame):
        eng = FeatureEngineer()
        features = iris_df.drop(columns=["target"])
        result = eng.scale_numeric(features, strategy="minmax")
        assert result.iloc[:, 0].min() >= -0.01
        assert result.iloc[:, 0].max() <= 1.01

    def test_robust_scaling(self, iris_df: pd.DataFrame):
        eng = FeatureEngineer()
        features = iris_df.drop(columns=["target"])
        result = eng.scale_numeric(features, strategy="robust")
        assert result.shape == features.shape

    def test_scale_transform(self, iris_df: pd.DataFrame):
        eng = FeatureEngineer()
        features = iris_df.drop(columns=["target"])
        eng.scale_numeric(features, strategy="standard", fit=True)
        result = eng.scale_numeric(features, strategy="standard", fit=False)
        assert abs(result.iloc[:, 0].mean()) < 0.1


class TestHandleMissing:
    def test_mean_imputation(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10, 20, 30, 40]})
        eng = FeatureEngineer()
        result = eng.handle_missing(df, strategy="mean")
        assert result["a"].isnull().sum() == 0
        assert abs(result["a"].iloc[2] - 2.333) < 0.1

    def test_median_imputation(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        eng = FeatureEngineer()
        result = eng.handle_missing(df, strategy="median")
        assert result["a"].iloc[2] == 2.0

    def test_mode_imputation(self):
        df = pd.DataFrame({"a": ["x", "x", "y", None]})
        eng = FeatureEngineer()
        result = eng.handle_missing(df, strategy="mode")
        assert result["a"].iloc[3] == "x"

    def test_drop_strategy(self):
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
        eng = FeatureEngineer()
        result = eng.handle_missing(df, strategy="drop")
        assert len(result) == 2


class TestDatetimeFeatures:
    def test_create_datetime_features(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-15 10:30:00", "2024-06-22 14:00:00"]),
        })
        result = FeatureEngineer.create_datetime_features(df, "ts")
        assert "ts_hour" in result.columns
        assert "ts_month" in result.columns
        assert "ts_is_weekend" in result.columns
        assert "ts" not in result.columns
        assert result["ts_hour"].iloc[0] == 10
        assert result["ts_month"].iloc[1] == 6


class TestInteractionFeatures:
    def test_create_interactions(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = FeatureEngineer.create_interaction_features(df, ["a", "b", "c"])
        assert "a_x_b" in result.columns
        assert "a_x_c" in result.columns
        assert "b_x_c" in result.columns
        assert result["a_x_b"].iloc[0] == 4

    def test_no_interaction_degree_1(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = FeatureEngineer.create_interaction_features(df, ["a", "b"], degree=1)
        assert "a_x_b" not in result.columns


class TestFeaturePipeline:
    def test_fit_transform(self, sample_df: pd.DataFrame):
        pipe = FeaturePipeline()
        pipe.add_step("missing", "handle_missing", strategy="median")
        pipe.add_step("encode", "encode_categorical", strategy="label")
        pipe.add_step("scale", "scale_numeric", strategy="standard")

        target = sample_df["target"]
        features = sample_df.drop(columns=["target"])
        result = pipe.fit_transform(features, target)
        assert result.shape[0] == features.shape[0]
        assert pd.api.types.is_numeric_dtype(result["cat1"])

    def test_transform_without_fit_raises(self, sample_df: pd.DataFrame):
        pipe = FeaturePipeline()
        pipe.add_step("scale", "scale_numeric", strategy="standard")
        with pytest.raises(RuntimeError, match="fitted"):
            pipe.transform(sample_df)

    def test_save_and_load(self, sample_df: pd.DataFrame, tmp_path: Path):
        pipe = FeaturePipeline()
        pipe.add_step("encode", "encode_categorical", strategy="label")
        target = sample_df["target"]
        features = sample_df.drop(columns=["target"])
        pipe.fit_transform(features, target)

        save_path = tmp_path / "pipe.joblib"
        pipe.save(save_path)

        loaded = FeaturePipeline.load(save_path)
        result = loaded.transform(features)
        assert pd.api.types.is_numeric_dtype(result["cat1"])
