"""Tests for data ingestion and validation — 18 tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import DataIngester, DataValidator, ValidationReport


class TestDataIngester:
    def test_ingest_csv(self, iris_csv: Path):
        ingester = DataIngester()
        df = ingester.ingest_csv(iris_csv)
        assert len(df) == 150
        assert "target" in df.columns

    def test_ingest_csv_not_found(self):
        ingester = DataIngester()
        with pytest.raises(FileNotFoundError):
            ingester.ingest_csv("/nonexistent/file.csv")

    def test_ingest_parquet(self, iris_df: pd.DataFrame, tmp_path: Path):
        path = tmp_path / "data.parquet"
        iris_df.to_parquet(path, index=False)
        ingester = DataIngester()
        df = ingester.ingest_parquet(path)
        assert len(df) == 150

    def test_ingest_parquet_not_found(self):
        ingester = DataIngester()
        with pytest.raises(FileNotFoundError):
            ingester.ingest_parquet("/nonexistent/file.parquet")

    @patch("src.data.ingestion.httpx.get")
    def test_ingest_api_list_response(self, mock_get: MagicMock):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            raise_for_status=lambda: None,
        )
        ingester = DataIngester()
        df = ingester.ingest_api("http://example.com/data")
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    @patch("src.data.ingestion.httpx.get")
    def test_ingest_api_data_key(self, mock_get: MagicMock):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"x": 1}]},
            raise_for_status=lambda: None,
        )
        ingester = DataIngester()
        df = ingester.ingest_api("http://example.com/data")
        assert len(df) == 1

    @patch("src.data.ingestion.httpx.get")
    def test_ingest_api_single_object(self, mock_get: MagicMock):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"key": "value"},
            raise_for_status=lambda: None,
        )
        ingester = DataIngester()
        df = ingester.ingest_api("http://example.com/data")
        assert len(df) == 1

    def test_ingest_csv_with_kwargs(self, tmp_path: Path):
        p = tmp_path / "sep.csv"
        p.write_text("a;b\n1;2\n3;4\n")
        ingester = DataIngester()
        df = ingester.ingest_csv(p, sep=";")
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2


class TestDataValidator:
    def test_validate_schema_pass(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        issues = validator.validate_schema(iris_df, expected_columns=["target"])
        assert len(issues) == 0

    def test_validate_schema_missing_column(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        issues = validator.validate_schema(iris_df, expected_columns=["nonexistent"])
        assert len(issues) == 1
        assert issues[0].check == "schema"

    def test_validate_schema_wrong_dtype(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        issues = validator.validate_schema(
            iris_df, expected_columns=["target"], dtypes={"target": "object"},
        )
        assert any(i.severity == "warning" for i in issues)

    def test_validate_completeness_pass(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        issues = validator.validate_completeness(iris_df, max_null_pct=0.1)
        assert len(issues) == 0

    def test_validate_completeness_fail(self):
        df = pd.DataFrame({"a": [1, None, None, None, None]})
        validator = DataValidator()
        issues = validator.validate_completeness(df, max_null_pct=0.5)
        assert len(issues) == 1

    def test_validate_completeness_empty(self):
        validator = DataValidator()
        issues = validator.validate_completeness(pd.DataFrame(), max_null_pct=0.1)
        assert len(issues) == 1
        assert "empty" in issues[0].message.lower()

    def test_validate_ranges_pass(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        issues = validator.validate_ranges(iris_df, {"target": (0, 2)})
        assert len(issues) == 0

    def test_validate_ranges_fail(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        issues = validator.validate_ranges(iris_df, {"target": (0, 1)})
        assert len(issues) == 1

    def test_validate_ranges_missing_column(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        issues = validator.validate_ranges(iris_df, {"missing_col": (0, 1)})
        assert len(issues) == 1

    def test_full_validate_report(self, iris_df: pd.DataFrame):
        validator = DataValidator()
        report = validator.validate(
            iris_df,
            expected_columns=["target"],
            max_null_pct=0.05,
            column_ranges={"target": (0, 2)},
        )
        assert isinstance(report, ValidationReport)
        assert report.is_valid
        assert report.stats["n_rows"] == 150
        assert report.error_count == 0
