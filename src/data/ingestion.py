"""Data ingestion and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataIngester:
    """Ingest data from multiple sources into pandas DataFrames."""

    def ingest_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Read a CSV file into a DataFrame."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        logger.info("ingesting_csv", path=str(path))
        return pd.read_csv(path, **kwargs)

    def ingest_parquet(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        """Read a Parquet file into a DataFrame."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        logger.info("ingesting_parquet", path=str(path))
        return pd.read_parquet(path, **kwargs)

    def ingest_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame."""
        logger.info("ingesting_database", query_preview=query[:100])
        return pd.read_sql(query, connection_string)

    def ingest_api(self, url: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Fetch JSON data from an API and convert to DataFrame."""
        logger.info("ingesting_api", url=url)
        response = httpx.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "data" in data:
            return pd.DataFrame(data["data"])
        return pd.DataFrame([data])


@dataclass
class ValidationIssue:
    """A single validation issue."""

    check: str
    column: str | None
    message: str
    severity: str = "error"


@dataclass
class ValidationReport:
    """Result of data validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


class DataValidator:
    """Validate DataFrames using Great Expectations-inspired checks."""

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: list[str],
        dtypes: dict[str, str] | None = None,
    ) -> list[ValidationIssue]:
        """Check that expected columns exist and have correct dtypes."""
        issues: list[ValidationIssue] = []
        actual_cols = set(df.columns)

        for col in expected_columns:
            if col not in actual_cols:
                issues.append(
                    ValidationIssue(
                        check="schema",
                        column=col,
                        message=f"Missing expected column: {col}",
                    )
                )

        if dtypes:
            for col, expected_dtype in dtypes.items():
                if col in actual_cols:
                    actual_dtype = str(df[col].dtype)
                    if expected_dtype not in actual_dtype:
                        issues.append(
                            ValidationIssue(
                                check="schema",
                                column=col,
                                message=f"Column {col}: expected dtype {expected_dtype}, got {actual_dtype}",
                                severity="warning",
                            )
                        )
        return issues

    def validate_completeness(
        self,
        df: pd.DataFrame,
        max_null_pct: float = 0.1,
    ) -> list[ValidationIssue]:
        """Check that null percentages are within acceptable limits."""
        issues: list[ValidationIssue] = []
        n_rows = len(df)
        if n_rows == 0:
            issues.append(
                ValidationIssue(check="completeness", column=None, message="DataFrame is empty")
            )
            return issues

        for col in df.columns:
            null_pct = df[col].isnull().sum() / n_rows
            if null_pct > max_null_pct:
                issues.append(
                    ValidationIssue(
                        check="completeness",
                        column=col,
                        message=f"Column {col}: {null_pct:.1%} null (max {max_null_pct:.1%})",
                    )
                )
        return issues

    def validate_ranges(
        self,
        df: pd.DataFrame,
        column_ranges: dict[str, tuple[float, float]],
    ) -> list[ValidationIssue]:
        """Check that numeric columns fall within specified ranges."""
        issues: list[ValidationIssue] = []
        for col, (min_val, max_val) in column_ranges.items():
            if col not in df.columns:
                issues.append(
                    ValidationIssue(
                        check="range",
                        column=col,
                        message=f"Column {col} not found for range check",
                    )
                )
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(
                    ValidationIssue(
                        check="range",
                        column=col,
                        message=f"Column {col} is not numeric",
                        severity="warning",
                    )
                )
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < min_val:
                issues.append(
                    ValidationIssue(
                        check="range",
                        column=col,
                        message=f"Column {col}: min value {col_min} < {min_val}",
                    )
                )
            if col_max > max_val:
                issues.append(
                    ValidationIssue(
                        check="range",
                        column=col,
                        message=f"Column {col}: max value {col_max} > {max_val}",
                    )
                )
        return issues

    def validate(
        self,
        df: pd.DataFrame,
        expected_columns: list[str] | None = None,
        dtypes: dict[str, str] | None = None,
        max_null_pct: float = 0.1,
        column_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> ValidationReport:
        """Run all validation checks and return a report."""
        all_issues: list[ValidationIssue] = []

        if expected_columns is not None:
            all_issues.extend(self.validate_schema(df, expected_columns, dtypes))

        all_issues.extend(self.validate_completeness(df, max_null_pct))

        if column_ranges:
            all_issues.extend(self.validate_ranges(df, column_ranges))

        is_valid = all(i.severity != "error" for i in all_issues)

        stats = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        return ValidationReport(is_valid=is_valid, issues=all_issues, stats=stats)
