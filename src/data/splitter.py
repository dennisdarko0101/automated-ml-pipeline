"""Data splitting utilities for train/validation/test sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    """Container for train/val/test splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "train": self.X_train.shape,
            "val": self.X_val.shape,
            "test": self.X_test.shape,
        }


class DataSplitter:
    """Split data into train/validation/test sets."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
    ) -> SplitResult:
        """Split data into train, validation, and test sets."""
        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify_col = y if stratify and self._is_categorical(y) else None

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_col,
        )

        # val_size is relative to the remaining data
        relative_val = val_size / (1 - test_size) if val_size > 0 else 0

        if relative_val > 0:
            stratify_temp = y_temp if stratify and self._is_categorical(y_temp) else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=relative_val,
                random_state=self.random_state, stratify=stratify_temp,
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, pd.DataFrame(), y_temp, pd.Series(dtype=y.dtype)

        logger.info(
            "data_split",
            train=len(X_train), val=len(X_val), test=len(X_test),
        )
        return SplitResult(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
        )

    def time_based_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        time_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> SplitResult:
        """Split data by time — no shuffling, preserves temporal order."""
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
        n = len(df_sorted)

        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size))

        train_df = df_sorted.iloc[:val_start]
        val_df = df_sorted.iloc[val_start:test_start]
        test_df = df_sorted.iloc[test_start:]

        return SplitResult(
            X_train=train_df.drop(columns=[target_column]),
            X_val=val_df.drop(columns=[target_column]),
            X_test=test_df.drop(columns=[target_column]),
            y_train=train_df[target_column],
            y_val=val_df[target_column],
            y_test=test_df[target_column],
        )

    def cross_validation_folds(
        self,
        df: pd.DataFrame,
        target_column: str,
        n_folds: int = 5,
        stratify: bool = True,
    ) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """Generate cross-validation folds."""
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if stratify and self._is_categorical(y):
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splitter = kf.split(X, y)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splitter = kf.split(X)

        for train_idx, val_idx in splitter:
            yield (
                X.iloc[train_idx],
                y.iloc[train_idx],
                X.iloc[val_idx],
                y.iloc[val_idx],
            )

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        """Check if a series is categorical (for stratification)."""
        return series.dtype == "object" or series.dtype.name == "category" or series.nunique() < 50
