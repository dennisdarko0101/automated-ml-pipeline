"""Feature engineering — automatic type detection, encoding, scaling, and pipeline chaining."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnTypes:
    """Detected column types."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)


class FeatureEngineer:
    """Stateless feature engineering operations.

    Each method returns a transformed DataFrame plus optional fitted artifacts
    stored on the instance for later transform-only usage.
    """

    def __init__(self) -> None:
        self._encoders: dict[str, Any] = {}
        self._scalers: dict[str, Any] = {}
        self._fill_values: dict[str, Any] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    @staticmethod
    def auto_detect_types(df: pd.DataFrame) -> ColumnTypes:
        """Heuristically detect column types."""
        ct = ColumnTypes()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                ct.datetime.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 15 and df[col].nunique() / max(len(df), 1) < 0.05:
                    ct.categorical.append(col)
                else:
                    ct.numeric.append(col)
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                if df[col].nunique() > 50:
                    ct.text.append(col)
                else:
                    ct.categorical.append(col)
            else:
                ct.categorical.append(col)
        return ct

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        strategy: Literal["onehot", "label", "target"] = "onehot",
        target: pd.Series | None = None,
        *,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Encode categorical columns."""
        df = df.copy()
        if columns is None:
            columns = [c for c in df.columns if df[c].dtype == object or df[c].dtype.name == "category"]
        if not columns:
            return df

        if strategy == "onehot":
            return self._onehot_encode(df, columns, fit=fit)
        if strategy == "label":
            return self._label_encode(df, columns, fit=fit)
        if strategy == "target":
            if target is None:
                raise ValueError("target series required for target encoding")
            return self._target_encode(df, columns, target, fit=fit)
        raise ValueError(f"Unknown encoding strategy: {strategy}")

    def _onehot_encode(self, df: pd.DataFrame, columns: list[str], *, fit: bool) -> pd.DataFrame:
        if fit:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = enc.fit_transform(df[columns])
            self._encoders["onehot"] = enc
            self._encoders["onehot_columns"] = columns
        else:
            enc = self._encoders["onehot"]
            encoded = enc.transform(df[columns])

        feature_names = enc.get_feature_names_out(columns)
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        df = df.drop(columns=columns)
        return pd.concat([df, encoded_df], axis=1)

    def _label_encode(self, df: pd.DataFrame, columns: list[str], *, fit: bool) -> pd.DataFrame:
        for col in columns:
            if fit:
                le = LabelEncoder()
                # Handle unseen values by fitting on the union of train categories
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                self._encoders[f"label_{col}"] = le
            else:
                le = self._encoders[f"label_{col}"]
                df[col] = df[col].astype(str)
                # Map unseen labels to -1
                classes = set(le.classes_)
                df[col] = df[col].apply(lambda x, c=classes, l=le: l.transform([x])[0] if x in c else -1)
        return df

    def _target_encode(
        self, df: pd.DataFrame, columns: list[str], target: pd.Series, *, fit: bool,
    ) -> pd.DataFrame:
        for col in columns:
            if fit:
                means = target.groupby(df[col]).mean()
                global_mean = target.mean()
                self._encoders[f"target_{col}"] = {"means": means.to_dict(), "global": global_mean}
                df[col] = df[col].map(means).fillna(global_mean)
            else:
                info = self._encoders[f"target_{col}"]
                df[col] = df[col].map(info["means"]).fillna(info["global"])
        return df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_numeric(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        strategy: Literal["standard", "minmax", "robust"] = "standard",
        *,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Scale numeric columns."""
        df = df.copy()
        if columns is None:
            columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not columns:
            return df

        if fit:
            scaler_cls = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[strategy]
            scaler = scaler_cls()
            df[columns] = scaler.fit_transform(df[columns])
            self._scalers["numeric"] = scaler
            self._scalers["numeric_columns"] = columns
        else:
            scaler = self._scalers["numeric"]
            df[columns] = scaler.transform(df[columns])
        return df

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    def handle_missing(
        self,
        df: pd.DataFrame,
        strategy: Literal["mean", "median", "mode", "drop"] = "median",
        columns: list[str] | None = None,
        *,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Handle missing values."""
        df = df.copy()
        if strategy == "drop":
            return df.dropna(subset=columns)

        target_cols = columns or [c for c in df.columns if df[c].isnull().any()]

        for col in target_cols:
            if col not in df.columns:
                continue
            if fit:
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    fill = df[col].mean()
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    fill = df[col].median()
                else:
                    fill = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
                self._fill_values[col] = fill
            else:
                fill = self._fill_values.get(col)
            if fill is not None:
                df[col] = df[col].fillna(fill)
        return df

    # ------------------------------------------------------------------
    # Datetime features
    # ------------------------------------------------------------------

    @staticmethod
    def create_datetime_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Extract temporal features from a datetime column."""
        df = df.copy()
        dt = pd.to_datetime(df[column])
        prefix = column
        df[f"{prefix}_hour"] = dt.dt.hour
        df[f"{prefix}_day"] = dt.dt.day
        df[f"{prefix}_month"] = dt.dt.month
        df[f"{prefix}_dayofweek"] = dt.dt.dayofweek
        df[f"{prefix}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        df = df.drop(columns=[column])
        return df

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------

    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame, columns: list[str], degree: int = 2,
    ) -> pd.DataFrame:
        """Create polynomial interaction features between specified columns."""
        df = df.copy()
        if degree < 2:
            return df
        for i, c1 in enumerate(columns):
            for c2 in columns[i + 1 :]:
                if pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2]):
                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
        return df


# ======================================================================
# Feature Pipeline — chainable fit/transform API
# ======================================================================

@dataclass
class PipelineStep:
    """One step in a feature pipeline."""

    name: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)


class FeaturePipeline:
    """Chain multiple FeatureEngineer operations into a fit/transform pipeline."""

    def __init__(self, steps: list[PipelineStep] | None = None) -> None:
        self.steps = steps or []
        self.engineer = FeatureEngineer()
        self._is_fitted = False

    def add_step(self, name: str, method: str, **params: Any) -> FeaturePipeline:
        self.steps.append(PipelineStep(name=name, method=method, params=params))
        return self

    def fit_transform(self, df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
        """Fit all steps and transform the data."""
        result = df.copy()
        for step in self.steps:
            fn = getattr(self.engineer, step.method)
            params = {**step.params, "fit": True} if "fit" in fn.__code__.co_varnames else step.params
            if "target" in fn.__code__.co_varnames and target is not None:
                params["target"] = target
            result = fn(result, **params)
        self._is_fitted = True
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted parameters (no fitting)."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        result = df.copy()
        for step in self.steps:
            fn = getattr(self.engineer, step.method)
            params = {**step.params}
            if "fit" in fn.__code__.co_varnames:
                params["fit"] = False
            result = fn(result, **params)
        return result

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "steps": [(s.name, s.method, s.params) for s in self.steps],
                "engineer_encoders": self.engineer._encoders,
                "engineer_scalers": self.engineer._scalers,
                "engineer_fill_values": self.engineer._fill_values,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FeaturePipeline:
        """Load a fitted pipeline from disk."""
        data = joblib.load(path)
        pipe = cls()
        pipe.steps = [PipelineStep(name=n, method=m, params=p) for n, m, p in data["steps"]]
        pipe.engineer._encoders = data["engineer_encoders"]
        pipe.engineer._scalers = data["engineer_scalers"]
        pipe.engineer._fill_values = data["engineer_fill_values"]
        pipe._is_fitted = True
        return pipe
