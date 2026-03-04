"""Feature selection — correlation, variance, importance, and RFE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureReport:
    """Report on feature selection results."""

    selected: list[str]
    dropped: list[str]
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.selected)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)


class FeatureSelector:
    """Select features using multiple strategies."""

    def correlation_filter(
        self, df: pd.DataFrame, threshold: float = 0.95,
    ) -> FeatureReport:
        """Remove highly correlated features (keeps the first of each pair)."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return FeatureReport(selected=list(df.columns), dropped=[])

        corr = numeric_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop: list[str] = []
        reasons: dict[str, str] = {}
        for col in upper.columns:
            high_corr = upper.index[upper[col] > threshold].tolist()
            if high_corr:
                to_drop.append(col)
                reasons[col] = f"Correlated (>{threshold}) with {high_corr[0]}"

        selected = [c for c in df.columns if c not in to_drop]
        return FeatureReport(selected=selected, dropped=to_drop, reasons=reasons)

    def variance_filter(
        self, df: pd.DataFrame, threshold: float = 0.01,
    ) -> FeatureReport:
        """Remove features with variance below threshold."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return FeatureReport(selected=list(df.columns), dropped=[])

        variances = numeric_df.var()
        low_var = variances[variances < threshold].index.tolist()

        reasons = {col: f"Variance {variances[col]:.6f} < {threshold}" for col in low_var}
        selected = [c for c in df.columns if c not in low_var]
        return FeatureReport(selected=selected, dropped=low_var, reasons=reasons)

    def importance_filter(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        model: object | None = None,
        top_k: int | None = None,
        task_type: str = "classification",
    ) -> FeatureReport:
        """Keep top-k features by tree-based feature importance."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return FeatureReport(selected=list(df.columns), dropped=[])

        if model is None:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        model.fit(numeric_df, target)  # type: ignore[union-attr]
        importances = pd.Series(model.feature_importances_, index=numeric_df.columns)  # type: ignore[union-attr]
        importances = importances.sort_values(ascending=False)

        if top_k is None:
            top_k = len(importances)

        selected_numeric = importances.head(top_k).index.tolist()
        dropped_numeric = importances.tail(len(importances) - top_k).index.tolist()

        # Keep non-numeric columns as-is
        non_numeric = [c for c in df.columns if c not in numeric_df.columns]
        selected = non_numeric + selected_numeric

        reasons = {
            col: f"Importance rank {i + top_k + 1}/{len(importances)}"
            for i, col in enumerate(dropped_numeric)
        }
        return FeatureReport(selected=selected, dropped=dropped_numeric, reasons=reasons)

    def recursive_elimination(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        model: object | None = None,
        min_features: int = 5,
        task_type: str = "classification",
    ) -> FeatureReport:
        """Recursive feature elimination using an estimator."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) <= min_features:
            return FeatureReport(selected=list(df.columns), dropped=[])

        if model is None:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        n_select = min(min_features, len(numeric_df.columns))
        rfe = RFE(estimator=model, n_features_to_select=n_select, step=1)  # type: ignore[arg-type]
        rfe.fit(numeric_df, target)

        mask = rfe.support_
        selected_numeric = numeric_df.columns[mask].tolist()
        dropped_numeric = numeric_df.columns[~mask].tolist()

        non_numeric = [c for c in df.columns if c not in numeric_df.columns]
        selected = non_numeric + selected_numeric

        rankings = rfe.ranking_
        reasons = {
            col: f"RFE ranking {rankings[i]}"
            for i, col in enumerate(numeric_df.columns)
            if not mask[i]
        }
        return FeatureReport(selected=selected, dropped=dropped_numeric, reasons=reasons)
