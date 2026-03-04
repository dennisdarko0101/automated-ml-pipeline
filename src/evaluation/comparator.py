"""Model comparison and ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.evaluation.evaluator import EvalReport, ModelEvaluator
from src.training.trainer import ModelCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonReport:
    """Report comparing multiple model candidates."""

    rankings: list[dict[str, Any]]
    best_model_name: str
    best_model: Any
    evaluation_reports: dict[str, EvalReport]
    significance_results: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    @property
    def comparison_table(self) -> pd.DataFrame:
        rows = []
        for entry in self.rankings:
            rows.append(
                {
                    "rank": entry["rank"],
                    "model": entry["model_name"],
                    **entry["metrics"],
                    "training_time": entry["training_time"],
                }
            )
        return pd.DataFrame(rows)


class ModelComparator:
    """Compare and rank model candidates."""

    def __init__(
        self,
        primary_metric: str = "f1",
        significance_level: float = 0.05,
        cv_folds: int = 5,
    ) -> None:
        self.primary_metric = primary_metric
        self.significance_level = significance_level
        self.evaluator = ModelEvaluator(cv_folds=cv_folds)

    def compare(
        self,
        candidates: list[ModelCandidate],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = "classification",
    ) -> ComparisonReport:
        """Compare all candidates and produce a ranked report."""
        eval_reports: dict[str, EvalReport] = {}
        for c in candidates:
            eval_reports[c.model_name] = self.evaluator.evaluate(
                c.model, X_test, y_test, task_type,
            )

        # Build rankings
        sort_ascending = self.primary_metric in ("mae", "mse", "rmse", "mape")
        ranked = sorted(
            candidates,
            key=lambda c: eval_reports[c.model_name].metrics.get(self.primary_metric, 0),
            reverse=not sort_ascending,
        )

        rankings = []
        for rank, c in enumerate(ranked, 1):
            rankings.append(
                {
                    "rank": rank,
                    "model_name": c.model_name,
                    "metrics": eval_reports[c.model_name].metrics,
                    "training_time": c.training_time,
                }
            )

        # Statistical significance between top 2
        sig_results: dict[str, Any] = {}
        if len(ranked) >= 2:
            sig_results = self._significance_test(
                eval_reports[ranked[0].model_name],
                eval_reports[ranked[1].model_name],
                ranked[0].model_name,
                ranked[1].model_name,
            )

        best = ranked[0]
        recommendation = self._generate_recommendation(
            best, eval_reports[best.model_name], sig_results,
        )

        return ComparisonReport(
            rankings=rankings,
            best_model_name=best.model_name,
            best_model=best.model,
            evaluation_reports=eval_reports,
            significance_results=sig_results,
            recommendation=recommendation,
        )

    def _significance_test(
        self,
        report_a: EvalReport,
        report_b: EvalReport,
        name_a: str,
        name_b: str,
    ) -> dict[str, Any]:
        """Paired t-test on CV scores between top two models."""
        scores_a = report_a.cv_scores
        scores_b = report_b.cv_scores
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {"test": "insufficient_data"}

        min_len = min(len(scores_a), len(scores_b))
        t_stat, p_value = sp_stats.ttest_rel(scores_a[:min_len], scores_b[:min_len])

        return {
            "test": "paired_t_test",
            "model_a": name_a,
            "model_b": name_b,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < self.significance_level,
        }

    def _generate_recommendation(
        self,
        best: ModelCandidate,
        report: EvalReport,
        sig: dict[str, Any],
    ) -> str:
        """Generate a human-readable recommendation."""
        metric_val = report.metrics.get(self.primary_metric, 0)
        rec = (
            f"Best model: {best.model_name} "
            f"({self.primary_metric}={metric_val:.4f}, "
            f"CV mean={report.cv_mean:.4f} +/- {report.cv_std:.4f})"
        )
        if sig.get("significant"):
            rec += f". Statistically significantly better than {sig['model_b']} (p={sig['p_value']:.4f})."
        elif sig.get("test") == "paired_t_test":
            rec += f". NOT statistically significantly better than {sig['model_b']} (p={sig['p_value']:.4f}); consider the simpler model."
        return rec
