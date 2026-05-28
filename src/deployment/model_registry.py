"""MLflow model registry wrapper for production model management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromotionResult:
    """Result of a model promotion decision."""

    should_promote: bool
    reason: str
    current_score: float
    new_score: float


class MLflowRegistry:
    """Wrapper around MLflow model registry for production workflows."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        # Default to the MLFLOW_TRACKING_URI env var so the registry talks to the
        # configured tracking server (e.g. the mlflow service in Docker) rather
        # than always assuming localhost.
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)

    def register_model(
        self,
        model: Any,
        name: str,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
    ) -> str:
        """Register a model with MLflow and return the run ID."""
        with mlflow.start_run(run_name=f"register_{name}") as run:
            mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.sklearn.log_model(
                model, artifact_path="model", registered_model_name=name,
            )
            logger.info("model_registered", name=name, run_id=run.info.run_id)
            return run.info.run_id

    def promote_model(self, name: str, version: int, stage: str = "Production") -> None:
        """Promote a model version to a given stage."""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name, version=str(version), stage=stage,
        )
        logger.info("model_promoted", name=name, version=version, stage=stage)

    def load_production_model(self, name: str) -> Any:
        """Load the latest Production-stage model."""
        model_uri = f"models:/{name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("production_model_loaded", name=name)
        return model

    def load_model_by_version(self, name: str, version: int) -> Any:
        """Load a specific model version."""
        model_uri = f"models:/{name}/{version}"
        return mlflow.sklearn.load_model(model_uri)

    def load_production_preprocessing(self, name: str) -> tuple[Any, list[str] | None]:
        """Load the fitted feature pipeline and selected columns for the champion.

        Returns ``(pipeline, columns)`` for the current Production version so the
        prediction path can apply the same preprocessing used during training.
        Returns ``(None, None)`` if no preprocessing artifacts were stored (for
        example, a model registered before preprocessing logging was added).
        """
        import json
        from pathlib import Path

        from src.features.engineer import FeaturePipeline

        client = mlflow.tracking.MlflowClient()
        prod = [
            v for v in client.search_model_versions(f"name='{name}'")
            if v.current_stage == "Production"
        ]
        if not prod:
            return None, None

        run_id = max(prod, key=lambda v: int(v.version)).run_id
        local_dir = Path(
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing")
        )
        pipeline = FeaturePipeline.load(local_dir / "feature_pipeline.joblib")
        columns = json.loads((local_dir / "selected_columns.json").read_text())
        return pipeline, columns

    def compare_with_production(
        self,
        new_model: Any,
        production_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = "classification",
        min_improvement: float = 0.01,
    ) -> PromotionResult:
        """Compare a new model against the current production model."""
        new_preds = new_model.predict(X_test)
        prod_preds = production_model.predict(X_test)

        if task_type == "classification":
            new_score = float(f1_score(y_test, new_preds, average="weighted", zero_division=0))
            prod_score = float(f1_score(y_test, prod_preds, average="weighted", zero_division=0))
            improvement = new_score - prod_score
            should_promote = improvement >= min_improvement
            direction = "higher is better"
        else:
            new_score = float(mean_squared_error(y_test, new_preds))
            prod_score = float(mean_squared_error(y_test, prod_preds))
            improvement = prod_score - new_score  # Lower MSE is better
            should_promote = improvement >= min_improvement
            direction = "lower is better"

        if should_promote:
            reason = f"New model improves by {improvement:.4f} ({direction})"
        else:
            reason = f"New model does not improve sufficiently (delta={improvement:.4f}, threshold={min_improvement})"

        return PromotionResult(
            should_promote=should_promote,
            reason=reason,
            current_score=prod_score,
            new_score=new_score,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models with their latest version per stage.

        Built from ``search_model_versions`` rather than
        ``search_registered_models`` so it stays correct when the MLflow client
        and tracking server are different versions: a newer client applies a
        registered-models search filter that an older server does not honour,
        which silently returns no models even though versions exist.
        """
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions()

        by_name: dict[str, dict[str, dict[str, Any]]] = {}
        for v in versions:
            stage = v.current_stage or "None"
            stages = by_name.setdefault(v.name, {})
            existing = stages.get(stage)
            if existing is None or int(v.version) > int(existing["version"]):
                stages[stage] = {"version": v.version, "stage": stage, "run_id": v.run_id}

        return [
            {"name": name, "latest_versions": list(stages.values())}
            for name, stages in sorted(by_name.items())
        ]
