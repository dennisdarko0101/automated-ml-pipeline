"""Prefect-based ML pipeline orchestration."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash

from src.config.pipeline_config import PipelineConfig
from src.data.ingestion import DataIngester, DataValidator, ValidationReport
from src.data.splitter import DataSplitter, SplitResult
from src.deployment.model_registry import MLflowRegistry
from src.evaluation.comparator import ComparisonReport, ModelComparator
from src.evaluation.evaluator import ModelEvaluator
from src.features.engineer import FeatureEngineer, FeaturePipeline, PipelineStep
from src.features.selector import FeatureReport, FeatureSelector
from src.training.trainer import AutoTrainer, TrainResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Prefect Tasks ──────────────────────────────────────────────────────


@task(name="ingest_data", retries=2, retry_delay_seconds=30)
def ingest_data(config: PipelineConfig) -> pd.DataFrame:
    """Ingest data from the configured source."""
    ingester = DataIngester()
    source = config.data_source

    if source.endswith(".parquet"):
        return ingester.ingest_parquet(source)
    if source.endswith(".csv"):
        return ingester.ingest_csv(source)
    if source.startswith("http"):
        return ingester.ingest_api(source)
    return ingester.ingest_csv(source)


@task(name="validate_data", retries=1)
def validate_data(df: pd.DataFrame, config: PipelineConfig) -> ValidationReport:
    """Validate ingested data."""
    validator = DataValidator()
    report = validator.validate(df, expected_columns=[config.target_column])
    if not report.is_valid:
        logger.warning("data_validation_issues", errors=report.error_count)
    return report


@task(name="engineer_features")
def engineer_features(
    df: pd.DataFrame, config: PipelineConfig,
) -> tuple[pd.DataFrame, FeaturePipeline]:
    """Run feature engineering pipeline."""
    fc = config.feature_config
    pipeline = FeaturePipeline()
    pipeline.add_step("missing", "handle_missing", strategy=fc.missing_strategy)
    pipeline.add_step("encode", "encode_categorical", strategy=fc.categorical_strategy)
    pipeline.add_step("scale", "scale_numeric", strategy=fc.numeric_strategy)

    target = df[config.target_column]
    features = df.drop(columns=[config.target_column])
    features_transformed = pipeline.fit_transform(features, target)

    result = pd.concat([features_transformed, target.reset_index(drop=True)], axis=1)
    return result, pipeline


@task(name="select_features")
def select_features(
    df: pd.DataFrame, config: PipelineConfig,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Select best features."""
    selector = FeatureSelector()
    target = df[config.target_column]
    features = df.drop(columns=[config.target_column])

    # Correlation filter first
    corr_report = selector.correlation_filter(features, config.feature_config.correlation_threshold)
    features = features[corr_report.selected]

    # Variance filter
    var_report = selector.variance_filter(features, config.feature_config.variance_threshold)
    features = features[var_report.selected]

    # Importance filter if top_k specified
    if config.feature_config.top_k_features:
        imp_report = selector.importance_filter(
            features,
            target,
            top_k=config.feature_config.top_k_features,
            task_type=config.training_config.task_type,
        )
        features = features[imp_report.selected]
        report = imp_report
    else:
        report = var_report

    result = pd.concat([features, target.reset_index(drop=True)], axis=1)
    return result, report


@task(name="split_data")
def split_data(df: pd.DataFrame, config: PipelineConfig) -> SplitResult:
    """Split data into train/val/test."""
    splitter = DataSplitter(random_state=config.random_state)
    return splitter.train_val_test_split(
        df,
        target_column=config.target_column,
        test_size=config.test_size,
        val_size=config.val_size,
    )


@task(name="train_models")
def train_models(split: SplitResult, config: PipelineConfig) -> TrainResult:
    """Train all configured models."""
    tc = config.training_config
    trainer = AutoTrainer(
        task_type=tc.task_type,
        models=tc.models,
        n_trials=tc.n_trials,
        cv_folds=tc.cv_folds,
        metric=tc.metric,
        random_state=config.random_state,
    )
    return trainer.train(split.X_train, split.y_train, split.X_val, split.y_val)


@task(name="evaluate_models")
def evaluate_models(
    train_result: TrainResult, split: SplitResult, config: PipelineConfig,
) -> ComparisonReport:
    """Evaluate and compare all trained models."""
    ec = config.evaluation_config
    comparator = ModelComparator(
        primary_metric=ec.primary_metric,
        significance_level=ec.significance_level,
    )
    return comparator.compare(
        train_result.all_results,
        split.X_test,
        split.y_test,
        task_type=config.training_config.task_type,
    )


def _log_preprocessing(
    feature_pipeline: FeaturePipeline, selected_columns: list[str],
) -> None:
    """Log the fitted feature pipeline and selected columns as run artifacts.

    Predictions must reproduce the exact preprocessing used at training time
    (missing-value handling, encoding, scaling, then column selection). Storing
    these alongside the model lets the prediction endpoint transform raw inputs
    the same way before calling ``model.predict``.
    """
    import json
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    fp_path = tmp / "feature_pipeline.joblib"
    feature_pipeline.save(fp_path)
    cols_path = tmp / "selected_columns.json"
    cols_path.write_text(json.dumps(selected_columns))
    mlflow.log_artifact(str(fp_path), artifact_path="preprocessing")
    mlflow.log_artifact(str(cols_path), artifact_path="preprocessing")


@task(name="register_best_model")
def register_best_model(
    comparison: ComparisonReport,
    config: PipelineConfig,
    feature_pipeline: FeaturePipeline,
    selected_columns: list[str],
) -> str:
    """Register the best model in MLflow with its fitted preprocessing."""
    try:
        with mlflow.start_run(run_name=f"{config.name}_best"):
            report = comparison.evaluation_reports[comparison.best_model_name]
            mlflow.log_metrics(report.metrics)
            mlflow.sklearn.log_model(
                comparison.best_model,
                artifact_path="model",
                registered_model_name=config.name,
            )
            try:
                _log_preprocessing(feature_pipeline, selected_columns)
            except Exception as exc:  # registration must not be lost over this
                logger.warning("preprocessing_logging_failed", error=str(exc))
            run_id = mlflow.active_run().info.run_id
            logger.info("model_registered", model=comparison.best_model_name, run_id=run_id)
            return run_id
    except Exception as exc:
        logger.warning("model_registration_failed", error=str(exc))
        return ""


@task(name="promote_model")
def promote_model(
    comparison: ComparisonReport,
    split: SplitResult,
    config: PipelineConfig,
    run_id: str,
) -> dict[str, Any]:
    """Champion/challenger promotion.

    The first registered version becomes the champion (Production). On later
    runs, the new version is compared against the current Production model on the
    held-out test set; it is promoted only if it beats the champion by at least
    the configured ``min_improvement``, otherwise it is parked in Staging as a
    challenger.
    """
    if not run_id:
        return {"promoted": False, "stage": "None", "reason": "model was not registered"}

    name = config.name
    task_type = config.training_config.task_type
    min_improvement = config.evaluation_config.min_improvement

    try:
        client = mlflow.tracking.MlflowClient()
        versions = [v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id]
        if not versions:
            return {"promoted": False, "stage": "None", "reason": "registered version not found"}
        new_version = max(int(v.version) for v in versions)

        registry = MLflowRegistry(tracking_uri=mlflow.get_tracking_uri())
        try:
            production_model = registry.load_production_model(name)
        except Exception:
            production_model = None

        if production_model is None:
            client.transition_model_version_stage(
                name=name, version=str(new_version), stage="Production",
                archive_existing_versions=True,
            )
            logger.info("model_promoted", name=name, version=new_version, stage="Production")
            return {
                "promoted": True, "stage": "Production", "version": new_version,
                "reason": "Initial champion (no prior production model)",
            }

        decision = registry.compare_with_production(
            comparison.best_model, production_model,
            split.X_test, split.y_test,
            task_type=task_type, min_improvement=min_improvement,
        )
        target_stage = "Production" if decision.should_promote else "Staging"
        client.transition_model_version_stage(
            name=name, version=str(new_version), stage=target_stage,
            archive_existing_versions=decision.should_promote,
        )
        logger.info(
            "promotion_decision", name=name, version=new_version,
            promoted=decision.should_promote, stage=target_stage,
        )
        return {
            "promoted": decision.should_promote, "stage": target_stage,
            "version": new_version, "reason": decision.reason,
            "champion_score": decision.current_score, "challenger_score": decision.new_score,
        }
    except Exception as exc:
        logger.warning("promotion_failed", error=str(exc))
        return {"promoted": False, "stage": "unknown", "reason": f"error: {exc}"}


# ── Main Flow ──────────────────────────────────────────────────────────


@flow(name="ml_pipeline", log_prints=True)
def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute the full ML pipeline end-to-end."""
    logger.info("pipeline_started", name=config.name)

    # 1. Ingest
    df = ingest_data(config)

    # 2. Validate
    validation = validate_data(df, config)

    # 3. Feature Engineering
    df_engineered, feature_pipeline = engineer_features(df, config)

    # 4. Feature Selection
    df_selected, feature_report = select_features(df_engineered, config)

    # 5. Split
    split = split_data(df_selected, config)

    # 6. Train
    train_result = train_models(split, config)

    # 7. Evaluate
    comparison = evaluate_models(train_result, split, config)

    # 8. Register (with the fitted preprocessing so predictions can reproduce it)
    selected_columns = [c for c in df_selected.columns if c != config.target_column]
    run_id = register_best_model(comparison, config, feature_pipeline, selected_columns)

    # 9. Champion/challenger promotion
    promotion = promote_model(comparison, split, config, run_id)

    logger.info(
        "pipeline_completed",
        name=config.name,
        best_model=comparison.best_model_name,
        recommendation=comparison.recommendation,
        promotion_stage=promotion.get("stage"),
    )

    return {
        "pipeline_name": config.name,
        "best_model": comparison.best_model_name,
        "metrics": comparison.evaluation_reports[comparison.best_model_name].metrics,
        "recommendation": comparison.recommendation,
        "run_id": run_id,
        "n_features_selected": feature_report.n_selected,
        "promotion": promotion,
    }


class MLPipeline:
    """Wrapper around the Prefect flow for programmatic usage."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        """Run the full pipeline."""
        return run_pipeline(self.config)
