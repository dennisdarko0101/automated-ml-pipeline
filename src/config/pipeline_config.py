"""Pipeline configuration — loaded from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    numeric_strategy: str = "standard"
    categorical_strategy: str = "onehot"
    missing_strategy: str = "median"
    datetime_columns: list[str] = field(default_factory=list)
    interaction_columns: list[str] = field(default_factory=list)
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    top_k_features: int | None = None


@dataclass
class TrainingConfig:
    """Training configuration."""

    task_type: str = "classification"
    models: list[str] = field(default_factory=lambda: ["random_forest", "xgboost", "lightgbm"])
    n_trials: int = 20
    cv_folds: int = 5
    metric: str = "f1"
    timeout_seconds: int = 3600


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    primary_metric: str = "f1"
    significance_level: float = 0.05
    min_improvement: float = 0.01


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    name: str
    data_source: str
    target_column: str
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors (empty if valid)."""
        errors: list[str] = []
        if not self.name:
            errors.append("Pipeline name is required")
        if not self.data_source:
            errors.append("Data source is required")
        if not self.target_column:
            errors.append("Target column is required")
        if self.test_size <= 0 or self.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        if self.val_size < 0 or self.val_size >= 1:
            errors.append("val_size must be between 0 and 1")
        if self.training_config.task_type not in ("classification", "regression"):
            errors.append("task_type must be 'classification' or 'regression'")
        if self.training_config.n_trials < 1:
            errors.append("n_trials must be >= 1")
        return errors


def _build_nested(raw: dict[str, Any], cls: type, key: str) -> Any:
    sub = raw.get(key, {})
    return cls(**sub) if isinstance(sub, dict) else sub


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at the top level")

    feature_config = _build_nested(raw, FeatureConfig, "feature_config")
    training_config = _build_nested(raw, TrainingConfig, "training_config")
    evaluation_config = _build_nested(raw, EvaluationConfig, "evaluation_config")

    config = PipelineConfig(
        name=raw.get("name", ""),
        data_source=raw.get("data_source", ""),
        target_column=raw.get("target_column", ""),
        feature_config=feature_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        test_size=raw.get("test_size", 0.2),
        val_size=raw.get("val_size", 0.1),
        random_state=raw.get("random_state", 42),
    )

    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid pipeline config: {'; '.join(errors)}")

    return config
