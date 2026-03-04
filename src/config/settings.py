"""Application settings via Pydantic Settings — loaded from env vars / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the AutoML pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "automl-pipeline"

    # Prefect
    prefect_api_url: str = "http://localhost:4200/api"

    # Storage
    s3_bucket: str = "automl-artifacts"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")

    # Logging
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Training defaults
    default_n_trials: int = 20
    default_cv_folds: int = 5
    default_test_size: float = 0.2
    default_val_size: float = 0.1


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings singleton."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings
