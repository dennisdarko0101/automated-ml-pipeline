"""Request and response schemas for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Pipeline ───────────────────────────────────────────────────────────


class PipelineRunRequest(BaseModel):
    """Request to trigger a pipeline run."""

    config_path: str = Field(..., description="Path to the pipeline YAML config file")


class PipelineRunResponse(BaseModel):
    """Response after triggering a pipeline run."""

    run_id: str
    pipeline_name: str
    status: str
    message: str = ""


class PipelineStatusResponse(BaseModel):
    """Status of a pipeline run."""

    run_id: str
    pipeline_name: str
    status: str
    start_time: datetime
    end_time: datetime | None = None
    metrics: dict[str, float] = {}
    error: str | None = None


class PipelineHistoryResponse(BaseModel):
    """History of pipeline runs."""

    runs: list[PipelineStatusResponse]


# ── Prediction ─────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request for model prediction."""

    model_name: str = Field(..., description="Name of the registered model")
    features: dict[str, Any] = Field(..., description="Feature key-value pairs")


class PredictResponse(BaseModel):
    """Prediction response."""

    model_name: str
    prediction: Any
    probabilities: list[float] | None = None


# ── Models ─────────────────────────────────────────────────────────────


class ModelVersionInfo(BaseModel):
    version: str
    stage: str
    run_id: str


class ModelInfo(BaseModel):
    name: str
    latest_versions: list[ModelVersionInfo] = []


class ModelListResponse(BaseModel):
    """List of registered models."""

    models: list[ModelInfo]


# ── Health ─────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"


# ── Errors ─────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    detail: str
