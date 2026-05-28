"""FastAPI application — REST API for the AutoML pipeline."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    ModelVersionInfo,
    PipelineHistoryResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
    PredictRequest,
    PredictResponse,
)
from src.config.pipeline_config import load_pipeline_config
from src.deployment.model_registry import MLflowRegistry
from src.orchestration.scheduler import PipelineRun, PipelineScheduler
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Module-level state
scheduler = PipelineScheduler()
_background_runs: dict[str, PipelineRun] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("api_started")
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="Automated ML Pipeline API",
    description="Production AutoML system — trigger pipelines, predict, manage models.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ─────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


# ── Pipeline ───────────────────────────────────────────────────────────


@app.post(
    "/api/v1/pipeline/run",
    response_model=PipelineRunResponse,
    responses={400: {"model": ErrorResponse}},
)
async def trigger_pipeline(request: PipelineRunRequest):
    """Trigger a pipeline run in the background."""
    try:
        config = load_pipeline_config(request.config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    def _run():
        run = scheduler.run_now(config)
        _background_runs[run.run_id] = run

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return PipelineRunResponse(
        run_id="pending",
        pipeline_name=config.name,
        status="started",
        message="Pipeline run triggered in background",
    )


@app.get(
    "/api/v1/pipeline/status/{run_id}",
    response_model=PipelineStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def pipeline_status(run_id: str):
    """Get status of a pipeline run."""
    run = _background_runs.get(run_id)
    if run is None:
        # Search in scheduler history
        for runs in scheduler._history.values():
            for r in runs:
                if r.run_id == run_id:
                    run = r
                    break
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return PipelineStatusResponse(
        run_id=run.run_id,
        pipeline_name=run.pipeline_name,
        status=run.status,
        start_time=run.start_time,
        end_time=run.end_time,
        metrics=run.metrics,
        error=run.error,
    )


@app.get("/api/v1/pipeline/history", response_model=PipelineHistoryResponse)
async def pipeline_history(pipeline_name: str | None = None):
    """Get pipeline run history."""
    all_runs: list[PipelineRun] = []
    if pipeline_name:
        all_runs = scheduler.get_history(pipeline_name)
    else:
        for runs in scheduler._history.values():
            all_runs.extend(runs)

    return PipelineHistoryResponse(
        runs=[
            PipelineStatusResponse(
                run_id=r.run_id,
                pipeline_name=r.pipeline_name,
                status=r.status,
                start_time=r.start_time,
                end_time=r.end_time,
                metrics=r.metrics,
                error=r.error,
            )
            for r in all_runs
        ]
    )


# ── Prediction ─────────────────────────────────────────────────────────


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def predict(request: PredictRequest):
    """Make a prediction using a registered model."""
    try:
        registry = MLflowRegistry()
        model = registry.load_production_model(request.model_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}")

    try:
        df = pd.DataFrame([request.features])
        # Apply the same preprocessing (scaling, encoding, feature selection) the
        # model was trained on; without this, raw inputs produce wrong predictions.
        pipeline, columns = registry.load_production_preprocessing(request.model_name)
        if pipeline is not None:
            df = pipeline.transform(df)
            if columns:
                df = df[columns]
        prediction = model.predict(df)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[0].tolist()

        return PredictResponse(
            model_name=request.model_name,
            prediction=prediction[0].item() if hasattr(prediction[0], "item") else prediction[0],
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


# ── Models ─────────────────────────────────────────────────────────────


@app.get(
    "/api/v1/models",
    response_model=ModelListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_models():
    """List all registered models."""
    try:
        registry = MLflowRegistry()
        raw = registry.list_models()
        models = [
            ModelInfo(
                name=m["name"],
                latest_versions=[
                    ModelVersionInfo(**v) for v in m.get("latest_versions", [])
                ],
            )
            for m in raw
        ]
        return ModelListResponse(models=models)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def start():
    """Entrypoint for the CLI."""
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
