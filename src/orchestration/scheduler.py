"""Pipeline scheduling and run history."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.config.pipeline_config import PipelineConfig, load_pipeline_config
from src.orchestration.pipeline import run_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineRun:
    """Record of a single pipeline execution."""

    run_id: str
    pipeline_name: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: datetime
    end_time: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class PipelineScheduler:
    """Manage pipeline scheduling and execution history."""

    def __init__(self) -> None:
        self._history: dict[str, list[PipelineRun]] = {}
        self._schedules: dict[str, str] = {}
        self._run_counter = 0

    def schedule(self, pipeline_name: str, cron_expression: str) -> None:
        """Register a cron schedule for a pipeline (stores schedule metadata)."""
        self._schedules[pipeline_name] = cron_expression
        logger.info("pipeline_scheduled", name=pipeline_name, cron=cron_expression)

    def run_now(self, config: PipelineConfig) -> PipelineRun:
        """Execute a pipeline immediately and record the run."""
        self._run_counter += 1
        run_id = f"run-{self._run_counter}-{int(time.time())}"
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=config.name,
            status="running",
            start_time=datetime.now(timezone.utc),
        )

        try:
            result = run_pipeline(config)
            run.status = "completed"
            run.metrics = result.get("metrics", {})
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            logger.error("pipeline_run_failed", name=config.name, error=str(exc))
        finally:
            run.end_time = datetime.now(timezone.utc)

        self._history.setdefault(config.name, []).append(run)
        return run

    def get_history(self, pipeline_name: str) -> list[PipelineRun]:
        """Get run history for a pipeline."""
        return self._history.get(pipeline_name, [])

    def get_schedule(self, pipeline_name: str) -> str | None:
        """Get the cron schedule for a pipeline."""
        return self._schedules.get(pipeline_name)

    def list_schedules(self) -> dict[str, str]:
        """List all registered schedules."""
        return dict(self._schedules)
