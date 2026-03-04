# Project Handoff

## Quick Start

```bash
pip install -e ".[dev]"
make test
```

## Key Decisions

1. **Prefect for orchestration**: Chosen for native Python decorators, retry support, and UI dashboard
2. **MLflow for tracking**: Industry-standard experiment tracking and model registry
3. **scikit-learn as base**: Consistent API across all model types; XGBoost and LightGBM wrap the same interface
4. **FastAPI**: Async-capable, auto-docs, Pydantic validation
5. **YAML configs**: Pipeline configurations are declarative and version-controllable

## Extension Points

- **New data source**: Add method to `DataIngester`
- **New model type**: Add to `MODEL_REGISTRY_CLS` in `trainer.py` and add search space in `hyperparams.py`
- **New feature step**: Add method to `FeatureEngineer`, reference it in `FeaturePipeline`
- **Custom metrics**: Extend `ModelEvaluator._evaluate_classification` / `_evaluate_regression`

## Known Limitations

- Hyperparameter tuning uses RandomizedSearchCV (no Bayesian optimization built in yet)
- Time-series models (ARIMA, Prophet) not included — extend via custom model types
- No GPU training support yet — add CUDA-enabled Docker image for GPU workloads
- Pipeline scheduler stores history in-memory — swap for database in production

## File Map

```
src/config/          Settings, YAML pipeline configs
src/data/            Ingestion, validation, splitting
src/features/        Engineering, selection, pipelines
src/training/        AutoTrainer, hyperparameter spaces
src/evaluation/      Metrics, model comparison
src/orchestration/   Prefect flows, scheduling
src/deployment/      MLflow registry
src/api/             FastAPI REST endpoints
```
