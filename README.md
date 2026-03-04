# Automated ML Pipeline

Production AutoML system that automates the full machine learning lifecycle: data ingestion, feature engineering, model training, evaluation, and deployment.

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Ingest  │──▶│ Feature  │──▶│  Train   │──▶│ Evaluate │──▶│  Deploy  │
│  + Valid │   │ Eng/Sel  │   │ (Multi)  │   │ + Compare│   │ (MLflow) │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
      ▲                              │                             │
      │         Prefect Orchestration + MLflow Tracking            │
      └────────────────────────────────────────────────────────────┘
```

## Features

- **Data Ingestion**: CSV, Parquet, SQL databases, REST APIs with built-in validation
- **Feature Engineering**: Auto type detection, encoding (one-hot/label/target), scaling, missing value handling, datetime features, interaction features
- **Feature Selection**: Correlation filter, variance filter, tree importance, recursive feature elimination
- **Multi-Model Training**: Logistic Regression, Random Forest, XGBoost, LightGBM with hyperparameter tuning
- **Evaluation**: Classification + regression metrics, cross-validation, statistical significance testing
- **Orchestration**: Prefect flows with retry logic, scheduling, and run history
- **Model Registry**: MLflow-based registration, promotion, and production model management
- **REST API**: FastAPI with pipeline triggers, predictions, model listing, and health checks
- **Production Ready**: Docker, CI/CD, structured logging, YAML configs

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
make test

# Run a pipeline
python scripts/run_pipeline.py configs/classification_pipeline.yaml

# Start API
uvicorn src.api.main:app --reload

# Docker (API + MLflow + Prefect)
make docker-up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/pipeline/run` | Trigger a pipeline run |
| POST | `/api/v1/predict` | Predict using production model |
| GET | `/api/v1/models` | List registered models |
| GET | `/api/v1/pipeline/status/{run_id}` | Pipeline run status |
| GET | `/api/v1/pipeline/history` | Past pipeline runs |
| GET | `/health` | Health check |

## Configuration

Pipelines are configured via YAML files in `configs/`:

```yaml
name: my-pipeline
data_source: data/dataset.csv
target_column: target
training_config:
  task_type: classification
  models: [random_forest, xgboost, lightgbm]
  n_trials: 20
  metric: f1
```

## Project Structure

```
src/
├── config/        Settings and pipeline config
├── data/          Ingestion, validation, splitting
├── features/      Engineering and selection
├── training/      AutoTrainer + hyperparameters
├── evaluation/    Metrics and model comparison
├── orchestration/ Prefect flows and scheduling
├── deployment/    MLflow model registry
├── api/           FastAPI REST endpoints
└── utils/         Logging utilities
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | Prefect |
| Experiment Tracking | MLflow |
| ML Models | scikit-learn, XGBoost, LightGBM |
| API | FastAPI |
| Config | Pydantic Settings, YAML |
| Logging | structlog |
| Testing | pytest |
| Containerization | Docker |

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Handoff](docs/HANDOFF.md)

## License

MIT
