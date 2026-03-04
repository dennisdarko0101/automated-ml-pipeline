# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
│  POST /pipeline/run  POST /predict  GET /models  GET /health│
└────────────┬────────────────────┬───────────────────────────┘
             │                    │
    ┌────────▼────────┐  ┌───────▼────────┐
    │  Prefect Flow   │  │  MLflow Model  │
    │  Orchestration  │  │  Registry      │
    └────────┬────────┘  └───────┬────────┘
             │                   │
┌────────────▼───────────────────▼───────────────────────┐
│                  ML Pipeline Tasks                      │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Ingest   │→│ Feature  │→│ Training │→│ Evaluate  │ │
│  │ + Valid. │ │ Eng/Sel  │ │ (Multi)  │ │ + Compare │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### Data Layer (`src/data/`)
- **DataIngester**: CSV, Parquet, Database (SQL), REST API ingestion
- **DataValidator**: Schema, completeness, range validation with reports
- **DataSplitter**: Train/val/test, time-based, cross-validation

### Feature Layer (`src/features/`)
- **FeatureEngineer**: Type detection, encoding, scaling, missing values, datetime features, interactions
- **FeaturePipeline**: Chainable fit/transform pipeline, serializable
- **FeatureSelector**: Correlation, variance, importance, RFE filtering

### Training Layer (`src/training/`)
- **AutoTrainer**: Multi-model training (LR, RF, XGBoost, LightGBM)
- **HyperparameterSpace**: Configurable search spaces per model

### Evaluation Layer (`src/evaluation/`)
- **ModelEvaluator**: Classification/regression metrics, CV scores
- **ModelComparator**: Ranking, statistical significance, recommendations

### Orchestration (`src/orchestration/`)
- **MLPipeline**: Prefect flow composing all tasks
- **PipelineScheduler**: Cron scheduling, execution history

### Deployment (`src/deployment/`)
- **MLflowRegistry**: Model registration, promotion, A/B comparison

### API (`src/api/`)
- FastAPI app with endpoints for pipeline runs, predictions, and model management

## Data Flow

1. YAML config → `PipelineConfig`
2. `DataIngester` → raw DataFrame
3. `DataValidator` → `ValidationReport`
4. `FeaturePipeline.fit_transform()` → engineered features
5. `FeatureSelector` → selected features + `FeatureReport`
6. `DataSplitter` → `SplitResult` (train/val/test)
7. `AutoTrainer.train()` → `TrainResult` (ranked models)
8. `ModelComparator.compare()` → `ComparisonReport`
9. Best model → MLflow registry
