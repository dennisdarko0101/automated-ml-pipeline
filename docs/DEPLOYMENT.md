# Deployment Guide

## Local Development

```bash
make dev
make test
python scripts/run_pipeline.py configs/classification_pipeline.yaml
```

## Docker

```bash
# Build and start all services
make docker-build
make docker-up

# Services:
#   API:     http://localhost:8000
#   MLflow:  http://localhost:5000
#   Prefect: http://localhost:4200
```

## Production Deployment

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `PREFECT_API_URL` | `http://localhost:4200/api` | Prefect server URL |
| `S3_BUCKET` | `automl-artifacts` | Artifact storage bucket |
| `DATA_DIR` | `data` | Data directory |
| `MODEL_DIR` | `models` | Model directory |
| `LOG_LEVEL` | `INFO` | Logging level |

### Kubernetes

1. Build and push Docker image via CI/CD
2. Deploy using Helm or raw manifests
3. Set env vars via ConfigMap/Secrets
4. Expose API via Ingress

### Monitoring

- Structured JSON logs via `structlog`
- Health check: `GET /health`
- MLflow UI for experiment tracking
- Prefect UI for pipeline monitoring
