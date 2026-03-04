"""Integration tests for the FastAPI application — 13 tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestPipelineEndpoints:
    def test_run_pipeline_bad_config(self, client: TestClient):
        resp = client.post("/api/v1/pipeline/run", json={"config_path": "/nonexistent.yaml"})
        assert resp.status_code == 400

    def test_run_pipeline_valid(self, client: TestClient, classification_config_path: Path):
        resp = client.post(
            "/api/v1/pipeline/run",
            json={"config_path": str(classification_config_path)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"

    def test_pipeline_status_not_found(self, client: TestClient):
        resp = client.get("/api/v1/pipeline/status/nonexistent-id")
        assert resp.status_code == 404

    def test_pipeline_history_empty(self, client: TestClient):
        resp = client.get("/api/v1/pipeline/history")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["runs"], list)

    def test_pipeline_history_with_name(self, client: TestClient):
        resp = client.get("/api/v1/pipeline/history", params={"pipeline_name": "test"})
        assert resp.status_code == 200


class TestPredictEndpoint:
    @patch("src.api.main.MLflowRegistry")
    def test_predict_model_not_found(self, mock_registry_cls: MagicMock, client: TestClient):
        mock_registry_cls.return_value.load_production_model.side_effect = Exception("Not found")
        resp = client.post(
            "/api/v1/predict",
            json={"model_name": "nonexistent", "features": {"a": 1}},
        )
        assert resp.status_code == 404

    @patch("src.api.main.MLflowRegistry")
    def test_predict_success(self, mock_registry_cls: MagicMock, client: TestClient):
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_registry_cls.return_value.load_production_model.return_value = mock_model

        resp = client.post(
            "/api/v1/predict",
            json={"model_name": "my-model", "features": {"a": 1, "b": 2}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] == 1
        assert data["probabilities"] is not None

    @patch("src.api.main.MLflowRegistry")
    def test_predict_no_proba(self, mock_registry_cls: MagicMock, client: TestClient):
        import numpy as np
        mock_model = MagicMock(spec=[])  # No predict_proba
        mock_model.predict = MagicMock(return_value=np.array([42.0]))
        mock_registry_cls.return_value.load_production_model.return_value = mock_model

        resp = client.post(
            "/api/v1/predict",
            json={"model_name": "reg-model", "features": {"a": 1}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["probabilities"] is None


class TestModelsEndpoint:
    @patch("src.api.main.MLflowRegistry")
    def test_list_models(self, mock_registry_cls: MagicMock, client: TestClient):
        mock_registry_cls.return_value.list_models.return_value = [
            {"name": "model-a", "latest_versions": [{"version": "1", "stage": "Production", "run_id": "r1"}]},
        ]
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "model-a"

    @patch("src.api.main.MLflowRegistry")
    def test_list_models_empty(self, mock_registry_cls: MagicMock, client: TestClient):
        mock_registry_cls.return_value.list_models.return_value = []
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert len(resp.json()["models"]) == 0

    @patch("src.api.main.MLflowRegistry")
    def test_list_models_error(self, mock_registry_cls: MagicMock, client: TestClient):
        mock_registry_cls.return_value.list_models.side_effect = Exception("Connection refused")
        resp = client.get("/api/v1/models")
        assert resp.status_code == 500

    def test_openapi_docs(self, client: TestClient):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "paths" in resp.json()
