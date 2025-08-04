"""Tests for the API endpoints"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_health_endpoint(api_client):
    """Test health endpoint"""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


def test_metrics_endpoint(api_client):
    """Test metrics endpoint"""
    response = api_client.get("/metrics")
    assert response.status_code == 200


def test_predict_endpoint(api_client, sample_housing_data):
    """Test prediction endpoint with valid data"""
    response = api_client.post(
        "/api/predict",
        data=json.dumps(sample_housing_data),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_endpoint_invalid_data(api_client):
    """Test prediction endpoint with invalid data"""
    response = api_client.post(
        "/api/predict",
        data=json.dumps({"invalid": "data"}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_batch_predict_endpoint(api_client, sample_housing_data):
    """Test batch prediction endpoint"""
    batch_data = [sample_housing_data, sample_housing_data]
    response = api_client.post(
        "/api/predict/batch",
        data=json.dumps(batch_data),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "predictions" in data
    assert len(data["predictions"]) == 2
