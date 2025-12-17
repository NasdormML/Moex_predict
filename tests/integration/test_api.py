from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint_success():
    """Test successful prediction"""
    response = client.post("/predict/SBER/2025-01-15")
    assert response.status_code == 200
    
    data = response.json()
    assert "ticker" in data
    assert "predictions" in data
    assert "forecast_dates" in data
    assert len(data["predictions"]) == 5

def test_predict_endpoint_invalid_ticker():
    """Test 404 for invalid ticker"""
    response = client.post("/predict/INVALID/2025-01-15")
    assert response.status_code == 404
    assert "не найдена" in response.json()["detail"]

def test_predict_endpoint_not_enough_data():
    """Test 422 for insufficient data"""
    response = client.post("/predict/SBER/2023-01-01")
    assert response.status_code == 422