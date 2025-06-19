from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"ticker": "SBER", "start_date": "2025-01-01", "end_date": "2025-01-10"},
    )
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert all("date" in item and "predicted_price" in item for item in data)
