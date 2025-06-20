from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient


@patch(
    "app.data.fetch_moex_data",
    return_value=pd.DataFrame(
        {
            "TRADEDATE": pd.date_range("2025-01-01", periods=3),
            "CLOSE": [100, 101, 102],
        }
    ),
)
def test_predict_endpoint(mock_fetch):
    import os

    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///:memory:"
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"ticker": "SBER", "start_date": "2025-01-01", "end_date": "2025-01-03"},
    )
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list) and len(data) == 3
    assert all("date" in item and "predicted_price" in item for item in data)

    mock_fetch.assert_called_once_with("SBER", "2025-01-01", "2025-01-03")
