import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestHealthEndpoint:
    """Тесты /health."""
    
    def test_health_ready(self, client, mock_mlflow):
        """Health check когда всё готово."""
        with patch("app.main.model_manager") as mock_mm:
            mock_mm.is_ready = Mock(return_value=True)
            mock_mm.__len__ = Mock(return_value=3)
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["models_loaded"] == 3
    
    def test_health_not_ready(self, client):
        """Health check когда не готово."""
        with patch("app.main.model_manager", None):
            response = client.get("/health")
            assert response.status_code == 503


class TestPredictEndpoint:
    """Тесты /predict/{ticker}/{target_date}."""
    
    @pytest.fixture
    def mock_model_bundle(self):
        """Мок bundle для предсказаний."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.detach = Mock(return_value=Mock())
        mock_model.return_value.detach.return_value.numpy = Mock(
            return_value=np.array([[250.5], [251.0], [251.5]])
        )
        
        from sklearn.preprocessing import RobustScaler
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        scaler_X.fit(np.random.randn(100, 10))
        scaler_y.fit(np.random.randn(100, 1))
        
        return {
            "model": mock_model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": 20,
            "model_version": "v1.0",
            "factory_key": "lstm",
            "model_params": {}
        }
    
    def test_predict_success(self, client, mock_model_bundle, sample_market_data):
        """Успешное предсказание."""
        with patch("app.main.model_manager") as mock_mm:
            mock_mm.get_model = Mock(return_value=mock_model_bundle)
            mock_mm.get_metadata = Mock(return_value={
                "data_upto": "2024-05-01"
            })
            mock_mm.maybe_retrain = AsyncMock(return_value=mock_model_bundle)
            
            with patch("app.main.fetch_moex_eod_data", new_callable=AsyncMock) as mock_fetch:
                with patch("app.main.fetch_usd_series", new_callable=AsyncMock) as mock_usd:
                    with patch("app.main.preprocess_data") as mock_prep:
                        with patch("app.main.predict_price", new_callable=AsyncMock) as mock_pred:
                            
                            mock_fetch.side_effect = [
                                sample_market_data,  # ticker
                                sample_market_data,  # IMOEX
                                sample_market_data   # USD (fallback)
                            ]
                            mock_usd.return_value = sample_market_data[["TRADEDATE", "CLOSE_USD"]]
                            mock_prep.return_value = sample_market_data.assign(
                                feature1=1.0, feature2=2.0
                            )
                            mock_pred.return_value = np.array([250.5, 251.0, 251.5])
                            
                            response = client.post("/predict/SBER/2024-05-10")
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert data["ticker"] == "SBER"
                            assert "predictions" in data
                            assert len(data["predictions"]) > 0
    
    def test_predict_model_not_found(self, client):
        """Ошибка когда модель не найдена."""
        with patch("app.main.model_manager") as mock_mm:
            mock_mm.get_model = Mock(return_value=None)
            
            response = client.post("/predict/UNKNOWN/2024-05-10")
            
            assert response.status_code == 404
            assert "not found" in response.json()["error"].lower()
    
    def test_predict_invalid_date(self, client, mock_model_bundle):
        """Ошибка при невалидной дате."""
        with patch("app.main.model_manager") as mock_mm:
            mock_mm.get_model = Mock(return_value=mock_model_bundle)
            mock_mm.get_metadata = Mock(return_value={
                "data_upto": "2024-05-10"
            })
            
            # target_date < last_known
            response = client.post("/predict/SBER/2024-05-01")
            
            assert response.status_code == 400


class TestTrainEndpoint:
    """Тесты /train/{ticker}."""
    
    def test_train_start_success(self, client):
        """Запуск обучения в фоне."""
        with patch("app.main.model_manager"):
            with patch("app.main.fetch_moex_eod_data", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = pd.DataFrame({
                    "TRADEDATE": pd.date_range("2024-01-01", periods=30),
                    "CLOSE": range(30)
                })
                
                response = client.post(
                    "/train/SBER",
                    params={
                        "model": "lstm",
                        "epochs": 10,
                        "enable_hpo": False
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "training_started"
                assert data["ticker"] == "SBER"
    
    def test_train_invalid_ticker(self, client):
        """Ошибка при невалидном тикере."""
        response = client.post("/train/INVALID_TICKER!!!")
        assert response.status_code == 400
    
    def test_train_no_data(self, client):
        """Ошибка когда нет данных."""
        with patch("app.main.fetch_moex_eod_data", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()  # empty
            
            response = client.post("/train/FAKE")
            assert response.status_code == 404


class TestValidation:
    """Тесты валидации входных данных."""
    
    @pytest.mark.parametrize("ticker,expected_status", [
        ("SBER", 200),           # OK
        ("sber", 200),           # lowercase -> uppercase
        ("SBER123", 200),        # alphanumeric
        ("SBER!!!", 400),        # invalid chars
        ("", 422),               # empty (FastAPI validation)
        ("VERYLONGTICKERNAME", 400),  # too long
    ])
    def test_ticker_validation(self, client, ticker, expected_status):
        """Валидация тикера."""
        with patch("app.main.model_manager"):
            response = client.post(f"/train/{ticker}")
            assert response.status_code == expected_status