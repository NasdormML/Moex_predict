import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MODEL_ARTIFACTS_DIR"] = "/tmp/test_models"
os.environ["DATA_CACHE_DIR"] = "/tmp/test_cache"
os.environ["HISTORY_DIR"] = "/tmp/test_history"

from app.model_manager import ModelManager


@pytest.fixture
def temp_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_model():
    """Мок модели PyTorch."""
    model = Mock(spec=nn.Module)
    model.eval = Mock()
    model.load_state_dict = Mock()
    model.state_dict = Mock(return_value={"weight": torch.tensor([1.0])})
    model.parameters = Mock(return_value=[torch.tensor([1.0], requires_grad=True)])
    model.named_parameters = Mock(return_value=[("layer.weight", torch.tensor([1.0], requires_grad=True))])
    return model


@pytest.fixture
def mock_scalers():
    """Мок скейлеров."""
    from sklearn.preprocessing import RobustScaler
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    scaler_X.fit(np.array([[1, 2], [3, 4]]))
    scaler_y.fit(np.array([[1], [2]]))
    
    return scaler_X, scaler_y


@pytest.fixture
def sample_model_bundle(mock_model, mock_scalers):
    """Готовый bundle модели."""
    scaler_X, scaler_y = mock_scalers
    return {
        "model": mock_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "seq_length": 20,
        "model_version": "v1.0",
        "factory_key": "lstm",
        "model_params": {
            "seq_length": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2
        }
    }


@pytest.fixture
def sample_metadata():
    """Пример метаданных."""
    return {
        "SBER": {
            "active_version": "v1.0",
            "versions": {
                "v1.0": {
                    "factory_key": "lstm",
                    "model_params": {
                        "seq_length": 20,
                        "hidden_size": 64
                    },
                    "train_date": "2024-01-01",
                    "data_upto": "2024-01-15"
                }
            }
        }
    }


@pytest.fixture
def mock_mlflow():
    """Мок MLflow."""
    with patch("app.main.mlflow") as mock:
        mock.start_run = MagicMock()
        mock.set_tag = MagicMock()
        mock.log_param = MagicMock()
        mock.log_metric = MagicMock()
        mock.get_tracking_uri = Mock(return_value="http://localhost:5001")
        yield mock


@pytest.fixture
async def model_manager(temp_dir):
    """Инициализированный ModelManager."""
    manager = ModelManager()
    manager._artifacts_root = temp_dir
    yield manager


@pytest.fixture
def sample_market_data():
    """Пример рыночных данных."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "TRADEDATE": dates,
        "OPEN_SBER": np.random.randn(100).cumsum() + 100,
        "HIGH_SBER": np.random.randn(100).cumsum() + 101,
        "LOW_SBER": np.random.randn(100).cumsum() + 99,
        "CLOSE_SBER": np.random.randn(100).cumsum() + 100,
        "VOL_SBER": np.random.randint(1000000, 10000000, 100),
        "CLOSE_IMOEX": np.random.randn(100).cumsum() + 3000,
        "CLOSE_USD": np.random.randn(100).cumsum() + 90
    })


@pytest.fixture
def client(temp_dir, mock_mlflow):
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from app.main import app
    
    with patch("app.main.ModelManager") as mock_mm_class:
        mock_mm = MagicMock()
        mock_mm_class.return_value = mock_mm
        mock_mm.load_all = Mock(return_value=asyncio.Future())
        mock_mm.load_all.return_value.set_result(None)
        mock_mm.__len__ = Mock(return_value=1)
        mock_mm.get_model = Mock(return_value=None)
        mock_mm.is_ready = Mock(return_value=True)
        
        async def mock_lifespan(app):
            yield
            
        app.router.lifespan_context = mock_lifespan
        
        with TestClient(app) as client:
            yield client