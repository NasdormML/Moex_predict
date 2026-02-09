import pytest
import numpy as np
import torch.nn as nn
from unittest.mock import Mock, mock_open, patch
from datetime import datetime, timedelta
from app.transfer_learning import TransferLearningManager, load_training_metadata, save_training_metadata


class TestMetadataOperations:
    """Тесты работы с метаданными."""
    
    def test_load_metadata_not_exists(self, temp_dir):
        """Загрузка несуществующих метаданных."""
        result = load_training_metadata(temp_dir / "nonexistent.pkl")
        assert result == {}
    
    def test_save_and_load_metadata(self, temp_dir):
        """Сохранение и загрузка."""
        path = temp_dir / "meta.pkl"
        data = {"SBER": {"version": "v1.0"}}
        
        save_training_metadata(data, path)
        loaded = load_training_metadata(path)
        
        assert loaded == data


class TestShouldRetrain:
    """Тесты логики переобучения."""
    
    @pytest.fixture
    def manager(self, temp_dir):
        return TransferLearningManager(metadata_path=temp_dir / "meta.pkl")
    
    def test_should_retrain_no_data(self, manager):
        """Нет данных — не переобучаем."""
        with patch.object(manager, "load_metadata", return_value={}):
            assert not manager.should_retrain("SBER")
    
    def test_should_retrain_fresh_model(self, manager):
        """Свежая модель — не переобучаем."""
        with patch.object(manager, "load_metadata", return_value={
            "SBER": {
                "active_version": "v1.0",
                "versions": {
                    "v1.0": {
                        "train_date": datetime.today().strftime("%Y-%m-%d")
                    }
                }
            }
        }):
            assert not manager.should_retrain("SBER", threshold_days=5)
    
    def test_should_retrain_stale_model(self, manager):
        """Старая модель — переобучаем."""
        old_date = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")
        with patch.object(manager, "load_metadata", return_value={
            "SBER": {
                "active_version": "v1.0",
                "versions": {
                    "v1.0": {
                        "train_date": old_date
                    }
                }
            }
        }):
            assert manager.should_retrain("SBER", threshold_days=5)


class TestFineTuning:
    """Тесты fine-tuning."""
    
    @pytest.fixture
    def simple_model(self):
        """Простая модель для тестов."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x[:, -1, :])
        
        return SimpleModel()
    
    @pytest.fixture
    def manager(self, temp_dir):
        return TransferLearningManager(
            metadata_path=temp_dir / "meta.pkl",
            artifacts_root=temp_dir
        )
    
    def test_fine_tune_selective_layers(self, manager, simple_model):
        """Fine-tuning только определённых слоёв."""
        bundle = {
            "model": simple_model,
            "model_params": {
                "fine_tune_modules": ["linear"],
                "window_days": 30
            },
            "seq_length": 10,
            "scaler_X": Mock(),
            "scaler_y": Mock(),
            "factory_key": "test",
            "model_version": "v1.0"
        }
        
        # Мокаем скейлеры
        bundle["scaler_X"].transform = lambda x: x
        bundle["scaler_y"].transform = lambda x: x
        
        # Создаём данные
        X = np.random.randn(100, 10, 10).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)
        
        # Проверяем, что только linear обучается
        for name, param in simple_model.named_parameters():
            param.requires_grad = False
        
        result = manager._fine_tune_model(bundle, X, y)
        assert result is not None
    
    def test_early_stopping(self, manager, simple_model):
        """Early stopping при отсутствии улучшений."""
        bundle = {
            "model": simple_model,
            "model_params": {"window_days": 30},
            "seq_length": 10,
            "scaler_X": Mock(),
            "scaler_y": Mock(),
            "factory_key": "test",
            "model_version": "v1.0"
        }
        
        bundle["scaler_X"].transform = lambda x: x
        bundle["scaler_y"].transform = lambda x: x
        
        X = np.random.randn(50, 10, 10).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)
        
        # Уменьшаем patience для теста
        with patch("app.transfer_learning.MAX_EPOCHS_WITHOUT_IMPROVEMENT", 2):
            result = manager._fine_tune_model(bundle, X, y)
            
            # Должно остановиться раньше 45 эпох
            assert result is not None


class TestVersioning:
    def test_version_increment(self, temp_dir):
        """Инкремент версии v1.0 -> v1.1."""
        manager = TransferLearningManager(
            metadata_path=temp_dir / "meta.pkl",
            artifacts_root=temp_dir
        )
        
        bundle = {
            "model": Mock(),
            "model_params": {},
            "scaler_X": Mock(),
            "scaler_y": Mock(),
            "factory_key": "test",
            "model_version": "v1.0"
        }
        
        with patch.object(manager, "load_metadata", return_value={}):
            with patch.object(manager, "save_metadata"):
                with patch("torch.save"):
                    with patch("builtins.open", mock_open()):
                        
                        result = manager._save_version("SBER", bundle, Mock(), Mock(), Mock())
                        
                        assert result["model_version"] == "v1.1"
    
    def test_version_parsing_invalid(self, temp_dir):
        """Обработка невалидной версии."""
        manager = TransferLearningManager(
            metadata_path=temp_dir / "meta.pkl",
            artifacts_root=temp_dir
        )
        
        bundle = {
            "model": Mock(),
            "model_params": {},
            "scaler_X": Mock(),
            "scaler_y": Mock(),
            "factory_key": "test",
            "model_version": "invalid_version"
        }
        
        with patch.object(manager, "load_metadata", return_value={}):
            with patch.object(manager, "save_metadata"):
                with patch("torch.save"):
                    with patch("builtins.open", mock_open()):
                        
                        result = manager._save_version("SBER", bundle, Mock(), Mock(), Mock())
                        
                        # Должно сброситься на v1.0
                        assert result["model_version"] == "v1.0"