import pickle
import pytest
import torch
from unittest.mock import patch
from app.model_manager import ModelManager


class TestModelManagerLoading:
    """Тесты загрузки моделей."""
    
    @pytest.fixture
    def setup_artifacts(self, temp_dir, mock_model, mock_scalers):
        """Создаёт структуру артефактов."""
        scaler_X, scaler_y = mock_scalers
        
        version_dir = temp_dir / "v1.0"
        version_dir.mkdir()
        
        # Сохраняем веса
        torch.save({"weight": torch.tensor([1.0])}, version_dir / "SBER_model.pth")
        
        # Сохраняем скейлеры
        with open(version_dir / "SBER_scaler_X.pkl", "wb") as f:
            pickle.dump(scaler_X, f)
        with open(version_dir / "SBER_scaler_y.pkl", "wb") as f:
            pickle.dump(scaler_y, f)
        
        return temp_dir
    
    @pytest.mark.asyncio
    async def test_load_all_success(self, setup_artifacts, mock_model, temp_dir):
        """Успешная загрузка всех моделей."""
        with patch("app.model_manager.Path") as mock_path:
            mock_path.return_value = temp_dir
            
            with patch("app.model_manager.get_model", return_value=mock_model):
                with patch("app.model_manager.torch.load", return_value={"weight": torch.tensor([1.0])}):
                    with patch("app.transfer_learning.TransferLearningManager.load_metadata") as mock_load_md:
                        
                        mock_load_md.return_value = {
                            "SBER": {
                                "active_version": "v1.0",
                                "versions": {
                                    "v1.0": {
                                        "factory_key": "lstm",
                                        "model_params": {"seq_length": 20}
                                    }
                                }
                            }
                        }
                        
                        manager = ModelManager()
                        manager._artifacts_root = setup_artifacts
                        
                        await manager.load_all()
                        
                        assert manager.is_ready()
                        assert len(manager) == 1
                        assert "SBER" in manager._models
    
    @pytest.mark.asyncio
    async def test_load_no_metadata(self, temp_dir):
        """Ошибка при отсутствии метаданных."""
        with patch("app.transfer_learning.TransferLearningManager.load_metadata", return_value={}):
            manager = ModelManager()
            manager._artifacts_root = temp_dir
            
            with pytest.raises(RuntimeError, match="No models loaded"):
                await manager.load_all()
    
    @pytest.mark.asyncio
    async def test_load_missing_version(self, temp_dir):
        """Пропуск модели без active_version."""
        with patch("app.transfer_learning.TransferLearningManager.load_metadata") as mock_load_md:
            mock_load_md.return_value = {
                "SBER": {
                    "versions": {}
                }
            }
            
            manager = ModelManager()
            manager._artifacts_root = temp_dir
            
            await manager.load_all()
            
            assert len(manager) == 0


class TestModelManagerSecurity:
    """Тесты безопасности."""
    
    def test_safe_load_pickle_size_limit(self, temp_dir):
        """Проверка лимита размера pickle."""
        manager = ModelManager()
        
        # Создаём большой файл
        big_file = temp_dir / "big.pkl"
        big_file.write_bytes(b"x" * (101 * 1024 * 1024))
        
        with pytest.raises(ValueError, match="too large"):
            manager._safe_load_pickle(big_file)
    
    def test_safe_load_pickle_not_exists(self, temp_dir):
        """Ошибка при отсутствии файла."""
        manager = ModelManager()
        
        with pytest.raises(FileNotFoundError):
            manager._safe_load_pickle(temp_dir / "nonexistent.pkl")


class TestModelManagerGetters:
    """Тесты getter-методов."""
    
    @pytest.fixture
    def manager_with_model(self, sample_model_bundle):
        """Manager с загруженной моделью."""
        manager = ModelManager()
        manager._models["SBER"] = sample_model_bundle
        manager._metadata["SBER"] = {"version": "v1.0"}
        manager._ready = True
        return manager
    
    def test_get_model_exists(self, manager_with_model, sample_model_bundle):
        """Получение существующей модели."""
        result = manager_with_model.get_model("SBER")
        assert result == sample_model_bundle
    
    def test_get_model_not_exists(self, manager_with_model):
        """Получение несуществующей модели."""
        result = manager_with_model.get_model("GAZP")
        assert result is None
    
    def test_get_metadata(self, manager_with_model):
        """Получение метаданных."""
        result = manager_with_model.get_metadata("SBER")
        assert result == {"version": "v1.0"}