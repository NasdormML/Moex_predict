import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.data import (
    fetch_moex_eod_data,
    fetch_usd_series,
    fetch_cbr_usd_rate,
    load_cached_data,
    save_cached_data,
    get_smart_features
)


class TestCaching:
    """Тесты кэширования."""
    
    def test_load_cache_hit(self, temp_dir):
        """Попадание в кэш."""
        from app.data import CACHE_DIR
        
        # Создаём кэш
        df = pd.DataFrame({"a": [1, 2, 3]})
        cache_path = temp_dir / "test.pkl"
        df.to_pickle(cache_path)
        
        with patch("app.data._cache_path", return_value=cache_path):
            with patch("app.data.CACHE_DIR", temp_dir):
                result = load_cached_data("TEST", "2024-01-01", "2024-01-10")
                
                assert result is not None
                assert len(result) == 3
    
    def test_load_cache_expired(self, temp_dir):
        """Истёкший кэш."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        cache_path = temp_dir / "test.pkl"
        df.to_pickle(cache_path)
        
        # Меняем время модификации на старое
        old_time = (datetime.now() - timedelta(days=10)).timestamp()
        import os
        os.utime(cache_path, (old_time, old_time))
        
        with patch("app.data._cache_path", return_value=cache_path):
            with patch("app.data.EXPIRATION_DAYS", 1):
                result = load_cached_data("TEST", "2024-01-01", "2024-01-10")
                
                assert result is None
    
    def test_save_cache_atomic(self, temp_dir):
        """Атомарное сохранение кэша."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with patch("app.data._cache_path", return_value=temp_dir / "test.pkl"):
            save_cached_data("TEST", "2024-01-01", "2024-01-10", df)
            
            # Проверяем, что файл создан
            assert (temp_dir / "test.pkl").exists()


class TestMOEXFetching:
    """Тесты загрузки с MOEX."""
    
    @patch("app.data._session")
    def test_fetch_success(self, mock_session):
        """Успешная загрузка."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "history": {
                "columns": ["TRADEDATE", "CLOSE"],
                "data": [["2024-01-01", 100.0], ["2024-01-02", 101.0]]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        
        result = fetch_moex_eod_data(
            "SBER", "stock", "shares", "TQBR",
            "2024-01-01", "2024-01-10",
            skip_cache=True
        )
        
        assert not result.empty
        assert "close" in result.columns
    
    @patch("app.data._session")
    def test_fetch_pagination(self, mock_session):
        """Пагинация при большом объёме."""
        # Первая страница
        response1 = Mock()
        response1.json.return_value = {
            "history": {
                "columns": ["TRADEDATE", "CLOSE"],
                "data": [["2024-01-01", 100.0]] * 100
            }
        }
        response1.raise_for_status = Mock()
        
        # Вторая страница (пустая)
        response2 = Mock()
        response2.json.return_value = {"history": {"columns": [], "data": []}}
        response2.raise_for_status = Mock()
        
        mock_session.get.side_effect = [response1, response2]
        
        result = fetch_moex_eod_data(
            "SBER", "stock", "shares", "TQBR",
            "2024-01-01", "2024-01-10",
            skip_cache=True
        )
        
        assert len(result) == 100
        assert mock_session.get.call_count == 2


class TestCBRFetching:
    """Тесты загрузки курсов ЦБ."""
    
    @patch("app.data._session")
    def test_fetch_cbr_success(self, mock_session):
        """Успешная загрузка курса USD."""
        mock_response = Mock()
        mock_response.content = b'''<?xml version="1.0"?>
        <ValCurs>
            <Valute>
                <CharCode>USD</CharCode>
                <Value>90,50</Value>
            </Valute>
        </ValCurs>'''
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        
        result = fetch_cbr_usd_rate("01/01/2024")
        
        assert result == 90.50
    
    @patch("app.data._session")
    def test_fetch_cbr_not_found(self, mock_session):
        """USD не найден в ответе."""
        mock_response = Mock()
        mock_response.content = b'''<?xml version="1.0"?>
        <ValCurs>
            <Valute>
                <CharCode>EUR</CharCode>
                <Value>100,00</Value>
            </Valute>
        </ValCurs>'''
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="USD rate not found"):
            fetch_cbr_usd_rate("01/01/2024")


class TestFeatureSelection:
    """Тесты отбора фичей."""
    
    def test_get_smart_features_basic(self):
        """Базовый отбор."""
        df = pd.DataFrame({
            "TRADEDATE": pd.date_range("2024-01-01", periods=100),
            "CLOSE_SBER": np.random.randn(100),
            "VOL_SBER": np.random.randint(1000, 10000, 100),
            "RSI14": np.random.randn(100),
            "MACD_LINE": np.random.randn(100),
            "sma20": np.random.randn(100),
            "log_ret_1": np.random.randn(100),
            "log_ret_1_lag1": np.random.randn(100),
            "log_ret_1_lag2": np.random.randn(100),
        })
        
        features = get_smart_features(df, "SBER", target_count=5)
        
        assert len(features) == 5
        assert "CLOSE_SBER" in features  # должен быть включён
    
    def test_get_smart_features_insufficient(self):
        """Недостаточно фичей."""
        df = pd.DataFrame({
            "TRADEDATE": pd.date_range("2024-01-01", periods=10),
            "CLOSE_SBER": np.random.randn(10),
        })
        
        features = get_smart_features(df, "SBER", target_count=20)
        
        # Должен вернуть сколько есть
        assert len(features) <= 20