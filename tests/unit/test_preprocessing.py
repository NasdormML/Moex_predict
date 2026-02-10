import pytest
import numpy as np
import pandas as pd

from app.preprocessing import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    preprocess_data
)


class TestTechnicalIndicators:
    """Тесты технических индикаторов."""
    
    def test_rsi_range(self):
        """RSI в диапазоне [0, 100]."""
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = compute_rsi(prices, period=5)
        
        assert rsi.min() >= 0
        assert rsi.max() <= 100
        assert not rsi.isna().all()
    
    def test_rsi_flat_prices(self):
        """RSI при постоянных ценах."""
        prices = pd.Series([100] * 20)
        rsi = compute_rsi(prices, period=5)
        
        # Должен обработать без ошибок
        assert not rsi.isna().any()
    
    def test_macd_crossover(self):
        """MACD корректно считает пересечения."""
        prices = pd.Series(np.sin(np.linspace(0, 4*np.pi, 100)) + 100)
        macd_line, signal_line, hist = compute_macd(prices)
        
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(hist) == len(prices)
    
    def test_bollinger_bands(self):
        """Bollinger Bands охватывают цены."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        upper, lower, mid = compute_bollinger_bands(prices, window=20)
        
        # Upper > Lower
        assert (upper > lower).all()
        # Mid между ними
        assert (mid >= lower).all() and (mid <= upper).all()
    
    def test_atr_positive(self):
        """ATR всегда положительный."""
        high = pd.Series(np.random.randn(100).cumsum() + 101)
        low = pd.Series(np.random.randn(100).cumsum() + 99)
        close = pd.Series(np.random.randn(100).cumsum() + 100)
        
        atr = compute_atr(high, low, close, window=14)
        
        assert (atr > 0).all()
        assert not atr.isna().any()


class TestPreprocessingPipeline:
    """Тесты полного пайплайна."""
    
    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame для препроцессинга."""
        return pd.DataFrame({
            "TRADEDATE": pd.date_range("2024-01-01", periods=50, freq="B"),
            "OPEN_SBER": np.random.randn(50).cumsum() + 250,
            "HIGH_SBER": np.random.randn(50).cumsum() + 251,
            "LOW_SBER": np.random.randn(50).cumsum() + 249,
            "CLOSE_SBER": np.random.randn(50).cumsum() + 250,
            "VOL_SBER": np.random.randint(1000000, 10000000, 50),
        })
    
    def test_preprocess_basic(self, sample_df):
        """Базовый препроцессинг."""
        result = preprocess_data(sample_df, "SBER")
        
        assert "RSI14" in result.columns
        assert "MACD_LINE" in result.columns
        assert "BB_UPPER" in result.columns
        assert "ATR" in result.columns
        assert "log_ret_1" in result.columns
    
    def test_preprocess_na_handling(self, sample_df):
        """Обработка пропусков."""
        # Добавляем NaN
        sample_df.loc[5:10, "CLOSE_SBER"] = np.nan
        
        result = preprocess_data(sample_df, "SBER")
        
        # Не должно быть NaN после обработки
        assert not result.isna().any().any()
    
    def test_preprocess_invalid_ticker(self, sample_df):
        """Ошибка при неверном тикере."""
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess_data(sample_df, "GAZP")
    
    def test_preprocess_returns_list(self, sample_df):
        """Возврат списка фичей."""
        result, features = preprocess_data(
            sample_df, "SBER", return_feature_list=True
        )
        
        assert isinstance(features, list)
        assert "CLOSE_SBER" in features
        assert "TRADEDATE" not in features