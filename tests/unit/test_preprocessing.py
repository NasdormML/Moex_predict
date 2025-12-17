import pytest
import pandas as pd
import numpy as np
from app.preprocessing import compute_rsi, compute_bollinger_bands, preprocess_data

@pytest.fixture
def sample_moex_data():
    """Sample MOEX data for testing"""
    return pd.DataFrame({
        'TRADEDATE': pd.date_range('2023-01-01', periods=100),
        'OPEN_SBER': [100 + i for i in range(100)],
        'HIGH_SBER': [101 + i for i in range(100)],
        'LOW_SBER': [99 + i for i in range(100)],
        'CLOSE_SBER': [100 + i*0.5 for i in range(100)],
        'VOL_SBER': [1000] * 100,
        'CLOSE_IMOEX': [2000] * 100,
        'CLOSE_USD': [90] * 100
    })

def test_rsi_calculation():
    """Test RSI indicator calculation - first 'period' values will be NaN"""
    prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101] * 5)
    
    rsi = compute_rsi(prices, period=3)
    
    assert len(rsi) == len(prices)
    assert rsi.name == prices.name
    
    assert rsi.isna().sum() == 3
    
    valid_rsi = rsi.dropna()
    assert valid_rsi.between(0, 100).all()

def test_bollinger_bands():
    """Test Bollinger Bands calculation - first 'window' values may be NaN"""
    prices = pd.Series(np.random.rand(100) * 10 + 100)
    
    upper, lower, middle = compute_bollinger_bands(prices, window=20)
    
    assert len(upper) == len(prices)
    assert len(middle) == len(prices)
    assert len(lower) == len(prices)
    
    valid_mask = upper.notna() & middle.notna() & lower.notna()
    if valid_mask.any():
        valid_upper = upper[valid_mask]
        valid_middle = middle[valid_mask]
        valid_lower = lower[valid_mask]
        
        # upper >= middle >= lower
        assert (valid_upper >= valid_middle).all()
        assert (valid_middle >= valid_lower).all()

def test_preprocess_data_output(sample_moex_data):
    """Test full preprocessing pipeline"""
    processed = preprocess_data(sample_moex_data, ticker="SBER")
    
    assert isinstance(processed, pd.DataFrame)
    assert len(processed) > 0
    
    required_base = ["target_close", "CLOSE_SBER", "RSI_14", "MACD_LINE"]
    for col in required_base:
        assert col in processed.columns, f"Missing base column: {col}"
    
    # ATR может быть удалена из-за высокой корреляции
    volatility_cols = [c for c in processed.columns if 'volatility' in c or 'ATR' in c]
    assert len(volatility_cols) > 0, "No volatility features found"
    
    na_count = processed.isna().sum().sum()
    assert na_count == 0, f"Found {na_count} NaN values after preprocessing"
    
    assert len(processed.columns) > 50
    
    non_numeric = processed.select_dtypes(exclude=[np.number]).columns
    assert len(non_numeric) == 0, f"Found non-numeric columns: {list(non_numeric)}"

def test_preprocess_data_length(sample_moex_data):
    """Test that preprocessing doesn't drop too many rows"""
    original_len = len(sample_moex_data)
    processed = preprocess_data(sample_moex_data, ticker="SBER")
    
    assert len(processed) >= original_len * 0.9