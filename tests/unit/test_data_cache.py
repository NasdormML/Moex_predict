import pytest
import pandas as pd
from app.data import save_cached_data, load_cached_data

# Fixture
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

def test_cache_save_and_load(sample_moex_data):
    """Test cache save/load functionality"""
    # Arrange
    df = sample_moex_data
    
    # Act
    save_cached_data("SBER", "2023-01-01", "2023-01-10", df, context="test")
    loaded = load_cached_data("SBER", "2023-01-01", "2023-01-10", context="test")
    
    # Assert
    assert loaded is not None
    assert len(loaded) == len(df)
    assert (loaded['CLOSE_SBER'] == df['CLOSE_SBER']).all()

def test_cache_file_created_in_correct_location(tmp_path, sample_moex_data, monkeypatch):
    """Test that cache files"""
    
    def mock_cache_path(ticker, start, end, context="data"):
        return tmp_path / f"{context}_{ticker}_{start}_{end}.pkl"
    
    monkeypatch.setattr('app.data._cache_path', mock_cache_path)
    
    # Save cache
    save_cached_data("SBER", "2023-01-01", "2023-01-10", sample_moex_data, context="data")
    
    # Check file exists
    expected_file = tmp_path / "data_SBER_2023-01-01_2023-01-10.pkl"
    print(f"Debug: Expected file: {expected_file}")
    print(f"Debug: Files in tmp_path: {list(tmp_path.iterdir())}")
    assert expected_file.exists()
    
    # Load and verify
    loaded = load_cached_data("SBER", "2023-01-01", "2023-01-10", context="data")
    assert loaded is not None

def test_cache_returns_none_for_nonexistent():
    """Test that loading non-existent cache returns None"""
    result = load_cached_data("NONEXISTENT", "2023-01-01", "2023-01-10", context="test")
    assert result is None