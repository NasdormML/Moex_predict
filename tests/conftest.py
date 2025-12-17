import pytest
import pandas as pd
from pathlib import Path

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

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory"""
    cache_dir = tmp_path / "data_cache"
    cache_dir.mkdir()
    return cache_dir