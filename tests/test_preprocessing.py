import unittest

import pandas as pd

from app.preprocessing import compute_rsi, preprocess_data


class TestPreprocessing(unittest.TestCase):
    def test_compute_rsi(self):
        series = pd.Series([1, 2, 3, 4, 5])
        rsi = compute_rsi(series, period=2)
        self.assertTrue(rsi.isnull().sum() > 0, "В начале должны быть NaN")

    def test_preprocess_data(self):
        data = {
            "TRADEDATE": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "CLOSE": [100, 105, 102],
            "OPEN": [98, 104, 101],
            "HIGH": [102, 107, 103],
            "LOW": [97, 103, 100],
            "VOLUME": [1000, 1500, 1200],
        }
        df = pd.DataFrame(data)
        ticker = "TEST"
        processed_df = preprocess_data(df, ticker)
        self.assertIn(f"CLOSE_{ticker}", processed_df.columns)
        self.assertIn(f"RSI_{ticker}", processed_df.columns)


if __name__ == "__main__":
    unittest.main()
