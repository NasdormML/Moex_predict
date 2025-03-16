import unittest
import pandas as pd
from app.data import fetch_moex_eod_data

class TestData(unittest.TestCase):
    def test_fetch_moex_eod_data(self):
        # Используем тикер SBER и короткий период для теста
        df = fetch_moex_eod_data("SBER", "stock", "shares", "TQBR", "2022-01-01", "2023-01-10")
        self.assertIsNotNone(df, "DataFrame не должен быть None")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("TRADEDATE", df.columns)

if __name__ == "__main__":
    unittest.main()
