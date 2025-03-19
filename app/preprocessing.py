import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Проверка обязательных колонок
    required_columns = [
        "date",
        f"close_{ticker}",
        "close_RTSI",
        "close_USD000UTSTOM"
    ]
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют колонки: {missing}")

    # Создаем копию DataFrame для обработки
    processed = df.copy()

    # 1. Добавляем технические индикаторы
    processed[f"rsi_{ticker}"] = compute_rsi(processed[f"close_{ticker}"])
    
    # 2. Вычисляем производные фичи
    processed[f"returns_{ticker}"] = processed[f"close_{ticker}"].pct_change()
    processed["usd_rts_spread"] = processed["close_RTSI"] / processed["close_USD000UTSTOM"]
    
    # 3. Удаляем первые 14 строк (для RSI)
    processed = processed.iloc[14:].reset_index(drop=True)
    
    # 4. Выбираем финальные признаки
    final_features = [
        "date",
        f"close_{ticker}",
        f"rsi_{ticker}",
        f"returns_{ticker}",
        "close_RTSI",
        "close_USD000UTSTOM",
        "usd_rts_spread"
    ]
    
    return processed[final_features].dropna()