import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Приводим имена столбцов к верхнему регистру
    df.columns = [col.upper() for col in df.columns]
    
    ticker = ticker.upper()
    if "OPEN" in df.columns:
        # Если присутствует столбец BEGIN, используем его для времени
        if "BEGIN" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["BEGIN"])
        # Если нет BEGIN, но есть TRADEDATE, используем его
        elif "TRADEDATE" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        # Если нет ни того ни другого, проверяем наличие столбца TIME
        elif "TIME" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["TIME"])
        else:
            raise ValueError("Не найден столбец для времени (BEGIN, TRADEDATE или TIME)")
            
        df.rename(columns={
            "OPEN": f"OPEN_{ticker}",
            "HIGH": f"HIGH_{ticker}",
            "LOW": f"LOW_{ticker}",
            "CLOSE": f"CLOSE_{ticker}",
            "VOLUME": f"VOL_{ticker}"
        }, inplace=True)
    else:
        # Если данных нет в формате с OPEN, предполагаем, что уже есть нужный формат
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df.rename(columns={
            "OPEN": f"OPEN_{ticker}",
            "HIGH": f"HIGH_{ticker}",
            "LOW": f"LOW_{ticker}",
            "CLOSE": f"CLOSE_{ticker}",
            "VOLUME": f"VOL_{ticker}"
        }, inplace=True)
    
    df.sort_values("TRADEDATE", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Вычисляем RSI по цене закрытия
    close_col = f"CLOSE_{ticker}"
    rsi_col = f"RSI_{ticker}"
    df[rsi_col] = compute_rsi(df[close_col], period=14)
    df.dropna(subset=[rsi_col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
