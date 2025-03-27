import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(series: pd.Series, window: int = 14) -> pd.Series:
    return series.rolling(window=window).mean()

def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df.columns = [col.upper() for col in df.columns]
    ticker = ticker.upper()
    
    if "OPEN" in df.columns:
        if "BEGIN" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["BEGIN"])
        elif "TRADEDATE" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
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
        if "TRADEDATE" not in df.columns:
            raise ValueError("Не найден столбец TRADEDATE")
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
    
    close_col = f"CLOSE_{ticker}"
    rsi_col = f"RSI_{ticker}"
    sma_col = f"SMA_{ticker}"
    
    df[rsi_col] = compute_rsi(df[close_col], period=14)
    df[sma_col] = compute_sma(df[close_col], window=14)
    
    # Заполнение пропусков для стабильности расчётов
    df[rsi_col] = df[rsi_col].ffill().bfill()
    df[sma_col] = df[sma_col].ffill().bfill()
    
    return df
