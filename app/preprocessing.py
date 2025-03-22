import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Вычисляет RSI с использованием экспоненциального сглаживания (метод Уайлдера).
    Для первых period значений, где данных недостаточно, заполняет их первым рассчитанным значением RSI.
    """
    delta = series.diff()

    # Вычисляем приросты и падения
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Рассчитываем экспоненциальное скользящее среднее приростов и потерь
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Вычисляем RS и RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Если данных достаточно, заполняем первые period значений
    if len(rsi) > period:
        first_valid = rsi.iloc[period]
        rsi.iloc[:period] = first_valid
    else:
        rsi[:] = np.nan  # либо можно заполнить значением 50, если это предпочтительно

    return rsi

def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Предобрабатывает DataFrame:
      - Приводит имена столбцов к верхнему регистру.
      - Определяет столбец времени (TRADEDATE) из BEGIN/TRADEDATE/TIME.
      - Переименовывает колонки для тикера.
      - Вычисляет RSI по цене закрытия с обновлённой функцией compute_rsi.
    """
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
    
    # Вычисляем RSI с помощью новой функции
    df[rsi_col] = compute_rsi(df[close_col], period=14)
    
    # На случай, если после вычисления остались NaN, заполним их
    df[rsi_col] = df[rsi_col].ffill().bfill()
    
    return df
