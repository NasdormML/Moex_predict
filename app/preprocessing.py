import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """""
    Parameters:
        series (pd.Series): Последовательность цен закрытия.
        period (int): Период для расчёта RSI.
    
    Returns:
        pd.Series: Значения RSI, где первые значения будут NaN.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """

    Выполняет шаги:
      - Преобразование столбца даты в тип datetime и сортировка по дате.
      - Переименование столбцов для ясности (добавление имени тикера).
      - Вычисление RSI по цене закрытия.
      - Добавление дополнительных признаков (тело свечи, верхняя и нижняя тени).
      - Удаление строк с отсутствующими значениями RSI.
    
    Parameters:
        df (pd.DataFrame): Исходный DataFrame с данными MOEX.
        ticker (str): Тикер, например, "SBER" или "GAZP".
    
    Returns:
        pd.DataFrame: Предобработанный DataFrame с новыми признаками.
    """
    # Приводим дату к типу datetime и сортируем по дате
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df.sort_values("TRADEDATE", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Переименование столбцов для уникальности
    close_col = f"CLOSE_{ticker}"
    open_col  = f"OPEN_{ticker}"
    high_col  = f"HIGH_{ticker}"
    low_col   = f"LOW_{ticker}"
    vol_col   = f"VOL_{ticker}"
    
    df.rename(columns={
        "CLOSE": close_col, 
        "OPEN": open_col, 
        "HIGH": high_col, 
        "LOW": low_col, 
        "VOLUME": vol_col
    }, inplace=True)
    
    # Вычисляем RSI для цены закрытия
    rsi_col = f"RSI_{ticker}"
    df[rsi_col] = compute_rsi(df[close_col], period=14)
    
    # Удаляем строки с NaN в RSI (первые строки из-за окна расчёта)
    df.dropna(subset=[rsi_col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Дополнительные свечные признаки
    body_col = f"BODY_{ticker}"
    upper_shadow_col = f"UPPER_SHADOW_{ticker}"
    lower_shadow_col = f"LOWER_SHADOW_{ticker}"
    
    df[body_col] = (df[close_col] - df[open_col]).abs()
    df[upper_shadow_col] = df[high_col] - df[[open_col, close_col]].max(axis=1)
    df[lower_shadow_col] = df[[open_col, close_col]].min(axis=1) - df[low_col]
    
    return df
