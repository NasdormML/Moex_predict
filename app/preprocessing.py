import numpy as np
import pandas as pd

RSI_PERIOD = 14
SMA_WINDOW = 14


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: int = 2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + std * num_std
    lower = sma - std * num_std
    return upper, lower, sma


def compute_atr(
    df: pd.DataFrame,
    ticker: str,
    window: int = SMA_WINDOW,
) -> pd.Series:
    high = df[f"HIGH_{ticker}"]
    low = df[f"LOW_{ticker}"]
    close = df[f"CLOSE_{ticker}"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    close = df[f"CLOSE_{ticker}"]
    vol = df[f"VOL_{ticker}"]

    df["dow"] = df["TRADEDATE"].dt.dayofweek
    df["month"] = df["TRADEDATE"].dt.month
    df["quarter"] = df["TRADEDATE"].dt.quarter

    df["log_ret_1"] = np.log(close).diff(1)
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)

    df["vol_5"] = df["log_ret_1"].rolling(5).std()
    df["vol_10"] = df["log_ret_1"].rolling(10).std()

    df["RSI14"] = compute_rsi(close)

    for span in (5, 10, 20, 50):
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()

    macd_line, macd_signal, macd_hist = compute_macd(close)
    df["MACD_LINE"], df["MACD_SIGNAL"], df["MACD_HIST"] = (
        macd_line,
        macd_signal,
        macd_hist,
    )

    bb_up, bb_low, bb_mid = compute_bollinger_bands(close)
    df["BB_UPPER"], df["BB_LOWER"], df["BB_MID"] = bb_up, bb_low, bb_mid

    df["ATR"] = compute_atr(df, ticker)
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_EMA"] = obv.ewm(span=20, adjust=False).mean()

    lag_feats = ["log_ret_1", "ret_1", "ret_5", "RSI14", "vol_5", "EMA_10", "MACD_HIST"]
    for feat in lag_feats:
        for lag in (1, 2, 3, 5):
            df[f"{feat}_lag{lag}"] = df[feat].shift(lag)

    return df.ffill().bfill().dropna().reset_index(drop=True)
