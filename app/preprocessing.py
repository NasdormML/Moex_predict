import pandas as pd
import numpy as np

RSI_PERIOD = 14
SMA_WINDOW = 14

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_sma(series: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    return series.rolling(window=window).mean()

def compute_log_returns(series: pd.Series, window: int = 1) -> pd.Series:
    return np.log(series).diff(window)

def compute_volatility(series: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    returns = compute_log_returns(series)
    return returns.rolling(window=window).std()

def compute_macd(series: pd.Series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_bollinger_bands(series: pd.Series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band, sma

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # Ожидаем, что для тикера SBER столбцы уже называются: HIGH_SBER, LOW_SBER, CLOSE_SBER
    high_low = df["HIGH_SBER"] - df["LOW_SBER"]
    high_prev_close = np.abs(df["HIGH_SBER"] - df["CLOSE_SBER"].shift(1))
    low_prev_close = np.abs(df["LOW_SBER"] - df["CLOSE_SBER"].shift(1))
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df['RSI'] = compute_rsi(df[f"CLOSE_{ticker}"])
    df['SMA_RETURNS'] = compute_log_returns(compute_sma(df[f"CLOSE_{ticker}"]))
    df['VOLATILITY'] = compute_volatility(df[f"CLOSE_{ticker}"])
    df['LOG_RETURNS'] = compute_log_returns(df[f"CLOSE_{ticker}"])
    
    macd_line, macd_signal, macd_hist = compute_macd(df[f"CLOSE_{ticker}"])
    df['MACD_LINE'] = macd_line
    df['MACD_SIGNAL'] = macd_signal
    df['MACD_HIST'] = macd_hist
    
    bb_upper, bb_lower, bb_middle = compute_bollinger_bands(df[f"CLOSE_{ticker}"])
    df['BB_UPPER'] = bb_upper
    df['BB_LOWER'] = bb_lower
    df['BB_MIDDLE'] = bb_middle
    
    df['ATR'] = compute_atr(df, window=SMA_WINDOW)

    # Заполняем пропуски вперёд и назад, затем удаляем оставшиеся NaN
    df = df.ffill().bfill().dropna().reset_index(drop=True)
    return df
