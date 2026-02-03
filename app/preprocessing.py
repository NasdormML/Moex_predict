import logging
from typing import Final, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RSI_PERIOD: Final[int] = 14
DEFAULT_SMA_WINDOW: Final[int] = 14
DEFAULT_BB_WINDOW: Final[int] = 20
DEFAULT_BB_STD: Final[float] = 2.0


def compute_rsi(series: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
    """
    Compute Relative Strength Index (RSI). Returns values in range [0,100].
    Uses EWM averages like common implementations.
    """
    series = series.astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line and histogram."""
    series = series.astype(float)
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_bollinger_bands(
    series: pd.Series,
    window: int = DEFAULT_BB_WINDOW,
    num_std: float = DEFAULT_BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper, lower, middle SMA)."""
    series = series.astype(float)
    sma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = sma + std * num_std
    lower = sma - std * num_std
    return upper, lower, sma


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = DEFAULT_SMA_WINDOW,
) -> pd.Series:
    """
    Compute Average True Range (ATR).
    Uses rolling mean of the true range.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def _validate_input_columns(df: pd.DataFrame, ticker: str) -> None:
    required = [
        f"CLOSE_{ticker}",
        f"VOL_{ticker}",
        f"HIGH_{ticker}",
        f"LOW_{ticker}",
        "TRADEDATE",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ticker '{ticker}': {missing}")


def _safe_log_series(series: pd.Series) -> pd.Series:
    eps = 1e-9
    if (series <= 0).any():
        logger.warning(
            "Non-positive values found in price series; clipping to eps before log."
        )
    return np.log(series.clip(lower=eps))


def preprocess_data(
    df: pd.DataFrame,
    ticker: str,
    rsi_period: int = DEFAULT_RSI_PERIOD,
    na_strategy: str = "ffill_drop",
    return_feature_list: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, List[str]]:
    """
    Args:
        df: DataFrame with columns including TRADEDATE,
        CLOSE_{ticker}, VOL_{ticker}, HIGH_{ticker}, LOW_{ticker}
        ticker: ticker suffix used in column names
        rsi_period: period for RSI
        na_strategy: how to handle NA values; only 'ffill_drop' is currently supported
        return_feature_list: if True, returns (df, feature_list)

    Returns:
        DataFrame with new features (and TRADEDATE). Optionally (df, feature_list).
    """
    if df is None:
        raise ValueError("Input dataframe is None")
    df = df.copy()
    if "TRADEDATE" not in df.columns:
        raise ValueError("TRADEDATE column is required")

    _validate_input_columns(df, ticker)

    # Ensure TRADEDATE dtype and sort
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)

    # Extract series and cast to float
    close = df[f"CLOSE_{ticker}"].astype(float)
    vol = df[f"VOL_{ticker}"].astype(float)
    high = df[f"HIGH_{ticker}"].astype(float)
    low = df[f"LOW_{ticker}"].astype(float)

    # Time features
    df["dow"] = df["TRADEDATE"].dt.dayofweek
    df["month"] = df["TRADEDATE"].dt.month
    df["quarter"] = df["TRADEDATE"].dt.quarter

    # Returns
    df["log_ret_1"] = _safe_log_series(close).diff(1)
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)

    # Volatility
    df["vol_5"] = df["log_ret_1"].rolling(5, min_periods=1).std()
    df["vol_10"] = df["log_ret_1"].rolling(10, min_periods=1).std()

    # Technical indicators
    df["RSI14"] = compute_rsi(close, rsi_period)

    for span in (5, 10, 20, 50):
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()

    macd_line, macd_signal, macd_hist = compute_macd(close)
    df["MACD_LINE"] = macd_line
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist

    bb_up, bb_low, bb_mid = compute_bollinger_bands(close)
    df["BB_UPPER"] = bb_up
    df["BB_LOWER"] = bb_low
    df["BB_MID"] = bb_mid

    df["ATR"] = compute_atr(high, low, close)

    # OBV (On-Balance Volume)
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_EMA"] = obv.ewm(span=20, adjust=False).mean()

    # Lag features
    lag_feats = ["log_ret_1", "ret_1", "ret_5", "RSI14", "vol_5", "EMA_10", "MACD_HIST"]
    for feat in lag_feats:
        for lag in (1, 2, 3, 5):
            df[f"{feat}_lag{lag}"] = df[feat].shift(lag)

    # NA handling: default forward-fill then drop remaining NaNs to avoid lookahead
    if na_strategy == "ffill_drop":
        df = df.ffill().dropna().reset_index(drop=True)
    else:
        raise ValueError(f"Unknown na_strategy: {na_strategy}")

    feature_list = [c for c in df.columns if c != "TRADEDATE"]

    if return_feature_list:
        return df, feature_list
    return df
