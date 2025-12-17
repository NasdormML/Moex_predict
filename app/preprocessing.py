import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    delta_clipped = delta.clip(lower=delta.quantile(0.05), upper=delta.quantile(0.95))
    gain = delta_clipped.clip(lower=0)
    loss = -delta_clipped.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    ema_fast = series.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    upper = sma + std * num_std
    lower = sma - std * num_std
    return upper, lower, sma


def compute_atr(df: pd.DataFrame, ticker: str, window: int = 14):
    high = df.get(f"HIGH_{ticker}")
    low = df.get(f"LOW_{ticker}")
    close = df.get(f"CLOSE_{ticker}")
    if high is None or low is None or close is None:
        return pd.Series(dtype=float)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()
    return atr


def compute_volume_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = df.get(f"CLOSE_{ticker}")
    volume = df.get(f"VOL_{ticker}")
    if close is None or volume is None:
        return df
    df = df.copy()
    df["volume_sma_5"] = volume.rolling(5, min_periods=1).mean()
    df["volume_sma_20"] = volume.rolling(20, min_periods=1).mean()
    df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-10)
    df["price_volume"] = close * volume
    price_change = close.diff()
    volume_direction = np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
    df["obv_enhanced"] = volume_direction.cumsum()
    df["volume_volatility"] = volume.rolling(10, min_periods=1).std()
    return df


def compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = pd.to_datetime(df["TRADEDATE"])
    df["day_sin"] = np.sin(2 * np.pi * date_col.dt.dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * date_col.dt.dayofyear / 365)
    df["month_sin"] = np.sin(2 * np.pi * date_col.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date_col.dt.month / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * date_col.dt.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * date_col.dt.dayofweek / 7)
    df["quarter_sin"] = np.sin(2 * np.pi * date_col.dt.quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * date_col.dt.quarter / 4)
    return df


def compute_statistical_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    close = df.get(f"CLOSE_{ticker}")
    if close is None:
        return df
    returns = np.log(close).diff()
    for window in [5, 10, 20]:
        df[f"returns_skew_{window}"] = returns.rolling(window, min_periods=2).skew()
        df[f"returns_kurt_{window}"] = returns.rolling(window, min_periods=4).kurt()
    rolling_mean = close.rolling(20, min_periods=1).mean()
    rolling_std = close.rolling(20, min_periods=1).std()
    df["price_zscore_20"] = (close - rolling_mean) / (rolling_std + 1e-10)
    df["price_q25_10"] = close.rolling(10, min_periods=1).quantile(0.25)
    df["price_q75_10"] = close.rolling(10, min_periods=1).quantile(0.75)
    df["price_iqr_10"] = df["price_q75_10"] - df["price_q25_10"]
    return df


def compute_trend_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    close = df.get(f"CLOSE_{ticker}")
    if close is None:
        return df
    ema_short = close.ewm(span=8, adjust=False).mean()
    ema_medium = close.ewm(span=21, adjust=False).mean()
    ema_long = close.ewm(span=55, adjust=False).mean()
    df["ema_ratio_short"] = close / (ema_short + 1e-10)
    df["ema_ratio_medium"] = close / (ema_medium + 1e-10)
    df["ema_ratio_long"] = close / (ema_long + 1e-10)
    rolling_min = close.rolling(10, min_periods=1).min()
    rolling_max = close.rolling(10, min_periods=1).max()
    df["price_position_10"] = (close - rolling_min) / (rolling_max - rolling_min + 1e-10)
    rolling_mean_20 = close.rolling(20, min_periods=1).mean()
    rolling_std_20 = close.rolling(20, min_periods=1).std()
    df["trend_strength"] = (close - rolling_mean_20) / (rolling_std_20 + 1e-10)
    return df


def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        z_scores = np.abs((col_data - col_data.mean()) / (col_data.std() + 1e-10))
        valid_indices = df[col].notna()
        outlier_mask = np.zeros(len(df[col]), dtype=bool)
        outlier_mask[valid_indices] = z_scores > threshold
        df.loc[outlier_mask, col] = np.nan
    return df


def create_advanced_lags(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 2, 3, 5, 8, 13]:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    for window in [3, 5, 8, 13]:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window, min_periods=1).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window, min_periods=1).std()
        df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window, min_periods=1).min()
        df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window, min_periods=1).max()
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "volume_ratio" in df.columns and "volatility_10" in df.columns:
        df["volume_volatility_interaction"] = df["volume_ratio"] * df["volatility_10"]
    if "RSI_14" in df.columns and "trend_strength" in df.columns:
        df["rsi_trend_interaction"] = df["RSI_14"] * df["trend_strength"]
    if "MACD_HIST" in df.columns and "BB_POSITION" in df.columns:
        df["macd_bb_interaction"] = df["MACD_HIST"] * df["BB_POSITION"]
    return df


def remove_highly_correlated_features(df: pd.DataFrame, ticker: str, threshold: float = 0.95) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return df
    protected_columns = [
        f"OPEN_{ticker}",
        f"HIGH_{ticker}",
        f"LOW_{ticker}",
        f"CLOSE_{ticker}",
        f"VOL_{ticker}",
        "CLOSE_IMOEX",
        "CLOSE_USD",
        "target_close",
    ]
    protected_columns = [col for col in protected_columns if col in numeric_df.columns]
    columns_to_analyze = [col for col in numeric_df.columns if col not in protected_columns]
    if len(columns_to_analyze) < 2:
        return df
    corr_matrix = numeric_df[columns_to_analyze].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    df_clean = df.drop(columns=high_corr_features)
    logger.info("Removed %d highly correlated features; protected %d columns", len(high_corr_features), len(protected_columns))
    return df_clean


def ensure_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_df = df[numeric_cols].copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    logger.debug("Preserved %d numeric columns out of %d", len(numeric_df.columns), len(df.columns))
    return numeric_df


def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    logger.info("Starting preprocessing for %s with %d rows", ticker, len(df))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    close_col = f"CLOSE_{ticker}"
    if close_col not in df.columns:
        logger.warning("Missing close column %s", close_col)
        return pd.DataFrame()
    close = df[close_col]

    # basic returns and vol
    df["log_ret_1"] = np.log(close).diff(1)
    df["log_ret_3"] = np.log(close).diff(3)
    df["log_ret_5"] = np.log(close).diff(5)
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["log_ret_1"].rolling(window, min_periods=1).std()

    # technicals
    df["RSI_14"] = compute_rsi(close)
    df["RSI_7"] = compute_rsi(close, period=7)
    for span in [5, 8, 13, 21, 34, 55]:
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()
    
    macd_line, macd_signal, macd_hist = compute_macd(close)
    df["MACD_LINE"] = macd_line
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist
    bb_up, bb_low, bb_mid = compute_bollinger_bands(close)

    df["BB_UPPER"] = bb_up
    df["BB_LOWER"] = bb_low
    df["BB_MID"] = bb_mid
    df["BB_WIDTH"] = (bb_up - bb_low) / (bb_mid + 1e-10)
    df["BB_POSITION"] = (close - bb_low) / (bb_up - bb_low + 1e-10)
    df["ATR"] = compute_atr(df, ticker)
    df["ATR_RATIO"] = df["ATR"] / (close + 1e-10)

    # extended features
    df = compute_volume_features(df, ticker)
    df = compute_cyclical_features(df)
    df = compute_statistical_features(df, ticker)
    df = compute_trend_features(df, ticker)

    # target and lags
    df["target_close"] = close.shift(-1)
    key_features = [
        "log_ret_1",
        "RSI_14",
        "MACD_HIST",
        "BB_POSITION",
        "volume_ratio",
        "price_zscore_20",
        "trend_strength",
    ]
    existing_features = [f for f in key_features if f in df.columns]
    for feature in existing_features:
        df = create_advanced_lags(df, feature)

    df = create_interaction_features(df)

    # outliers and cleaning
    df = remove_outliers(df, threshold=3.0)
    initial_len = len(df)
    df = df.ffill().bfill().dropna().reset_index(drop=True)
    final_len = len(df)
    logger.info("Dropped %d rows with missing values", initial_len - final_len)

    df = remove_highly_correlated_features(df, ticker, threshold=0.95)
    df = ensure_numeric_dataframe(df)

    required_columns = [f"CLOSE_{ticker}", "target_close"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning("Missing key columns after preprocess: %s", missing_columns)

    logger.info("Preprocessing completed. Final feature count: %d. Rows: %d", len(df.columns), len(df))
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logger.warning("Non-numeric columns remain: %s", list(non_numeric_cols))
        df = df.select_dtypes(include=[np.number])

    return df
