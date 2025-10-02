import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Optimized indicator parameters
RSI_PERIOD = 14
SMA_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Enhanced RSI with outlier handling"""
    delta = series.diff()

    # Clipping outliers in differences
    delta_clipped = np.clip(delta, delta.quantile(0.05), delta.quantile(0.95))

    gain = delta_clipped.clip(lower=0)
    loss = -delta_clipped.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    series: pd.Series,
    fast_period: int = MACD_FAST,
    slow_period: int = MACD_SLOW,
    signal_period: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD with exponential smoothing"""
    ema_fast = series.ewm(
        span=fast_period, adjust=False, min_periods=fast_period
    ).mean()
    ema_slow = series.ewm(
        span=slow_period, adjust=False, min_periods=slow_period
    ).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(
        span=signal_period, adjust=False, min_periods=signal_period
    ).mean()
    macd_hist = macd_line - signal_line

    return macd_line, signal_line, macd_hist


def compute_bollinger_bands(
    series: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std: int = BOLLINGER_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands with robust standard deviation"""
    sma = series.rolling(window=window, min_periods=1).mean()

    # Standard deviation instead of MAD for stability
    std = series.rolling(window=window, min_periods=1).std()

    upper = sma + std * num_std
    lower = sma - std * num_std

    return upper, lower, sma


def compute_atr(
    df: pd.DataFrame,
    ticker: str,
    window: int = SMA_WINDOW,
) -> pd.Series:
    """Average True Range with enhanced processing"""
    high = df[f"HIGH_{ticker}"]
    low = df[f"LOW_{ticker}"]
    close = df[f"CLOSE_{ticker}"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()

    return atr


def compute_volume_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extended volume-based features"""
    close = df[f"CLOSE_{ticker}"]
    volume = df[f"VOL_{ticker}"]

    # Basic volume features
    df["volume_sma_5"] = volume.rolling(5, min_periods=1).mean()
    df["volume_sma_20"] = volume.rolling(20, min_periods=1).mean()
    df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-10)

    # Price * Volume (VWAP-like)
    df["price_volume"] = close * volume

    # OBV with enhancements
    price_change = close.diff()
    volume_direction = np.where(
        price_change > 0, volume, np.where(price_change < 0, -volume, 0)
    )
    df["obv_enhanced"] = volume_direction.cumsum()

    # Volume volatility
    df["volume_volatility"] = volume.rolling(10, min_periods=1).std()

    return df


def compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time features"""
    date_col = pd.to_datetime(df["TRADEDATE"])

    # Sine/cosine transformations for cyclicity
    df["day_sin"] = np.sin(2 * np.pi * date_col.dt.dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * date_col.dt.dayofyear / 365)

    df["month_sin"] = np.sin(2 * np.pi * date_col.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date_col.dt.month / 12)

    # Day of week in cyclical form
    df["weekday_sin"] = np.sin(2 * np.pi * date_col.dt.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * date_col.dt.dayofweek / 7)

    # Quarterly features
    df["quarter_sin"] = np.sin(2 * np.pi * date_col.dt.quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * date_col.dt.quarter / 4)

    return df


def compute_statistical_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Statistical features and distribution moments"""
    close = df[f"CLOSE_{ticker}"]
    returns = np.log(close).diff()

    # Rolling distribution moments
    for window in [5, 10, 20]:
        df[f"returns_skew_{window}"] = returns.rolling(window, min_periods=2).skew()
        df[f"returns_kurt_{window}"] = returns.rolling(window, min_periods=4).kurt()

    # Price Z-score
    rolling_mean = close.rolling(20, min_periods=1).mean()
    rolling_std = close.rolling(20, min_periods=1).std()
    df["price_zscore_20"] = (close - rolling_mean) / (rolling_std + 1e-10)

    # Quantile features
    df["price_q25_10"] = close.rolling(10, min_periods=1).quantile(0.25)
    df["price_q75_10"] = close.rolling(10, min_periods=1).quantile(0.75)
    df["price_iqr_10"] = df["price_q75_10"] - df["price_q25_10"]

    return df


def compute_trend_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Trend and momentum features"""
    close = df[f"CLOSE_{ticker}"]

    # Multiple EMA ratios
    ema_short = close.ewm(span=8, adjust=False).mean()
    ema_medium = close.ewm(span=21, adjust=False).mean()
    ema_long = close.ewm(span=55, adjust=False).mean()

    df["ema_ratio_short"] = close / (ema_short + 1e-10)
    df["ema_ratio_medium"] = close / (ema_medium + 1e-10)
    df["ema_ratio_long"] = close / (ema_long + 1e-10)

    # Price position in recent range
    rolling_min = close.rolling(10, min_periods=1).min()
    rolling_max = close.rolling(10, min_periods=1).max()
    df["price_position_10"] = (close - rolling_min) / (
        rolling_max - rolling_min + 1e-10
    )

    # Trend strength
    rolling_mean_20 = close.rolling(20, min_periods=1).mean()
    rolling_std_20 = close.rolling(20, min_periods=1).std()
    df["trend_strength"] = (close - rolling_mean_20) / (rolling_std_20 + 1e-10)

    return df


def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Outlier removal using z-score"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Calculate z-score, ignoring NaN
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        z_scores = np.abs(stats.zscore(col_data, nan_policy="omit"))

        # Create mask for outliers
        outlier_mask = np.zeros(len(df[col]), dtype=bool)
        valid_indices = df[col].notna()
        outlier_mask[valid_indices] = z_scores > threshold

        # Replace outliers with NaN
        df.loc[outlier_mask, col] = np.nan

    return df


def create_advanced_lags(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Creating extended lags and window statistics"""
    # Basic lags
    for lag in [1, 2, 3, 5, 8, 13]:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # Rolling statistics for target variable
    for window in [3, 5, 8, 13]:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df[target_col].rolling(window, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            df[target_col].rolling(window, min_periods=1).std()
        )
        df[f"{target_col}_rolling_min_{window}"] = (
            df[target_col].rolling(window, min_periods=1).min()
        )
        df[f"{target_col}_rolling_max_{window}"] = (
            df[target_col].rolling(window, min_periods=1).max()
        )

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creating interaction features between key variables"""
    df = df.copy()

    # Volume-volatility interaction
    if "volume_ratio" in df.columns and "volatility_10" in df.columns:
        df["volume_volatility_interaction"] = df["volume_ratio"] * df["volatility_10"]

    # RSI-trend interaction
    if "RSI_14" in df.columns and "trend_strength" in df.columns:
        df["rsi_trend_interaction"] = df["RSI_14"] * df["trend_strength"]

    # MACD-Bollinger Bands interaction
    if "MACD_HIST" in df.columns and "BB_POSITION" in df.columns:
        df["macd_bb_interaction"] = df["MACD_HIST"] * df["BB_POSITION"]

    return df


def remove_highly_correlated_features(
    df: pd.DataFrame, ticker: str, threshold: float = 0.95
) -> pd.DataFrame:
    """Removing highly correlated features with protection of key columns"""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or len(numeric_df.columns) < 2:
        return df

    # Protect key columns from removal
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

    # Keep only existing protected columns
    protected_columns = [col for col in protected_columns if col in numeric_df.columns]

    # Columns for correlation analysis (excluding protected ones)
    columns_to_analyze = [
        col for col in numeric_df.columns if col not in protected_columns
    ]

    if len(columns_to_analyze) < 2:
        return df

    # Calculate correlation matrix only for analyzed columns
    corr_matrix = numeric_df[columns_to_analyze].corr().abs()

    # Upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation above threshold
    high_corr_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]

    # Remove these features
    df_clean = df.drop(columns=high_corr_features)

    print(f"Removed {len(high_corr_features)} highly correlated features")
    print(f"Protected {len(protected_columns)} key columns")

    return df_clean


def ensure_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns are numeric and remove non-numeric ones"""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Create new DataFrame with only numeric columns
    numeric_df = df[numeric_cols].copy()

    # Ensure all data can be converted to float
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    print(
        f"Preserved {len(numeric_df.columns)} numeric columns out of {len(df.columns)}"
    )
    return numeric_df


def preprocess_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Enhanced data preprocessing with extended feature engineering
    """
    print("Starting enhanced data preprocessing...")
    df = df.copy()

    # Save original columns for debugging
    original_columns = df.columns.tolist()
    print(f"Original columns: {len(original_columns)}")

    # Basic time features
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    close = df[f"CLOSE_{ticker}"]

    print("1. Computing basic features...")
    # Basic returns and volatility
    df["log_ret_1"] = np.log(close).diff(1)
    df["log_ret_3"] = np.log(close).diff(3)
    df["log_ret_5"] = np.log(close).diff(5)

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)

    # Volatility of different periods
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = (
            df["log_ret_1"].rolling(window, min_periods=1).std()
        )

    print("2. Adding technical indicators...")
    # RSI
    df["RSI_14"] = compute_rsi(close)
    df["RSI_7"] = compute_rsi(close, period=7)

    # EMA of different periods
    for span in [5, 8, 13, 21, 34, 55]:
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()

    # MACD
    macd_line, macd_signal, macd_hist = compute_macd(close)
    df["MACD_LINE"], df["MACD_SIGNAL"], df["MACD_HIST"] = (
        macd_line,
        macd_signal,
        macd_hist,
    )

    # Bollinger Bands
    bb_up, bb_low, bb_mid = compute_bollinger_bands(close)
    df["BB_UPPER"], df["BB_LOWER"], df["BB_MID"] = bb_up, bb_low, bb_mid
    df["BB_WIDTH"] = (bb_up - bb_low) / (bb_mid + 1e-10)
    df["BB_POSITION"] = (close - bb_low) / (bb_up - bb_low + 1e-10)

    # ATR
    df["ATR"] = compute_atr(df, ticker)
    df["ATR_RATIO"] = df["ATR"] / (close + 1e-10)

    print("3. Adding extended features...")
    # Volume features
    df = compute_volume_features(df, ticker)

    # Cyclical time features
    df = compute_cyclical_features(df)

    # Statistical features
    df = compute_statistical_features(df, ticker)

    # Trend features
    df = compute_trend_features(df, ticker)

    print("4. Creating lags and window statistics...")
    # Target variable for prediction
    df["target_close"] = close.shift(-1)

    # Lags for key features
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

    print("5. Adding interaction features...")
    df = create_interaction_features(df)

    print("6. Handling outliers...")
    df = remove_outliers(df, threshold=3.0)

    print("7. Final cleaning...")
    # Remove rows with missing values
    initial_len = len(df)
    df = df.ffill().bfill().dropna().reset_index(drop=True)
    final_len = len(df)

    print(f"Removed {initial_len - final_len} rows with missing values")

    print("8. Removing highly correlated features...")
    df = remove_highly_correlated_features(df, ticker, threshold=0.95)

    print("9. Ensuring numeric data format...")
    df = ensure_numeric_dataframe(df)

    # Check that key columns are preserved
    required_columns = [f"CLOSE_{ticker}", "target_close"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"WARNING: Missing key columns: {missing_columns}")

    print(f"Preprocessing completed. Final feature count: {len(df.columns)}")
    print(f"Number of observations: {len(df)}")

    # Check that all data is numeric
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"WARNING: Non-numeric columns remain: {list(non_numeric_cols)}")
        df = df.select_dtypes(include=[np.number])

    return df
