import logging
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
from app.preprocessing import preprocess_data

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "data_cache"))
EXPIRATION_DAYS = int(os.getenv("DATA_CACHE_EXP_DAYS", "1"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_session = requests.Session()
_retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)



def _cache_path(ticker: str, start: str, end: str, context: str = "data") -> Path:
    safe_ticker = ticker.replace("/", "_")
    return CACHE_DIR / f"{context}_{safe_ticker}_{start}_{end}.pkl"


def load_cached_data(ticker: str, start: str, end: str, context: str = "data") -> Optional[pd.DataFrame]:
    p = _cache_path(ticker, start, end, context)
    try:
        if not p.exists():
            return None
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=EXPIRATION_DAYS):
            logger.debug("Cache expired: %s", p)
            return None
        return pd.read_pickle(p)
    except Exception as e:
        logger.warning("Failed to load cache %s: %s", p, e)
        return None


def save_cached_data(ticker: str, start: str, end: str, df: pd.DataFrame, context: str = "data") -> None:
    p = _cache_path(ticker, start, end, context)
    temp = p.with_suffix(".tmp")
    try:
        df.to_pickle(temp)
        os.replace(temp, p)
        logger.debug("Saved cache %s", p)
    except Exception as e:
        logger.warning("Failed to write cache %s: %s", p, e)
        if temp.exists():
            try:
                temp.unlink()
            except Exception:
                pass


@lru_cache(maxsize=500)
def fetch_cbr_usd_rate(date_str: str) -> float:
    url = f"http://www.cbr.ru/scripts/XML_daily.asp?date_req={date_str}"
    resp = _session.get(url, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    for v in root.findall("Valute"):
        cc = v.find("CharCode")
        val = v.find("Value")
        if cc is not None and cc.text == "USD" and val is not None:
            return float(val.text.replace(",", "."))
    raise RuntimeError(f"USD rate not found for {date_str}")


def fetch_moex_eod_data(
    security: str,
    engine: str,
    market: str,
    board: str,
    start: str,
    end: str,
    *,
    skip_cache: bool = False,
    context: str = "data",
    session: requests.Session = _session,
) -> pd.DataFrame:
    if not skip_cache:
        df_cached = load_cached_data(security, start, end, context)
        if df_cached is not None:
            return df_cached

    url = (
        f"https://iss.moex.com/iss/history/engines/{engine}/markets/{market}/"
        f"boards/{board}/securities/{security}.json"
    )
    all_data = []
    params = {"from": start, "till": end, "start": 0, "limit": 100}
    while True:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        hist = payload.get("history", {})
        data = hist.get("data", [])
        if not data:
            break
        all_data.extend(data)
        if len(data) < params["limit"]:
            break
        params["start"] += params["limit"]

    if not all_data:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(all_data, columns=hist.get("columns", []))

    if not skip_cache and not df.empty:
        try:
            save_cached_data(security, start, end, df, context)
        except Exception as e:
            logger.warning("Failed to cache moex data %s: %s", security, e)
    return df


def fetch_usd_series(start: str, end: str, *, skip_cache: bool = False, context: str = "data", max_workers: int = 8) -> pd.DataFrame:
    ticker = "USD000UTSTOM"
    if not skip_cache:
        df_cached = load_cached_data(ticker, start, end, context)
        if df_cached is not None:
            return df_cached

    dates = pd.date_range(start, end)
    records: List[Tuple[pd.Timestamp, float]] = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {
            executor.submit(fetch_cbr_usd_rate, d.strftime("%d/%m/%Y")): d for d in dates
        }
        for future in as_completed(future_to_date):
            d = future_to_date[future]
            try:
                rate = future.result()
            except Exception as exc:
                logger.warning("Failed to fetch USD for %s: %s", d, exc)
                rate = float("nan")
                errors.append((d, str(exc)))
            records.append((d, rate))

    df = pd.DataFrame(records, columns=["TRADEDATE", "CLOSE"]) if records else pd.DataFrame()
    if not df.empty:
        df.sort_values("TRADEDATE", inplace=True)
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    if not skip_cache and not df.empty:
        save_cached_data(ticker, start, end, df, context)
    if errors:
        logger.debug("USD fetch errors (sample): %s", errors[:5])
    return df


def get_smart_features(proc: pd.DataFrame, ticker: str, target_count: Optional[int] = None) -> List[str]:
    excluded_cols = {"TRADEDATE", "target_close", "date"}
    all_features = [col for col in proc.columns if col not in excluded_cols]

    logger.info("Total available features: %d", len(all_features))
    if target_count is None or target_count > len(all_features):
        target_count = len(all_features)

    feature_groups = {
        "price_basic": [
            f"OPEN_{ticker}",
            f"HIGH_{ticker}",
            f"LOW_{ticker}",
            f"CLOSE_{ticker}",
            f"VOL_{ticker}",
            "CLOSE_IMOEX",
            "CLOSE_USD",
        ],
        "returns": [f for f in all_features if f.startswith(("log_ret_", "ret_"))],
        "volatility": [f for f in all_features if f.startswith("volatility_")],
        "rsi": [f for f in all_features if f.startswith("RSI_")],
        "ema": [f for f in all_features if f.startswith("EMA_")],
        "macd": [f for f in all_features if f.startswith("MACD_")],
        "bollinger": [f for f in all_features if f.startswith("BB_")],
        "atr": [f for f in all_features if f.startswith("ATR")],
        "volume": [f for f in all_features if "volume" in f.lower()],
        "statistical": [
            f
            for f in all_features
            if any(x in f for x in ["skew", "kurt", "zscore", "q25", "q75", "iqr", "mad"])
        ],
        "trend": [f for f in all_features if any(x in f for x in ["trend", "ratio", "position"])],
        "cyclical": [f for f in all_features if any(x in f for x in ["sin", "cos"])],
        "lags": [f for f in all_features if any(x in f for x in ["_lag_", "_rolling_"])],
        "interactions": [f for f in all_features if "interaction" in f],
    }

    existing_groups = {g: [f for f in feats if f in all_features] for g, feats in feature_groups.items()}
    existing_groups = {k: v for k, v in existing_groups.items() if v}

    if not existing_groups:
        # fallback — take first target_count features
        logger.warning("No grouped features matched; falling back to first features")
        return all_features[:target_count]

    total_available = sum(len(features) for features in existing_groups.values())
    if total_available == 0:
        logger.warning("Total available features in groups is zero, fallback to basic list")
        return all_features[:target_count]

    group_quotas = {}
    for group_name, features in existing_groups.items():
        group_ratio = len(features) / total_available
        group_quotas[group_name] = max(1, int(target_count * group_ratio))

    total_allocated = sum(group_quotas.values())
    if total_allocated != target_count:
        difference = target_count - total_allocated

        # order groups by size descending
        sorted_groups = sorted(existing_groups.items(), key=lambda x: len(x[1]), reverse=True)
        i = 0
        while difference != 0:
            group_name = sorted_groups[i % len(sorted_groups)][0]
            if difference > 0:
                group_quotas[group_name] += 1
                difference -= 1
            else:
                if group_quotas[group_name] > 1:
                    group_quotas[group_name] -= 1
                    difference += 1
            i += 1

    def extract_param(feature: str) -> int:
        try:
            if "_lag_" in feature:
                return int(feature.split("_lag_")[-1])
            if "_rolling_" in feature:
                parts = feature.split("_rolling_")[-1].split("_")
                for part in parts:
                    if part.isdigit():
                        return int(part)
                return 0
            if "EMA_" in feature:
                return int(feature.split("EMA_")[-1])
            if "volatility_" in feature:
                return int(feature.split("volatility_")[-1])
        except Exception:
            return 0
        return 0

    selected_features: List[str] = []

    for group_name, quota in group_quotas.items():
        group_features = existing_groups[group_name]
        if len(group_features) <= quota:
            selected_features.extend(group_features)
            continue

        # pick representative features
        if group_name in {"price_basic", "rsi", "macd", "bollinger", "atr"}:
            selected_features.extend(group_features[:quota])
            continue

        # for lag-like groups
        if any(x in group_features[0] for x in ["_lag_", "_rolling_", "EMA_", "volatility_"]):
            try:
                sorted_features = sorted(group_features, key=extract_param)
                step = max(1, len(sorted_features) // quota)
                indices = [min(i * step, len(sorted_features) - 1) for i in range(quota)]
                selected_features.extend([sorted_features[i] for i in indices])
            except Exception as e:
                logger.debug("Selection fallback for group %s: %s", group_name, e)
                selected_features.extend(group_features[:quota])
        else:
            step = max(1, len(group_features) // quota)
            indices = [min(i * step, len(group_features) - 1) for i in range(quota)]
            selected_features.extend([group_features[i] for i in indices])

    # keep order and unique
    selected_features = list(dict.fromkeys(selected_features))

    # adjust length
    if len(selected_features) > target_count:
        selected_features = selected_features[:target_count]
    elif len(selected_features) < target_count:
        remaining = [f for f in all_features if f not in selected_features]
        needed = target_count - len(selected_features)
        if remaining:
            selected_features.extend(remaining[:needed])

    logger.info("Selected %d features from %d available", len(selected_features), len(all_features))
    return selected_features


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).float()


def get_dataloaders(
    ticker: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seq_length: int = 20,
    lookback_days: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_scalers: bool = False,
) -> Tuple:
    """Build train/val dataloaders for single-horizon forecasting.

    Returns (train_dl, val_dl) or (train_dl, val_dl, scaler_X, scaler_y) if return_scalers=True
    """
    end_dt = datetime.fromisoformat(end_date) if end_date else datetime.now()
    start_dt = datetime.fromisoformat(start_date) if start_date else end_dt - timedelta(days=lookback_days)
    s_str, e_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", s_str, e_str)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", s_str, e_str)
    df_u = fetch_usd_series(s_str, e_str)

    def _prep(df: pd.DataFrame, ren: dict) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df.get("TRADEDATE")))
        df["TRADEDATE"] = df["TRADEDATE"].dt.normalize()
        return df.rename(columns=ren)

    df_t = _prep(df_t, {"OPEN": f"OPEN_{ticker}", "HIGH": f"HIGH_{ticker}", "LOW": f"LOW_{ticker}", "CLOSE": f"CLOSE_{ticker}", "VOLUME": f"VOL_{ticker}"})
    df_i = _prep(df_i, {"CLOSE": "CLOSE_IMOEX"})
    df_u = _prep(df_u, {"CLOSE": "CLOSE_USD"})

    if df_t.empty:
        raise ValueError(f"No data returned for ticker {ticker} in range {s_str}..{e_str}")

    merged = (
        df_t.merge(df_i[["TRADEDATE", "CLOSE_IMOEX"]], on="TRADEDATE", how="outer")
        .merge(df_u[["TRADEDATE", "CLOSE_USD"]], on="TRADEDATE", how="outer")
        .sort_values("TRADEDATE")
        .ffill()
        .bfill()
        .dropna()
        .reset_index(drop=True)
    )

    proc = preprocess_data(merged, ticker)
    features = get_smart_features(proc, ticker, target_count=132)

    data = proc[features].values.astype(float)

    if data.shape[0] <= seq_length:
        raise ValueError(f"Insufficient points: {data.shape[0]} for seq_length={seq_length}")

    idx = features.index(f"CLOSE_{ticker}")

    X = np.lib.stride_tricks.sliding_window_view(data, (seq_length, data.shape[1]))[:, 0, :, :]
    X = X[:-1]
    y = data[seq_length:, idx].reshape(-1, 1)

    # Train/val split
    n = len(X)
    split_idx = int(0.8 * n)
    train_X = X[:split_idx]
    train_y = y[:split_idx]
    val_X = X[split_idx:]
    val_y = y[split_idx:]

    # Fit scalers on train data
    flat_train_X = train_X.reshape(-1, train_X.shape[-1])
    scaler_X = RobustScaler().fit(flat_train_X)
    scaler_y = RobustScaler().fit(train_y)

    train_Xs = scaler_X.transform(flat_train_X).reshape(train_X.shape)
    train_ys = scaler_y.transform(train_y)

    # Transform val
    flat_val_X = val_X.reshape(-1, val_X.shape[-1])
    val_Xs = scaler_X.transform(flat_val_X).reshape(val_X.shape)
    val_ys = scaler_y.transform(val_y)

    train_ds = TimeSeriesDataset(train_Xs, train_ys)
    val_ds = TimeSeriesDataset(val_Xs, val_ys)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if return_scalers:
        return train_dl, val_dl, scaler_X, scaler_y
    return train_dl, val_dl


# -----------------------------
# Multi-horizon loader
# -----------------------------
def get_dataloaders_multi(
    ticker: str,
    seq_length: int,
    horizon: int,
    batch_size: int,
    lookback_days: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_scalers: bool = False,
    num_workers: int = 0,
) -> Tuple:
    """
    Multi-horizon dataloaders.
    Returns (train_dl, val_dl) or (train_dl, val_dl, scaler_X, scaler_y) if return_scalers=True.
    """
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.fromisoformat(end_date)
    if start_date is None:
        start_dt = end_dt - timedelta(days=lookback_days)
    else:
        start_dt = datetime.fromisoformat(start_date)
    s_str, e_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", s_str, e_str)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", s_str, e_str)
    df_u = fetch_usd_series(s_str, e_str)

    def prep(df, ren):
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df.get("TRADEDATE")))
        df["TRADEDATE"] = df["TRADEDATE"].dt.normalize()
        return df.rename(columns=ren)

    df_t = prep(
        df_t,
        {
            "OPEN": f"OPEN_{ticker}",
            "HIGH": f"HIGH_{ticker}",
            "LOW": f"LOW_{ticker}",
            "CLOSE": f"CLOSE_{ticker}",
            "VOLUME": f"VOL_{ticker}",
        },
    )
    df_t = df_t[
        [
            "TRADEDATE",
            f"OPEN_{ticker}",
            f"HIGH_{ticker}",
            f"LOW_{ticker}",
            f"CLOSE_{ticker}",
            f"VOL_{ticker}",
        ]
    ] if not df_t.empty else df_t
    df_i = prep(df_i, {"CLOSE": "CLOSE_IMOEX"})[["TRADEDATE", "CLOSE_IMOEX"]] if not df_i.empty else df_i
    df_u = prep(df_u, {"CLOSE": "CLOSE_USD"})[["TRADEDATE", "CLOSE_USD"]] if not df_u.empty else df_u

    if df_t.empty:
        raise ValueError(f"No data for ticker {ticker} in {s_str}..{e_str}")

    merged = (
        df_t.merge(df_i, on="TRADEDATE", how="outer")
        .merge(df_u, on="TRADEDATE", how="outer")
        .sort_values("TRADEDATE")
        .ffill()
        .bfill()
        .dropna()
        .reset_index(drop=True)
    )

    proc = preprocess_data(merged, ticker)
    features = get_smart_features(proc, ticker, target_count=132)

    data = proc[features].values.astype(float)
    close_idx = features.index(f"CLOSE_{ticker}")

    n = len(data)
    max_i = n - seq_length - horizon + 1
    if max_i <= 0:
        raise ValueError(
            f"Недостаточно точек ({n}) для seq_length={seq_length} + horizon={horizon}"
        )
    X_list, Y_list = [], []
    for i in range(max_i):
        X_list.append(data[i : i + seq_length])
        Y_list.append(data[i + seq_length : i + seq_length + horizon, close_idx])
    X = np.array(X_list)
    Y = np.array(Y_list)

    X_tensor = X 
    y_tensor = Y
    
    # Train/val split (time-ordered)
    split_idx = int(0.8 * len(X_tensor))
    train_X = X_tensor[:split_idx]
    train_y = y_tensor[:split_idx]
    val_X = X_tensor[split_idx:]
    val_y = y_tensor[split_idx:]

    flat_X = train_X.reshape(-1, train_X.shape[-1])
    scaler_X = RobustScaler().fit(flat_X)
    scaler_y = RobustScaler().fit(train_y)

    Xs_train = scaler_X.transform(flat_X).reshape(train_X.shape)
    ys_train = scaler_y.transform(train_y)

    val_flat = val_X.reshape(-1, val_X.shape[-1])
    Xs_val = scaler_X.transform(val_flat).reshape(val_X.shape)
    ys_val = scaler_y.transform(val_y)

    train_ds = TimeSeriesDataset(Xs_train, ys_train)
    val_ds = TimeSeriesDataset(Xs_val, ys_val)

    tr_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if return_scalers:
        return tr_dl, va_dl, scaler_X, scaler_y
    return tr_dl, va_dl
