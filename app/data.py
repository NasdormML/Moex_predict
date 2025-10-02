import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from app.preprocessing import preprocess_data

# настройка каталога для кэша
CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "data_cache"))
EXPIRATION_DAYS = int(os.getenv("DATA_CACHE_EXP_DAYS", "1"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# общий HTTP-сеанс с повтором
session = requests.Session()
retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)


def _cache_path(ticker: str, start: str, end: str, context: str = "data") -> Path:
    return CACHE_DIR / f"{context}_{ticker}_{start}_{end}.pkl"


def load_cached_data(
    ticker: str, start: str, end: str, context: str = "data"
) -> Optional[pd.DataFrame]:
    p = _cache_path(ticker, start, end, context)
    if not p.exists():
        return None
    if datetime.now() - datetime.fromtimestamp(p.stat().st_mtime) > timedelta(
        days=EXPIRATION_DAYS
    ):
        return None
    return pd.read_pickle(p)


def save_cached_data(
    ticker: str, start: str, end: str, df: pd.DataFrame, context: str = "data"
):
    p = _cache_path(ticker, start, end, context)
    df.to_pickle(p)


@lru_cache(maxsize=500)
def fetch_cbr_usd_rate(date_str: str) -> float:
    url = f"http://www.cbr.ru/scripts/XML_daily.asp?date_req={date_str}"
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    for v in root.findall("Valute"):
        if v.find("CharCode").text == "USD":
            return float(v.find("Value").text.replace(",", "."))
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
        hist = resp.json().get("history", {})
        data = hist.get("data", [])
        if not data:
            break
        all_data.extend(data)
        if len(data) < params["limit"]:
            break
        params["start"] += params["limit"]

    df = pd.DataFrame(all_data, columns=hist.get("columns", []))
    if not skip_cache and not df.empty:
        save_cached_data(security, start, end, df, context)
    return df


def fetch_usd_series(
    start: str,
    end: str,
    *,
    skip_cache: bool = False,
    context: str = "data",
) -> pd.DataFrame:
    ticker = "USD000UTSTOM"
    if not skip_cache:
        df_cached = load_cached_data(ticker, start, end, context)
        if df_cached is not None:
            return df_cached

    dates = pd.date_range(start, end)
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_date = {
            executor.submit(fetch_cbr_usd_rate, d.strftime("%d/%m/%Y")): d
            for d in dates
        }
        for future in as_completed(future_to_date):
            d = future_to_date[future]
            rate = future.result()
            records.append((d, rate))

    df = pd.DataFrame(records, columns=["TRADEDATE", "CLOSE"])
    df.sort_values("TRADEDATE", inplace=True)
    if not skip_cache:
        save_cached_data(ticker, start, end, df, context)
    return df


def get_smart_features(
    proc: pd.DataFrame, ticker: str, target_count: int = None
) -> list:
    excluded_cols = {"TRADEDATE", "target_close", "date"}
    all_features = [col for col in proc.columns if col not in excluded_cols]

    print(f"Всего доступно признаков: {len(all_features)}")

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
            if any(
                x in f for x in ["skew", "kurt", "zscore", "q25", "q75", "iqr", "mad"]
            )
        ],
        "trend": [
            f
            for f in all_features
            if any(x in f for x in ["trend", "ratio", "position"])
        ],
        "cyclical": [f for f in all_features if any(x in f for x in ["sin", "cos"])],
        "lags": [
            f for f in all_features if any(x in f for x in ["_lag_", "_rolling_"])
        ],
        "interactions": [f for f in all_features if "interaction" in f],
    }

    existing_groups = {}
    for group_name, patterns in feature_groups.items():
        existing_features = [f for f in patterns if f in all_features]
        if existing_features:
            existing_groups[group_name] = existing_features
            print(f"Группа {group_name}: {len(existing_features)} признаков")

    total_available = sum(len(features) for features in existing_groups.values())
    group_quotas = {}

    for group_name, features in existing_groups.items():
        group_ratio = len(features) / total_available
        group_quotas[group_name] = max(1, int(target_count * group_ratio))

    total_allocated = sum(group_quotas.values())
    if total_allocated != target_count:
        difference = target_count - total_allocated
        sorted_groups = sorted(
            group_quotas.items(), key=lambda x: len(existing_groups[x[0]]), reverse=True
        )

        for i in range(abs(difference)):
            group_name, quota = sorted_groups[i % len(sorted_groups)]
            if difference > 0:
                group_quotas[group_name] += 1
            else:
                group_quotas[group_name] = max(1, group_quotas[group_name] - 1)

    def extract_param(feature):
        try:
            if "_lag_" in feature:
                return int(feature.split("_lag_")[-1])
            elif "_rolling_" in feature:
                parts = feature.split("_rolling_")[-1].split("_")
                for part in parts:
                    if part.isdigit():
                        return int(part)
                return 0
            elif "EMA_" in feature:
                return int(feature.split("EMA_")[-1])
            elif "volatility_" in feature:
                return int(feature.split("volatility_")[-1])
            else:
                return 0
        except (ValueError, IndexError):
            return 0

    selected_features = []

    for group_name, quota in group_quotas.items():
        group_features = existing_groups[group_name]

        if len(group_features) <= quota:
            selected_features.extend(group_features)
        else:
            if group_name in ["price_basic", "rsi", "macd", "bollinger", "atr"]:
                selected_features.extend(group_features[:quota])
            else:
                if any(
                    x in group_features[0]
                    for x in ["_lag_", "_rolling_", "EMA_", "volatility_"]
                ):
                    try:
                        sorted_features = sorted(group_features, key=extract_param)
                        step = max(1, len(sorted_features) // quota)
                        indices = [i * step for i in range(quota)]
                        indices = [min(i, len(sorted_features) - 1) for i in indices]
                        selected_features.extend([sorted_features[i] for i in indices])
                    except Exception as e:
                        print(f"Ошибка при сортировке группы {group_name}: {e}")
                        selected_features.extend(group_features[:quota])
                else:
                    step = max(1, len(group_features) // quota)
                    indices = [i * step for i in range(quota)]
                    indices = [min(i, len(group_features) - 1) for i in indices]
                    selected_features.extend([group_features[i] for i in indices])

    selected_features = list(dict.fromkeys(selected_features))

    if len(selected_features) > target_count:
        selected_features = selected_features[:target_count]
    elif len(selected_features) < target_count:
        remaining = [f for f in all_features if f not in selected_features]
        needed = target_count - len(selected_features)
        if remaining:
            selected_features.extend(remaining[:needed])

    print(f"{len(selected_features)} features from {len(all_features)}")
    return selected_features


def get_dataloaders(
    ticker: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seq_length: int = 20,
    lookback_days: int = 365,
    start_date: str = None,
    end_date: str = None,
    return_scalers: bool = False,
):
    end_dt = datetime.fromisoformat(end_date) if end_date else datetime.now()
    start_dt = (
        datetime.fromisoformat(start_date)
        if start_date
        else end_dt - timedelta(days=lookback_days)
    )
    s_str, e_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", s_str, e_str)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", s_str, e_str)
    df_u = fetch_usd_series(s_str, e_str)

    def prep(df, ren):
        df = df.copy()
        df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df["TRADEDATE"]))
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
    df_i = prep(df_i, {"CLOSE": "CLOSE_IMOEX"})
    df_u = prep(df_u, {"CLOSE": "CLOSE_USD"})

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

    idx = features.index(f"CLOSE_{ticker}")
    windows = np.lib.stride_tricks.sliding_window_view(
        data, (seq_length, data.shape[1])
    )[:, 0, :, :]
    X = windows[:-1]
    y = data[seq_length:, idx].reshape(-1, 1)

    X_flat = X.reshape(-1, X.shape[2])
    scaler_X = RobustScaler().fit(X_flat)
    Xs = scaler_X.transform(X_flat).reshape(X.shape)
    scaler_y = RobustScaler().fit(y)
    ys = scaler_y.transform(y)

    tensor_x = torch.tensor(Xs, dtype=torch.float32)
    tensor_y = torch.tensor(ys, dtype=torch.float32)
    ds = TensorDataset(tensor_x, tensor_y)
    train_ds, val_ds = random_split(
        ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))]
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return (
        (train_dl, val_dl, scaler_X, scaler_y) if return_scalers else (train_dl, val_dl)
    )


def get_dataloaders_multi(
    ticker: str,
    seq_length: int,
    horizon: int,
    batch_size: int,
    lookback_days: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_scalers: bool = False,
):
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
        df = df.copy()
        df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df["TRADEDATE"]))
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
    ]
    df_i = prep(df_i, {"CLOSE": "CLOSE_IMOEX"})[["TRADEDATE", "CLOSE_IMOEX"]]
    df_u = prep(df_u, {"CLOSE": "CLOSE_USD"})[["TRADEDATE", "CLOSE_USD"]]

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

    flat_X = X.reshape(-1, X.shape[-1])
    scaler_X = RobustScaler().fit(flat_X)
    Xs = scaler_X.transform(flat_X).reshape(X.shape)
    scaler_y = RobustScaler().fit(Y)
    Ys = scaler_y.transform(Y)

    tensor_x = torch.tensor(Xs, dtype=torch.float32)
    tensor_y = torch.tensor(Ys, dtype=torch.float32)
    ds = TensorDataset(tensor_x, tensor_y)
    tr_n = int(0.8 * len(ds))
    tr_ds, va_ds = random_split(ds, [tr_n, len(ds) - tr_n])
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    return (tr_dl, va_dl, scaler_X, scaler_y) if return_scalers else (tr_dl, va_dl)
