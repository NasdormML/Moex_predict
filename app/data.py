import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from functools import lru_cache
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# функция кэша
def _cache_path(ticker: str, start: str, end: str, context: str = "data") -> Path:
    return CACHE_DIR / f"{context}_{ticker}_{start}_{end}.pkl"

def load_cached_data(ticker: str, start: str, end: str, context: str = "data") -> pd.DataFrame:
    p = _cache_path(ticker, start, end, context)
    if not p.exists():
        return None
    if datetime.now() - datetime.fromtimestamp(p.stat().st_mtime) > timedelta(days=EXPIRATION_DAYS):
        return None
    return pd.read_pickle(p)


def save_cached_data(ticker: str, start: str, end: str, df: pd.DataFrame, context: str = "data"):
    p = _cache_path(ticker, start, end, context)
    df.to_pickle(p)

@lru_cache(maxsize=500)
def fetch_cbr_usd_rate(date_str: str) -> float:
    url = f"http://www.cbr.ru/scripts/XML_daily.asp?date_req={date_str}"
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    for v in root.findall('Valute'):
        if v.find('CharCode').text == 'USD':
            return float(v.find('Value').text.replace(',', '.'))
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
    context: str = "data", ) -> pd.DataFrame:

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
    context: str = "data", ) -> pd.DataFrame:

    ticker = "USD000UTSTOM"
    if not skip_cache:
        df_cached = load_cached_data(ticker, start, end, context)
        if df_cached is not None:
            return df_cached

    dates = pd.date_range(start, end)
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_date = {executor.submit(fetch_cbr_usd_rate, d.strftime("%d/%m/%Y")): d for d in dates}
        for future in as_completed(future_to_date):
            d = future_to_date[future]
            rate = future.result()
            records.append((d, rate))

    df = pd.DataFrame(records, columns=["TRADEDATE", "CLOSE"])  
    df.sort_values("TRADEDATE", inplace=True)
    if not skip_cache:
        save_cached_data(ticker, start, end, df, context)
    return df

from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(
    ticker: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seq_len: int = 20,
    lookback_days: int = 365,
    start_date: str = None,
    end_date: str = None,
    return_scalers: bool = False, ):

    end_dt = datetime.fromisoformat(end_date) if end_date else datetime.now()
    start_dt = datetime.fromisoformat(start_date) if start_date else end_dt - timedelta(days=lookback_days)
    s_str, e_str = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    # выгрузка чере параллель
    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", s_str, e_str)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", s_str, e_str)
    df_u = fetch_usd_series(s_str, e_str)

    # предобработка и обьединение
    def prep(df, ren):
        df = df.copy()
        df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df["TRADEDATE"]))
        df["TRADEDATE"] = df["TRADEDATE"].dt.normalize()
        return df.rename(columns=ren)

    df_t = prep(df_t, {"OPEN":f"OPEN_{ticker}", "HIGH":f"HIGH_{ticker}", "LOW":f"LOW_{ticker}",
                       "CLOSE":f"CLOSE_{ticker}", "VOLUME":f"VOL_{ticker}"})
    df_i = prep(df_i, {"CLOSE":"CLOSE_IMOEX"})
    df_u = prep(df_u, {"CLOSE":"CLOSE_USD"})

    merged = (
        df_t[["TRADEDATE"] + [c for c in df_t.columns if c.startswith(("OPEN_","HIGH_","LOW_","CLOSE_","VOL_"))]]
        .merge(df_i[["TRADEDATE","CLOSE_IMOEX"]], on="TRADEDATE", how="outer")
        .merge(df_u[["TRADEDATE","CLOSE_USD"]], on="TRADEDATE", how="outer")
        .sort_values("TRADEDATE").ffill().bfill().dropna().reset_index(drop=True)
    )
    from app.preprocessing import preprocess_data
    proc = preprocess_data(merged, ticker)

    features = [
        f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
        "CLOSE_IMOEX", "CLOSE_USD",
        "RSI","SMA_RETURNS","VOLATILITY","LOG_RETURNS",
        "MACD_LINE","MACD_SIGNAL","MACD_HIST",
        "BB_UPPER","BB_LOWER","BB_MIDDLE","ATR"
    ]
    data = proc[features].values.astype(float)

    # создание последовательности через скользящее окно numpy
    idx = features.index(f"CLOSE_{ticker}")
    windows = np.lib.stride_tricks.sliding_window_view(data, (seq_len, data.shape[1]))[:,0,:,:]
    # дроп последнего окна для состыковки длины
    X = windows[:-1]
    y = data[seq_len:, idx].reshape(-1,1)

    # scale
    X_flat = X.reshape(-1, X.shape[2])
    scaler_X = RobustScaler().fit(X_flat)
    Xs = scaler_X.transform(X_flat).reshape(X.shape)
    scaler_y = RobustScaler().fit(y)
    ys = scaler_y.transform(y)

    tensor_x = torch.tensor(Xs, dtype=torch.float32)
    tensor_y = torch.tensor(ys, dtype=torch.float32)
    ds = TensorDataset(tensor_x, tensor_y)
    train_size = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, len(ds) - train_size])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (train_dl, val_dl, scaler_X, scaler_y) if return_scalers else (train_dl, val_dl)
