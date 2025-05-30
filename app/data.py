import requests
import pandas as pd
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime, timedelta

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

def get_with_retries(url, params, timeout=30, retries=3):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print("Ошибка запроса:", e)
        return None

def fetch_moex_eod_data(security, engine, market, board, start_date, end_date):
    base_url = (
        f"https://iss.moex.com/iss/history/engines/{engine}/markets/{market}/"
        f"boards/{board}/securities/{security}.json"
    )
    all_data, columns, offset, limit = [], None, 0, 100
    while True:
        params = {"from": start_date, "till": end_date, "start": offset, "limit": limit}
        resp = get_with_retries(base_url, params)
        if not resp:
            break
        data = resp.json()
        if columns is None:
            columns = data.get("history", {}).get("columns")
        page = data.get("history", {}).get("data", [])
        if not page:
            break
        all_data.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return pd.DataFrame(all_data, columns=columns) if all_data else pd.DataFrame()

def fetch_cbr_usd_rate(date_obj: datetime) -> float:
    date_str = date_obj.strftime("%d/%m/%Y")
    url = f"http://www.cbr.ru/scripts/XML_daily.asp?date_req={date_str}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for v in root.findall('Valute'):
            if v.find('CharCode').text == 'USD':
                return float(v.find('Value').text.replace(',', '.'))
    except Exception:
        pass
    return None

def get_dataloaders(
    ticker: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    return_scalers: bool = False,
    seq_len: int = 20,
    lookback_days: int = 365,
    start_date: str = None,
    end_date: str = None,
):
    # Определяем даты
    if end_date:
        end_dt = datetime.fromisoformat(end_date)
    else:
        end_dt = datetime.now()
    if start_date:
        start_dt = datetime.fromisoformat(start_date)
    else:
        start_dt = end_dt - timedelta(days=lookback_days)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")
    # Тикер
    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start_str, end_str)
    # Индекс IMOEX
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start_str, end_str)
    # USD
    df_u = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", start_str, end_str)
    if df_u is None or df_u.empty:
        dates = pd.date_range(start_str, end_str)
        df_u = pd.DataFrame({
            "TRADEDATE": dates,
            "CLOSE": [fetch_cbr_usd_rate(d) for d in dates]
        })

    # Переименование и нормализация
    for df, ren in [
        (df_t, {"OPEN":f"OPEN_{ticker}", "HIGH":f"HIGH_{ticker}", "LOW":f"LOW_{ticker}", 
                "CLOSE":f"CLOSE_{ticker}", "VOLUME":f"VOL_{ticker}"}),
        (df_i, {"CLOSE":"CLOSE_IMOEX"}),
        (df_u, {"CLOSE":"CLOSE_USD"})
    ]:
        df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df["TRADEDATE"])).dt.normalize()
        df.rename(columns=ren, inplace=True)

    merged = (
        df_t[["TRADEDATE", f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}"]]
        .merge(df_i[["TRADEDATE", "CLOSE_IMOEX"]], on="TRADEDATE", how="outer")
        .merge(df_u[["TRADEDATE", "CLOSE_USD"]], on="TRADEDATE", how="outer")
        .sort_values("TRADEDATE")
        .ffill().bfill().dropna()
        .reset_index(drop=True)
    )

    # Предобработка
    from app.preprocessing import preprocess_data
    proc = preprocess_data(merged, ticker)

    # 5) Формирование X и y
    features = [
        f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
        "CLOSE_IMOEX", "CLOSE_USD",
        "RSI", "SMA_RETURNS", "VOLATILITY", "LOG_RETURNS",
        "MACD_LINE", "MACD_SIGNAL", "MACD_HIST",
        "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
        "ATR"
    ]
    data = proc[features].values.astype(float)

    if len(data) < seq_len + 1:
        raise RuntimeError(f"Недостаточно данных ({len(data)}) для seq_len={seq_len}")

    X, y = [], []
    idx_close = features.index(f"CLOSE_{ticker}")
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][idx_close])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Масштабирование
    scaler_X = RobustScaler().fit(X.reshape(-1, X.shape[2]))
    scaler_y = RobustScaler().fit(y)
    Xs = scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    ys = scaler_y.transform(y)

    # DataLoader
    tensor_x = torch.tensor(Xs, dtype=torch.float32)
    tensor_y = torch.tensor(ys, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    n = len(dataset)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [int(0.8*n), n-int(0.8*n)])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if return_scalers:
        return train_dl, val_dl, scaler_X, scaler_y
    return train_dl, val_dl
