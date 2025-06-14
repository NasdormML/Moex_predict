import os
import pickle
from datetime import datetime, timedelta
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from app.data import fetch_cbr_usd_rate, fetch_moex_eod_data
from app.preprocessing import preprocess_data


def get_metadata_path(version: str = "v1"):
    metadata_dir = os.path.join("history", "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    return os.path.join(metadata_dir, f"training_metadata_{version}.pkl")


def load_training_metadata(version: str = None) -> dict:
    # если version не указан, берём из ENV или по-умолчанию v1
    version = version or os.getenv("MODEL_VERSION", "v1")
    path = get_metadata_path(version)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_training_metadata(metadata: dict, version: str = None):
    version = version or os.getenv("MODEL_VERSION", "v1")
    path = get_metadata_path(version)
    with open(path, "wb") as f:
        pickle.dump(metadata, f)


# Внутри retrain_model считываем factory_key и params из metadata
def retrain_model(
    ticker: str,
    last_train_datetime: datetime,
    new_end_datetime: datetime,
    current_model_bundle: dict,
    cfg: Optional[DictConfig] = None,
    window_days: int = 180,
    default_lr: float = 1e-7,
    default_epochs: int = 45,
    hpo_trials: int = 0,
) -> dict:
    # определяем версию метаданных
    version = cfg.train.version if cfg is not None else None
    metadata = load_training_metadata(version)

    # вытаскиваем из metadata или из текущего бандла
    factory_key = metadata.get(
        f"{ticker}_factory_key", current_model_bundle.get("factory_key")
    )
    model_params = metadata.get(
        f"{ticker}_model_params", current_model_bundle.get("model_params")
    )

    # проверяем, нужно ли дообучение
    last_str = metadata.get(ticker)
    if last_str:
        last_date = datetime.strptime(last_str, "%Y-%m-%d").date()
        if new_end_datetime.date() <= last_date:
            return current_model_bundle

    # --- подготовка данных ---
    seq_length = model_params.get("seq_length", 20)
    available_days = (new_end_datetime.date() - last_train_datetime.date()).days
    window_days = min(window_days, max(available_days, seq_length + 1))
    start_date = (new_end_datetime - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_date = new_end_datetime.strftime("%Y-%m-%d")

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start_date, end_date)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start_date, end_date)
    df_u = fetch_moex_eod_data(
        "USD000UTSTOM", "currency", "selt", "CETS", start_date, end_date
    )
    if df_u is None or df_u.empty:
        dates = pd.date_range(start_date, end_date)
        df_u = pd.DataFrame(
            {"TRADEDATE": dates, "CLOSE": [fetch_cbr_usd_rate(d) for d in dates]}
        )

    # нормализация и rename
    def prep(df, ren):
        df["TRADEDATE"] = pd.to_datetime(
            df.get("BEGIN", df["TRADEDATE"])
        ).dt.normalize()
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
        df_t[
            [
                "TRADEDATE",
                f"OPEN_{ticker}",
                f"HIGH_{ticker}",
                f"LOW_{ticker}",
                f"CLOSE_{ticker}",
                f"VOL_{ticker}",
            ]
        ]
        .merge(df_i[["TRADEDATE", "CLOSE_IMOEX"]], on="TRADEDATE")
        .merge(df_u[["TRADEDATE", "CLOSE_USD"]], on="TRADEDATE")
        .sort_values("TRADEDATE")
        .ffill()
        .bfill()
        .dropna()
        .reset_index(drop=True)
    )
    df_proc = preprocess_data(merged, ticker)

    # формируем X, y
    features = [
        col
        for col in df_proc.columns
        if col.startswith((f"OPEN_{ticker}", f"CLOSE_{ticker}"))
    ]
    data = df_proc[features].values.astype(float)
    X_list, y_list = [], []
    close_idx = features.index(f"CLOSE_{ticker}")
    for i in range(len(data) - seq_length):
        X_list.append(data[i : i + seq_length])
        y_list.append(data[i + seq_length][close_idx])
    X_arr = np.array(X_list)
    y_arr = np.array(y_list).reshape(-1, 1)

    # масштабирование
    scaler_X = current_model_bundle["scaler_X"]
    scaler_y = current_model_bundle["scaler_y"]
    X_scaled = scaler_X.transform(X_arr.reshape(-1, X_arr.shape[2])).reshape(
        X_arr.shape
    )
    y_scaled = scaler_y.transform(y_arr)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # обучение только fc-слоёв
    model = current_model_bundle["model"]
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=default_lr
    )
    loss_fn = nn.HuberLoss()
    model.train()
    for e in range(default_epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        mlflow.log_metric("retraining_loss", loss.item(), step=e)

    # сохраняем новую версию
    maj, min_ = version.lstrip("v").split(".")
    new_ver = f"v{maj}.{int(min_)+1}"
    artifact_root = os.getenv("MODEL_ARTIFACTS_DIR", "saved_models")
    out_dir = os.path.join(artifact_root, new_ver)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{ticker}_model.pth"))
    pickle.dump(scaler_X, open(os.path.join(out_dir, f"{ticker}_scaler_X.pkl"), "wb"))
    pickle.dump(scaler_y, open(os.path.join(out_dir, f"{ticker}_scaler_y.pkl"), "wb"))

    # обновляем метаданные
    metadata[ticker] = new_end_datetime.strftime("%Y-%m-%d")
    metadata[f"{ticker}_model_version"] = new_ver
    metadata[f"{ticker}_factory_key"] = factory_key
    metadata[f"{ticker}_model_params"] = model_params
    save_training_metadata(metadata, new_ver)

    # возвращаем bundle
    return {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "seq_length": seq_length,
        "factory_key": factory_key,
        "model_params": model_params,
        "model_version": new_ver,
    }
