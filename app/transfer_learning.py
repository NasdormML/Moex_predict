import os
import pickle
import copy
from datetime import datetime, timedelta

import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna

from app.models import PricePredictionModel, TCNModel
from app.data import fetch_moex_eod_data, fetch_cbr_usd_rate
from app.preprocessing import preprocess_data


def get_metadata_path(version: str = "v1"):
    metadata_dir = os.path.join("history", "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    return os.path.join(metadata_dir, f"training_metadata_{version}.pkl")


def load_training_metadata(version: str = "v1") -> dict:
    path = get_metadata_path(version)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_training_metadata(metadata: dict, version: str = "v1"):
    path = get_metadata_path(version)
    with open(path, "wb") as f:
        pickle.dump(metadata, f)


def process_tradedate(df: pd.DataFrame) -> pd.DataFrame:
    if "BEGIN" in df.columns:
        df["TRADEDATE"] = pd.to_datetime(df["BEGIN"])
    elif "TRADETIME" in df.columns:
        df["TRADEDATE"] = pd.to_datetime(df["TRADETIME"])
    elif "TRADEDATE" in df.columns:
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    else:
        raise ValueError("Не найден столбец с датой")
    df["TRADEDATE"] = df["TRADEDATE"].dt.normalize()
    return df


def tune_hyperparams(X_tensor, y_tensor, model_template, frozen_prefix="fc", n_trials=20):
    """
    Запускает HPO с Optuna и возвращает лучшие параметры,
    или {} если ни одна конфигурация не дала валидных результатов.
    """
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-8, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
        epochs = trial.suggest_int("epochs", 10, 30)

        # создаём копию модели и замораживаем все слои кроме fc
        model = copy.deepcopy(model_template)
        for name, param in model.named_parameters():
            param.requires_grad = (frozen_prefix in name)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.HuberLoss()

        # разбиваем на train/val внутри HPO
        n_samples = X_tensor.size(0)
        split = max(int(0.8 * n_samples), 1)
        X_tr, X_val = X_tensor[:split], X_tensor[split:]
        y_tr, y_val = y_tensor[:split], y_tensor[split:]

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_tr)
            loss = loss_fn(preds, y_tr)
            # если loss nan — прерываем trial
            if not torch.isfinite(loss):
                raise optuna.TrialPruned()
            loss.backward()
            optimizer.step()
            trial.report(loss.item(), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = loss_fn(val_preds, y_val)
        return val_loss.item()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials)
    try:
        return study.best_params
    except (ValueError, KeyError):
        # ни один trial не завершился успешно
        return {}


def retrain_model(
    ticker: str,
    last_train_datetime: datetime,
    new_end_datetime: datetime,
    current_model_bundle: dict,
    window_days: int = 180,
    default_lr: float = 1e-7,
    default_epochs: int = 45,
    hpo_trials: int = 20
) -> dict:
    metadata = load_training_metadata()
    last_str = metadata.get(ticker)
    if last_str:
        last_date = datetime.strptime(last_str, "%Y-%m-%d").date()
        if new_end_datetime.date() <= last_date:
            return current_model_bundle

    available_days = (new_end_datetime.date() - last_train_datetime.date()).days
    seq_length = current_model_bundle.get("seq_length", 20)
    min_needed = seq_length + 1
    window_days = min(window_days, max(available_days, min_needed))
    start_date = (new_end_datetime - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_date = new_end_datetime.strftime("%Y-%m-%d")

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start_date, end_date)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start_date, end_date)
    df_u = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", start_date, end_date)
    if df_u is None or df_u.empty or 'CLOSE' not in df_u.columns:
        dates = pd.date_range(start_date, end_date)
        rates = [fetch_cbr_usd_rate(d) for d in dates]
        df_u = pd.DataFrame({"TRADEDATE": dates, "CLOSE": rates})

    df_t = process_tradedate(df_t)
    df_i = process_tradedate(df_i)
    df_u = process_tradedate(df_u)

    df_t.rename(columns={
        "OPEN": f"OPEN_{ticker}", "HIGH": f"HIGH_{ticker}",
        "LOW": f"LOW_{ticker}",  "CLOSE": f"CLOSE_{ticker}",
        "VOLUME": f"VOL_{ticker}"}, inplace=True)
    df_i.rename(columns={"CLOSE": "CLOSE_IMOEX"}, inplace=True)
    df_u.rename(columns={"CLOSE": "CLOSE_USD"}, inplace=True)

    merged = (
        df_t[["TRADEDATE", f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}"]]
        .merge(df_i[["TRADEDATE", "CLOSE_IMOEX"]], on="TRADEDATE")
        .merge(df_u[["TRADEDATE", "CLOSE_USD"]], on="TRADEDATE")
        .sort_values("TRADEDATE").ffill().bfill().dropna().reset_index(drop=True)
    )
    df_proc = preprocess_data(merged, ticker)
    features = [
        f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
        "CLOSE_IMOEX", "CLOSE_USD",
        "RSI", "SMA_RETURNS", "VOLATILITY", "LOG_RETURNS",
        "MACD_LINE", "MACD_SIGNAL", "MACD_HIST",
        "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
        "ATR"
    ]
    data = df_proc[features].values.astype(float)
    X_list, y_list = [], []
    close_idx = features.index(f"CLOSE_{ticker}")
    for i in range(len(data) - seq_length):
        X_list.append(data[i : i + seq_length])
        y_list.append(data[i + seq_length][close_idx])
    if len(X_list) < seq_length + 1:
        return current_model_bundle

    X_arr = np.array(X_list)
    y_arr = np.array(y_list).reshape(-1, 1)
    scaler_X = current_model_bundle["scaler_X"]
    scaler_y = current_model_bundle["scaler_y"]
    X_scaled = scaler_X.transform(X_arr.reshape(-1, X_arr.shape[2])).reshape(X_arr.shape)
    y_scaled = scaler_y.transform(y_arr)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    model = current_model_bundle["model"]

    try:
        best = tune_hyperparams(X_tensor, y_tensor, model, frozen_prefix="fc", n_trials=hpo_trials)
    except Exception:
        best = {}

    lr_opt  = best.get("lr", default_lr)
    wd_opt  = best.get("weight_decay", 0.0)
    epochs_opt = best.get("epochs", default_epochs)
    mlflow.log_params({"lr": lr_opt, "weight_decay": wd_opt, "epochs": epochs_opt})

    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_opt, weight_decay=wd_opt)
    loss_fn = nn.HuberLoss()
    model.train()
    for e in range(epochs_opt):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        mlflow.log_metric("retraining_loss", loss.item(), step=e)

    md = load_training_metadata()
    curr = md.get(f"{ticker}_model_version", os.getenv("MODEL_VERSION", "v1"))
    part = curr.lstrip("v").split('.')
    maj, min_ = part[0], int(part[1]) if len(part)>1 else 0
    new_ver = f"v{maj}.{min_+1}"
    out_dir = os.path.join("models", new_ver)
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, f"{ticker}_model.pth"))
    pickle.dump(scaler_X, open(os.path.join(out_dir, f"{ticker}_scaler_X.pkl"), 'wb'))
    pickle.dump(scaler_y, open(os.path.join(out_dir, f"{ticker}_scaler_y.pkl"), 'wb'))

    md[ticker] = new_end_datetime.strftime("%Y-%m-%d")
    md[f"{ticker}_model_version"] = new_ver
    save_training_metadata(md)

    bundle = current_model_bundle.copy()
    bundle["model"] = model
    bundle["model_version"] = new_ver
    return bundle
