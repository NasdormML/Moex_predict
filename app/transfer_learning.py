from datetime import datetime, timedelta
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os

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


def retrain_model(
    ticker: str,
    last_train_datetime: datetime,
    new_end_datetime: datetime,
    current_model_bundle: dict,
    window_days: int = 180,
    lr: float = 1e-5,
    epochs: int = 10
) -> dict:
    """ 
    Постепенная доработка: объединение новых данных EOD, IMOEX, USD, вычислительных показателей,
    масштабируем с помощью существующих скалеров, заморозим все слои, кроме полносвязных 
    и точная настройка набора данных скользящего окна.
    """
    print(f"Дообучение {ticker} с {last_train_datetime} до {new_end_datetime}")

    # Load raw data for ticker, index, and USD
    start_date = (new_end_datetime - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_date = new_end_datetime.strftime("%Y-%m-%d")
    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start_date, end_date)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start_date, end_date)
    df_u = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", start_date, end_date)

    # Fallback USD from CBR
    if df_u is None or df_u.empty or 'CLOSE' not in df_u.columns:
        dates = pd.date_range(start_date, end_date)
        rates = [fetch_cbr_usd_rate(d) for d in dates]
        df_u = pd.DataFrame({"TRADEDATE": dates, "CLOSE": rates})

    # Normalize dates
    df_t = process_tradedate(df_t)
    df_i = process_tradedate(df_i)
    df_u = process_tradedate(df_u)

    df_t.rename(columns={
        "OPEN": f"OPEN_{ticker}", "HIGH": f"HIGH_{ticker}",
        "LOW": f"LOW_{ticker}",   "CLOSE": f"CLOSE_{ticker}",
        "VOLUME": f"VOL_{ticker}"
    }, inplace=True)
    df_i.rename(columns={"CLOSE": "CLOSE_IMOEX"}, inplace=True)
    df_u.rename(columns={"CLOSE": "CLOSE_USD"}, inplace=True)

    df = df_t[["TRADEDATE", f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}"]]
    df = df.merge(df_i[["TRADEDATE", "CLOSE_IMOEX"]], on="TRADEDATE", how="outer")
    df = df.merge(df_u[["TRADEDATE", "CLOSE_USD"]], on="TRADEDATE", how="outer")
    df = df.sort_values("TRADEDATE").ffill().bfill().dropna().reset_index(drop=True)

    # Compute technical indicators
    df_proc = preprocess_data(df, ticker)
    features = [
        f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
        "CLOSE_IMOEX", "CLOSE_USD",
        "RSI", "SMA_RETURNS", "VOLATILITY", "LOG_RETURNS",
        "MACD_LINE", "MACD_SIGNAL", "MACD_HIST",
        "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
        "ATR"
    ]
    data = df_proc[features].values.astype(float)
    seq_length = current_model_bundle.get("seq_length", 20)

    # Build sliding window dataset
    X_list, y_list = [], []
    close_idx = features.index(f"CLOSE_{ticker}")
    for i in range(len(data) - seq_length):
        X_list.append(data[i : i + seq_length])
        y_list.append(data[i + seq_length][close_idx])
    if len(X_list) < 10:
        print(f"Недостаточно примеров для дообучения: {len(X_list)}")
        return current_model_bundle

    X_arr = np.array(X_list)  # shape (n_samples, seq_length, num_features)
    y_arr = np.array(y_list).reshape(-1, 1)  # shape (n_samples, 1)

    # Scale using existing scalers
    scaler_X = current_model_bundle["scaler_X"]
    scaler_y = current_model_bundle["scaler_y"]
    n_samples = X_arr.shape[0]
    # Reshape for scaler: (n_samples * seq_length, num_features)
    X_flat = X_arr.reshape(-1, X_arr.shape[2])
    X_scaled_flat = scaler_X.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n_samples, seq_length, X_arr.shape[2])
    y_scaled = scaler_y.transform(y_arr)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    model = current_model_bundle["model"]
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    # Fine‑tuning loop
    model.train()
    for epoch in range(epochs):
         optimizer.zero_grad()
         preds = model(X_tensor)
         loss = loss_fn(preds, y_tensor)
         loss.backward()
         optimizer.step()
         mlflow.log_metric("retraining_loss", loss.item(), step=epoch)
         print(f"Epoch {epoch+1}/{epochs} - loss={loss.item():.6f}")

    metadata = load_training_metadata()
    current_ver = metadata.get(f"{ticker}_model_version", os.getenv("MODEL_VERSION", "v1"))
    # Generate new version: v<major>.<minor+1>
    ver_nums = current_ver.lstrip("v").split(".")
    major = ver_nums[0]
    minor = int(ver_nums[1]) if len(ver_nums) > 1 else 0
    new_ver = f"v{major}.{minor+1}"

    # New folder
    new_dir = os.path.join("models", new_ver)
    os.makedirs(new_dir, exist_ok=True)

    # Save model and scaler
    torch.save(model.state_dict(), os.path.join(new_dir, f"{ticker}_model.pth"))
    with open(os.path.join(new_dir, f"{ticker}_scaler_X.pkl"), "wb") as f:
        pickle.dump(current_model_bundle["scaler_X"], f)
    with open(os.path.join(new_dir, f"{ticker}_scaler_y.pkl"), "wb") as f:
        pickle.dump(current_model_bundle["scaler_y"], f)

    # Refresh metadata
    metadata[ticker] = new_end_datetime.strftime("%Y-%m-%d")
    metadata[f"{ticker}_model_version"] = new_ver
    save_training_metadata(metadata)
    print(f"Saved new version {new_ver} for {ticker} in {new_dir}")

    updated_bundle = current_model_bundle.copy()
    updated_bundle["model"] = model
    updated_bundle["model_version"] = new_ver
    return updated_bundle