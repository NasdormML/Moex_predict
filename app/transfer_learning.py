import logging
import os
import pickle
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from app.data import fetch_cbr_usd_rate, fetch_moex_eod_data
from app.preprocessing import preprocess_data

# Logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

METADATA_PATH = os.path.join("history", "metadata", "training_metadata.pkl")
ARTIFACTS_ROOT = os.getenv("MODEL_ARTIFACTS_DIR", "saved_models")
DEFAULT_LR = 1e-7
DEFAULT_EPOCHS = 45


def load_training_metadata():
    try:
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_training_metadata(md: dict):
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(md, f)


def retrain_model(ticker: str, bundle: dict, retrain_threshold: int = 5):
    md_all = load_training_metadata()
    md = md_all.get(ticker, {})
    ver = md.get("active_version")
    ver_md = md.get("versions", {}).get(ver, {})
    train_date = ver_md.get("train_date")
    data_upto = ver_md.get("data_upto")

    if not train_date or not data_upto:
        logger.info(f"[{ticker}] missing metadata, skip retrain")
        return bundle

    td = datetime.strptime(train_date, "%Y-%m-%d").date()
    bizdays = len(pd.bdate_range(td, datetime.today().date())) - 1
    if bizdays < retrain_threshold:
        logger.info(f"[{ticker}] fresh ({bizdays} days), skip retrain")
        return bundle

    logger.info(f"[{ticker}] retraining, last train {bizdays} bd ago")
    # load window data
    window_days = bundle["model_params"].get("window_days", 180)
    start = (datetime.today().date() - pd.tseries.offsets.BDay(window_days)).strftime(
        "%Y-%m-%d"
    )
    end = datetime.today().date().strftime("%Y-%m-%d")

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start, end)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start, end)
    df_u = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", start, end)
    if df_u is None or df_u.empty:
        df_u = pd.DataFrame(
            {
                "TRADEDATE": pd.date_range(start, end),
                "CLOSE": [fetch_cbr_usd_rate(d) for d in pd.date_range(start, end)],
            }
        )

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
                f"CLOSE_{ticker}",
                f"OPEN_{ticker}",
                f"HIGH_{ticker}",
                f"LOW_{ticker}",
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
    proc = preprocess_data(merged, ticker)

    features = bundle["model_params"].get(
        "feature_list", [c for c in proc.columns if c != "TRADEDATE"]
    )
    data = proc[features].values.astype(float)
    seq = bundle["seq_length"]
    if data.shape[0] <= seq:
        logger.info(f"[{ticker}] insufficient data, skip retrain")
        return bundle

    X_arr = np.lib.stride_tricks.sliding_window_view(data, (seq, data.shape[1]))[
        :, 0, :, :
    ]
    y_arr = data[seq:, features.index(f"CLOSE_{ticker}")].reshape(-1, 1)

    scaler_X = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    Xs = scaler_X.transform(X_arr.reshape(-1, X_arr.shape[2])).reshape(X_arr.shape)
    ys = scaler_y.transform(y_arr)

    # fine-tune params
    model = bundle["model"]
    ft = bundle["model_params"].get("fine_tune_modules", [])
    params = [
        p for n, p in model.named_parameters() if not ft or any(m in n for m in ft)
    ]
    for p in model.parameters():
        p.requires_grad = p in params
    if not params:
        params = list(model.parameters())
        [p.requires_grad_() for p in params]

    opt = optim.AdamW(params, lr=DEFAULT_LR)
    loss_fn = nn.HuberLoss()
    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)
    model.train()
    for epoch in range(DEFAULT_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
        mlflow.log_metric("retraining_loss", loss.item(), step=epoch)

    # save
    maj, min_ = ver.lstrip("v").split(".")
    new_ver = f"v{maj}.{int(min_)+1}"
    out = os.path.join(ARTIFACTS_ROOT, new_ver)
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, f"{ticker}_model.pth"))
    with open(os.path.join(out, f"{ticker}_scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(out, f"{ticker}_scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    # update metadata
    md_all[ticker]["versions"][new_ver] = {
        "train_date": datetime.today().strftime("%Y-%m-%d"),
        "data_upto": datetime.today().strftime("%Y-%m-%d"),
        "factory_key": bundle["factory_key"],
        "model_params": bundle["model_params"],
    }
    md_all[ticker]["active_version"] = new_ver
    save_training_metadata(md_all)
    bundle.update(
        {
            "model": model,
            "model_version": new_ver,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
        }
    )
    logger.info(f"[{ticker}] retrain done {new_ver}")
    return bundle
