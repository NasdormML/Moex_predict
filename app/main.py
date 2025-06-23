# app/main.py
import os
from datetime import date, timedelta

import mlflow
import mlflow.pytorch
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from app.data import fetch_cbr_usd_rate, fetch_moex_eod_data
from app.model_manager import load_models
from app.predict import predict_price
from app.preprocessing import preprocess_data
from app.transfer_learning import load_training_metadata, retrain_model

# --- FastAPI & MLflow setup ---
app = FastAPI(title="MOEX Price Prediction API")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "MOEX_Price_Prediction"))
mlflow.pytorch.autolog()

# --- Load models once ---
models_dict = load_models()

history_dir = os.getenv("HISTORY_DIR", "history")
os.makedirs(history_dir, exist_ok=True)


@app.post("/predict/{ticker}/{target_date}")
def predict(ticker: str, target_date: date):
    ticker = ticker.upper()
    if ticker not in models_dict:
        raise HTTPException(404, f"Модель для {ticker} не найдена")

    # 1) Possibly retrain if older than 5 bd
    bundle = models_dict[ticker]
    models_dict[ticker] = retrain_model(ticker, bundle, retrain_threshold=5)
    mlflow.set_tag("model_version", models_dict[ticker]["model_version"])

    # 2) Fetch data up to today or target_date if earlier
    md_ticket = load_training_metadata().get(ticker, {})
    ver = md_ticket.get("active_version")
    ver_md = md_ticket.get("versions", {}).get(ver, {})
    data_upto = ver_md.get("data_upto")
    if data_upto:
        fetch_end = data_upto
    else:
        fetch_end = (date.today() - timedelta(days=1)).isoformat()

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", None, fetch_end)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", None, fetch_end)
    df_u = fetch_moex_eod_data(
        "USD000UTSTOM", "currency", "selt", "CETS", None, fetch_end
    )
    if df_u is None or df_u.empty:
        df_u = pd.DataFrame(
            {
                "TRADEDATE": pd.date_range(fetch_end, fetch_end),
                "CLOSE": [
                    fetch_cbr_usd_rate(d) for d in pd.date_range(fetch_end, fetch_end)
                ],
            }
        )

    # 3) Normalize & merge, selecting only numeric cols
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
    )[
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
    )

    # 4) Preprocess, build X window
    proc = preprocess_data(merged, ticker)
    feat = [c for c in proc.columns if c != "TRADEDATE"]
    X_all = proc[feat].values.astype(float)
    seq = models_dict[ticker]["seq_length"]
    if len(X_all) < seq:
        raise HTTPException(422, f"Недостаточно данных: {len(X_all)} < {seq}")

    # 5) Predict
    pred = predict_price(
        models_dict[ticker]["model"],
        models_dict[ticker]["scaler_X"],
        models_dict[ticker]["scaler_y"],
        X_all,  # predict_price will slice last seq rows internally
        seq,
    )
    mlflow.log_metric("predicted_price", pred)

    # 6) Save history
    rec = pd.DataFrame(
        {"TRADEDATE": [target_date.isoformat()], "predicted_price": [pred]}
    )
    pf = os.path.join(history_dir, f"predictions_{ticker}.csv")
    rec.to_csv(pf, mode="a", header=not os.path.exists(pf), index=False)

    return {
        "ticker": ticker,
        "target_date": target_date.isoformat(),
        "prediction": pred,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", 5000)))
