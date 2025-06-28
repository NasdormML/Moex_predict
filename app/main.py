import os
from datetime import date, datetime, timedelta

import mlflow
import mlflow.pytorch
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from app.data import fetch_moex_eod_data, fetch_usd_series
from app.model_manager import load_models
from app.predict import predict_price
from app.preprocessing import preprocess_data
from app.transfer_learning import load_training_metadata, retrain_model

app = FastAPI(title="MOEX Price Prediction API")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "MOEX_Price_Prediction"))
mlflow.pytorch.autolog()

models_dict = load_models()
history_dir = os.getenv("HISTORY_DIR", "history")
os.makedirs(history_dir, exist_ok=True)


@app.post("/predict/{ticker}/{target_date}")
def predict(ticker: str, target_date: date):
    ticker = ticker.upper()
    if ticker not in models_dict:
        raise HTTPException(404, f"Модель для {ticker} не найдена")

    # Fine-tune при необходимости
    bundle = models_dict[ticker]
    updated = retrain_model(ticker, bundle, retrain_threshold=5)
    if updated is not bundle:
        models_dict[ticker] = updated
        bundle = updated
    mlflow.set_tag("model_version", bundle["model_version"])

    md_ticket = load_training_metadata().get(ticker, {})
    ver = md_ticket.get("active_version", "")
    ver_md = md_ticket.get("versions", {}).get(ver, {})
    data_upto_str = ver_md.get("data_upto")
    if data_upto_str:
        last_known = datetime.strptime(data_upto_str, "%Y-%m-%d").date()
    else:
        last_known = date.today() - timedelta(days=1)

    seq = bundle["seq_length"]
    window = seq * 2
    start = (last_known - timedelta(days=window)).isoformat()
    end = last_known.isoformat()

    df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start, end)
    df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start, end)
    df_u = fetch_usd_series(start, end)

    def prep(df, ren, cols):
        df = df.copy()
        df["TRADEDATE"] = pd.to_datetime(
            df.get("BEGIN", df["TRADEDATE"])
        ).dt.normalize()
        return df.rename(columns=ren)[cols]

    t_cols = [
        "TRADEDATE",
        f"OPEN_{ticker}",
        f"HIGH_{ticker}",
        f"LOW_{ticker}",
        f"CLOSE_{ticker}",
        f"VOL_{ticker}",
    ]
    df_t = prep(
        df_t,
        {
            "OPEN": f"OPEN_{ticker}",
            "HIGH": f"HIGH_{ticker}",
            "LOW": f"LOW_{ticker}",
            "CLOSE": f"CLOSE_{ticker}",
            "VOLUME": f"VOL_{ticker}",
        },
        t_cols,
    )
    df_i = prep(df_i, {"CLOSE": "CLOSE_IMOEX"}, ["TRADEDATE", "CLOSE_IMOEX"])
    df_u = prep(df_u, {"CLOSE": "CLOSE_USD"}, ["TRADEDATE", "CLOSE_USD"])

    merged = (
        df_t.merge(df_i, on="TRADEDATE", how="outer")
        .merge(df_u, on="TRADEDATE", how="outer")
        .sort_values("TRADEDATE")
        .ffill()
        .bfill()
        .dropna()
    )

    proc = preprocess_data(merged, ticker)
    feat = [c for c in proc.columns if c != "TRADEDATE"]
    X_all = proc[feat].values.astype(float)
    if len(X_all) < seq:
        raise HTTPException(422, f"Недостаточно данных: {len(X_all)} < {seq}")

    preds = predict_price(
        bundle["model"], bundle["scaler_X"], bundle["scaler_y"], X_all, seq
    )

    for idx, val in enumerate(preds, start=1):
        mlflow.log_metric(f"pred_step_{idx}", val)

    future_dates = []
    d = last_known + timedelta(days=1)
    while len(future_dates) < len(preds):
        if d.weekday() < 5:
            future_dates.append(d)
        d += timedelta(days=1)

    # Сохраняем в историю
    rec = pd.DataFrame({"DATE": future_dates, "predicted_price": preds})
    pf = os.path.join(history_dir, f"predictions_{ticker}.csv")
    rec.to_csv(pf, mode="a", header=not os.path.exists(pf), index=False)

    return {
        "ticker": ticker,
        "known_up_to": last_known.isoformat(),
        "forecast_dates": [d.isoformat() for d in future_dates],
        "predictions": preds,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", 5000)))
