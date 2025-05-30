import os
import math
from datetime import datetime

import uvicorn
import pandas as pd
import mlflow
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.data import fetch_moex_eod_data, fetch_cbr_usd_rate
from app.preprocessing import preprocess_data
from app.predict import predict_price
from app.model_manager import load_models
from app.transfer_learning import retrain_model, load_training_metadata
from app.monitoring import validate_model_performance

app = FastAPI(title="MOEX Price Prediction API")

# MLflow config
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "MOEX_Price_Prediction"))
mlflow.pytorch.autolog()

# загружаем все модели ( factory_key + params из metadata)
models_dict = load_models()

history_dir = os.getenv("HISTORY_DIR", "history")
os.makedirs(history_dir, exist_ok=True)

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API предсказания цен MOEX"}

@app.get("/model_info/{ticker}")
def model_info(ticker: str):
    ticker = ticker.upper()
    if ticker not in models_dict:
        raise HTTPException(404, f"Модель для {ticker} не найдена")
    metadata = load_training_metadata()
    last_train = metadata.get(ticker)
    return {
        "ticker": ticker,
        "model_version": models_dict[ticker]["model_version"],
        "last_train_date": last_train
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    ticker = request.ticker.upper()
    if ticker not in models_dict:
        raise HTTPException(404, f"Модель для {ticker} не найдена")
    try:
        req_start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        req_end   = datetime.strptime(request.end_date,   "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(422, "Неверный формат даты: используйте YYYY-MM-DD")

    with mlflow.start_run(run_name=f"Predict_{ticker}_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_params({
            "ticker":      ticker,
            "start_date":  request.start_date,
            "end_date":    request.end_date
        })

        # переобучение если необходимо (используем metadata)
        metadata       = load_training_metadata()
        last_train_str = metadata.get(ticker)
        last_train     = datetime.strptime(last_train_str, "%Y-%m-%d") if last_train_str else datetime.min
        models_dict[ticker] = retrain_model(
            ticker,
            last_train,
            datetime.combine(req_end, datetime.min.time()),
            models_dict[ticker]
        )
        mlflow.set_tag("model_version", models_dict[ticker]["model_version"])

        df_t = fetch_moex_eod_data(ticker,        "stock",    "shares", "TQBR", request.start_date, request.end_date)
        df_i = fetch_moex_eod_data("IMOEX",       "stock",    "index",  "SNDX", request.start_date, request.end_date)
        df_u = fetch_moex_eod_data("USD000UTSTOM","currency","selt",   "CETS", request.start_date, request.end_date)
        if df_u is None or df_u.empty:
            dates = pd.date_range(request.start_date, request.end_date)
            df_u = pd.DataFrame({
                "TRADEDATE": dates,
                "CLOSE":     [fetch_cbr_usd_rate(d) for d in dates]
            })

        def normalize(df, cols_map):
            df["TRADEDATE"] = pd.to_datetime(df.get("BEGIN", df["TRADEDATE"])).dt.normalize()
            return df.rename(columns=cols_map)

        df_t = normalize(df_t, {
            "OPEN":  f"OPEN_{ticker}",
            "HIGH":  f"HIGH_{ticker}",
            "LOW":   f"LOW_{ticker}",
            "CLOSE": f"CLOSE_{ticker}",
            "VOLUME":f"VOL_{ticker}"
        })
        df_i = normalize(df_i, {"CLOSE": "CLOSE_IMOEX"})
        df_u = normalize(df_u, {"CLOSE": "CLOSE_USD"})

        merged = (
            df_t[["TRADEDATE", f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}"]]
            .merge(df_i[["TRADEDATE", "CLOSE_IMOEX"]], on="TRADEDATE", how="outer")
            .merge(df_u[["TRADEDATE", "CLOSE_USD"]], on="TRADEDATE", how="outer")
            .sort_values("TRADEDATE")
            .dropna()
        )
        proc = preprocess_data(merged, ticker)

        feat_list = [
            f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
            "CLOSE_IMOEX", "CLOSE_USD",
            "RSI","SMA_RETURNS","VOLATILITY","LOG_RETURNS",
            "MACD_LINE","MACD_SIGNAL","MACD_HIST",
            "BB_UPPER","BB_LOWER","BB_MIDDLE","ATR"
        ]
        X = proc[feat_list].values.astype(float)
        seq = models_dict[ticker]["seq_length"]
        if X.shape[0] < seq:
            raise HTTPException(422, "Недостаточно данных после предобработки")

        pred = predict_price(
            models_dict[ticker]["model"],
            models_dict[ticker]["scaler_X"],
            models_dict[ticker]["scaler_y"],
            X, seq
        )
        pred = float(pred)
        if not math.isfinite(pred):
            raise HTTPException(500, "Invalid prediction: non-finite value")
        mlflow.log_metric("predicted_price", pred)

        # сохраняем историю
        date_str = datetime.today().strftime("%Y-%m-%d")
        rec = pd.DataFrame({"TRADEDATE":[date_str],"predicted_price":[pred]})
        pf = os.path.join(history_dir, f"predictions_{ticker}.csv")
        rec.to_csv(pf, mode='a', header=not os.path.exists(pf), index=False)

        # performance check
        real_path = os.path.join(history_dir, f"real_{ticker}.csv")
        if os.path.exists(real_path):
            real = pd.read_csv(real_path)
            hist = pd.read_csv(pf)
            ok   = validate_model_performance(ticker, real, hist)
            mlflow.log_metric("performance_issue", 0 if ok else 1)
            if not ok:
                # не нашли cfg: retrain_model будет читать metadata
                models_dict[ticker] = retrain_model(
                    ticker,
                    datetime.strptime(load_training_metadata()[ticker], "%Y-%m-%d"),
                    datetime.now(),
                    models_dict[ticker]
                )

    return {"ticker": ticker, "predicted_price": pred, "date": date_str}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", 5000)))
