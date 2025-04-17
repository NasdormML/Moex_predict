import scheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import math
from datetime import datetime, timedelta
import pandas as pd
import mlflow
import os

from app.data import fetch_moex_eod_data, fetch_cbr_usd_rate
from app.preprocessing import preprocess_data
from app.predict import predict_price
from app.model_manager import load_models
from app.transfer_learning import retrain_model, load_training_metadata
from app.monitoring import validate_model_performance

app = FastAPI(title="MOEX Price Prediction API")

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("MOEX_Price_Prediction")

models_dict = load_models()

os.makedirs("history", exist_ok=True)

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str  # YYYY-MM-DD
    end_date: str

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

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API предсказания цен MOEX"}

@app.post("/predict")
def predict(request: PredictionRequest):
    ticker = request.ticker.upper()
    if ticker not in models_dict:
        raise HTTPException(404, f"Модель для {ticker} не найдена")

    try:
        req_start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        req_end   = datetime.strptime(request.end_date,   "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(422, "Неверный формат даты")

    with mlflow.start_run(run_name=f"Predict_{ticker}_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_params({
            "ticker": ticker,
            "start_date": request.start_date,
            "end_date": request.end_date
        })

        # дообучение по дате
        metadata = load_training_metadata()
        last_train = datetime.strptime(metadata.get(ticker, request.start_date), "%Y-%m-%d").date()
        if (req_end - last_train).days > 3:
            mlflow.log_param("retraining_trigger", True)
            models_dict[ticker] = retrain_model(
                ticker,
                datetime.combine(last_train, datetime.min.time()),
                datetime.combine(req_end, datetime.min.time()),
                models_dict[ticker]
            )
        else:
            mlflow.log_param("retraining_trigger", False)

        # загрузка EOD
        df_t = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", request.start_date, request.end_date)
        df_i = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", request.start_date, request.end_date)
        df_u = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", request.start_date, request.end_date)

        # fallback USD
        dates = pd.date_range(request.start_date, request.end_date)
        if df_u is None or df_u.empty or "CLOSE" not in df_u.columns:
            rates = [fetch_cbr_usd_rate(d) for d in dates]
            df_u = pd.DataFrame({"TRADEDATE": dates, "CLOSE": rates})

        # привести даты
        df_t = process_tradedate(df_t)
        df_i = process_tradedate(df_i)
        df_u = process_tradedate(df_u)

        df_t.rename(columns={
            "OPEN": f"OPEN_{ticker}", "HIGH": f"HIGH_{ticker}",
            "LOW": f"LOW_{ticker}",   "CLOSE": f"CLOSE_{ticker}",
            "VOLUME": f"VOL_{ticker}"
        }, inplace=True)
        df_i.rename(columns={"CLOSE": "CLOSE_IMOEX"}, inplace=True)
        df_u.rename(columns={"CLOSE": "CLOSE_USD"},   inplace=True)

        merged = (df_t[["TRADEDATE", f"OPEN_{ticker}", f"HIGH_{ticker}",
                        f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}"]]
                  .merge(df_i[["TRADEDATE","CLOSE_IMOEX"]], on="TRADEDATE", how="outer")
                  .merge(df_u[["TRADEDATE","CLOSE_USD"]],   on="TRADEDATE", how="outer")
                  .sort_values("TRADEDATE")
                  .dropna())

        df_proc = preprocess_data(merged, ticker)
        features = [
            f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
            "CLOSE_IMOEX", "CLOSE_USD",
            "RSI", "SMA_RETURNS", "VOLATILITY", "LOG_RETURNS",
            "MACD_LINE", "MACD_SIGNAL", "MACD_HIST",
            "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
            "ATR"
        ]
        X = df_proc[features].values.astype(float)
        seq = models_dict[ticker]["seq_length"]
        if X.shape[0] < seq:
            raise HTTPException(422, "Недостаточно данных после предобработки")

        pred = predict_price(models_dict[ticker]["model"],
                             models_dict[ticker]["scaler_X"],
                             models_dict[ticker]["scaler_y"],
                             X, seq)
        if not math.isfinite(pred):
            raise HTTPException(500, "Invalid prediction")

        mlflow.log_metric("predicted_price", pred)
        date_str = datetime.today().strftime("%Y-%m-%d")

        # сохранить предсказание
        rec = pd.DataFrame({"TRADEDATE":[date_str], "predicted_price":[pred]})
        pf  = os.path.join("history", f"predictions_{ticker}.csv")
        rec.to_csv(pf, mode='a' if os.path.exists(pf) else 'w',
                   header=not os.path.exists(pf), index=False)

        # проверка качества
        rf = os.path.join("history", f"real_{ticker}.csv")
        if os.path.exists(rf):
            real = pd.read_csv(rf)
            hist = pd.read_csv(pf)
            ok   = validate_model_performance(ticker, real, hist)
            mlflow.log_param("performance_issue", not ok)
            if not ok:
                models_dict[ticker] = retrain_model(
                    ticker,
                    datetime.strptime(load_training_metadata()[ticker], "%Y-%m-%d"),
                    datetime.now(),
                    models_dict[ticker]
                )

        return {"ticker": ticker, "predicted_price": pred, "date": date_str}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
