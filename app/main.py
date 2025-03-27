from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import math
from datetime import datetime, date
import pandas as pd

from app.data import fetch_moex_eod_data, fetch_cbr_usd_rate
from app.preprocessing import preprocess_data
from app.predict import predict_price
from app.model_manager import load_models

app = FastAPI(title="MOEX Price Prediction API")

# Загрузка сохранённой модели (для тикера SBER)
models_dict = load_models()

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API предсказания цен MOEX"}

@app.post("/predict")
def predict(request: PredictionRequest):
    ticker = request.ticker.upper()
    if ticker not in models_dict:
        raise HTTPException(status_code=404, detail=f"Модель для тикера {ticker} не найдена")
    
    try:
        req_start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        req_end = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="Неверный формат даты. Используйте 'YYYY-MM-DD'")
    
    # Загружаем данные SBER и IMOEX с MOEX
    df_sber = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", request.start_date, request.end_date)
    df_imoex = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", request.start_date, request.end_date)
    
    # Загружаем данные USD с MOEX
    df_usd = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", request.start_date, request.end_date)
    
    # Если MOEX не вернул данные по USD или они неполные – заполняем их курсом ЦБ РФ
    dates = pd.date_range(start=request.start_date, end=request.end_date)
    if df_usd is None or df_usd.empty or 'CLOSE' not in df_usd.columns:
        usd_rates = [fetch_cbr_usd_rate(d) for d in dates]
        df_usd = pd.DataFrame({"TRADEDATE": dates, "CLOSE": usd_rates})
    else:
        df_usd["TRADEDATE"] = pd.to_datetime(df_usd["TRADEDATE"]).dt.normalize()
        df_usd.sort_values("TRADEDATE", inplace=True)
        df_usd.reset_index(drop=True, inplace=True)
        df_usd_full = pd.DataFrame({"TRADEDATE": dates})
        df_usd = df_usd_full.merge(df_usd, on="TRADEDATE", how="left")
        missing_mask = df_usd["CLOSE"].isna()
        if missing_mask.any():
            df_usd.loc[missing_mask, "CLOSE"] = [fetch_cbr_usd_rate(d) for d in df_usd.loc[missing_mask, "TRADEDATE"]]
    
    if df_sber is None or df_sber.empty:
        raise HTTPException(status_code=404, detail=f"Данные для тикера {ticker} не найдены")
    if (df_imoex is None or df_imoex.empty) or (df_usd is None or df_usd.empty):
        raise HTTPException(status_code=404, detail="Данные для дополнительных индикаторов не найдены")
    
    df_sber.columns = [col.upper() for col in df_sber.columns]
    df_imoex.columns = [col.upper() for col in df_imoex.columns]
    df_usd.columns = [col.upper() for col in df_usd.columns]
    
    def process_tradedate(df):
        if "BEGIN" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["BEGIN"])
        elif "TRADETIME" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["TRADETIME"])
        elif "TRADEDATE" in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        else:
            raise ValueError("Не найден столбец с датой")
        df["TRADEDATE"] = df["TRADEDATE"].dt.normalize()
        df.sort_values("TRADEDATE", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    df_sber = process_tradedate(df_sber)
    df_imoex = process_tradedate(df_imoex)
    df_usd = process_tradedate(df_usd)
    
    merged_df = df_sber.merge(
        df_imoex[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": "CLOSE_IMOEX"}),
        on="TRADEDATE", how="left"
    ).merge(
        df_usd[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": "CLOSE_USD"}),
        on="TRADEDATE", how="left"
    )
    
    merged_df['CLOSE_IMOEX'] = merged_df['CLOSE_IMOEX'].ffill().bfill()
    merged_df['CLOSE_USD'] = merged_df['CLOSE_USD'].ffill().bfill()
    merged_df.reset_index(drop=True, inplace=True)
    
    df_processed = preprocess_data(merged_df, ticker)
    
    features = [
        f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
        "CLOSE_IMOEX", "CLOSE_USD", f"RSI_{ticker}", f"SMA_{ticker}"
    ]
    missing = [col for col in features if col not in df_processed.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Отсутствуют признаки: {missing}")
    
    data = df_processed[features].values.astype(float)
    seq_length = 20
    if data.shape[0] < seq_length:
        raise HTTPException(status_code=422, detail=f"Недостаточно данных. Требуется минимум {seq_length} записей, получено {data.shape[0]}")
    
    model_info = models_dict[ticker]
    try:
        prediction = predict_price(model_info["model"], model_info["scaler_X"], model_info["scaler_y"], data, seq_length=seq_length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if not math.isfinite(prediction):
        raise HTTPException(status_code=500, detail="Предсказанная цена не является допустимым числом")
    
    return {
        "ticker": ticker,
        "predicted_price": prediction,
        "date": datetime.today().strftime("%Y-%m-%d")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
