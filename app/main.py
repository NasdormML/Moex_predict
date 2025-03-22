from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import math
from datetime import datetime, date
import pandas as pd

from app.data import fetch_moex_intraday_data, fetch_moex_eod_data, fetch_cbr_usd_rate
from app.preprocessing import preprocess_data
from app.predict import predict_price
from app.model_manager import load_models

app = FastAPI(title="MOEX Price Prediction API")

# Загружаем модели (например, для SBER и GAZP)
models_dict = load_models()

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str  # Формат "YYYY-MM-DD"
    end_date: str    # Формат "YYYY-MM-DD"

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API предсказания цен MOEX"}

@app.post("/predict")
def predict(request: PredictionRequest):
    ticker = request.ticker.upper()
    if ticker not in models_dict:
        raise HTTPException(status_code=404, detail=f"Модель для тикера {ticker} не найдена")
    
    # Преобразуем даты запроса в date
    try:
        req_start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        req_end = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="Неверный формат даты. Используйте 'YYYY-MM-DD'")
    
    today_date = date.today()
    
    # Получаем данные по SBER, IMOEX и USD
    if req_end < today_date:
        df_sber = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", request.start_date, request.end_date)
        df_imoex = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", request.start_date, request.end_date)
        df_usd = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", request.start_date, request.end_date)
    else:
        df_sber = fetch_moex_intraday_data(ticker, interval=1, from_date=request.start_date, till_date=request.end_date)
        df_imoex = fetch_moex_intraday_data("IMOEX", interval=1, from_date=request.start_date, till_date=request.end_date)
        if df_imoex is None or df_imoex.empty:
            df_imoex = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", request.start_date, request.end_date)
        df_usd = fetch_moex_intraday_data("USD000UTSTOM", interval=1, from_date=request.start_date, till_date=request.end_date)
        if df_usd is None or df_usd.empty:
            df_usd = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", request.start_date, request.end_date)
    
    # Если данные по USD отсутствуют, используем данные ЦБ РФ
    if df_usd is None or df_usd.empty:
        dates = pd.date_range(start=request.start_date, end=request.end_date)
        usd_rates = []
        for d in dates:
            rate = fetch_cbr_usd_rate(d)
            usd_rates.append(rate)
        df_usd = pd.DataFrame({"TRADEDATE": dates, "CLOSE": usd_rates})
    
    # Проверяем, что данные для основного тикера получены
    if df_sber is None or df_sber.empty:
        raise HTTPException(status_code=404, detail=f"Данные для тикера {ticker} не найдены")
    if (df_imoex is None or df_imoex.empty) or (df_usd is None or df_usd.empty):
        raise HTTPException(status_code=404, detail="Данные для дополнительных индикаторов не найдены")
    
    # Приводим имена столбцов во всех DataFrame к верхнему регистру
    df_sber.columns = [col.upper() for col in df_sber.columns]
    df_imoex.columns = [col.upper() for col in df_imoex.columns]
    df_usd.columns = [col.upper() for col in df_usd.columns]
    
    # Обработка столбца даты: если intraday, ожидаем столбец BEGIN, иначе – TRADEDATE
    if "BEGIN" in df_sber.columns:
        df_sber["TRADEDATE"] = pd.to_datetime(df_sber["BEGIN"])
    else:
        df_sber["TRADEDATE"] = pd.to_datetime(df_sber["TRADEDATE"])
    df_sber.sort_values("TRADEDATE", inplace=True)
    df_sber.reset_index(drop=True, inplace=True)
    
    if "BEGIN" in df_imoex.columns:
        df_imoex["TRADEDATE"] = pd.to_datetime(df_imoex["BEGIN"])
    else:
        df_imoex["TRADEDATE"] = pd.to_datetime(df_imoex.get("TRADETIME", df_imoex.get("TRADEDATE")))
    df_imoex.sort_values("TRADEDATE", inplace=True)
    df_imoex.reset_index(drop=True, inplace=True)
    
    if "BEGIN" in df_usd.columns:
        df_usd["TRADEDATE"] = pd.to_datetime(df_usd["BEGIN"])
    else:
        df_usd["TRADEDATE"] = pd.to_datetime(df_usd.get("TRADETIME", df_usd.get("TRADEDATE")))
    df_usd.sort_values("TRADEDATE", inplace=True)
    df_usd.reset_index(drop=True, inplace=True)
    
    # Объединяем данные по TRADEDATE (используем данные SBER как базовые)
    merged_df = df_sber.merge(
        df_imoex[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": "CLOSE_IMOEX"}),
        on="TRADEDATE", how="left")
    merged_df = merged_df.merge(
        df_usd[["TRADEDATE", "CLOSE"]].rename(columns={"CLOSE": "CLOSE_USD"}),
        on="TRADEDATE", how="left")
    
    # Заполняем пропуски методом ffill и bfill для дополнительных индикаторов
    merged_df['CLOSE_IMOEX'] = merged_df['CLOSE_IMOEX'].ffill().bfill()
    merged_df['CLOSE_USD'] = merged_df['CLOSE_USD'].ffill().bfill()
    merged_df.reset_index(drop=True, inplace=True)
    
    # Предобработка данных для тикера (включает вычисление RSI)
    df_processed = preprocess_data(merged_df, ticker=ticker)
    
    # Формирование набора признаков для модели
    features = [
        f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
        "CLOSE_IMOEX", "CLOSE_USD", f"RSI_{ticker}"
    ]
    missing = [col for col in features if col not in df_processed.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Отсутствуют признаки: {missing}")
    
    data = df_processed[features].values.astype(float)
    
    # Проверяем, что данных достаточно для формирования последовательности (seq_length = 20)
    if data.shape[0] < 20:
        raise HTTPException(status_code=422, detail=f"Недостаточно данных для последовательности. Требуется минимум 20 записей, получено {data.shape[0]}")
    
    model_info = models_dict[ticker]
    try:
        prediction = predict_price(model_info["model"], model_info["scaler_X"], model_info["scaler_y"], data, seq_length=20)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Проверяем, что предсказание является конечным числом
    if not math.isfinite(prediction):
        raise HTTPException(status_code=500, detail="Предсказанная цена не является допустимым числом")
    
    return {
        "ticker": ticker,
        "predicted_price": prediction,
        "date": datetime.today().strftime("%Y-%m-%d")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
