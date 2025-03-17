from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from datetime import datetime

from app.data import fetch_moex_eod_data
from app.preprocessing import preprocess_data
from app.predict import predict_price

# Импортируем менеджер моделей, который загрузит все модели при старте
from app.model_manager import load_models

app = FastAPI(title="MOEX Price Prediction API")

# Загружаем модели для всех тикеров из папки models
models_dict = load_models()

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str     # "YYYY-MM-DD"
    end_date: str       # "YYYY-MM-DD"

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API предсказания цен MOEX"}

@app.post("/predict")
def predict(request: PredictionRequest):
    ticker = request.ticker.upper()
    # Проверяем, что для указанного тикера загружена модель
    if ticker not in models_dict:
        raise HTTPException(status_code=404, detail=f"Модель для тикера {ticker} не найдена")

    # Загружаем данные с MOEX
    df = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", request.start_date, request.end_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Данные не найдены")

    # Предобработка данных
    df_processed = preprocess_data(df, ticker=ticker)
    features = [col for col in df_processed.columns if col.startswith(("OPEN", "HIGH", "LOW", "CLOSE", "VOL", "RSI", "BODY", "UPPER_SHADOW", "LOWER_SHADOW"))]
    # Преобразуем в numpy-массив
    data = df_processed[features].values.astype(float)

    # Выполняем предсказание с помощью загруженной модели
    model_info = models_dict[ticker]
    prediction = predict_price(model_info["model"], model_info["scaler_X"], model_info["scaler_y"], data, seq_length=20)

    return {
        "ticker": ticker,
        "predicted_price": prediction,
        "date": datetime.today().strftime("%Y-%m-%d")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
