from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

from app.data import fetch_all_data
from app.preprocessing import preprocess_data
from app.predict import predict_price
from app.model_manager import load_models

app = FastAPI(title="MOEX Prediction API")

class PredictionRequest(BaseModel):
    ticker: str
    days_back: int = 30  # По умолчанию прогноз на основе 30 дней истории

@app.on_event("startup")
async def startup_event():
    try:
        app.state.models = load_models()
        print(f"Loaded models: {list(app.state.models.keys())}")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        ticker = request.ticker.upper()
        
        # Конфигурация источников данных
        configs = [
            {
                "security": ticker,
                "market": "shares",
                "interval": 24,  # Дневные свечи
                "days_back": request.days_back + 5  # Загружаем больше данных для RSI
            },
            {
                "security": "RTSI",
                "market": "index",
                "days_back": request.days_back
            },
            {
                "security": "USD000UTSTOM",
                "market": "currency",
                "days_back": request.days_back
            }
        ]

        # Загрузка и объединение данных
        merged_df = fetch_all_data(configs)
        if merged_df is None or merged_df.empty:
            raise HTTPException(404, detail="Не удалось загрузить данные")

        # Предобработка данных
        processed_df = preprocess_data(merged_df, ticker)
        
        # Проверка модели
        if ticker not in app.state.models:
            raise HTTPException(404, detail=f"Модель для {ticker} не найдена")

        # Подготовка данных для модели
        if len(processed_df) < 20:
            raise HTTPException(400, 
                detail="Недостаточно данных для прогноза (минимум 20 дней)")

        # Извлекаем признаки (исключая дату)
        features = processed_df.drop(columns=["date"]).values.astype(np.float32)

        # Прогноз
        model_info = app.state.models[ticker]
        prediction = predict_price(
            model_info["model"],
            model_info["scaler_X"],
            model_info["scaler_y"],
            features,
            seq_length=20
        )

        return {
            "ticker": ticker,
            "predicted_price": round(float(prediction), 2),
            "currency": "RUB",
            "last_date": processed_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "used_days": len(processed_df)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, detail=f"Ошибка прогноза: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)