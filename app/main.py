from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Импортируем функции из других модулей приложения
from app.data import fetch_moex_eod_data
from app.preprocessing import preprocess_data
from app.predict import predict_price

app = FastAPI(title="MOEX Price Prediction API")

# Модель запроса для предсказания
class PredictionRequest(BaseModel):
    ticker: str
    start_date: str   # Формат YYYY-MM-DD
    end_date: str     # Формат YYYY-MM-DD

@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API предсказания цен MOEX"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Загружаем данные для указанного тикера
    df = fetch_moex_eod_data(
        security=request.ticker,
        engine="stock",
        market="shares",
        board="TQBR",
        start_date=request.start_date,
        end_date=request.end_date
    )
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Данные не найдены для указанного тикера")
    
    # Предобработка данных (например, сортировка, расчет RSI и др.)
    df_processed = preprocess_data(df, ticker=request.ticker)
    
    # Получение предсказания от модели
    try:
        prediction = predict_price(df_processed, ticker=request.ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {e}")
    
    return {"ticker": request.ticker, "predicted_price": prediction}

if __name__ == "__main__":
    # Запуск FastAPI приложения через uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
