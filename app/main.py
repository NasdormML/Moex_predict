from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import math
from datetime import datetime
import pandas as pd
import mlflow

from app.data import fetch_moex_eod_data, fetch_cbr_usd_rate
from app.preprocessing import preprocess_data
from app.predict import predict_price
from app.model_manager import load_models
from app.transfer_learning import retrain_model, load_training_metadata

app = FastAPI(title="MOEX Price Prediction API")

# Настройка MLflow Tracking: убедитесь, что MLflow UI запущен на указанном порту (например, 5002)
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("MOEX_Price_Prediction")

# Загружаем сохранённую модель (например, для тикера SBER)
models_dict = load_models()

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

def process_tradedate(df):
    # Приведение даты к единому формату
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
        raise HTTPException(status_code=404, detail=f"Модель для тикера {ticker} не найдена")
    
    # Разбор дат из запроса
    try:
        req_start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        req_end = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="Неверный формат даты. Используйте 'YYYY-MM-DD'")
    
    with mlflow.start_run(run_name=f"Predict_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("start_date", request.start_date)
        mlflow.log_param("end_date", request.end_date)
        
        # Получаем метаданные о последнем обучении
        metadata = load_training_metadata()
        last_train_str = metadata.get(ticker, "2025-04-01")
        last_train_date = datetime.strptime(last_train_str, "%Y-%m-%d").date()
        
        # Если конечная дата запроса превышает последнюю тренировку более чем на 7 дней – запускаем дообучение
        if (req_end - last_train_date).days > 7:
            mlflow.log_param("retraining_trigger", True)
            mlflow.log_param("last_train_date", last_train_str)
            print("Запуск дообучения для", ticker)
            models_dict[ticker] = retrain_model(
                ticker,
                datetime.combine(last_train_date, datetime.min.time()),
                datetime.combine(req_end, datetime.min.time()),
                models_dict[ticker]
            )
        else:
            mlflow.log_param("retraining_trigger", False)
        
        # Получаем данные с MOEX для тикера и дополнительных индикаторов
        df_sber = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", request.start_date, request.end_date)
        df_imoex = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", request.start_date, request.end_date)
        df_usd = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", request.start_date, request.end_date)
        
        # Если для USD нет данных, заполняем их с ЦБ РФ
        dates = pd.date_range(start=request.start_date, end=request.end_date)
        if df_usd is None or df_usd.empty or 'CLOSE' not in df_usd.columns:
            usd_rates = [fetch_cbr_usd_rate(d) for d in dates]
            df_usd = pd.DataFrame({"TRADEDATE": dates, "CLOSE": usd_rates})
        else:
            df_usd["TRADEDATE"] = pd.to_datetime(df_usd["TRADEDATE"]).dt.normalize()
            df_usd.sort_values("TRADEDATE", inplace=True)
            df_usd.reset_index(drop=True, inplace=True)
        
        # Приводим столбцы с датой к единому формату
        df_sber = process_tradedate(df_sber)
        df_imoex = process_tradedate(df_imoex)
        df_usd   = process_tradedate(df_usd)
        
        # Переименовываем столбцы для SBER, IMOEX и USD
        df_sber.rename(columns={
            "OPEN": "OPEN_SBER",
            "HIGH": "HIGH_SBER",
            "LOW": "LOW_SBER",
            "CLOSE": "CLOSE_SBER",
            "VOLUME": "VOL_SBER"
        }, inplace=True)
        df_imoex.rename(columns={"CLOSE": "CLOSE_IMOEX"}, inplace=True)
        df_usd.rename(columns={"CLOSE": "CLOSE_USD"}, inplace=True)
        
        # Отбираем необходимые столбцы перед объединением
        df_sber = df_sber[["TRADEDATE", "OPEN_SBER", "HIGH_SBER", "LOW_SBER", "CLOSE_SBER", "VOL_SBER"]]
        df_imoex = df_imoex[["TRADEDATE", "CLOSE_IMOEX"]]
        df_usd   = df_usd[["TRADEDATE", "CLOSE_USD"]]
        
        # Объединяем данные по дате
        merged_df = pd.merge(df_sber, df_imoex, on="TRADEDATE", how="outer")
        merged_df = pd.merge(merged_df, df_usd, on="TRADEDATE", how="outer")
        merged_df.sort_values("TRADEDATE", inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        mlflow.log_metric("records_after_merge", merged_df.shape[0])
        
        # Удаляем строки, где отсутствуют ключевые значения
        merged_df.dropna(subset=["CLOSE_SBER", "CLOSE_IMOEX", "CLOSE_USD"], inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        
        if merged_df.shape[0] < 20:
            raise HTTPException(status_code=422, detail=f"Недостаточно данных после объединения: получено {merged_df.shape[0]} строк, требуется минимум 20.")
        
        # Предобработка: вычисляем технические индикаторы
        df_processed = preprocess_data(merged_df, ticker)
        
        features = [
            "OPEN_SBER", "HIGH_SBER", "LOW_SBER", "CLOSE_SBER", "VOL_SBER",
            "CLOSE_IMOEX", "CLOSE_USD",
            "RSI", "SMA_RETURNS", "VOLATILITY", "LOG_RETURNS",
            "MACD_LINE", "MACD_SIGNAL", "MACD_HIST",
            "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
            "ATR"
        ]
        missing = [col for col in features if col not in df_processed.columns]
        if missing:
            raise HTTPException(status_code=500, detail=f"Отсутствуют признаки: {missing}")
        
        data = df_processed[features].values.astype(float)
        seq_length = 20
        if data.shape[0] < seq_length:
            raise HTTPException(status_code=422, detail=f"Недостаточно данных после предобработки. Требуется минимум {seq_length} записей, получено {data.shape[0]}")
        
        try:
            prediction = predict_price(
                models_dict[ticker]["model"],
                models_dict[ticker]["scaler_X"],
                models_dict[ticker]["scaler_y"],
                data,
                seq_length=seq_length
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        if not math.isfinite(prediction):
            raise HTTPException(status_code=500, detail="Предсказанная цена не является допустимым числом")
        
        mlflow.log_metric("predicted_price", prediction)
        mlflow.set_tag("prediction_date", datetime.today().strftime("%Y-%m-%d"))
        
        return {
            "ticker": ticker,
            "predicted_price": prediction,
            "date": datetime.today().strftime("%Y-%m-%d")
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
