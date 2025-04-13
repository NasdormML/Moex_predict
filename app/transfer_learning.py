from datetime import datetime
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
from app.models import PricePredictionModel
from app.data import fetch_moex_eod_data
from app.preprocessing import preprocess_data
import pandas as pd
import os
import pickle

def retrain_model(ticker, last_train_datetime, new_end_datetime, current_model_bundle):
    """
    Выполняет дообучение модели на данных с last_train_datetime до new_end_datetime.
    Возвращает обновлённый словарь с моделью и скейлерами.
    """
    print(f"Дообучение модели для {ticker} с {last_train_datetime} до {new_end_datetime}")
    
    # Загрузка новых данных для тикера
    start_date = last_train_datetime.strftime("%Y-%m-%d")
    end_date = new_end_datetime.strftime("%Y-%m-%d")
    df_new = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start_date, end_date)
    if df_new is None or df_new.empty:
        print("Нет новых данных для дообучения.")
        return current_model_bundle

    # Переименование столбцов для тикера
    # Это необходимо, чтобы в DataFrame появились столбцы вида "OPEN_<ticker>", "CLOSE_<ticker>" и т.д.
    df_new.rename(columns={
        "OPEN": f"OPEN_{ticker}",
        "HIGH": f"HIGH_{ticker}",
        "LOW": f"LOW_{ticker}",
        "CLOSE": f"CLOSE_{ticker}",
        "VOLUME": f"VOL_{ticker}"
    }, inplace=True)
    
    # Обработка данных – теперь ожидается, что в DataFrame присутствуют нужные имена столбцов
    df_new_processed = preprocess_data(df_new, ticker)
    
    features = [
         f"OPEN_{ticker}", f"HIGH_{ticker}", f"LOW_{ticker}", f"CLOSE_{ticker}", f"VOL_{ticker}",
         "CLOSE_IMOEX", "CLOSE_USD",
         "RSI", "SMA_RETURNS", "VOLATILITY", "LOG_RETURNS",
         "MACD_LINE", "MACD_SIGNAL", "MACD_HIST",
         "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
         "ATR"
    ]
    missing = [col for col in features if col not in df_new_processed.columns]
    if missing:
        print("Отсутствуют признаки для дообучения:", missing)
        return current_model_bundle
    
    data = df_new_processed[features].values.astype(float)
    seq_length = 20
    if data.shape[0] < seq_length:
        print("Недостаточно данных для дообучения.")
        return current_model_bundle
    
    X_train = data[-seq_length:]
    # Целевая переменная – последняя цена закрытия для тикера
    y_train = data[-1][features.index(f"CLOSE_{ticker}")]
    model = current_model_bundle["model"]
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # Преобразуем входную последовательность в тензор
    X_train_tensor = torch.tensor(X_train.reshape(1, seq_length, -1), dtype=torch.float32)
    y_train_tensor = torch.tensor([[y_train]], dtype=torch.float32)
    
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        mlflow.log_metric("retraining_loss", loss.item(), step=epoch)
    
    # Обновляем модель в словаре
    model_bundle = current_model_bundle.copy()
    model_bundle["model"] = model
    
    # Обновляем метаданные о последнем обучении
    metadata = load_training_metadata()
    metadata[ticker] = new_end_datetime.strftime("%Y-%m-%d")
    with open("training_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    return model_bundle

def load_training_metadata():
    if os.path.exists("training_metadata.pkl"):
        with open("training_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = {}
    return metadata
