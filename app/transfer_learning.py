import os
import json
import pickle
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.data import fetch_moex_eod_data, fetch_cbr_usd_rate
from app.preprocessing import preprocess_data
from app.predict import predict_price
from app.model_manager import load_models
from sklearn.preprocessing import RobustScaler

# Файл для хранения метаданных обучения (например, последний обучающий день и агрегированные данные)
TRAINING_METADATA_FILE = os.path.join("models", "training_metadata.json")
HISTORICAL_DATA_FILE = os.path.join("models", "historical_data.pkl")

def load_training_metadata():
    if os.path.exists(TRAINING_METADATA_FILE):
        with open(TRAINING_METADATA_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_training_metadata(metadata):
    with open(TRAINING_METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def load_historical_data(ticker: str):
    if os.path.exists(HISTORICAL_DATA_FILE):
        with open(HISTORICAL_DATA_FILE, "rb") as f:
            data = pickle.load(f)
        return data.get(ticker)
    return None

def save_historical_data(ticker: str, data):
    historical_data = {}
    if os.path.exists(HISTORICAL_DATA_FILE):
        with open(HISTORICAL_DATA_FILE, "rb") as f:
            historical_data = pickle.load(f)
    historical_data[ticker] = data
    with open(HISTORICAL_DATA_FILE, "wb") as f:
        pickle.dump(historical_data, f)

def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length, 3])  # Предположим, что 4-й столбец – цена закрытия
    return np.array(sequences), np.array(targets)

def retrain_model(ticker, last_train_date: datetime, new_end_date: datetime, model_info, seq_length=20, n_epochs=50, mix_ratio=0.2):
    """
    Дообучение модели:
      - Загружаются новые данные (с last_train_date+1 до new_end_date)
      - Если доступны исторические данные, смешиваются с новыми (чтобы избежать потери знаний)
      - Пересчитываются скейлеры на объединённом наборе
      - Модель дообучается с небольшим learning rate и L2-регуляризацией
    """
    ticker = ticker.upper()
    start_date = (last_train_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = new_end_date.strftime("%Y-%m-%d")
    
    # Загружаем новые данные
    df_ticker = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start_date, end_date)
    df_imoex = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start_date, end_date)
    df_usd = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", start_date, end_date)
    
    dates = pd.date_range(start=start_date, end=end_date)
    if df_usd is None or df_usd.empty or 'CLOSE' not in df_usd.columns:
        usd_rates = [fetch_cbr_usd_rate(d) for d in dates]
        df_usd = pd.DataFrame({"TRADEDATE": dates, "CLOSE": usd_rates})
    
    # Приведение к единому виду и сортировка
    for df in [df_ticker, df_imoex, df_usd]:
        df.columns = [col.upper() for col in df.columns]
        if "TRADEDATE" not in df.columns:
            df["TRADEDATE"] = pd.to_datetime(df.iloc[:,0])
        else:
            df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df.sort_values("TRADEDATE", inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    merged_df = df_ticker.merge(
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
        raise ValueError(f"Отсутствуют признаки: {missing}")
    
    new_data = df_processed[features].values.astype(float)
    if new_data.shape[0] < seq_length + 1:
        raise ValueError(f"Недостаточно данных для переобучения. Требуется минимум {seq_length + 1} записей.")
    
    # Получаем старые данные (если имеются) и объединяем с новыми
    historical_data = load_historical_data(ticker)
    if historical_data is not None:
        # Для смешивания выбираем случайную выборку из исторических данных
        n_samples = int(mix_ratio * len(new_data))
        idx = np.random.choice(len(historical_data), size=n_samples, replace=False)
        sampled_old = historical_data[idx]
        combined_data = np.concatenate([sampled_old, new_data], axis=0)
    else:
        combined_data = new_data.copy()
    
    # Сохраним обновлённые исторические данные для будущих дообучений
    save_historical_data(ticker, combined_data)
    
    # Пересчитываем скейлеры на новом наборе данных
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    scaler_X.fit(combined_data)
    scaler_y.fit(combined_data[:, [3]])  # цена закрытия
    
    # Обновление скейлеров в model_info
    model_info["scaler_X"] = scaler_X
    model_info["scaler_y"] = scaler_y
    
    # Создаем последовательности для дообучения
    sequences, targets = create_sequences(combined_data, seq_length)
    sequences = torch.tensor(scaler_X.transform(sequences.reshape(-1, combined_data.shape[1])).reshape(sequences.shape), dtype=torch.float32)
    targets = torch.tensor(scaler_y.transform(targets.reshape(-1, 1)), dtype=torch.float32)
    
    model = model_info["model"]
    model.train()
    
    # Используем более низкий learning rate и L2-регуляризацию для стабильного дообучения
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    
    # Дообучение
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    # Сохраняем обновлённую модель и скейлеры
    model_path = os.path.join("models", f"{ticker}_model.pth")
    torch.save(model.state_dict(), model_path)
    with open(os.path.join("models", f"{ticker}_scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join("models", f"{ticker}_scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
    
    # Обновляем метаданные о последнем обучении
    metadata = load_training_metadata()
    metadata[ticker] = new_end_date.strftime("%Y-%m-%d")
    save_training_metadata(metadata)
    
    return model_info
