import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.data import fetch_moex_eod_data
from app.preprocessing import preprocess_data
from app.models import PricePredictionModel

# Устанавливаем seed для воспроизводимости
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def create_sequences(X, y, seq_length=20):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

if __name__ == "__main__":
    ticker = "SBER"
    start_date = "2010-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    print("Загружаем данные SBER с MOEX...")
    sber_df = fetch_moex_eod_data(ticker, "stock", "shares", "TQBR", start_date, end_date)
    print("Загружаем данные IMOEX...")
    imoex_df = fetch_moex_eod_data("IMOEX", "stock", "index", "SNDX", start_date, end_date)
    print("Загружаем данные USD/RUB...")
    usd_df = fetch_moex_eod_data("USD000UTSTOM", "currency", "selt", "CETS", start_date, end_date)
    
    if sber_df is None or imoex_df is None:
        raise ValueError("Ошибка при загрузке данных SBER или IMOEX.")
    
    # Если USD данные отсутствуют или неполные, выбрасываем ошибку (при обучении логика замены не применяется)
    if usd_df is None or usd_df.empty or 'CLOSE' not in usd_df.columns:
        raise ValueError("Ошибка при загрузке данных USD/RUB с MOEX.")
    
    # Приводим даты к типу datetime и сортируем
    for df in [sber_df, imoex_df, usd_df]:
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df.sort_values("TRADEDATE", inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # Предобработка данных: вычисляем RSI и SMA, заполняем пропуски
    sber_df = preprocess_data(sber_df, ticker)
    
    # Для остальных датасетов переименовываем столбец закрытия
    imoex_df.rename(columns={"CLOSE": "CLOSE_IMOEX"}, inplace=True)
    usd_df.rename(columns={"CLOSE": "CLOSE_USD"}, inplace=True)
    
    sber_df = sber_df[["TRADEDATE", "OPEN_SBER", "HIGH_SBER", "LOW_SBER", "CLOSE_SBER", "VOL_SBER", "RSI_SBER", "SMA_SBER"]]
    imoex_df = imoex_df[["TRADEDATE", "CLOSE_IMOEX"]]
    usd_df   = usd_df[["TRADEDATE", "CLOSE_USD"]]
    
    merged_df = sber_df.merge(imoex_df, on="TRADEDATE", how="outer") \
                        .merge(usd_df, on="TRADEDATE", how="outer")
    merged_df.sort_values("TRADEDATE", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.dropna(subset=["CLOSE_SBER", "CLOSE_IMOEX", "CLOSE_USD", "RSI_SBER", "SMA_SBER"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    features = ["OPEN_SBER", "HIGH_SBER", "LOW_SBER", "CLOSE_SBER", "VOL_SBER",
                "CLOSE_IMOEX", "CLOSE_USD", "RSI_SBER", "SMA_SBER"]
    target_col = "CLOSE_SBER"
    
    data = merged_df[features].values.astype(float)
    targets = merged_df[[target_col]].values.astype(float)
    
    total_len = len(data)
    train_end = int(0.8 * total_len)
    train_data = data[:train_end]
    test_data = data[train_end:]
    train_targets = targets[:train_end]
    test_targets = targets[train_end:]
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(train_data)
    scaler_y.fit(train_targets)
    train_data_scaled = scaler_X.transform(train_data)
    test_data_scaled  = scaler_X.transform(test_data)
    train_targets_scaled = scaler_y.transform(train_targets)
    test_targets_scaled  = scaler_y.transform(test_targets)
    
    seq_length = 20
    X_train, y_train = create_sequences(train_data_scaled, train_targets_scaled, seq_length)
    X_test, y_test = create_sequences(test_data_scaled, test_targets_scaled, seq_length)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = PricePredictionModel(seq_length, len(features), output_dim=1,
                                 lstm_units=150, fc_units=64, dropout_rate=0.15)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    num_epochs = 40
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break
    
    model.load_state_dict(best_model_state)
    
    model.eval()
    test_losses = []
    preds_list = []
    actual_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_losses.append(loss.item())
            preds_list.append(outputs.cpu().numpy())
            actual_list.append(y_batch.cpu().numpy())
    avg_test_loss = sum(test_losses) / len(test_losses)
    print("Test Loss (MSE):", avg_test_loss)
    
    preds = np.concatenate(preds_list, axis=0)
    actuals = np.concatenate(actual_list, axis=0)
    
    preds_inv = scaler_y.inverse_transform(preds)
    actuals_inv = scaler_y.inverse_transform(actuals)
    
    mse_orig = mean_squared_error(actuals_inv, preds_inv)
    rmse_orig = np.sqrt(mse_orig)
    mae_orig = np.mean(np.abs(actuals_inv - preds_inv))
    mape_orig = np.mean(np.abs((actuals_inv - preds_inv) / actuals_inv)) * 100
    
    print(f"Метрики для SBER:")
    print(f"MSE (рубли^2): {mse_orig:.3f}")
    print(f"RMSE (рубли):  {rmse_orig:.3f}")
    print(f"MAE (рубли):   {mae_orig:.3f}")
    print(f"MAPE:          {mape_orig:.2f}%")
    
    plt.figure(figsize=(12,6))
    plt.plot(actuals_inv.flatten(), label="Actual SBER Close")
    plt.plot(preds_inv.flatten(), label="Predicted SBER Close")
    plt.title("SBER Closing Price Prediction (Test Set) - PyTorch Model")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    # Сохранение обученной модели и скейлеров для FastAPI
    model_save_path = "saved_model.pth"
    scaler_X_save_path = "scaler_X.pkl"
    scaler_y_save_path = "scaler_y.pkl"
    
    # Сохраняем состояние модели
    torch.save(model.state_dict(), model_save_path)
    
    # Сохраняем скейлеры
    with open(scaler_X_save_path, "wb") as f:
        pickle.dump(scaler_X, f)
    with open(scaler_y_save_path, "wb") as f:
        pickle.dump(scaler_y, f)
    
    print("Модель и скейлеры успешно сохранены.")
