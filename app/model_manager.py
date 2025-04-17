import os
import pickle
import torch
from app.models import PricePredictionModel, TCNModel

# Версия модели export MODEL_VERSION=v1
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Путь к директории с моделями для выбранной версии
BASE_MODEL_DIR = os.path.join("models", MODEL_VERSION)

def load_models():
    models_dict = {}
    ticker_configs = {
        "SBER": {
            "model_class": "PricePredictionModel",
            "model_file": os.path.join(BASE_MODEL_DIR, "SBER_model.pth"),
            "scaler_X":   os.path.join(BASE_MODEL_DIR, "SBER_scaler_X.pkl"),
            "scaler_y":   os.path.join(BASE_MODEL_DIR, "SBER_scaler_y.pkl"),
            "seq_length": 20,
            "num_features": 18,
            "output_dim": 1,
            "lstm_units": 196,
            "fc_units": 151,
            "dropout_rate": 0.1335
        },
        "GAZP": {
            "model_class": "TCNModel",
            "model_file": os.path.join(BASE_MODEL_DIR, "GAZP_model.pth"),
            "scaler_X":   os.path.join(BASE_MODEL_DIR, "GAZP_scaler_X.pkl"),
            "scaler_y":   os.path.join(BASE_MODEL_DIR, "GAZP_scaler_y.pkl"),
            "seq_length": 20,
            "num_features": 18,
            "num_channels": [32, 64, 128],
            "kernel_size": 4,
            "dropout": 0.28,
            "fc_units": 18
        }
    }

    for ticker, cfg in ticker_configs.items():
        # Проверяем, что все файлы есть
        if not (os.path.exists(cfg["model_file"]) and 
                os.path.exists(cfg["scaler_X"]) and 
                os.path.exists(cfg["scaler_y"])):
            print(f"[WARN] Версия '{MODEL_VERSION}': не найдены файлы для {ticker}")
            continue

        # Создаём экземпляр нужной модели
        if cfg["model_class"] == "PricePredictionModel":
            model = PricePredictionModel(
                seq_length=cfg["seq_length"],
                num_features=cfg["num_features"],
                output_dim=cfg["output_dim"],
                lstm_units=cfg["lstm_units"],
                fc_units=cfg["fc_units"],
                dropout_rate=cfg["dropout_rate"]
            )
        elif cfg["model_class"] == "TCNModel":
            model = TCNModel(
                num_features=cfg["num_features"],
                num_channels=cfg["num_channels"],
                kernel_size=cfg["kernel_size"],
                dropout=cfg["dropout"],
                fc_units=cfg["fc_units"]
            )
        else:
            raise ValueError(f"Неизвестная архитектура для {ticker}: {cfg['model_class']}")

        # Загружаем веса
        model.load_state_dict(torch.load(cfg["model_file"], map_location=torch.device('cpu')))
        model.eval()

        # Загружаем скейлеры
        with open(cfg["scaler_X"], "rb") as f:
            scaler_X = pickle.load(f)
        with open(cfg["scaler_y"], "rb") as f:
            scaler_y = pickle.load(f)

        # Сохраняем в словарь
        models_dict[ticker] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": cfg["seq_length"]
        }

    if not models_dict:
        raise RuntimeError(f"Ни одна модель не загружена из версии '{MODEL_VERSION}'")
    return models_dict
