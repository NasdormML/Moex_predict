# model_manager.py
import os
import pickle
import torch
from app.models import PricePredictionModel, TCNModel

def load_models():
    models_dict = {}
    ticker_configs = {
        "SBER": {
            "model_class": "PricePredictionModel",
            "model_file": os.path.join("models", "SBER_model.pth"),
            "scaler_X": os.path.join("models", "SBER_scaler_X.pkl"),
            "scaler_y": os.path.join("models", "SBER_scaler_y.pkl"),
            "seq_length": 20,
            "num_features": 18,
            "output_dim": 1,
            "lstm_units": 196,
            "fc_units": 151,
            "dropout_rate": 0.1335
        },
        "GAZP": {
            "model_class": "TCNModel",
            "model_file": os.path.join("models", "GAZP_model.pth"),
            "scaler_X": os.path.join("models", "GAZP_scaler_X.pkl"),
            "scaler_y": os.path.join("models", "GAZP_scaler_y.pkl"),
            "seq_length": 20,
            "num_features": 18,
            "num_channels": [32, 64, 128],
            "kernel_size": 4,
            "dropout": 0.28,
            "fc_units": 18
        }
    }
    
    for ticker, cfg in ticker_configs.items():
        if (os.path.exists(cfg["model_file"]) and 
            os.path.exists(cfg["scaler_X"]) and 
            os.path.exists(cfg["scaler_y"])):
            
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
                raise ValueError(f"Неизвестная архитектура для {ticker}")
            
            model.load_state_dict(torch.load(cfg["model_file"], map_location=torch.device('cpu')))
            model.eval()
            
            with open(cfg["scaler_X"], "rb") as f:
                scaler_X = pickle.load(f)
            with open(cfg["scaler_y"], "rb") as f:
                scaler_y = pickle.load(f)
            
            models_dict[ticker] = {
                "model": model,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "seq_length": cfg["seq_length"]
            }
        else:
            print(f"Модель или скейлеры для {ticker} не найдены.")
    return models_dict
