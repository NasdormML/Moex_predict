import os
import pickle
import torch
from app.models import PricePredictionModel, TCNModel
from app.transfer_learning import load_training_metadata

DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

TICKER_CONFIGS = {
    "SBER": {
        "model_class": "PricePredictionModel",
        "model_name": "SBER_model.pth",
        "scaler_X_name": "SBER_scaler_X.pkl",
        "scaler_y_name": "SBER_scaler_y.pkl",
        "seq_length": 20,
        "num_features": 18,
        "output_dim": 1,
        "lstm_units": 240,
        "fc_units": 127,
        "dropout_rate": 0.13351299266216415
    },
    "GAZP": {
        "model_class": "TCNModel",
        "model_name": "GAZP_model.pth",
        "scaler_X_name": "GAZP_scaler_X.pkl",
        "scaler_y_name": "GAZP_scaler_y.pkl",
        "seq_length": 20,
        "num_features": 18,
        "num_channels": [64, 128, 256, 512],
        "kernel_size": 5,
        "dropout": 0.28,
        "fc_units": 30
    }
}

def load_models():
    """
    Загружает для каждого тикера последнюю версию модели из папки models/<version>/
    и возвращает словарь {ticker: {model, scaler_X, scaler_y, seq_length, model_version}}
    """
    models_dict = {}
    metadata = load_training_metadata()

    for ticker, cfg in TICKER_CONFIGS.items():
        version = metadata.get(f"{ticker}_model_version", DEFAULT_MODEL_VERSION)
        model_dir = os.path.join("models", version)

        model_path    = os.path.join(model_dir, cfg["model_name"])
        scaler_X_path = os.path.join(model_dir, cfg["scaler_X_name"])
        scaler_y_path = os.path.join(model_dir, cfg["scaler_y_name"])
        if not (os.path.exists(model_path) and 
                os.path.exists(scaler_X_path) and 
                os.path.exists(scaler_y_path)):
            print(f"[WARN] для {ticker} не найдены файлы в {model_dir}, пропускаем")
            continue

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
            raise ValueError(f"Неизвестная модель для {ticker}: {cfg['model_class']}")

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with open(scaler_X_path, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)

        models_dict[ticker] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": cfg.get("seq_length"),
            "model_version": version
        }

    if not models_dict:
        raise RuntimeError("Ни одна модель не загружена!")
    return models_dict
