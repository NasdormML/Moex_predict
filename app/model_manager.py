import os
import pickle
import torch

from app.transfer_learning import load_training_metadata
from app.models import PricePredictionModel, TCNModel

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
        "dropout_rate": 0.13351299266216415,
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
    models_dict = {}
    metadata = load_training_metadata()

    for ticker, cfg in TICKER_CONFIGS.items():
        version = metadata.get(f"{ticker}_model_version", DEFAULT_MODEL_VERSION)
        model_dir = os.path.join("models", version)
        path_m  = os.path.join(model_dir, cfg["model_name"])
        path_sx = os.path.join(model_dir, cfg["scaler_X_name"])
        path_sy = os.path.join(model_dir, cfg["scaler_y_name"])
        if not (os.path.exists(path_m) and os.path.exists(path_sx) and os.path.exists(path_sy)):
            continue

        if cfg["model_class"] == "PricePredictionModel":
            model = PricePredictionModel(
                cfg["seq_length"],
                cfg["num_features"],
                cfg["output_dim"],
                cfg["lstm_units"],
                cfg["fc_units"],
                cfg["dropout_rate"],
            )
        else:
            model = TCNModel(
                num_features=cfg["num_features"],
                num_channels=cfg["num_channels"],
                kernel_size=cfg["kernel_size"],
                dropout=cfg["dropout"],
                fc_units=cfg["fc_units"],
            )

        raw_sd = torch.load(path_m, map_location="cpu")
        sd = {}
        for k, v in raw_sd.items():
            if k.startswith("attention."):
                new_k = "attn." + k.split(".", 1)[1]
            else:
                new_k = k
            sd[new_k] = v

        model.load_state_dict(sd)
        model.eval()

        with open(path_sx, "rb") as f:
            scaler_X = pickle.load(f)
        with open(path_sy, "rb") as f:
            scaler_y = pickle.load(f)

        models_dict[ticker] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": cfg["seq_length"],
            "model_version": version
        }

    if not models_dict:
        raise RuntimeError("Ни одна модель не загружена!")
    return models_dict
