import os
import pickle

import torch

from app.models.factory import get_model
from app.transfer_learning import load_training_metadata

ARTIFACTS_ROOT = os.getenv("MODEL_ARTIFACTS_DIR", "saved_models")
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Конфигурация по тикерам с именами файлов и параметрами по умолчанию
TICKER_CONFIGS = {
    "SBER": {
        "default_factory": "lstm",
        "model_file": "SBER_model.pth",
        "scaler_x": "SBER_scaler_X.pkl",
        "scaler_y": "SBER_scaler_y.pkl",
        "params": {
            "seq_length": 20,
            "num_features": 18,
            "output_dim": 1,
            "lstm_units": 240,
            "fc_units": 127,
            "dropout_rate": 0.1335,
        },
    },
    "GAZP": {
        "default_factory": "tcn",
        "model_file": "GAZP_model.pth",
        "scaler_x": "GAZP_scaler_X.pkl",
        "scaler_y": "GAZP_scaler_y.pkl",
        "params": {
            "num_features": 18,
            "num_channels": [64, 128, 256, 512],
            "kernel_size": 5,
            "dropout": 0.28,
            "fc_units": 30,
        },
    },
    "ROSN": {
        "default_factory": "tft",
        "model_file": "ROSN_model.pth",
        "scaler_x": "ROSN_scaler_X.pkl",
        "scaler_y": "ROSN_scaler_y.pkl",
        "params": {
            "seq_length": 15,
            "num_features": 18,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 384,
            "dropout": 0.09031434953930437,
        },
    },
}


def load_models():
    models = {}
    metadata = load_training_metadata()
    for ticker, cfg in TICKER_CONFIGS.items():
        version = metadata.get(f"{ticker}_model_version", DEFAULT_MODEL_VERSION)
        factory_key = metadata.get(f"{ticker}_factory_key", cfg["default_factory"])
        params = metadata.get(f"{ticker}_model_params", cfg["params"])

        model_dir = os.path.join(ARTIFACTS_ROOT, version)
        path_model = os.path.join(model_dir, cfg["model_file"])
        path_sx = os.path.join(model_dir, cfg["scaler_x"])
        path_sy = os.path.join(model_dir, cfg["scaler_y"])
        if not all(os.path.exists(p) for p in (path_model, path_sx, path_sy)):
            continue

        try:
            model = get_model(factory_key, **params)
            state = torch.load(path_model, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            print(f"[ERROR] Failed to load model for {ticker}: {e}")
            continue

        with open(path_sx, "rb") as f:
            scaler_X = pickle.load(f)
        with open(path_sy, "rb") as f:
            scaler_y = pickle.load(f)

        models[ticker] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": params.get("seq_length"),
            "model_version": version,
        }

    if not models:
        raise RuntimeError("No models loaded")
    return models
