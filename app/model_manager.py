import glob
import logging
import os
import pickle

import torch

from app.models.factory import get_model
from app.transfer_learning import load_training_metadata

ARTIFACTS_ROOT = os.getenv("MODEL_ARTIFACTS_DIR", "saved_models")

# Настройка логгера
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_models():
    """
    Загружает все активные модели, описанные в вложенных метаданных:
    {
      "<TICKER>": {
        "active_version": "vX.Y",
        "versions": {
          "vX.Y": {
            "factory_key": "...",
            "model_params": { ... }
          },
          ...
        }
      },
      ...
    }
    Артефакты лежат в saved_models/<version>/
    Ищутся файлы:
      - <ticker>_model.pth
      - <ticker>_best.pth
    и скейлеры:
      - <ticker>_scaler_X.pkl
      - <ticker>_scaler_y.pkl
    """
    metadata = load_training_metadata()
    models = {}

    for ticker, ticker_md in metadata.items():
        # Получаем активную версию для тикера
        version = ticker_md.get("active_version")
        if not version:
            logger.warning(f"Skipping {ticker}: no active_version set")
            continue

        # Берём параметры этой версии
        ver_md = ticker_md.get("versions", {}).get(version, {})
        factory_key = ver_md.get("factory_key")
        params = ver_md.get("model_params")
        if not (factory_key and params):
            logger.warning(f"Skipping {ticker}@{version}: incomplete metadata")
            continue

        model_dir = os.path.join(ARTIFACTS_ROOT, version)
        if not os.path.isdir(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            continue

        # Ищем файл весов
        weight_file = None
        for pattern in (f"{ticker}_model.pth", f"{ticker}_best.pth"):
            candidates = glob.glob(os.path.join(model_dir, pattern))
            if candidates:
                weight_file = candidates[0]
                break
        if not weight_file:
            logger.warning(f"Weights file not found for {ticker}@{version}")
            continue

        # Проверяем скейлеры
        path_sx = os.path.join(model_dir, f"{ticker}_scaler_X.pkl")
        path_sy = os.path.join(model_dir, f"{ticker}_scaler_y.pkl")
        if not os.path.exists(path_sx) or not os.path.exists(path_sy):
            logger.warning(f"Scaler files not found for {ticker}@{version}")
            continue

        # Загрузка модели
        try:
            model = get_model(factory_key, **params)
            state = torch.load(weight_file, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to instantiate or load {ticker}@{version}: {e}")
            continue

        # Загрузка скейлеров
        try:
            with open(path_sx, "rb") as fx:
                scaler_X = pickle.load(fx)
            with open(path_sy, "rb") as fy:
                scaler_y = pickle.load(fy)
        except Exception as e:
            logger.error(f"Failed to load scalers for {ticker}@{version}: {e}")
            continue

        # Проверяем, есть ли обязательный параметр seq_length
        seq_length = params.get("seq_length")
        if seq_length is None:
            logger.warning(f"seq_length not found in params for {ticker}@{version}")
            continue

        # Регистрируем модель
        models[ticker] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": seq_length,
            "model_version": version,
            "factory_key": factory_key,
            "model_params": params,
        }
        logger.info(f"Loaded model {ticker}@{version}")

    if not models:
        logger.error("No models loaded, aborting")
        raise RuntimeError("No models loaded")

    logger.info(f"Total loaded models: {len(models)} ({', '.join(models.keys())})")
    return models
