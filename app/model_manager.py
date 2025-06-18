import glob
import logging
import os
import pickle

import torch

from app.models.factory import get_model
from app.transfer_learning import load_training_metadata

ARTIFACTS_ROOT = os.getenv("MODEL_ARTIFACTS_DIR", "saved_models")

# Настраиваем логгер модуля
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_models():
    """
    Загружает все модели, которые есть в training_metadata.pkl.
    Для каждого <TICKER> из метаданных ищем артефакты в saved_models/<version>/:
      - веса: либо '<ticker>_best.pth', либо '<ticker>_model.pth'
      - скейлеры: '<ticker>_scaler_X.pkl', '<ticker>_scaler_y.pkl'
    Логирует по мере загрузки.
    """
    models = {}
    metadata = load_training_metadata()

    for key, version in metadata.items():
        if not key.endswith("_model_version"):
            continue
        ticker = key[: -len("_model_version")]
        factory_key = metadata.get(f"{ticker}_factory_key")
        params = metadata.get(f"{ticker}_model_params")

        if factory_key is None or params is None:
            logger.warning(f"Пропускаем {ticker}: неполные метаданные")
            continue

        model_dir = os.path.join(ARTIFACTS_ROOT, version)
        if not os.path.isdir(model_dir):
            logger.warning(f"Директория не найдена: {model_dir}")
            continue

        # ищем файл весов
        matches = glob.glob(os.path.join(model_dir, f"{ticker}_best.pth")) + glob.glob(
            os.path.join(model_dir, f"{ticker}_model.pth")
        )
        if not matches:
            logger.warning(f"Весов для {ticker}@{version} не найдено")
            continue
        path_model = matches[0]

        # пути к скейлерам
        path_sx = os.path.join(model_dir, f"{ticker}_scaler_X.pkl")
        path_sy = os.path.join(model_dir, f"{ticker}_scaler_y.pkl")
        if not (os.path.exists(path_sx) and os.path.exists(path_sy)):
            logger.warning(f"Скейлеры для {ticker}@{version} не найдены")
            continue

        try:
            model = get_model(factory_key, **params)
            state = torch.load(path_model, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            logger.error(f"Не удалось загрузить {ticker}@{version}: {e}")
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
            "factory_key": factory_key,
            "model_params": params,
        }
        logger.info(f"Загружена модель {ticker}@{version}")

    if not models:
        logger.error("Не загружено ни одной модели")
        raise RuntimeError("No models loaded")

    logger.info(f"Всего загружено моделей: {len(models)} ({', '.join(models.keys())})")
    return models
