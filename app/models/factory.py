import inspect
from importlib import import_module
from typing import Any

AVAILABLE = {
    "lstm": "app.models.attention_lstm",
    "tcn": "app.models.tcn",
    "tft": "app.models.tft",
    "autoformer": "app.models.autoformer",
    # Квантильные версии
    "quantile_lstm": "app.models.attention_lstm",
    "quantile_tcn": "app.models.tcn",
    "quantile_tft": "app.models.tft",
    "quantile_autoformer": "app.models.autoformer",
}


def get_model(name: str, **params: Any):
    """Создает модель с автоматическим определением квантильного режима"""
    
    if name not in AVAILABLE:
        raise ValueError(f"Unknown model {name}, choose from {list(AVAILABLE)}")
    
    # Определяем режим
    is_quantile = name.startswith("quantile_")
    
    if not is_quantile and "quantiles" in params:
        print(f"[WARNING] Removing 'quantiles' for non-quantile model {name}")
        params.pop("quantiles")
    
    # Для квантильных моделей добавляем n_quantiles
    if is_quantile:
        quantiles = params.get("quantiles", [0.05, 0.5, 0.95])
        params["n_quantiles"] = len(quantiles)
        print(f"[Factory] Quantile mode ON: {quantiles}")
    
    # Загружаем модуль и фильтруем параметры
    module = import_module(AVAILABLE[name])
    sig = inspect.signature(module.build_model)
    accepted_args = sig.parameters.keys()

    filtered_params = {k: v for k, v in params.items() if k in accepted_args}

    model = module.build_model(**filtered_params)
    
    return model