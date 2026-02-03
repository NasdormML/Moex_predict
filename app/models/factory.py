import inspect
import logging
from functools import lru_cache
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)

AVAILABLE: dict[str, str] = {
    "lstm": "app.models.attention_lstm",
    "tcn": "app.models.tcn",
    "tft": "app.models.tft",
    "autoformer": "app.models.autoformer",
    "quantile_lstm": "app.models.attention_lstm",
    "quantile_tcn": "app.models.tcn",
    "quantile_tft": "app.models.tft",
    "quantile_autoformer": "app.models.autoformer",
}


@lru_cache(maxsize=32)
def _get_model_signature(module_path: str):
    """Cache model signatures."""
    module = import_module(module_path)
    return inspect.signature(module.build_model)


def get_model(name: str, **params: Any):
    """
    Create model with automatic parameter filtering.
    """
    if name not in AVAILABLE:
        raise ValueError(f"Unknown model {name}")

    is_quantile = name.startswith("quantile_")

    # Handle quantiles
    if not is_quantile and "quantiles" in params:
        params = {k: v for k, v in params.items() if k != "quantiles"}

    if is_quantile:
        quantiles = params.get("quantiles", [0.05, 0.5, 0.95])
        params["n_quantiles"] = len(quantiles)

    # FIX: фильтруем параметры ДО логирования, warning только если реально нужно
    module_path = AVAILABLE[name]
    sig = _get_model_signature(module_path)
    accepted = set(sig.parameters.keys())

    # Разделяем на accepted и rejected
    accepted_params = {}
    rejected_params = {}

    for k, v in params.items():
        if k in accepted:
            accepted_params[k] = v
        else:
            rejected_params[k] = v

    # Логируем только нестандартные rejected (не output_dim и т.п.)
    ignored_standard = {"output_dim", "input_dim", "num_features"}  # частые legacy
    non_standard_rejected = set(rejected_params.keys()) - ignored_standard

    if non_standard_rejected:
        logger.warning(
            f"Ignored unknown parameters for {name}: {non_standard_rejected}"
        )

    # Логируем отладку только при DEBUG
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Building {name} with params: {accepted_params}")

    module = import_module(module_path)
    return module.build_model(**accepted_params)
