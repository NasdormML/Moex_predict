import inspect
from importlib import import_module

AVAILABLE = {
    "lstm": "app.models.attention_lstm",
    "tcn": "app.models.tcn",
    "tft": "app.models.tft",
    "autoformer": "app.models.autoformer",
}


def get_model(name: str, **params):
    if name not in AVAILABLE:
        raise ValueError(f"Unknown model {name}, choose from {list(AVAILABLE)}")

    module = import_module(AVAILABLE[name])
    sig = inspect.signature(module.build_model)
    accepted_args = sig.parameters.keys()
    filtered_params = {k: v for k, v in params.items() if k in accepted_args}

    return module.build_model(**filtered_params)
