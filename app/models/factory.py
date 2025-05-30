from importlib import import_module

AVAILABLE = {
  "lstm": "app.models.attention_lstm",
  "tcn":  "app.models.tcn",
  "tft":  "app.models.tft",
}

def get_model(name: str, **params):
    if name not in AVAILABLE:
        raise ValueError(f"Unknown model {name}, choose from {list(AVAILABLE)}")
    module = import_module(AVAILABLE[name])
    return module.build_model(**params)
