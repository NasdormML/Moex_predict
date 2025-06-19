import torch

from app.models.factory import get_model


def test_factory_returns_model():
    model = get_model("lstm", input_size=5)
    assert hasattr(model, "forward")
    x = torch.randn(2, 10, 5)
    out = model(x)
    assert out.shape == (2, 1)
