import torch
import torch.nn as nn


class TFTModel(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_features: int,
        horizon: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(seq_length, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        z = self.input_proj(x) + self.pos_emb.unsqueeze(0)  # [B, T, D]
        z = self.transformer(z)  # [B, T, D]
        last = z[:, -1, :]  # [B, D]
        return self.head(last)  # [B, horizon]


def build_model(
    seq_length: int, num_features: int, horizon: int, **arch_kwargs
) -> nn.Module:
    return TFTModel(
        seq_length=seq_length, num_features=num_features, horizon=horizon, **arch_kwargs
    )
