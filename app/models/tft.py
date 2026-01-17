import torch
import torch.nn as nn
from typing import Optional


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
        n_quantiles: Optional[int] = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        
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

        output_dim = horizon * n_quantiles if n_quantiles else horizon
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        z = self.input_proj(x) + self.pos_emb.unsqueeze(0)  # [B, T, D]
        z = self.transformer(z)  # [B, T, D]
        last = z[:, -1, :]  # [B, D]
        
        output = self.head(last)
        
        if self.n_quantiles:
            output = output.view(-1, self.horizon, self.n_quantiles)  # [B, H, Q]
        
        return output


def build_model(
    seq_length: int,
    num_features: int,
    horizon: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    dropout: float = 0.1,
    n_quantiles: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    
    return TFTModel(
        seq_length=seq_length,
        num_features=num_features,
        horizon=horizon,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        n_quantiles=n_quantiles,
    )
