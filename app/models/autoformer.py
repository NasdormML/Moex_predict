import torch
import torch.nn as nn


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor):
        # x: [B, F, T]
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class AutoformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel_size=25)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.decomp2 = SeriesDecomp(kernel_size=25)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        xf = x.transpose(1, 2)  # [B, D, T]
        trend1, seas1 = self.decomp1(xf)
        x1 = (trend1 + seas1).transpose(1, 2)
        attn_out, _ = self.attn(x1, x1, x1)
        x2 = self.norm1(attn_out + x1)
        ff_out = self.norm2(self.ffn(x2) + x2)
        xf2 = ff_out.transpose(1, 2)
        trend2, seas2 = self.decomp2(xf2)
        return (trend2 + seas2).transpose(1, 2)


class AutoformerModel(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_features: int,
        horizon: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.layers = nn.ModuleList(
            [
                AutoformerLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        last = h[:, -1, :]  # [B, D]
        return self.head(last)  # [B, horizon]


def build_model(
    seq_length: int, num_features: int, horizon: int, **arch_kwargs
) -> nn.Module:
    return AutoformerModel(
        seq_length=seq_length, num_features=num_features, horizon=horizon, **arch_kwargs
    )
