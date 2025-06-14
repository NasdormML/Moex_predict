import torch
import torch.nn as nn


class TFTModel(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        # Проекцируем входные признаки в размерность модели
        self.input_proj = nn.Linear(num_features, d_model)
        # Позиционные эмбеддинги
        self.pos_emb = nn.Parameter(torch.randn(seq_length, d_model))
        # Transformer-энкодер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # PyTorch Transformer expects seq, batch, feature
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Проекция и добавление позиционных эмбеддингов
        x = self.input_proj(x) + self.pos_emb.unsqueeze(0)  # -> [B, T, D]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        out = self.decoder(x[-1])
        return out.view(-1, 1)


def build_model(
    seq_length: int,
    num_features: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
) -> nn.Module:
    """
    Фабричная функция для создания TFTModel.
    """
    return TFTModel(
        seq_length=seq_length,
        num_features=num_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )
