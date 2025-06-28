import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, timesteps):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.b = nn.Parameter(torch.zeros(timesteps, 1))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        e = torch.matmul(x, self.W) + self.b  # [B, T, 1]
        alpha = torch.softmax(e.squeeze(-1), dim=1).unsqueeze(-1)  # [B, T, 1]
        return torch.sum(x * alpha, dim=1)  # [B, H]


class LSTMModel(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_features: int,
        horizon: int,
        lstm_units: int = 128,
        fc_units: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(num_features, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = AttentionLayer(lstm_units, seq_length)
        self.fc1 = nn.Linear(lstm_units, fc_units)
        self.fc2 = nn.Linear(fc_units, horizon)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        o, _ = self.lstm(x)  # [B, T, H]
        o = self.dropout(o)
        a = self.attn(o)  # [B, H]
        h = self.relu(self.fc1(a))
        return self.fc2(self.dropout(h))  # [B, horizon]


def build_model(
    seq_length: int, num_features: int, horizon: int, **arch_kwargs
) -> nn.Module:
    return LSTMModel(
        seq_length=seq_length, num_features=num_features, horizon=horizon, **arch_kwargs
    )
