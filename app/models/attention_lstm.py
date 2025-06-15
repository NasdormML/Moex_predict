import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, timesteps):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.b = nn.Parameter(torch.zeros(timesteps, 1))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        e = torch.matmul(x, self.W) + self.b
        alpha = torch.softmax(e.squeeze(-1), dim=1).unsqueeze(-1)
        return torch.sum(x * alpha, dim=1)


class LSTMModel(nn.Module):
    def __init__(
        self, seq_length, num_features, output_dim, lstm_units, fc_units, dropout_rate
    ):
        super().__init__()
        self.lstm = nn.LSTM(num_features, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = AttentionLayer(lstm_units, seq_length)
        self.fc1 = nn.Linear(lstm_units, fc_units)
        self.fc2 = nn.Linear(fc_units, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        o, _ = self.lstm(x)
        o = self.dropout(o)
        a = self.attn(o)
        h = self.relu(self.fc1(a))
        return self.fc2(self.dropout(h))


def build_model(**kwargs):
    return LSTMModel(**kwargs)
