import torch
import torch.nn as nn

# ==========================
# Модель LSTM с attention
# ==========================

class AttentionLayer(nn.Module):
    """
    Слой внимания (Attention Layer) для объединения временных шагов.
    """
    def __init__(self, hidden_dim, timesteps):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        self.W = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.b = nn.Parameter(torch.zeros(timesteps, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, x):
        # x: (batch, timesteps, hidden_dim)
        e = torch.matmul(x, self.W) + self.b  # (batch, timesteps, 1)
        e = e.squeeze(-1)                     # (batch, timesteps)
        alpha = torch.softmax(e, dim=1)       # (batch, timesteps)
        alpha = alpha.unsqueeze(-1)           # (batch, timesteps, 1)
        context = torch.sum(x * alpha, dim=1)   # (batch, hidden_dim)
        return context

class PricePredictionModel(nn.Module):
    """
    Модель для предсказания цены на основе LSTM с attention.
    
    Параметры:
    - seq_length: длина входной последовательности.
    - num_features: число входных признаков.
    - output_dim: размер выхода (обычно 1).
    - lstm_units: число скрытых единиц в слое LSTM.
    - fc_units: число нейронов в полносвязном слое.
    - dropout_rate: вероятность dropout-а.
    """
    def __init__(self, seq_length, num_features, output_dim, lstm_units, fc_units, dropout_rate):
        super(PricePredictionModel, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_units, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = AttentionLayer(hidden_dim=lstm_units, timesteps=seq_length)
        self.fc1 = nn.Linear(lstm_units, fc_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_units, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_length, num_features)
        lstm_out, _ = self.lstm(x)        # (batch, seq_length, lstm_units)
        x = self.dropout(lstm_out)
        attn_out = self.attention(x)      # (batch, lstm_units)
        x = self.fc1(attn_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================
# Модель TCN (Temporal Convolutional Network)
# ==========================

class Chomp1d(nn.Module):
    """
    Класс для «обрезания» лишних значений после свёрточных операций.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        # x имеет форму (batch, channels, sequence_length)
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Базовый временной блок TCN с двумя свёрточными слоями, BatchNorm, ReLU и dropout.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(self.bn1(out))
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu(self.bn2(out))
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Реализация TCN, где слои строятся последовательно согласно заданной конфигурации.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1,
                                       dilation=dilation_size,
                                       padding=(kernel_size-1)*dilation_size,
                                       dropout=dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, num_inputs, seq_length)
        return self.network(x)

class TCNModel(nn.Module):
    """
    Модель на основе TCN для предсказания цены.
    
    Параметры:
    - num_features: число входных признаков.
    - num_channels: список с числом каналов для каждого временного блока.
    - kernel_size: размер ядра свёртки.
    - dropout: вероятность dropout.
    - fc_units: число нейронов в полносвязном слое.
    """
    def __init__(self, num_features, num_channels, kernel_size=2, dropout=0.2, fc_units=32):
        super(TCNModel, self).__init__()
        self.tcn = TCN(num_inputs=num_features, num_channels=num_channels,
                       kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], fc_units)
        self.out = nn.Linear(fc_units, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, seq_length, num_features)
        x = x.permute(0, 2, 1)            # (batch, num_features, seq_length)
        tcn_out = self.tcn(x)             # (batch, last_channel, seq_length_after_chomp)
        last_out = tcn_out[:, :, -1]      # Выбираем выход последнего временного шага
        x = self.relu(self.fc(last_out))
        return self.out(x)