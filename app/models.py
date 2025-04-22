import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, timesteps):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_dim,1))
        self.b = nn.Parameter(torch.zeros(timesteps,1))
        nn.init.xavier_uniform_(self.W)
    def forward(self,x):
        e = torch.matmul(x,self.W)+self.b
        alpha = torch.softmax(e.squeeze(-1),dim=1).unsqueeze(-1)
        return torch.sum(x*alpha,dim=1)

class PricePredictionModel(nn.Module):
    def __init__(self,seq_length,num_features,output_dim,lstm_units,fc_units,dropout):
        super().__init__()
        self.lstm = nn.LSTM(num_features,lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionLayer(lstm_units,seq_length)
        self.fc1 = nn.Linear(lstm_units,fc_units)
        self.fc2 = nn.Linear(fc_units,output_dim)
        self.relu= nn.ReLU()
    def forward(self,x):
        o,_=self.lstm(x)
        o=self.dropout(o)
        a=self.attn(o)
        h=self.relu(self.fc1(a))
        return self.fc2(self.dropout(h))


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