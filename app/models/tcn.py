import torch
import torch.nn as nn
from typing import Optional


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class TCNModel(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_features: int,
        horizon: int,
        num_channels: list[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        fc_units: int = 64,
        n_quantiles: Optional[int] = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        
        # TCN блоки
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_features if i == 0 else num_channels[i - 1]
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], fc_units)
        
        output_dim = horizon * n_quantiles if n_quantiles else horizon
        self.out = nn.Linear(fc_units, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        y = self.tcn(x)  # [B, C, T]
        last = y[:, :, -1]  # [B, C]
        h = self.relu(self.fc(last))
        
        output = self.out(h)
        
        # 🔥 РЕШАПИНГ для квантилей
        if self.n_quantiles:
            output = output.view(-1, self.horizon, self.n_quantiles)  # [B, H, Q]
        
        return output  # [B, H] или [B, H, Q]


def build_model(
    seq_length: int,
    num_features: int,
    horizon: int,
    num_channels: list[int],
    kernel_size: int = 2,
    dropout: float = 0.2,
    fc_units: int = 64,
    n_quantiles: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    
    return TCNModel(
        seq_length=seq_length,
        num_features=num_features,
        horizon=horizon,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        fc_units=fc_units,
        n_quantiles=n_quantiles,
    )