import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class QuantileLoss(nn.Module):
    """
    Quantile Loss для многоквантильного прогноза.
    Поддерживает target размерностью [B, H] или [B, H, 1].
    """
    
    def __init__(self, quantiles: List[float] = [0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, H, Q] или [B, H]
            target: [B, H] или [B, H, 1]
        
        Returns:
            Средний loss по всем квантилям
        """

        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(-1)
        
        # Если target [B, H], делаем [B, H, 1]
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        
        # Теперь оба тензора 3D: [B, H, Q] и [B, H, 1]
        # Расширяем target до [B, H, Q]
        target = target.expand(-1, -1, predictions.shape[-1])
        
        # Считаем ошибку
        errors = target - predictions
        
        # Quantile loss для каждого квантиля
        losses = []
        for i, q in enumerate(self.quantiles):
            q_tensor = torch.tensor(q, device=predictions.device, dtype=predictions.dtype)
            loss = torch.max((q_tensor - 1) * errors[..., i], q_tensor * errors[..., i])
            losses.append(loss.mean())
        
        return torch.stack(losses).mean()


class CoverageMetric:
    """
    Метрика покрытия (Coverage) для валидации квантильных прогнозов.
    """
    
    @staticmethod
    def calculate_coverage(
        predictions: torch.Tensor, 
        target: torch.Tensor, 
        quantiles: List[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        if predictions.dim() != 3 or predictions.shape[-1] != len(quantiles):
            return None, None
        
        if target.dim() == 3:
            target = target.squeeze(-1)
        
        lower_bound = predictions[:, :, 0]  # [B, H]
        upper_bound = predictions[:, :, -1]  # [B, H]
        target = target.squeeze(-1)  # [B, H]
        
        # Расчет покрытия
        coverage_lower = (target < lower_bound).float().mean().item()
        coverage_upper = (target <= upper_bound).float().mean().item()
        
        return coverage_lower, coverage_upper
