import numpy as np
import torch
from typing import List, Dict, Union


def predict_price(
    model, scaler_X, scaler_y, data, seq_length: int = 20
) -> Union[List[float], Dict[str, List[float]]]:
    """
    Делает мульти-шаговый прогноз:
      — model: обученная модель
      — scaler_X, scaler_y: fitted scaler’ы
      — data: numpy-массив (T, F)
      — seq_length: длина входного окна
    
    Возвращает:
      - Для point-forecast: [float, float, ...]
      - Для quantile-forecast: {"mean": [...], "lower": [...], "upper": [...]}
    """
    if data.shape[0] < seq_length:
        raise ValueError(
            f"Недостаточно данных для последовательности. "
            f"Требуется минимум {seq_length} записей, получено {data.shape[0]}"
        )

    # Берём последние seq_length строк
    sequence = data[-seq_length:]
    num_features = sequence.shape[1]

    # Масштабирование
    sequence_scaled = scaler_X.transform(sequence).reshape(1, seq_length, num_features)
    sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_tensor = model(sequence_tensor)  # [1, horizon] или [1, horizon, Q]

    # Обратное масштабирование
    pred_scaled = pred_tensor.cpu().numpy()
    
    # Определяем режим
    is_quantile = pred_scaled.ndim == 3 and pred_scaled.shape[-1] > 1
    
    if is_quantile:
        # Для квантильных моделей: [1, horizon, Q] -> [horizon, Q]
        pred_scaled = pred_scaled[0]  # [horizon, Q]
        
        preds = []
        for q_idx in range(pred_scaled.shape[1]):
            pred_q = pred_scaled[:, q_idx].reshape(-1, 1)  # [horizon, 1]
            inv_pred = scaler_y.inverse_transform(pred_q)  # [horizon, 1]
            preds.append(inv_pred.flatten())
        
        return {
            "mean": preds[1].tolist(),
            "lower": preds[0].tolist(),
            "upper": preds[-1].tolist(),
        }
    else:
        pred_scaled = pred_scaled.reshape(-1, pred_scaled.shape[-1])
        pred = scaler_y.inverse_transform(pred_scaled)
        
        if not np.isfinite(pred).all():
            raise ValueError("Model returned non-finite prediction.")
        
        return pred[0].tolist()
