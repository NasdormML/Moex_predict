import numpy as np
import torch


def predict_price(model, scaler_X, scaler_y, data, seq_length: int = 20):
    """
    Делает мульти-шаговый прогноз:
      — model: обученная модель, возвращающая тензор [B, horizon]
      — scaler_X, scaler_y: fitted scaler’ы для X и Y
      — data: numpy-массив формы (T, F)
      — seq_length: длина входного окна
    Возвращает список предсказаний длины horizon.
    """
    if data.shape[0] < seq_length:
        raise ValueError(
            f"Недостаточно данных для последовательности. "
            f"Требуется минимум {seq_length} записей, получено {data.shape[0]}"
        )

    # Берём последние seq_length строк
    sequence = data[-seq_length:]
    num_features = sequence.shape[1]

    # Масштабирование и подготовка тензора
    sequence_scaled = scaler_X.transform(sequence).reshape(1, seq_length, num_features)
    sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_tensor = model(sequence_tensor)  # [1, horizon]

    pred_scaled = pred_tensor.cpu().numpy()  # shape (1, horizon)
    pred = scaler_y.inverse_transform(pred_scaled)  # всё ещё shape (1, horizon)

    if not np.isfinite(pred).all():
        raise ValueError("Model returned non-finite prediction.")

    # Вернём простой список float
    return pred[0].tolist()
