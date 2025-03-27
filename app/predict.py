import numpy as np
import torch

def predict_price(model, scaler_X, scaler_y, data, seq_length=20):
    if data.shape[0] < seq_length:
        raise ValueError(f"Недостаточно данных для последовательности. Требуется минимум {seq_length} записей, получено {data.shape[0]}")
    
    sequence = data[-seq_length:]
    num_features = sequence.shape[1]
    
    # Масштабирование последовательности
    sequence_scaled = scaler_X.transform(sequence).reshape(1, seq_length, num_features)
    
    sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_tensor = model(sequence_tensor)
    pred_scaled = pred_tensor.cpu().numpy()
    pred = scaler_y.inverse_transform(pred_scaled)
    return float(pred[0][0])
