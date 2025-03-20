import numpy as np

def predict_price(model, scaler_X, scaler_y, data, seq_length=20):
    if data.shape[0] < seq_length:
        raise ValueError(f"Недостаточно данных для последовательности. Требуется минимум {seq_length} записей, получено {data.shape[0]}")
    
    # Берем последние seq_length записей
    sequence = data[-seq_length:]
    num_features = sequence.shape[1]
    
    # Масштабируем входные данные:
    # Приводим данные к 2D, применяем scaler_X и возвращаем в 3D форму
    sequence_reshaped = sequence.reshape(-1, num_features)
    sequence_scaled = scaler_X.transform(sequence_reshaped).reshape(1, seq_length, num_features)
    
    pred_scaled = model.predict(sequence_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)
    return float(pred[0][0])
