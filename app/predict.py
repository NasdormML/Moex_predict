import pickle
import numpy as np
import tensorflow as tf
from app.models import AttentionLayer

def load_model_and_scalers(ticker):
    """
    Загружает модель и объекты масштабирования для заданного тикера.
    
    Ожидается, что артефакты сохранены в папке models/ с именами вида:
      - {ticker}_final_model.h5
      - {ticker}_scaler_X.pkl
      - {ticker}_scaler_y.pkl
    """
    model_path = f"models/{ticker}_final_model.h5"
    scaler_X_path = f"models/{ticker}_scaler_X.pkl"
    scaler_y_path = f"models/{ticker}_scaler_y.pkl"
    
    model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    
    with open(scaler_X_path, "rb") as f:
         scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
         scaler_y = pickle.load(f)
    
    return model, scaler_X, scaler_y

def predict_price(model, scaler_X, scaler_y, data, seq_length=20):
    """
    Выполняет предсказание цены.
    
    Parameters:
        model: Загруженная модель TensorFlow.
        scaler_X: Объект масштабирования для признаков.
        scaler_y: Объект масштабирования для целевой переменной.
        data (np.array): Массив признаков (предобработанный).
        seq_length (int): Длина последовательности для инференса.
        
    Returns:
        float: Предсказанная цена.
    """
    # Для инференса выбираем последние seq_length записей
    sequence = data[-seq_length:]
    sequence = np.expand_dims(sequence, axis=0)
    pred_scaled = model.predict(sequence)
    pred = scaler_y.inverse_transform(pred_scaled)
    return float(pred[0][0])
