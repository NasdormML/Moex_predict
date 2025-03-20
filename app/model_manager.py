import os
import pickle
import tensorflow as tf
from app.models import AttentionLayer

def load_models():
    models_dict = {}
    ticker = "SBER"
    model_path = os.path.join("models", f"{ticker}_final_model.keras")
    scaler_X_path = os.path.join("models", f"{ticker}_scaler_X.pkl")
    scaler_y_path = os.path.join("models", f"{ticker}_scaler_y.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
        model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        with open(scaler_X_path, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)
        models_dict[ticker] = {"model": model, "scaler_X": scaler_X, "scaler_y": scaler_y}
    else:
        print("Модель или скейлеры не найдены в папке models.")
    
    return models_dict
