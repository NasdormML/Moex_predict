import os
import pickle
import torch
from app.models import PricePredictionModel

def load_models():
    models_dict = {}
    ticker = "SBER"
    model_path = os.path.join("models", f"{ticker}_model.pth")
    scaler_X_path = os.path.join("models", f"{ticker}_scaler_X.pkl")
    scaler_y_path = os.path.join("models", f"{ticker}_scaler_y.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
        # Параметры, использованные при обучении
        seq_length = 20
        num_features = 18  # число признаков, как в подготовленных данных
        output_dim = 1
        lstm_units = 196
        fc_units = 151
        dropout_rate = 0.13351299266216415
        
        model = PricePredictionModel(seq_length, num_features, output_dim, lstm_units, fc_units, dropout_rate)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        with open(scaler_X_path, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)
        models_dict[ticker] = {"model": model, "scaler_X": scaler_X, "scaler_y": scaler_y}
    else:
        print("Модель или скейлеры не найдены в папке models.")
    
    return models_dict
