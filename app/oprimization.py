import optuna
from tensorflow.keras.callbacks import EarlyStopping
from app.models import build_model

def optimize_model(X_train, y_train, X_val, y_val, seq_length, num_features, n_trials=30):
    """
    Выполняет подбор гиперпараметров с использованием Optuna.
    
    Parameters:
        X_train, y_train: Обучающие данные.
        X_val, y_val: Данные для валидации.
        seq_length (int): Длина входной последовательности.
        num_features (int): Число признаков.
        n_trials (int): Количество испытаний.
        
    Returns:
        best_params (dict): Лучшие гиперпараметры.
        study: Объект исследования Optuna.
    """
    def objective(trial):
        lstm_units = trial.suggest_int("lstm_units", 64, 256, step=32)
        gru_units = trial.suggest_int("gru_units", 32, 128, step=16)
        dense_units = trial.suggest_int("dense_units", 32, 128, step=16)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        
        model = build_model(seq_length, num_features, lstm_units, gru_units, dense_units, dropout_rate, learning_rate)
        cb = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, callbacks=cb, verbose=0)
        return min(history.history["val_loss"])
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    return best_params, study
