import pandas as pd
import mlflow

def validate_model_performance(ticker: str, real_data: pd.DataFrame, prediction_history: pd.DataFrame, threshold: float = 0.05) -> bool:
    """
    Сравнивает фактические данные и историю предсказаний для тикера.
    
    real_data должен содержать столбцы:
      - TRADEDATE: даты торгов (в формате YYYY-MM-DD)
      - close: фактическая цена закрытия
      
    prediction_history должен содержать:
      - TRADEDATE: даты, для которых были сделаны предсказания
      - predicted_price: предсказанные цены модели

    Функция вычисляет среднюю абсолютную процентную ошибку (MAPE) и логирует её через MLflow под именем <ticker>_mean_pct_error.
    Если MAPE меньше или равна threshold, возвращается True (качество удовлетворительное), иначе – False.
    """

    real_data['TRADEDATE'] = pd.to_datetime(real_data['TRADEDATE']).dt.strftime("%Y-%m-%d")
    prediction_history['TRADEDATE'] = pd.to_datetime(prediction_history['TRADEDATE']).dt.strftime("%Y-%m-%d")
    
    # Объединяем данные по дате
    merged = pd.merge(real_data, prediction_history, on='TRADEDATE', how='inner')
    if merged.empty:
        print("Нет совпадений дат между реальными данными и предсказаниями.")
        return True

    # Вычисляем абсолютное процентное отклонение и среднее отклонение
    merged["pct_error"] = (merged["predicted_price"] - merged["CLOSE"]).abs() / merged["CLOSE"]
    mean_error = merged["pct_error"].mean()
    
    mlflow.log_metric(f"{ticker}_mean_pct_error", mean_error)
    print(f"Средняя процентная ошибка для {ticker}: {mean_error:.4f}")
    
    return mean_error <= threshold