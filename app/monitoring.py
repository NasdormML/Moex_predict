import mlflow
import pandas as pd


def validate_model_performance(
    ticker: str,
    real_data: pd.DataFrame,
    prediction_history: pd.DataFrame,
    threshold: float = 0.05,
) -> bool:
    """
    Валидирует производительность модели с учетом квантильных прогнозов.
    
    Для квантильных моделей проверяет:
    1. Точность среднего прогноза (MAPE)
    2. Покрытие интервалов (Coverage)
    
    Returns:
        True если модель проходит валидацию
    """
    
    real_data["TRADEDATE"] = pd.to_datetime(real_data["TRADEDATE"]).dt.strftime("%Y-%m-%d")
    prediction_history["DATE"] = pd.to_datetime(prediction_history["DATE"]).dt.strftime("%Y-%m-%d")

    merged = pd.merge(
        real_data.rename(columns={"CLOSE": "real_price"}),
        prediction_history,
        left_on="TRADEDATE",
        right_on="DATE",
        how="inner"
    )
    if merged.empty:
        print(f"[Monitor] Нет совпадений дат для {ticker}.")
        return True

    # Определяем тип модели (point vs quantile)
    is_quantile = "predicted_lower" in merged.columns and "predicted_upper" in merged.columns
    print(f"[Monitor] Validation for {ticker} (quantile={is_quantile})")

    # Точность среднего прогноза
    merged["pct_error"] = (merged["predicted_price"] - merged["real_price"]).abs() / merged["real_price"]
    mean_error = merged["pct_error"].mean()
    mlflow.log_metric(f"{ticker}_mape", mean_error)
    print(f"[Monitor] MAPE for {ticker}: {mean_error:.4f}")

    # Проверка Coverage (только для квантильных)
    coverage_ok = True
    if is_quantile:
        coverage_lower = (merged["real_price"] < merged["predicted_lower"]).mean()
        coverage_upper = (merged["real_price"] <= merged["predicted_upper"]).mean()
        
        # lower ~5%, upper ~95%
        lower_ok = abs(coverage_lower - 0.05) <= 0.02
        upper_ok = abs(coverage_upper - 0.95) <= 0.02
        
        mlflow.log_metrics({
            f"{ticker}_coverage_lower": coverage_lower,
            f"{ticker}_coverage_upper": coverage_upper,
            f"{ticker}_coverage_error": abs(coverage_lower - 0.05) + abs(coverage_upper - 0.95)
        })
        
        print(f"[Monitor] Coverage: {coverage_lower:.1%} (target 5%) / {coverage_upper:.1%} (target 95%)")
        print(f"[Monitor] Coverage OK: lower={lower_ok}, upper={upper_ok}")
        
        coverage_ok = lower_ok and upper_ok

    # === 3. Итоговое решение ===
    accuracy_ok = mean_error <= threshold
    overall_ok = accuracy_ok and coverage_ok
    
    mlflow.log_metric(f"{ticker}_validation_passed", int(overall_ok))
    
    if not overall_ok:
        print(f"[Monitor] ⚠️ Validation failed for {ticker}")
        if not accuracy_ok:
            print(f"  → MAPE {mean_error:.4f} > threshold {threshold}")
        if not coverage_ok:
            print(f"  → Coverage out of range")
    else:
        print(f"[Monitor] ✅ Validation passed for {ticker}")
    
    return overall_ok


# Тест
if __name__ == "__main__":
    ticker = "SBER"
    
    real_df = pd.DataFrame({
        "TRADEDATE": pd.date_range("2025-01-01", periods=5),
        "CLOSE": [300, 305, 310, 308, 312]
    })
    
    # Предсказания (quantile)
    pred_df = pd.DataFrame({
        "DATE": pd.date_range("2025-01-01", periods=5),
        "predicted_price": [302, 306, 309, 307, 311],
        "predicted_lower": [295, 300, 305, 302, 306],
        "predicted_upper": [310, 315, 318, 315, 318]
    })
    
    # Запуск валидации
    result = validate_model_performance(ticker, real_df, pred_df, threshold=0.05)
    print(f"Test result: {result}")