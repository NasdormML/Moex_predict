import mlflow
import pandas as pd


def validate_model_performance(
    ticker: str,
    real_data: pd.DataFrame,
    prediction_history: pd.DataFrame,
    threshold: float = 0.05,
) -> bool:
    real_data["TRADEDATE"] = pd.to_datetime(real_data["TRADEDATE"]).dt.strftime(
        "%Y-%m-%d"
    )
    prediction_history["TRADEDATE"] = pd.to_datetime(
        prediction_history["TRADEDATE"]
    ).dt.strftime("%Y-%m-%d")

    merged = pd.merge(real_data, prediction_history, on="TRADEDATE", how="inner")
    if merged.empty:
        print("Нет совпадений дат.")
        return True

    merged["pct_error"] = (merged["predicted_price"] - merged["close"]).abs() / merged[
        "close"
    ]
    mean_error = merged["pct_error"].mean()
    mlflow.log_metric(f"{ticker}_mean_pct_error", mean_error)
    print(f"MAPE для {ticker}: {mean_error:.4f}")
    return mean_error <= threshold
