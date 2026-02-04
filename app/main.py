import asyncio
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Annotated, Optional

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from app.data import fetch_moex_eod_data, fetch_usd_series
from app.model_manager import ModelManager
from app.predict import predict_price
from app.preprocessing import preprocess_data

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------
# MLflow setup
# -------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001").strip()
mlflow.set_tracking_uri(MLFLOW_URI)

try:
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "MOEX_Price_Prediction"))
except Exception as e:
    logger.warning(f"Failed to set MLflow experiment: {e}")

model_manager: Optional[ModelManager] = None
history_dir = os.getenv("HISTORY_DIR", "history")
os.makedirs(history_dir, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
async def _maybe_async_call(func, *args, **kwargs):
    """Call func - if coroutine, await it; otherwise run in thread."""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def _business_days_between(start: date, end: date) -> int:
    """Count business days between dates (exclusive of start)."""
    if end <= start:
        return 0
    import numpy as np

    return np.busday_count(start.isoformat(), end.isoformat())


_history_locks: dict[str, asyncio.Lock] = {}


async def _append_history_async(ticker: str, dates: list[date], preds: list[float]):
    """Write CSV safely with per-ticker locking."""
    if ticker not in _history_locks:
        _history_locks[ticker] = asyncio.Lock()

    async with _history_locks[ticker]:
        df = pd.DataFrame(
            {"DATE": [d.isoformat() for d in dates], "predicted_price": preds}
        )
        pf = os.path.join(history_dir, f"predictions_{ticker}.csv")

        def _write():
            from filelock import FileLock

            lock_path = f"{pf}.lock"
            with FileLock(lock_path):
                header = not os.path.exists(pf) or os.path.getsize(pf) == 0
                with open(pf, "a", newline="") as f:
                    df.to_csv(f, header=header, index=False)

        await asyncio.to_thread(_write)


async def run_training(
    ticker: str,
    model: str = "lstm",
    start_date: str = "2013-01-01",
    epochs: int = 50,
    version: str = "v1",
    enable_hpo: bool = False,
    n_trials: int = 10,
):
    """Run training in background subprocess."""
    cmd = [
        sys.executable,
        "train.py",
        f"model={model}",
        f"data.ticker={ticker}",
        f"data.start_date={start_date}",
        f"train.epochs={epochs}",
        f"train.version={version}",
    ]
    
    if enable_hpo:
        cmd.extend([
            "optimization.enable=true",
            f"optimization.n_trials={n_trials}",
            f"optimization.epochs_per_trial={max(10, epochs // 2)}",
        ])
    
    logger.info(f"Starting training: {' '.join(cmd)}")
    
    def _train():
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd="/app",
            )
            logger.info(f"Training completed with code {result.returncode}")
            if result.returncode != 0:
                logger.error(f"Training stderr: {result.stderr}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            return False
        except Exception as e:
            logger.exception("Training failed")
            return False
    
    return await asyncio.to_thread(_train)


# -------------------------
# Lifecycle
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown."""
    global model_manager
    logger.info("Initializing ModelManager...")

    model_manager = ModelManager()
    try:
        await _maybe_async_call(model_manager.load_all)
        logger.info(f"Loaded {len(model_manager)} models")
    except Exception as e:
        logger.warning(f"No models loaded: {e}. Training available via /train endpoint.")
        os.makedirs(os.getenv("MODEL_ARTIFACTS_DIR", "/data/models"), exist_ok=True)

    try:
        yield
    finally:
        if model_manager and hasattr(model_manager, "shutdown"):
            try:
                await _maybe_async_call(model_manager.shutdown)
            except Exception:
                logger.exception("Shutdown error")
        model_manager = None
        logger.info("Shutdown complete")


app = FastAPI(
    title="MOEX Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------
# Validation
# -------------------------
def validate_ticker(t: str) -> str:
    t = t.upper().strip()
    if not t.isalnum() or len(t) > 10:
        raise HTTPException(400, f"Invalid ticker: {t}")
    return t


def get_last_known_date(metadata: dict) -> date:
    data_upto = metadata.get("data_upto")
    if data_upto:
        try:
            return datetime.strptime(data_upto, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Invalid data_upto: %r", data_upto)
    return date.today() - timedelta(days=1)


async def fetch_market_data(
    ticker: str, last_known: date, seq_length: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window = seq_length * 2
    start = (last_known - timedelta(days=window)).isoformat()
    end = last_known.isoformat()

    tasks = [
        _maybe_async_call(
            fetch_moex_eod_data, ticker, "stock", "shares", "TQBR", start, end
        ),
        _maybe_async_call(
            fetch_moex_eod_data, "IMOEX", "stock", "index", "SNDX", start, end
        ),
        _maybe_async_call(fetch_usd_series, start, end),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    names = (ticker, "IMOEX", "USD")
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            logger.error("Fetch %s failed: %s", name, res)
            raise HTTPException(503, f"Data source error: {name}")
        if res is None or (hasattr(res, "empty") and res.empty):
            raise HTTPException(503, f"No data: {name}")

    return results[0], results[1], results[2]


def prepare_dataframe(
    df: pd.DataFrame, rename_map: dict, columns: list[str]
) -> pd.DataFrame:
    if df is None or df.empty:
        raise HTTPException(503, "Empty dataframe")

    df = df.copy()
    date_col = "BEGIN" if "BEGIN" in df.columns else "TRADEDATE"
    if date_col not in df.columns:
        raise HTTPException(500, "No date column")

    df["TRADEDATE"] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.rename(columns=rename_map)

    missing = set(columns) - set(df.columns)
    if missing:
        raise HTTPException(500, f"Missing columns: {missing}")

    return df[columns]


def merge_dataframes(df_t, df_i, df_u, ticker: str) -> pd.DataFrame:
    t_cols = [
        "TRADEDATE",
        f"OPEN_{ticker}",
        f"HIGH_{ticker}",
        f"LOW_{ticker}",
        f"CLOSE_{ticker}",
        f"VOL_{ticker}",
    ]

    df_t = prepare_dataframe(
        df_t,
        {
            "OPEN": f"OPEN_{ticker}",
            "HIGH": f"HIGH_{ticker}",
            "LOW": f"LOW_{ticker}",
            "CLOSE": f"CLOSE_{ticker}",
            "VOLUME": f"VOL_{ticker}",
        },
        t_cols,
    )

    df_i = prepare_dataframe(
        df_i, {"CLOSE": "CLOSE_IMOEX"}, ["TRADEDATE", "CLOSE_IMOEX"]
    )
    df_u = prepare_dataframe(df_u, {"CLOSE": "CLOSE_USD"}, ["TRADEDATE", "CLOSE_USD"])

    merged = (
        df_t.merge(df_i, on="TRADEDATE", how="outer")
        .merge(df_u, on="TRADEDATE", how="outer")
        .sort_values("TRADEDATE")
        .ffill()
        .dropna()
        .reset_index(drop=True)
    )
    return merged


def _log_to_mlflow(ticker: str, bundle: dict, seq: int, X_rows: int, preds: list):
    """Background task for MLflow logging."""
    try:
        import urllib.request

        parsed = urllib.parse.urlparse(MLFLOW_URI)
        test_uri = f"{parsed.scheme}://{parsed.netloc}/health"

        try:
            urllib.request.urlopen(test_uri, timeout=2)
        except Exception:
            logger.debug("MLflow server not reachable, skipping logging")
            return

        with mlflow.start_run(
            run_name=f"predict_{ticker}_{datetime.utcnow().isoformat()}"
        ):
            mlflow.set_tag("model_version", bundle.get("model_version"))
            mlflow.set_tag("ticker", ticker)
            mlflow.log_param("seq_length", seq)
            mlflow.log_param("n_input_rows", X_rows)
            for idx, val in enumerate(preds, start=1):
                mlflow.log_metric(f"pred_step_{idx}", val)
    except Exception:
        logger.exception("MLflow logging failed")


# -------------------------
# Training Endpoints
# -------------------------
@app.post("/train/{ticker}")
async def train_model(
    ticker: Annotated[str, Path(...)],
    model: str = Query("lstm", enum=["lstm", "tcn", "tft"]),
    start_date: str = Query("2013-01-01"),
    epochs: int = Query(50, ge=10, le=200),
    version: str = Query("v1"),
    enable_hpo: bool = Query(False),
    n_trials: int = Query(10, ge=5, le=50),
    background_tasks: BackgroundTasks = None,
):
    """
    Train new model for ticker.
    
    - **ticker**: Stock ticker (SBER, GAZP, etc.)
    - **model**: Model architecture (lstm, tcn, tft)
    - **start_date**: Historical data start date
    - **epochs**: Training epochs
    - **version**: Model version tag
    - **enable_hpo**: Enable Optuna hyperparameter optimization
    - **n_trials**: Number of HPO trials (if HPO enabled)
    """
    ticker = validate_ticker(ticker)
    
    # Проверяем доступность данных
    try:
        test_df = await _maybe_async_call(
            fetch_moex_eod_data, ticker, "stock", "shares", "TQBR", 
            (date.today() - timedelta(days=30)).isoformat(), 
            date.today().isoformat()
        )
        if test_df is None or test_df.empty:
            raise HTTPException(404, f"No market data for ticker: {ticker}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data check failed: {e}")
        raise HTTPException(503, "Market data unavailable")

    # Запускаем в фоне
    if background_tasks:
        background_tasks.add_task(
            run_training, ticker, model, start_date, epochs, version, enable_hpo, n_trials
        )
        return {
            "status": "training_started",
            "ticker": ticker,
            "model": model,
            "version": version,
            "epochs": epochs,
            "hpo_enabled": enable_hpo,
            "message": "Training started in background. Check MLflow for progress."
        }
    else:
        success = await run_training(ticker, model, start_date, epochs, version, enable_hpo, n_trials)
        if success:
            if model_manager:
                try:
                    await _maybe_async_call(model_manager.load_all)
                except Exception as e:
                    logger.warning(f"Model reload after training failed: {e}")
            return {
                "status": "training_completed",
                "ticker": ticker,
                "model": model,
                "version": version,
            }
        else:
            raise HTTPException(500, "Training failed. Check logs.")


@app.get("/train/status/{ticker}")
async def training_status(ticker: str):
    """Check if model exists for ticker."""
    ticker = validate_ticker(ticker)
    
    if model_manager is None:
        raise HTTPException(503, "Service unavailable")
    
    bundle = await _maybe_async_call(model_manager.get_model, ticker)
    
    if bundle:
        return {
            "ticker": ticker,
            "status": "trained",
            "model_version": bundle.get("model_version"),
            "model_type": bundle.get("model_type", "unknown"),
        }
    else:
        return {
            "ticker": ticker,
            "status": "not_trained",
            "message": f"Model for {ticker} not found. Use POST /train/{ticker} to train."
        }


@app.get("/models")
async def list_models():
    """List all available trained models."""
    if model_manager is None:
        raise HTTPException(503, "Service unavailable")
    
    try:
        models = await _maybe_async_call(getattr, model_manager, "list_models") or []
        return {"models": models, "count": len(models)}
    except Exception as e:
        return {
            "models": list(getattr(model_manager, "_models", {}).keys()),
            "count": len(getattr(model_manager, "_models", {}))
        }


# -------------------------
# Prediction Endpoints
# -------------------------
@app.post("/predict/{ticker}/{target_date}")
async def predict(
    ticker: Annotated[str, Path(...)],
    target_date: Annotated[date, Path(...)],
    background_tasks: BackgroundTasks,
):
    if model_manager is None:
        raise HTTPException(503, "Service unavailable")

    ticker = validate_ticker(ticker)

    bundle = await _maybe_async_call(model_manager.get_model, ticker)
    if bundle is None:
        raise HTTPException(
            404, 
            {
                "error": f"Model not found: {ticker}",
                "message": f"Use POST /train/{ticker} to train a new model first."
            }
        )

    # Maybe retrain
    try:
        updated = await _maybe_async_call(model_manager.maybe_retrain, ticker, bundle)
        if updated and updated.get("model_version") != bundle.get("model_version"):
            logger.info(
                "Model updated: %s -> %s",
                bundle.get("model_version"),
                updated.get("model_version"),
            )
            bundle = updated
    except Exception:
        logger.exception("Retrain failed, using existing model")

    # Metadata
    metadata = await _maybe_async_call(model_manager.get_metadata, ticker) or {}
    last_known = get_last_known_date(metadata)

    if target_date < last_known:
        raise HTTPException(400, "target_date before last known data")

    seq = bundle.get("seq_length", 0)
    if seq <= 0:
        raise HTTPException(500, "Invalid seq_length")

    # Fetch and process
    df_t, df_i, df_u = await fetch_market_data(ticker, last_known, seq)
    merged = merge_dataframes(df_t, df_i, df_u, ticker)

    proc = preprocess_data(merged, ticker)
    features = [c for c in proc.columns if c != "TRADEDATE"]
    X_all = proc[features].values.astype(float)

    if len(X_all) < seq:
        raise HTTPException(422, f"Insufficient data: {len(X_all)} < {seq}")

    # Predict
    try:
        preds = await _maybe_async_call(
            predict_price,
            bundle["model"],
            bundle["scaler_X"],
            bundle["scaler_y"],
            X_all,
            seq,
        )
        preds_list = [float(x) for x in preds]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Prediction error")

    # Build forecast dates
    future_dates = []
    d = last_known + timedelta(days=1)
    while len(future_dates) < len(preds_list):
        if d.weekday() < 5:
            future_dates.append(d)
        d += timedelta(days=1)

    # Limit by target_date
    needed_bd = _business_days_between(last_known, target_date)
    if needed_bd == 0:
        forecast_dates, forecast_preds = [], []
    elif needed_bd <= len(preds_list):
        forecast_dates = future_dates[:needed_bd]
        forecast_preds = preds_list[:needed_bd]
    else:
        logger.warning("Need %d days, have %d", needed_bd, len(preds_list))
        forecast_dates = future_dates
        forecast_preds = preds_list

    # Background tasks
    background_tasks.add_task(
        _log_to_mlflow, ticker, bundle, seq, len(X_all), forecast_preds
    )
    background_tasks.add_task(
        _append_history_async, ticker, forecast_dates, forecast_preds
    )

    return JSONResponse(
        {
            "ticker": ticker,
            "known_up_to": last_known.isoformat(),
            "requested_target_date": target_date.isoformat(),
            "forecast_dates": [d.isoformat() for d in forecast_dates],
            "predictions": forecast_preds,
        }
    )


@app.get("/health")
async def health_check():
    if model_manager is None:
        raise HTTPException(503, "Not ready")
    
    models_count = 0
    try:
        models_count = len(model_manager)
    except:
        pass

    return {
        "status": "healthy",
        "models_loaded": models_count,
        "training_available": True,
        "mlflow_connected": bool(mlflow.get_tracking_uri())
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        reload=False,
    )
