import asyncio
import logging
import os
import subprocess
import sys
import json
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Annotated, Optional, Dict, Any
from dataclasses import dataclass, asdict
import threading

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Path, Query, Depends
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
# Training Status Tracking
# -------------------------
@dataclass
class TrainingJob:
    ticker: str
    model: str
    version: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: datetime
    end_time: Optional[datetime] = None
    message: str = ""
    progress: float = 0.0
    process: Optional[subprocess.Popen] = None
    result: Optional[Dict[str, Any]] = None

_training_jobs: Dict[str, TrainingJob] = {}
_jobs_lock = threading.Lock()

def get_job_key(ticker: str, version: str) -> str:
    return f"{ticker}_{version}"

def update_job_status(key: str, **kwargs):
    with _jobs_lock:
        if key in _training_jobs:
            for k, v in kwargs.items():
                setattr(_training_jobs[key], k, v)

def get_active_jobs() -> Dict[str, TrainingJob]:
    with _jobs_lock:
        return {k: v for k, v in _training_jobs.items() if v.status in ["pending", "running"]}

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
) -> bool:
    """Run training in background subprocess with progress tracking.
    """
    job_key = get_job_key(ticker, version)
    
    cmd = [
        sys.executable,
        "-u",
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
    update_job_status(job_key, status="running", message="Training started")
    
    def _train():
        try:
            # Start process with piping stdout/stderr for progress check
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/app",
                bufsize=1,
            )
            
            # Saving the process for cancellation
            with _jobs_lock:
                if job_key in _training_jobs:
                    _training_jobs[job_key].process = process
            
            stdout_lines = []
            stderr_lines = []
            
            # Read stdout in real time to track progress
            for line in process.stdout:
                stdout_lines.append(line)
                if "Epoch" in line and "/" in line:
                    try:
                        parts = line.split("Epoch")[1].split("/")[0].strip()
                        current = int(parts.split()[0])
                        progress = min(95.0, (current / epochs) * 100)
                        update_job_status(job_key, progress=progress)
                    except:
                        pass
            
            # Читаем stderr
            for line in process.stderr:
                stderr_lines.append(line)
            
            process.wait(timeout=3600)
            
            if process.returncode == 0:
                update_job_status(
                    job_key, 
                    status="completed", 
                    progress=100.0,
                    end_time=datetime.utcnow(),
                    message="Training completed successfully"
                )
                return True
            else:
                error_msg = "".join(stderr_lines[-20:])
                update_job_status(
                    job_key,
                    status="failed",
                    end_time=datetime.utcnow(),
                    message=f"Training failed: {error_msg[:500]}"
                )
                logger.error(f"Training failed: {error_msg}")
                return False
                
        except subprocess.TimeoutExpired:
            update_job_status(
                job_key,
                status="failed",
                end_time=datetime.utcnow(),
                message="Training timed out (3600s)"
            )
            logger.error("Training timed out")
            return False
        except Exception as e:
            update_job_status(
                job_key,
                status="failed",
                end_time=datetime.utcnow(),
                message=f"Exception: {str(e)}"
            )
            logger.exception("Training failed")
            return False
        finally:
            # Cleanup process reference
            with _jobs_lock:
                if job_key in _training_jobs:
                    _training_jobs[job_key].process = None
    
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
        active = get_active_jobs()
        for key, job in active.items():
            if job.process and job.process.poll() is None:
                logger.info(f"Terminating training job {key}")
                job.process.terminate()
                try:
                    job.process.wait(timeout=5)
                except:
                    job.process.kill()
        
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
    background_tasks: BackgroundTasks,
    model: str = Query("lstm", enum=["lstm", "tcn", "tft"]),
    start_date: str = Query("2013-01-01"),
    epochs: int = Query(50, ge=10, le=200),
    version: str = Query("v1"),
    enable_hpo: bool = Query(False),
    n_trials: int = Query(10, ge=5, le=50),
    force: bool = Query(False, description="Force retrain even if model exists"),
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
    - **force**: Retrain even if version already exists
    """
    ticker = validate_ticker(ticker)
    job_key = get_job_key(ticker, version)
    
    with _jobs_lock:
        if job_key in _training_jobs and _training_jobs[job_key].status in ["pending", "running"]:
            raise HTTPException(
                409, 
                {
                    "error": "Training already in progress",
                    "job_key": job_key,
                    "status": _training_jobs[job_key].status,
                    "started_at": _training_jobs[job_key].start_time.isoformat()
                }
            )
    
    # Model existence check 
    if not force and model_manager:
        existing = await _maybe_async_call(model_manager.get_model, ticker)
        if existing and existing.get("model_version") == version:
            raise HTTPException(
                409,
                {
                    "error": "Model version already exists",
                    "message": f"Model {ticker}@{version} already trained. Use force=true to retrain."
                }
            )

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

    # Create a record of the task
    with _jobs_lock:
        _training_jobs[job_key] = TrainingJob(
            ticker=ticker,
            model=model,
            version=version,
            status="pending",
            start_time=datetime.utcnow(),
            message="Queued for training"
        )
    
    background_tasks.add_task(
        run_training, ticker, model, start_date, epochs, version, enable_hpo, n_trials
    )
    
    return {
        "status": "training_started",
        "job_key": job_key,
        "ticker": ticker,
        "model": model,
        "version": version,
        "epochs": epochs,
        "hpo_enabled": enable_hpo,
        "started_at": datetime.utcnow().isoformat(),
        "message": "Training started in background. Check /train/status/{ticker}/{version} for progress."
    }


@app.get("/train/status/{ticker}")
async def training_status_ticker(
    ticker: str,
    version: Optional[str] = Query(None, description="Specific version or latest")
):
    """Get training status for ticker (all versions or specific)."""
    ticker = validate_ticker(ticker)
    
    with _jobs_lock:
        if version:
            key = get_job_key(ticker, version)
            if key not in _training_jobs:
                raise HTTPException(404, f"No training job found for {ticker}@{version}")
            job = _training_jobs[key]
            return {
                "job_key": key,
                **asdict(job),
                "is_active": job.status in ["pending", "running"]
            }
        else:
            jobs = {
                k: {**asdict(v), "is_active": v.status in ["pending", "running"]}
                for k, v in _training_jobs.items() 
                if v.ticker == ticker
            }
            return {"ticker": ticker, "jobs": jobs, "count": len(jobs)}


@app.get("/train/jobs")
async def list_training_jobs(
    status_filter: Optional[str] = Query(None, enum=["pending", "running", "completed", "failed"]),
    ticker_filter: Optional[str] = Query(None)
):
    """List all training jobs with optional filtering."""
    with _jobs_lock:
        jobs = {}
        for k, v in _training_jobs.items():
            if status_filter and v.status != status_filter:
                continue
            if ticker_filter and v.ticker != ticker_filter.upper():
                continue
            jobs[k] = {**asdict(v), "is_active": v.status in ["pending", "running"]}
        
        stats = {
            "total": len(_training_jobs),
            "pending": sum(1 for j in _training_jobs.values() if j.status == "pending"),
            "running": sum(1 for j in _training_jobs.values() if j.status == "running"),
            "completed": sum(1 for j in _training_jobs.values() if j.status == "completed"),
            "failed": sum(1 for j in _training_jobs.values() if j.status == "failed"),
        }
        
        return {"jobs": jobs, "stats": stats, "filtered_count": len(jobs)}


@app.delete("/train/jobs/{ticker}/{version}")
async def cancel_training(ticker: str, version: str):
    """Cancel running training job."""
    ticker = validate_ticker(ticker)
    job_key = get_job_key(ticker, version)
    
    with _jobs_lock:
        if job_key not in _training_jobs:
            raise HTTPException(404, "Job not found")
        
        job = _training_jobs[job_key]
        if job.status not in ["pending", "running"]:
            raise HTTPException(409, f"Cannot cancel job with status: {job.status}")
        
        if job.process and job.process.poll() is None:
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except:
                job.process.kill()
        
        job.status = "failed"
        job.end_time = datetime.utcnow()
        job.message = "Cancelled by user"
        
        return {"status": "cancelled", "job_key": job_key}


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

    active_trainings = len(get_active_jobs())

    return {
        "status": "healthy",
        "models_loaded": models_count,
        "active_trainings": active_trainings,
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