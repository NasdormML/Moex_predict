"""Prediction API routes"""

import asyncio
import logging
from datetime import date, timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from app.config import HISTORY_DIR, MODELS_DIR
from app.data import fetch_moex_eod_data, fetch_usd_series
from app.predict import predict_price
from app.preprocessing import preprocess_data
from app.utils.helpers import (
    _maybe_async_call,
    business_days_between,
    get_last_known_date,
    merge_dataframes,
    validate_ticker,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


async def append_history(ticker: str, dates: list[date], preds: list[float]):
    """Write predictions to CSV safely."""
    import pandas as pd
    from filelock import FileLock

    df = pd.DataFrame(
        {"DATE": [d.isoformat() for d in dates], "predicted_price": preds}
    )

    pf = HISTORY_DIR / f"predictions_{ticker}.csv"
    lock_path = f"{pf}.lock"

    def _write():
        with FileLock(lock_path):
            header = not pf.exists() or pf.stat().st_size == 0
            with open(pf, "a", newline="") as f:
                df.to_csv(f, header=header, index=False)

    await _maybe_async_call(_write)


@router.post("/predict/{ticker}/{target_date}")
async def predict(
    request: Request,
    ticker: Annotated[str, Path()],
    target_date: Annotated[date, Path()],
    background_tasks: Optional[BackgroundTasks] = None,
):
    """Make price prediction for target date."""
    ticker = validate_ticker(ticker)

    model_manager = request.app.state.model_manager

    bundle = await _maybe_async_call(model_manager.get_model, ticker)
    if bundle is None:
        raise HTTPException(
            404,
            {
                "error": f"Model not found: {ticker}",
                "models_dir": str(MODELS_DIR),
                "message": f"Use POST /train/{ticker} to train first.",
            },
        )

    # Maybe retrain
    try:
        updated = await _maybe_async_call(model_manager.maybe_retrain, ticker, bundle)
        if updated and updated.get("model_version") != bundle.get("model_version"):
            logger.info(
                f"Model updated: {bundle.get('model_version')} -> "
                f"{updated.get('model_version')}"
            )
            bundle = updated
    except Exception:
        logger.exception("Retrain failed, using existing")

    # Metadata
    metadata = await _maybe_async_call(model_manager.get_metadata, ticker) or {}
    last_known = get_last_known_date(metadata)

    if target_date < last_known:
        raise HTTPException(400, "target_date before last known data")

    seq = bundle.get("seq_length", 0)
    if seq <= 0:
        raise HTTPException(500, "Invalid seq_length")

    # Fetch data
    window = seq * 2
    start = (last_known - timedelta(days=window)).isoformat()
    end = last_known.isoformat()

    try:
        df_t, df_i, df_u = await asyncio.gather(
            _maybe_async_call(
                fetch_moex_eod_data, ticker, "stock", "shares", "TQBR", start, end
            ),
            _maybe_async_call(
                fetch_moex_eod_data, "IMOEX", "stock", "index", "SNDX", start, end
            ),
            _maybe_async_call(fetch_usd_series, start, end),
        )
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise HTTPException(503, "Data source error")

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
    needed = business_days_between(last_known, target_date)
    if needed == 0:
        forecast_dates, forecast_preds = [], []
    elif needed <= len(preds_list):
        forecast_dates = future_dates[:needed]
        forecast_preds = preds_list[:needed]
    else:
        forecast_dates = future_dates
        forecast_preds = preds_list

    # Background save
    if background_tasks:
        background_tasks.add_task(
            append_history, ticker, forecast_dates, forecast_preds
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
