"""Training API routes"""

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from app.config import MODELS_DIR
from app.data import fetch_moex_eod_data
from app.training.jobs import TrainingJob, job_manager
from app.training.runner import run_training
from app.utils.helpers import _maybe_async_call, validate_ticker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/{ticker}")
async def train_model(
    ticker: Annotated[str, Path()],
    background_tasks: BackgroundTasks,
    model: Annotated[str, Query()] = "lstm",
    start_date: Annotated[str, Query()] = "2013-01-01",
    epochs: Annotated[int, Query(ge=10, le=200)] = 50,
    version: Annotated[str, Query()] = "v1",
    enable_hpo: Annotated[bool, Query()] = False,
    n_trials: Annotated[int, Query(ge=5, le=50)] = 10,
    force: Annotated[bool, Query()] = False,
):
    """Start training for ticker."""
    ticker = validate_ticker(ticker)
    job_key = job_manager.get_key(ticker, version)

    existing = job_manager.get(job_key)
    if existing and existing.status in ["pending", "running"]:
        raise HTTPException(
            409,
            {
                "error": "Training in progress",
                "job_key": job_key,
                "status": existing.status,
            },
        )

    expected_path = MODELS_DIR / version / f"{ticker}_model.pth"
    if not force and expected_path.exists():
        raise HTTPException(
            409,
            {
                "error": "Model exists",
                "path": str(expected_path),
                "message": f"Use force=true for {ticker}@{version}",
            },
        )

    # Check data availability
    try:
        test_df = await _maybe_async_call(
            fetch_moex_eod_data,
            ticker,
            "stock",
            "shares",
            "TQBR",
            (date.today() - timedelta(days=30)).isoformat(),
            date.today().isoformat(),
        )
        if test_df is None or test_df.empty:
            raise HTTPException(404, f"No market data for ticker: {ticker}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data check failed: {e}")
        raise HTTPException(503, "Market data unavailable")

    # Create job
    job = TrainingJob(
        ticker=ticker,
        model=model,
        version=version,
        status="pending",
        start_time=datetime.now(timezone.utc),
        message="Queued for training",
    )
    job_manager.add(job)

    background_tasks.add_task(
        run_training, ticker, model, start_date, epochs, version, enable_hpo, n_trials
    )

    return {
        "status": "training_started",
        "job_key": job_key,
        "ticker": ticker,
        "model": model,
        "version": version,
        "save_path": str(MODELS_DIR / version),
        "epochs": epochs,
        "hpo_enabled": enable_hpo,
        "environment": "docker" if os.path.exists("/.dockerenv") else "local",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/stream/{ticker}")
async def train_stream(
    ticker: Annotated[str, Path()],
    model: Annotated[str, Query()] = "lstm",
    start_date: Annotated[str, Query()] = "2013-01-01",
    epochs: Annotated[int, Query(ge=10, le=200)] = 50,
    version: Annotated[str, Query()] = "v1",
    enable_hpo: Annotated[bool, Query()] = False,
    n_trials: Annotated[int, Query(ge=5, le=50)] = 10,
    force: Annotated[bool, Query()] = False,
):
    """Stream training progress via SSE."""
    ticker = validate_ticker(ticker)
    job_key = job_manager.get_key(ticker, version)

    # Check conflicts
    existing = job_manager.get(job_key)
    if existing and existing.status in ["pending", "running"]:
        raise HTTPException(
            409,
            {
                "error": "Training in progress",
                "job_key": job_key,
                "status": existing.status,
            },
        )

    expected_path = MODELS_DIR / version / f"{ticker}_model.pth"
    if not force and expected_path.exists():
        raise HTTPException(
            409,
            {
                "error": "Model exists",
                "path": str(expected_path),
                "message": f"Use force=true for {ticker}@{version}",
            },
        )

    async def event_generator():
        # Create job
        job = TrainingJob(
            ticker=ticker,
            model=model,
            version=version,
            status="pending",
            start_time=datetime.now(timezone.utc),
            message="Starting...",
        )
        job_manager.add(job)

        # Start training in background
        import asyncio

        task = asyncio.create_task(
            run_training(
                ticker, model, start_date, epochs, version, enable_hpo, n_trials
            )
        )

        last_progress = -1

        try:
            while True:
                await asyncio.sleep(0.5)
                job = job_manager.get(job_key)

                if job is None:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

                # Send update if progress changed or status finalized
                if job.progress != last_progress or job.status in [
                    "completed",
                    "failed",
                ]:
                    last_progress = job.progress

                    event_data = {
                        "job_key": job_key,
                        "status": job.status,
                        "progress": job.progress,
                        "message": job.message,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    if job.status == "completed":
                        event_data["result"] = {
                            "ticker": ticker,
                            "version": version,
                            "model_path": str(MODELS_DIR / version),
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        break
                    elif job.status == "failed":
                        event_data["error"] = job.message
                        yield f"data: {json.dumps(event_data)}\n\n"
                        break
                    else:
                        yield f"data: {json.dumps(event_data)}\n\n"

        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/status/{ticker}")
async def training_status(
    ticker: str,
    version: Annotated[Optional[str], Query(None)],
):
    """Get training status for ticker."""
    ticker = validate_ticker(ticker)

    if version:
        key = job_manager.get_key(ticker, version)
        job = job_manager.get(key)

        if not job:
            # Check if model exists on disk
            model_path = MODELS_DIR / version / f"{ticker}_model.pth"
            if model_path.exists():
                return {
                    "ticker": ticker,
                    "version": version,
                    "status": "trained",
                    "model_path": str(model_path),
                }
            raise HTTPException(404, f"No training job found for {ticker}@{version}")

        return {
            "ticker": job.ticker,
            "model": job.model,
            "version": job.version,
            "status": job.status,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "message": job.message,
            "progress": job.progress,
            "is_active": job.status in ["pending", "running"],
        }

    # All jobs for ticker
    jobs = job_manager.get_by_ticker(ticker)
    return {
        "ticker": ticker,
        "jobs": {
            k: {
                "ticker": v.ticker,
                "model": v.model,
                "version": v.version,
                "status": v.status,
                "start_time": v.start_time.isoformat() if v.start_time else None,
                "end_time": v.end_time.isoformat() if v.end_time else None,
                "message": v.message,
                "progress": v.progress,
                "is_active": v.status in ["pending", "running"],
            }
            for k, v in jobs.items()
        },
        "count": len(jobs),
    }


@router.get("/jobs")
async def list_jobs(
    status_filter: Annotated[Optional[str], Query()] = None,
    ticker_filter: Annotated[Optional[str], Query()] = None,
):
    """List all training jobs."""
    all_jobs = job_manager.list_all()

    filtered = {}
    for k, v in all_jobs.items():
        if status_filter and v.status != status_filter:
            continue
        if ticker_filter and v.ticker != ticker_filter.upper():
            continue

        filtered[k] = {
            "ticker": v.ticker,
            "model": v.model,
            "version": v.version,
            "status": v.status,
            "start_time": v.start_time.isoformat() if v.start_time else None,
            "end_time": v.end_time.isoformat() if v.end_time else None,
            "message": v.message,
            "progress": v.progress,
            "is_active": v.status in ["pending", "running"],
        }

    return {
        "jobs": filtered,
        "stats": job_manager.get_stats(),
        "filtered_count": len(filtered),
    }


@router.delete("/jobs/{ticker}/{version}")
async def cancel_training(
    ticker: str,
    version: str,
):
    """Cancel running training job."""
    ticker = validate_ticker(ticker)
    key = job_manager.get_key(ticker, version)

    job = job_manager.get(key)
    if not job:
        raise HTTPException(404, "Job not found")

    if job.status not in ["pending", "running"]:
        raise HTTPException(409, f"Cannot cancel job with status: {job.status}")

    if job.process and job.process.poll() is None:
        job.process.terminate()
        try:
            job.process.wait(timeout=5)
        except Exception:
            job.process.kill()

    job_manager.update(
        key,
        status="failed",
        end_time=datetime.now(timezone.utc),
        message="Cancelled by user",
    )

    return {"status": "cancelled", "job_key": key}
