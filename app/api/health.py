"""Health check endpoint"""

import os

from fastapi import APIRouter, Request

from app.config import MODELS_DIR, PROJECT_ROOT
from app.training.jobs import job_manager

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request):
    """API health status."""
    model_manager = request.app.state.model_manager

    if model_manager is None:
        from fastapi import HTTPException

        raise HTTPException(503, "Not ready")

    models_count = 0
    try:
        models_count = len(model_manager)
    except Exception:
        pass

    active_trainings = len(job_manager.get_active())

    return {
        "status": "healthy",
        "models_loaded": models_count,
        "active_trainings": active_trainings,
        "training_available": True,
        "mlflow_connected": bool(os.getenv("MLFLOW_TRACKING_URI")),
        "environment": "docker" if os.path.exists("/.dockerenv") else "local",
        "models_dir": str(MODELS_DIR),
        "project_root": str(PROJECT_ROOT),
    }
