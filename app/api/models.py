"""Models API routes"""

import logging

from fastapi import APIRouter, Request

from app.config import MODELS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(request: Request):
    """List all available trained models."""
    model_manager = request.app.state.model_manager

    models = []  # <-- Определили список

    # Scan disk
    if MODELS_DIR.exists():
        for version_dir in MODELS_DIR.iterdir():
            if not version_dir.is_dir():
                continue
            for model_file in version_dir.glob("*_model.pth"):
                ticker = model_file.stem.replace("_model", "")
                models.append(
                    {
                        "ticker": ticker,
                        "version": version_dir.name,
                        "path": str(model_file),
                        "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                    }
                )

    # Loaded in memory
    loaded = []
    if model_manager:
        try:
            loaded = list(getattr(model_manager, "_models", {}).keys())
        except Exception:
            pass

    return {
        "models_on_disk": models,
        "models_loaded": loaded,
        "models_dir": str(MODELS_DIR),
        "total_models": len(models),
    }
