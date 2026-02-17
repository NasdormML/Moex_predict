# app/api/__init__.py
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.predict import router as predict_router

__all__ = ["predict_router", "models_router", "health_router"]
