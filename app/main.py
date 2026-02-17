"""Main FastAPI application"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.predict import router as predict_router
from app.config import MODELS_DIR
from app.model_manager import ModelManager
from app.training.jobs import job_manager
from app.training.routes import router as train_router

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown."""
    global model_manager

    logger.info("Initializing ModelManager...")
    os.environ["MODEL_ARTIFACTS_DIR"] = str(MODELS_DIR)

    model_manager = ModelManager()

    try:
        await model_manager.load_all()
        logger.info(f"Loaded {len(model_manager)} models")
    except Exception as e:
        logger.warning(f"No models loaded: {e}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    app.state.model_manager = model_manager

    try:
        yield
    finally:
        active = job_manager.get_active()
        for key, job in active.items():
            if job.process and job.process.poll() is None:
                logger.info(f"Terminating {key}")
                job.process.terminate()
                try:
                    job.process.wait(timeout=5)
                except Exception:
                    job.process.kill()

        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="MOEX Price Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(train_router)
    app.include_router(predict_router)
    app.include_router(models_router)
    app.include_router(health_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        reload=False,
    )
