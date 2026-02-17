"""Application configuration and path detection"""

import logging
import os
from pathlib import Path as PathLib

logger = logging.getLogger(__name__)


def get_project_paths():
    """Auto-detect project paths for both local and Docker environments."""
    docker_app_dir = os.getenv("APP_DIR", "/app")
    docker_model_dir = os.getenv("MODEL_ARTIFACTS_DIR", "/data/models")
    docker_history_dir = os.getenv("HISTORY_DIR", "/data/history")

    in_docker = (
        os.path.exists("/.dockerenv")
        or os.getenv("DOCKER_CONTAINER", "false").lower() == "true"
    )

    if in_docker and os.path.exists(docker_app_dir):
        project_root = PathLib(docker_app_dir)
        train_py = project_root / "train.py"
        models_dir = PathLib(docker_model_dir)
        hist_dir = PathLib(docker_history_dir)
        logger.info(f"Running in Docker mode: {project_root}")
    else:
        current_file = PathLib(__file__).resolve()
        app_dir = current_file.parent
        project_root = app_dir.parent

        if not (project_root / "train.py").exists():
            project_root = PathLib.cwd()

        train_py = project_root / "train.py"
        models_dir = PathLib(
            os.getenv("MODEL_ARTIFACTS_DIR", project_root / "saved_models")
        )
        hist_dir = PathLib(os.getenv("HISTORY_DIR", project_root / "history"))

        logger.info(f"Running in local mode: {project_root}")

    models_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    return project_root, train_py, models_dir, hist_dir


PROJECT_ROOT, TRAIN_PY, MODELS_DIR, HISTORY_DIR = get_project_paths()

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Train script: {TRAIN_PY}")
logger.info(f"Models dir: {MODELS_DIR}")
logger.info(f"History dir: {HISTORY_DIR}")
