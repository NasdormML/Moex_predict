"""Training runner with subprocess and progress tracking"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

from app.config import MODELS_DIR, PROJECT_ROOT, TRAIN_PY
from app.training.jobs import job_manager

logger = logging.getLogger(__name__)


async def run_training(
    ticker: str,
    model: str = "lstm",
    start_date: str = "2013-01-01",
    epochs: int = 50,
    version: str = "v1",
    enable_hpo: bool = False,
    n_trials: int = 10,
) -> bool:
    """Run training in background subprocess with progress tracking."""
    job_key = job_manager.get_key(ticker, version)

    if not TRAIN_PY.exists():
        logger.error(f"train.py not found at {TRAIN_PY}")
        job_manager.update(job_key, status="failed", message="train.py not found")
        return False

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_PY),
        f"model={model}",
        f"data.ticker={ticker}",
        f"data.start_date={start_date}",
        f"train.epochs={epochs}",
        f"train.version={version}",
        f"train.model_artifacts_dir={MODELS_DIR}",
    ]

    if enable_hpo:
        cmd.extend(
            [
                "optimization.enable=true",
                f"optimization.n_trials={n_trials}",
                f"optimization.epochs_per_trial={max(10, epochs // 2)}",
            ]
        )

    logger.info(f"Starting training: {' '.join(cmd)}")
    logger.info(f"Models will be saved to: {MODELS_DIR / version}")
    job_manager.update(job_key, status="running", message="Training started")

    def _train():
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            )
            env["MODEL_ARTIFACTS_DIR"] = str(MODELS_DIR)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
                env=env,
            )

            job_manager.update(job_key, process=process)

            # Read stdout
            for line in process.stdout:
                line = line.strip()

                if "Epoch" in line and "/" in line:
                    try:
                        epoch_part = line.split("Epoch")[1].split("|")[0].strip()
                        current, total = map(int, epoch_part.split("/"))
                        progress = min(95.0, (current / total) * 100)
                        job_manager.update(job_key, progress=progress)
                    except Exception:
                        pass

            # Read stderr (JSON progress from train.py)
            for line in process.stderr:
                try:
                    data = json.loads(line.strip())
                    if data.get("type") == "progress":
                        job_manager.update(
                            job_key, progress=data.get("progress_percent", 0)
                        )
                    elif data.get("type") == "status":
                        logger.info(f"Train status: {data.get('message')}")
                except json.JSONDecodeError:
                    pass

            process.wait(timeout=3600)

            if process.returncode == 0:
                expected_model = MODELS_DIR / version / f"{ticker}_model.pth"
                if expected_model.exists():
                    logger.info(f"Model saved: {expected_model}")

                job_manager.update(
                    job_key,
                    status="completed",
                    progress=100.0,
                    end_time=datetime.utcnow(),
                    message="Training completed successfully",
                )
                return True
            else:
                job_manager.update(
                    job_key,
                    status="failed",
                    end_time=datetime.utcnow(),
                    message=f"Training failed with code {process.returncode}",
                )
                return False

        except subprocess.TimeoutExpired:
            job_manager.update(
                job_key,
                status="failed",
                end_time=datetime.utcnow(),
                message="Training timed out (3600s)",
            )
            return False
        except Exception as e:
            job_manager.update(
                job_key,
                status="failed",
                end_time=datetime.utcnow(),
                message=f"Exception: {str(e)}",
            )
            logger.exception("Training failed")
            return False
        finally:
            job_manager.update(job_key, process=None)

    return await asyncio.to_thread(_train)
