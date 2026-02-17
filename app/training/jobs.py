"""Training job tracking and status management"""

import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


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


class JobManager:
    """Thread-safe training job manager."""

    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def get_key(self, ticker: str, version: str) -> str:
        return f"{ticker}_{version}"

    def add(self, job: TrainingJob) -> str:
        key = self.get_key(job.ticker, job.version)
        with self._lock:
            self._jobs[key] = job
        return key

    def update(self, key: str, **kwargs) -> bool:
        with self._lock:
            if key not in self._jobs:
                return False
            for k, v in kwargs.items():
                setattr(self._jobs[key], k, v)
            return True

    def get(self, key: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(key)

    def get_active(self) -> Dict[str, TrainingJob]:
        with self._lock:
            return {
                k: v
                for k, v in self._jobs.items()
                if v.status in ["pending", "running"]
            }

    def get_by_ticker(self, ticker: str) -> Dict[str, TrainingJob]:
        with self._lock:
            return {k: v for k, v in self._jobs.items() if v.ticker == ticker}

    def list_all(self) -> Dict[str, TrainingJob]:
        with self._lock:
            return dict(self._jobs)

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total": len(self._jobs),
                "pending": sum(1 for j in self._jobs.values() if j.status == "pending"),
                "running": sum(1 for j in self._jobs.values() if j.status == "running"),
                "completed": sum(
                    1 for j in self._jobs.values() if j.status == "completed"
                ),
                "failed": sum(1 for j in self._jobs.values() if j.status == "failed"),
            }


# Глобальный экземпляр
job_manager = JobManager()
