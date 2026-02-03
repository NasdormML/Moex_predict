import logging
import os
import pickle
from pathlib import Path

import torch

from app.models.factory import get_model
from app.transfer_learning import TransferLearningManager

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe model loading and management."""

    def __init__(self):
        self._models: dict[str, dict] = {}
        self._metadata: dict[str, dict] = {}
        self._tl_manager = TransferLearningManager()
        self._ready = False

    def __len__(self) -> int:
        return len(self._models)

    def is_ready(self) -> bool:
        return self._ready

    async def load_all(self) -> None:
        """Load all models asynchronously."""
        artifacts_root = Path(os.getenv("MODEL_ARTIFACTS_DIR", "saved_models"))

        if not artifacts_root.exists():
            raise RuntimeError(f"Artifacts directory not found: {artifacts_root}")

        metadata = self._tl_manager.load_metadata()

        for ticker, ticker_md in metadata.items():
            try:
                model_bundle = self._load_model(ticker, ticker_md, artifacts_root)
                if model_bundle:
                    self._models[ticker] = model_bundle
                    self._metadata[ticker] = ticker_md
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}", exc_info=True)

        if not self._models:
            raise RuntimeError("No models loaded")

        self._ready = True
        logger.info(f"Loaded {len(self._models)} models: {list(self._models.keys())}")

    def _load_model(
        self, ticker: str, ticker_md: dict, artifacts_root: Path
    ) -> dict | None:
        """Load single model with validation."""
        version = ticker_md.get("active_version")
        if not version:
            logger.warning(f"Skipping {ticker}: no active_version")
            return None

        ver_md = ticker_md.get("versions", {}).get(version, {})
        factory_key = ver_md.get("factory_key")
        params = ver_md.get("model_params")

        if not (factory_key and params):
            logger.warning(f"Skipping {ticker}@{version}: incomplete metadata")
            return None

        model_dir = artifacts_root / version
        if not model_dir.is_dir():
            logger.warning(f"Model directory not found: {model_dir}")
            return None

        # Find weights file
        weight_file = self._find_weight_file(model_dir, ticker)
        if not weight_file:
            return None

        # Load scalers securely
        scaler_x_path = model_dir / f"{ticker}_scaler_X.pkl"
        scaler_y_path = model_dir / f"{ticker}_scaler_y.pkl"

        try:
            scaler_X = self._safe_load_pickle(scaler_x_path)
            scaler_y = self._safe_load_pickle(scaler_y_path)
        except Exception as e:
            logger.error(f"Failed to load scalers for {ticker}: {e}")
            return None

        # Load model
        try:
            model = get_model(factory_key, **params)
            state = torch.load(
                weight_file,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load model {ticker}@{version}: {e}")
            return None

        seq_length = params.get("seq_length")
        if seq_length is None:
            logger.warning(f"seq_length not found for {ticker}")
            return None

        return {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": seq_length,
            "model_version": version,
            "factory_key": factory_key,
            "model_params": params,
        }

    def _find_weight_file(self, model_dir: Path, ticker: str) -> Path | None:
        """Find model weights file."""
        for pattern in [f"{ticker}_model.pth", f"{ticker}_best.pth"]:
            candidates = list(model_dir.glob(pattern))
            if candidates:
                return candidates[0]
        logger.warning(f"Weights file not found for {ticker} in {model_dir}")
        return None

    def _safe_load_pickle(self, path: Path) -> object:
        """Safely load pickle with validation."""
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")

        # Security: check file size
        max_size = 100 * 1024 * 1024
        if path.stat().st_size > max_size:
            raise ValueError(f"Pickle file too large: {path}")

        with open(path, "rb") as f:
            return pickle.load(f)

    def get_model(self, ticker: str) -> dict | None:
        """Get loaded model bundle."""
        return self._models.get(ticker)

    def get_metadata(self, ticker: str) -> dict:
        """Get model metadata."""
        return self._metadata.get(ticker, {})

    async def maybe_retrain(self, ticker: str, bundle: dict) -> dict:
        """Check and perform retraining if needed."""
        updated = await self._tl_manager.retrain_if_needed(ticker, bundle)
        if updated is not bundle:
            self._models[ticker] = updated
        return updated
