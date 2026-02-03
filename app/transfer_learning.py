import asyncio
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from app.data import fetch_cbr_usd_rate, fetch_moex_eod_data
from app.preprocessing import preprocess_data

logger = logging.getLogger(__name__)

DEFAULT_METADATA_PATH = Path("history/metadata/training_metadata.pkl")
DEFAULT_ARTIFACTS_ROOT = Path(os.getenv("MODEL_ARTIFACTS_DIR", "saved_models"))
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 45
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 10


def load_training_metadata(path: str | Path | None = None) -> dict:
    """
    Load training metadata from pickle file.

    Args:
        path: Path to metadata file. If None, uses default location.

    Returns:
        Dictionary with training metadata.
    """
    metadata_path = Path(path) if path else DEFAULT_METADATA_PATH

    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return {}


def save_training_metadata(metadata: dict, path: str | Path | None = None) -> None:
    """
    Save training metadata to pickle file.

    Args:
        metadata: Dictionary with training metadata.
        path: Path to metadata file. If None, uses default location.
    """
    metadata_path = Path(path) if path else DEFAULT_METADATA_PATH

    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
    except Exception as e:
        logger.error(f"Failed to save metadata to {metadata_path}: {e}")
        raise


class TransferLearningManager:
    """Manages model retraining and versioning."""

    def __init__(
        self,
        metadata_path: Path | None = None,
        artifacts_root: Path | None = None,
    ):
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self.artifacts_root = artifacts_root or DEFAULT_ARTIFACTS_ROOT

    def load_metadata(self) -> dict:
        """Load training metadata (delegate to module function)."""
        return load_training_metadata(self.metadata_path)

    def save_metadata(self, metadata: dict) -> None:
        """Save training metadata (delegate to module function)."""
        save_training_metadata(metadata, self.metadata_path)

    def should_retrain(self, ticker: str, threshold_days: int = 5) -> bool:
        """Check if model needs retraining."""
        metadata = self.load_metadata()
        ticker_md = metadata.get(ticker, {})

        ver = ticker_md.get("active_version")
        ver_md = ticker_md.get("versions", {}).get(ver, {})

        train_date_str = ver_md.get("train_date")
        if not train_date_str:
            return False

        try:
            train_date = datetime.strptime(train_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid train_date format for {ticker}: {train_date_str}")
            return False

        business_days = len(pd.bdate_range(train_date, datetime.today().date())) - 1

        return business_days >= threshold_days

    async def retrain_if_needed(self, ticker: str, bundle: dict) -> dict:
        """Retrain model if data is stale."""
        if not self.should_retrain(ticker):
            return bundle

        logger.info(f"Retraining {ticker}...")
        return await self._retrain(ticker, bundle)

    async def _retrain(self, ticker: str, bundle: dict) -> dict:
        """Perform model retraining."""
        window_days = bundle["model_params"].get("window_days", 180)
        end_date = datetime.today().date()
        start_date = end_date - pd.tseries.offsets.BDay(window_days)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            df_t, df_i, df_u = await asyncio.gather(
                fetch_moex_eod_data(
                    ticker, "stock", "shares", "TQBR", start_str, end_str
                ),
                fetch_moex_eod_data(
                    "IMOEX", "stock", "index", "SNDX", start_str, end_str
                ),
                self._fetch_usd_data(start_str, end_str),
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return bundle

        merged = self._merge_data(df_t, df_i, df_u, ticker)
        if merged is None or len(merged) == 0:
            logger.warning(f"No data for {ticker}, skipping retrain")
            return bundle

        proc = preprocess_data(merged, ticker)

        features = bundle["model_params"].get(
            "feature_list", [c for c in proc.columns if c != "TRADEDATE"]
        )

        X, y = self._create_sequences(proc, features, ticker, bundle["seq_length"])
        if X is None:
            return bundle

        scaler_X = bundle["scaler_X"]
        scaler_y = bundle["scaler_y"]

        try:
            X_scaled = scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
            y_scaled = scaler_y.transform(y)
        except Exception as e:
            logger.error(f"Scaling failed for {ticker}: {e}")
            return bundle

        model = self._fine_tune_model(bundle, X_scaled, y_scaled)
        new_bundle = self._save_version(ticker, bundle, model, scaler_X, scaler_y)

        logger.info(f"Retrained {ticker} to version {new_bundle['model_version']}")
        return new_bundle

    async def _fetch_usd_data(self, start: str, end: str) -> pd.DataFrame:
        """Fetch USD data with fallback to CBR."""
        try:
            df_u = await fetch_moex_eod_data(
                "USD000UTSTOM", "currency", "selt", "CETS", start, end
            )
        except Exception as e:
            logger.warning(f"MOEX USD fetch failed: {e}")
            df_u = pd.DataFrame()

        if df_u is None or df_u.empty:
            dates = pd.date_range(start, end)
            rates = await self._fetch_cbr_rates_batch(dates)
            df_u = pd.DataFrame(
                {
                    "TRADEDATE": dates,
                    "CLOSE": rates,
                }
            )

        return df_u

    async def _fetch_cbr_rates_batch(self, dates: pd.DatetimeIndex) -> list:
        """Fetch CBR rates with concurrency limit."""
        semaphore = asyncio.Semaphore(5)

        async def fetch_one(d):
            async with semaphore:
                return await asyncio.to_thread(fetch_cbr_usd_rate, d)

        return await asyncio.gather(*[fetch_one(d) for d in dates])

    def _merge_data(
        self,
        df_t: pd.DataFrame,
        df_i: pd.DataFrame,
        df_u: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame | None:
        """Merge and clean market data."""

        def prep(df, rename_map):
            if df is None or df.empty:
                raise ValueError("Empty dataframe")
            df = df.copy()
            date_col = "BEGIN" if "BEGIN" in df.columns else "TRADEDATE"
            df["TRADEDATE"] = pd.to_datetime(df[date_col]).dt.normalize()
            return df.rename(columns=rename_map)

        try:
            df_t = prep(
                df_t,
                {
                    "OPEN": f"OPEN_{ticker}",
                    "HIGH": f"HIGH_{ticker}",
                    "LOW": f"LOW_{ticker}",
                    "CLOSE": f"CLOSE_{ticker}",
                    "VOLUME": f"VOL_{ticker}",
                },
            )

            df_i = prep(df_i, {"CLOSE": "CLOSE_IMOEX"})
            df_u = prep(df_u, {"CLOSE": "CLOSE_USD"})

            merged = (
                df_t[
                    [
                        "TRADEDATE",
                        f"CLOSE_{ticker}",
                        f"OPEN_{ticker}",
                        f"HIGH_{ticker}",
                        f"LOW_{ticker}",
                        f"VOL_{ticker}",
                    ]
                ]
                .merge(df_i[["TRADEDATE", "CLOSE_IMOEX"]], on="TRADEDATE", how="outer")
                .merge(df_u[["TRADEDATE", "CLOSE_USD"]], on="TRADEDATE", how="outer")
                .sort_values("TRADEDATE")
                .ffill()
                .dropna()
                .reset_index(drop=True)
            )

            return merged
        except Exception as e:
            logger.error(f"Data merge failed: {e}")
            return None

    def _create_sequences(
        self,
        proc: pd.DataFrame,
        features: list[str],
        ticker: str,
        seq_length: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Create training sequences."""
        data = proc[features].values.astype(float)

        if len(data) <= seq_length:
            logger.warning(f"Insufficient data: {len(data)} <= {seq_length}")
            return None, None

        target_idx = features.index(f"CLOSE_{ticker}")

        X_list, y_list = [], []
        for i in range(len(data) - seq_length):
            X_list.append(data[i : i + seq_length])
            y_list.append(data[i + seq_length, target_idx])

        return np.array(X_list), np.array(y_list).reshape(-1, 1)

    def _fine_tune_model(
        self,
        bundle: dict,
        X: np.ndarray,
        y: np.ndarray,
    ) -> torch.nn.Module:
        """Fine-tune model with early stopping."""
        model = bundle["model"]
        ft_modules = bundle["model_params"].get("fine_tune_modules", [])

        # Setup fine-tuning
        if ft_modules:
            for name, param in model.named_parameters():
                param.requires_grad = any(m in name for m in ft_modules)
        else:
            for param in model.parameters():
                param.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning("No trainable parameters, using all")
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = list(model.parameters())

        # Split train/val
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        optimizer = optim.AdamW(trainable_params, lr=DEFAULT_LR)
        loss_fn = nn.HuberLoss()

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        model.train()

        for epoch in range(DEFAULT_EPOCHS):
            # Train
            optimizer.zero_grad()
            train_pred = model(X_train_t)
            train_loss = loss_fn(train_pred, y_train_t)
            train_loss.backward()
            optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()
            model.train()

            # Log to MLflow
            try:
                mlflow.log_metrics(
                    {
                        "retrain_train_loss": train_loss.item(),
                        "retrain_val_loss": val_loss,
                    },
                    step=epoch,
                )
            except Exception:
                pass

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= MAX_EPOCHS_WITHOUT_IMPROVEMENT:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        return model

    def _save_version(
        self,
        ticker: str,
        bundle: dict,
        model: torch.nn.Module,
        scaler_X,
        scaler_y,
    ) -> dict:
        """Save new model version with proper versioning."""
        current_ver = bundle.get("model_version", "v1.0")

        try:
            if current_ver.startswith("v"):
                parts = current_ver[1:].split(".")
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                new_ver = f"v{major}.{minor + 1}"
            else:
                new_ver = "v1.0"
        except (ValueError, IndexError):
            new_ver = "v1.0"

        out_dir = self.artifacts_root / new_ver
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), out_dir / f"{ticker}_model.pth")

        with open(out_dir / f"{ticker}_scaler_X.pkl", "wb") as f:
            pickle.dump(scaler_X, f)
        with open(out_dir / f"{ticker}_scaler_y.pkl", "wb") as f:
            pickle.dump(scaler_y, f)

        # Update metadata
        metadata = self.load_metadata()
        if ticker not in metadata:
            metadata[ticker] = {"versions": {}}

        metadata[ticker]["versions"][new_ver] = {
            "train_date": datetime.today().strftime("%Y-%m-%d"),
            "data_upto": datetime.today().strftime("%Y-%m-%d"),
            "factory_key": bundle["factory_key"],
            "model_params": bundle["model_params"],
        }
        metadata[ticker]["active_version"] = new_ver

        self.save_metadata(metadata)

        new_bundle = bundle.copy()
        new_bundle.update(
            {
                "model": model,
                "model_version": new_ver,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
            }
        )

        return new_bundle
