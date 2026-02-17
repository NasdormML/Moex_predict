import json
import os
import pickle
import random
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.data import get_dataloaders_multi
from app.models.factory import get_model
from app.optimization import optimize_model
from app.transfer_learning import load_training_metadata, save_training_metadata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _ensure_str_keys(obj):
    """Recursively ensure all dict keys are strings (not bytes)."""
    if isinstance(obj, dict):
        return {
            (
                k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
            ): _ensure_str_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_ensure_str_keys(v) for v in obj]
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8")
    else:
        return obj


def log_progress(
    epoch: int, total_epochs: int, metrics: Dict[str, float], phase: str = "training"
):
    """Output progress in JSON format for API parsing"""
    progress_data = {
        "type": "progress",
        "epoch": epoch,
        "total_epochs": total_epochs,
        "progress_percent": round((epoch / total_epochs) * 100, 2),
        "metrics": metrics,
        "phase": phase,
        "timestamp": datetime.time().isoformat(),
    }
    print(json.dumps(progress_data), file=sys.stderr, flush=True)


def log_status(status: str, message: str, data: Optional[Dict] = None):
    """Output status updates for API"""
    status_data = {
        "type": "status",
        "status": status,
        "message": message,
        "timestamp": datetime.time().isoformat(),
    }
    if data:
        status_data.update(data)
    print(json.dumps(status_data), file=sys.stderr, flush=True)


def validate_config(cfg: DictConfig) -> bool:
    """Validate configuration before training"""
    required = ["data.ticker", "model.name", "train.epochs", "train.version"]
    for key in required:
        parts = key.split(".")
        val = cfg
        for p in parts:
            if p not in val:
                print(f"ERROR: Missing required config: {key}", file=sys.stderr)
                return False
            val = val[p]

    try:
        if cfg.data.start_date:
            datetime.strptime(str(cfg.data.start_date), "%Y-%m-%d")
    except ValueError:
        print("ERROR: Invalid start_date format. Use YYYY-MM-DD", file=sys.stderr)
        return False

    return True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""

    if not validate_config(cfg):
        sys.exit(1)

    if cfg.data.end_date is None:
        cfg.data.end_date = date.today().isoformat()

    set_seed(cfg.train.seed)

    models_dir = cfg.train.get("model_artifacts_dir") or os.getenv(
        "MODEL_ARTIFACTS_DIR", "data/models"
    )
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {cfg.data.ticker} with model {cfg.model.name}")
    print(f"Models will be saved to: {models_path}")
    print(f"Config: {OmegaConf.to_yaml(cfg)}", flush=True)

    log_status(
        "initializing",
        "Loading data and preparing model",
        {
            "ticker": cfg.data.ticker,
            "model": cfg.model.name,
            "version": cfg.train.version,
            "models_dir": str(models_path),
        },
    )

    # Device selection
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}", flush=True)

    try:
        train_dl, val_dl, scaler_X, scaler_y = get_dataloaders_multi(
            ticker=cfg.data.ticker,
            seq_length=cfg.model.params.seq_length,
            horizon=cfg.train.horizon,
            batch_size=cfg.data.batch_size,
            lookback_days=cfg.data.lookback_days,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            return_scalers=True,
        )
        log_status(
            "data_loaded",
            f"Loaded {len(train_dl)} train batches, {len(val_dl)} val batches",
        )
    except Exception as e:
        log_status("error", f"Failed to load data: {str(e)}")
        raise

    # Optuna HPO
    if cfg.optimization.enable:
        log_status("hpo_start", f"Starting HPO with {cfg.optimization.n_trials} trials")
        try:
            best_params, _ = optimize_model(train_dl, val_dl, cfg, device)
            for k, v in best_params.items():
                if k in cfg.model.params:
                    cfg.model.params[k] = v
                elif k == "learning_rate":
                    cfg.train.lr = v
            log_status("hpo_complete", "HPO finished", {"best_params": best_params})
        except Exception as e:
            log_status("warning", f"HPO failed: {e}, continuing with default params")

    X0, _ = next(iter(train_dl))
    num_feat = X0.shape[-1]

    model = get_model(
        cfg.model.name,
        num_features=num_feat,
        horizon=cfg.train.horizon,
        **cfg.model.params,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=(
            cfg.optimization.weight_decay
            if cfg.optimization.enable
            else cfg.train.get("weight_decay", 0.01)
        ),
    )
    criterion = nn.HuberLoss()

    # Scheduler
    scheduler = None
    if cfg.optimization.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.optimization.scheduler_factor,
            patience=cfg.optimization.scheduler_patience,
            mode="min",
        )

    out_dir = models_path / cfg.train.version
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model will be saved to: {out_dir}", flush=True)

    # Training loop
    best_val = float("inf")
    no_improve = 0
    early_patience = cfg.train.get("early_stopping_patience", 5)
    epochs = cfg.train.epochs

    log_status("training_start", f"Starting training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # — Training phase
        model.train()
        train_loss = 0.0

        for X, Y in train_dl:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        # — Validation phase
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for Xv, Yv in val_dl:
                Xv, Yv = Xv.to(device), Yv.to(device)
                p = model(Xv)
                val_loss += criterion(p, Yv).item()
                all_preds.append(p.cpu().numpy())
                all_targets.append(Yv.cpu().numpy())

        val_loss /= len(val_dl)

        # — Metrics calculation
        preds_np = np.concatenate(all_preds)
        tgt_np = np.concatenate(all_targets)
        inv_p = scaler_y.inverse_transform(preds_np)
        inv_t = scaler_y.inverse_transform(tgt_np)
        rmse = np.sqrt(mean_squared_error(inv_t, inv_p))
        mae = mean_absolute_error(inv_t, inv_p)

        epoch_time = time.time() - epoch_start

        # — Logging
        metrics = {
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "epoch_time": round(epoch_time, 2),
        }

        log_progress(epoch, epochs, metrics, "training")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"RMSE={rmse:.4f} | MAE={mae:.4f} | "
            f"time={epoch_time:.2f}s",
            flush=True,
        )

        # — Checkpointing
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0

            model_path = out_dir / f"{cfg.data.ticker}_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                model_path,
            )

            print(f"  -> Saved new best model (val_loss={val_loss:.4f})", flush=True)
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{early_patience})", flush=True)

        # — Scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # — Early stopping
        if no_improve >= early_patience:
            print(f"\nEarly stopping at epoch {epoch}", flush=True)
            log_status(
                "early_stopping",
                f"Stopped at epoch {epoch}",
                {"best_val_loss": round(best_val, 6)},
            )
            break

    # — Save final artifacts
    log_status("saving", "Saving model artifacts")

    scaler_X_path = out_dir / f"{cfg.data.ticker}_scaler_X.pkl"
    scaler_y_path = out_dir / f"{cfg.data.ticker}_scaler_y.pkl"

    with open(scaler_X_path, "wb") as fx:
        pickle.dump(scaler_X, fx)
    with open(scaler_y_path, "wb") as fy:
        pickle.dump(scaler_y, fy)

    # — Update metadata
    try:
        md = load_training_metadata()
        ticker_md = md.setdefault(
            cfg.data.ticker, {"active_version": None, "versions": {}}
        )
        version = cfg.train.version

        model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
        model_params["output_dim"] = cfg.train.horizon
        model_params["horizon"] = cfg.train.horizon
        model_params["num_features"] = num_feat

        ticker_md["versions"][version] = {
            "train_date": cfg.data.end_date,
            "data_upto": cfg.data.end_date,
            "factory_key": cfg.model.name,
            "model_params": model_params,
            "metrics": {
                "final_val_loss": float(best_val),
                "final_rmse": float(rmse),
                "final_mae": float(mae),
                "epochs_trained": epoch,
            },
        }
        ticker_md["active_version"] = version

        save_training_metadata(_ensure_str_keys(md))

        log_status(
            "completed",
            "Training completed successfully",
            {
                "ticker": cfg.data.ticker,
                "version": version,
                "model_path": str(out_dir),
                "final_metrics": {
                    "val_loss": float(best_val),
                    "rmse": float(rmse),
                    "mae": float(mae),
                },
            },
        )

        print(f"\n✓ Training completed for {cfg.data.ticker}@{version}")
        print(f"  Model saved to: {out_dir}")
        print(f"  Final metrics: RMSE={rmse:.4f}, MAE={mae:.4f}")

    except Exception as e:
        log_status("error", f"Failed to save metadata: {str(e)}")
        raise

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code if exit_code is not None else 0)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        log_status(
            "error", f"Training failed: {str(e)}", {"error_type": type(e).__name__}
        )
        raise
