import os
import pickle
import random

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.data import get_dataloaders_multi
from app.models.factory import get_model
from app.models.quantile_loss import QuantileLoss, CoverageMetric
from app.optimization import optimize_model
from app.transfer_learning import load_training_metadata, save_training_metadata


class AsymmetricHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0, alpha: float = 1.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
    
    def forward(self, pred, target):
        error = target - pred
        weight = torch.where(error > 0, self.alpha, 1.0)
        abs_err = torch.abs(error)
        is_small = abs_err <= self.delta
        small_loss = 0.5 * error**2
        large_loss = self.delta * (abs_err - 0.5 * self.delta)
        return torch.mean(weight * torch.where(is_small, small_loss, large_loss))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _ensure_str_keys(obj):
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from datetime import date

    if cfg.data.end_date is None:
        cfg.data.end_date = date.today().isoformat()

    set_seed(cfg.train.seed)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )

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

    # Optuna HPO
    if cfg.optimization.enable:
        best_params, _ = optimize_model(train_dl, val_dl, cfg, device)
        for k, v in best_params.items():
            if k in cfg.model.params:
                cfg.model.params[k] = v
            elif k == "learning_rate":
                cfg.train.lr = v

    # Model + optimizer + criterion
    X0, _ = next(iter(train_dl))
    num_feat = X0.shape[-1]
    
    # quantiles в params
    model_params = dict(cfg.model.params)
    if cfg.model.name.startswith("quantile_"):
        model_params["quantiles"] = cfg.model.quantiles
        print(f"[Train] Quantile mode: {cfg.model.quantiles}")
    
    model = get_model(
        cfg.model.name,
        num_features=num_feat,
        horizon=cfg.train.horizon,
        **model_params,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=(
            cfg.optimization.weight_decay
            if cfg.optimization.enable
            else cfg.train.weight_decay
        ),
    )
    
    # Динамический выбор loss
    if cfg.model.name.startswith("quantile_"):
        criterion = QuantileLoss(cfg.model.quantiles)
        print(f"[Train] Using QuantileLoss")
    else:
        criterion = AsymmetricHuberLoss(delta=1.0, alpha=1.5)
        print(f"[Train] Using AsymmetricHuberLoss")

    # ReduceLROnPlateau
    if cfg.optimization.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.optimization.scheduler_factor,
            patience=cfg.optimization.scheduler_patience,
            mode="min",
        )
    else:
        scheduler = None

    out_dir = os.path.join(cfg.train.model_artifacts_dir, cfg.train.version)
    os.makedirs(out_dir, exist_ok=True)

    best_val = float("inf")
    no_improve = 0
    early_patience = 8

    # MLflow init
    mlflow_enabled = cfg.get("mlflow", {}).get("enabled", False)
    if mlflow_enabled:
        mlflow.start_run(run_name=f"{cfg.data.ticker}_{cfg.model.name}_{cfg.train.version}")

    for epoch in range(1, cfg.train.epochs + 1):
        # — train
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

        # — validation
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

        # — metrics
        preds_np = np.concatenate(all_preds)
        tgt_np = np.concatenate(all_targets)
        
        # Для квантилей (0.5)
        is_quantile = cfg.model.name.startswith("quantile_")
        if is_quantile:
            preds_for_metrics = preds_np[:, :, 1]  # [B, H] — 50% квантиль
        else:
            preds_for_metrics = preds_np
        
        inv_p = scaler_y.inverse_transform(preds_for_metrics.reshape(-1, preds_for_metrics.shape[-1]))
        inv_t = scaler_y.inverse_transform(tgt_np.reshape(-1, tgt_np.shape[-1]))
        rmse = np.sqrt(mean_squared_error(inv_t, inv_p))
        mae = mean_absolute_error(inv_t, inv_p)

        # — coverage для квантильных
        coverage_metrics = {}
        if is_quantile:
            coverage_lower, coverage_upper = CoverageMetric.calculate_coverage(
                torch.tensor(preds_np), 
                torch.tensor(tgt_np), 
                cfg.model.quantiles
            )
            if coverage_lower is not None:
                coverage_metrics = {
                    "val_coverage_lower": coverage_lower,
                    "val_coverage_upper": coverage_upper,
                    "val_coverage_error": abs(coverage_lower - 0.05) + abs(coverage_upper - 0.95)
                }
                print(f"  Coverage: {coverage_lower:.1%} / {coverage_upper:.1%}")

        print(
            f"Epoch {epoch}/{cfg.train.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"RMSE={rmse:.4f} | MAE={mae:.4f}"
        )

        # — MLflow logging
        if mlflow_enabled:
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": rmse,
                "val_mae": mae,
                **coverage_metrics
            }
            mlflow.log_metrics(metrics, step=epoch)

        # — checkpoint
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"{cfg.data.ticker}_model.pth"),
            )
        else:
            no_improve += 1

        if scheduler:
            scheduler.step(val_loss)

        if no_improve >= early_patience:
            print(f"Early stopping at epoch {epoch}, no improvement for {early_patience} epochs.")
            break

    # Save scalers
    with open(os.path.join(out_dir, f"{cfg.data.ticker}_scaler_X.pkl"), "wb") as fx:
        pickle.dump(scaler_X, fx)
    with open(os.path.join(out_dir, f"{cfg.data.ticker}_scaler_y.pkl"), "wb") as fy:
        pickle.dump(scaler_y, fy)

    # Update metadata
    md = load_training_metadata()
    ticker_md = md.setdefault(cfg.data.ticker, {"active_version": None, "versions": {}})
    version = cfg.train.version

    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    model_params["output_dim"] = cfg.train.horizon
    model_params["horizon"] = cfg.train.horizon
    model_params["num_features"] = num_feat
    
    # Сохраняем quantiles в metadata
    if is_quantile:
        model_params["quantiles"] = cfg.model.quantiles

    ticker_md["versions"][version] = {
        "train_date": cfg.data.end_date,
        "data_upto": cfg.data.end_date,
        "factory_key": cfg.model.name,
        "model_params": model_params,
    }
    ticker_md["active_version"] = version

    save_training_metadata(_ensure_str_keys(md))
    print(f"Метаданные для {cfg.data.ticker}@{version} сохранены.")
    
    if mlflow_enabled:
        mlflow.end_run()


if __name__ == "__main__":
    main()
