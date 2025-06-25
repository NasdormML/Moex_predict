import os
import pickle
import random

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
    # default end_date today
    from datetime import date

    if cfg.data.end_date is None:
        cfg.data.end_date = date.today().isoformat()

    set_seed(cfg.train.seed)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Multi-step data loaders
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

    # Hyperparameter optimization
    if cfg.optimization.enable:
        best_params, _ = optimize_model(train_dl, val_dl, cfg, device)
        for k, v in best_params.items():
            if k in cfg.model.params:
                cfg.model.params[k] = v
            elif k == "learning_rate":
                cfg.train.lr = v

    # Instantiate model with horizon
    try:
        X0, _ = next(iter(train_dl))
        num_feat = X0.shape[-1]
    except Exception:
        raise RuntimeError("Не удалось определить num_features из train_dl")

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
            else cfg.train.weight_decay
        ),
    )
    criterion = nn.HuberLoss()

    out_dir = os.path.join(cfg.train.model_artifacts_dir, cfg.train.version)
    os.makedirs(out_dir, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss = 0.0
        for X, Y in train_dl:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            preds = model(X)  # [B, horizon]
            loss = criterion(preds, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for Xv, Yv in val_dl:
                Xv, Yv = Xv.to(device), Yv.to(device)
                preds = model(Xv)
                val_loss += criterion(preds, Yv).item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(Yv.cpu().numpy())
        val_loss /= len(val_dl)

        # inverse scaling and metrics
        preds_np = np.concatenate(all_preds)  # (N, horizon)
        tgt_np = np.concatenate(all_targets)
        inv_p = scaler_y.inverse_transform(preds_np)
        inv_t = scaler_y.inverse_transform(tgt_np)
        rmse = np.sqrt(mean_squared_error(inv_t, inv_p))
        mae = mean_absolute_error(inv_t, inv_p)

        print(
            f"Epoch {epoch}/{cfg.train.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss_scaled={val_loss:.4f} | "
            f"RMSE={rmse:.4f} | MAE={mae:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"{cfg.data.ticker}_model.pth"),
            )

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

    ticker_md["versions"][version] = {
        "train_date": cfg.data.end_date,
        "data_upto": cfg.data.end_date,
        "factory_key": cfg.model.name,
        "model_params": model_params,
    }
    ticker_md["active_version"] = version

    md_clean = _ensure_str_keys(md)
    save_training_metadata(md_clean)
    print(f"Метаданные для {cfg.data.ticker}@{version} сохранены.")


if __name__ == "__main__":
    main()
