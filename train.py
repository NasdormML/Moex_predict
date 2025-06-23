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

from app.data import get_dataloaders
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
    set_seed(cfg.train.seed)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_dl, val_dl, scaler_X, scaler_y = get_dataloaders(
        ticker=cfg.data.ticker,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        return_scalers=True,
        seq_len=cfg.model.params.seq_length,
        lookback_days=cfg.data.lookback_days,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )

    if cfg.optimization.enable:
        best_params, _ = optimize_model(train_dl, val_dl, cfg, device)
        for k, v in best_params.items():
            if k in cfg.model.params:
                cfg.model.params[k] = v
            elif k == "learning_rate":
                cfg.train.lr = v

    model = get_model(cfg.model.name, **cfg.model.params).to(device)
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
        for X, y in train_dl:
            X, y = X.to(device), y.to(device).squeeze(-1)
            optimizer.zero_grad()
            preds = model(X).squeeze(-1)
            loss = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for Xv, yv in val_dl:
                Xv, yv = Xv.to(device), yv.to(device).squeeze(-1)
                preds = model(Xv).squeeze(-1)
                val_loss += criterion(preds, yv).item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yv.cpu().numpy())
        val_loss /= len(val_dl)

        inv_p = scaler_y.inverse_transform(np.concatenate(all_preds).reshape(-1, 1))
        inv_t = scaler_y.inverse_transform(np.concatenate(all_targets).reshape(-1, 1))
        mse_val = mean_squared_error(inv_t, inv_p)
        rmse = np.sqrt(mse_val)
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

    with open(os.path.join(out_dir, f"{cfg.data.ticker}_scaler_X.pkl"), "wb") as fx:
        pickle.dump(scaler_X, fx)
    with open(os.path.join(out_dir, f"{cfg.data.ticker}_scaler_y.pkl"), "wb") as fy:
        pickle.dump(scaler_y, fy)

    # Обновляем метаданные и приводим ключи к str
    md = load_training_metadata()
    ticker_md = md.setdefault(cfg.data.ticker, {"active_version": None, "versions": {}})
    version = cfg.train.version
    ticker_md["versions"][version] = {
        "train_date": cfg.data.end_date,
        "data_upto": cfg.data.end_date,
        "factory_key": cfg.model.name,
        "model_params": OmegaConf.to_container(cfg.model.params, resolve=True),
    }
    ticker_md["active_version"] = version

    # чистим и сохраняем
    md_clean = _ensure_str_keys(md)
    save_training_metadata(md_clean)
    try:
        save_training_metadata(md_clean)
        print("Метаданные успешно сохранены.")
    except Exception as e:
        print("Ошибка при сохранении метаданных:", e)


if __name__ == "__main__":
    main()
