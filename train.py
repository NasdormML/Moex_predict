import os
import pickle
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig

from app.data import get_dataloaders
from app.models.factory import get_model
from app.transfer_learning import load_training_metadata, save_training_metadata


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # извлекаем тикер и параметры из конфига
    ticker = cfg.data.ticker

    # получаем даталоадеры и скейлеры
    train_dl, val_dl, scaler_X, scaler_y = get_dataloaders(
        ticker=ticker,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        return_scalers=True,
        seq_len=cfg.model.params.seq_length,
        lookback_days=cfg.data.lookback_days,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )

    # создаём модель и оптимизатор
    model = get_model(cfg.model.name, **cfg.model.params)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = torch.nn.MSELoss()

    # обучение
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_dl:
            pred = model(Xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{cfg.train.epochs}, loss={avg:.4f}")

    # сохраняем артефакты
    root = cfg.train.model_artifacts_dir
    version = cfg.train.version
    out_dir = os.path.join(root, version)
    os.makedirs(out_dir, exist_ok=True)

    # сохраняем веса и скейлеры под именем тикера
    torch.save(model.state_dict(), os.path.join(out_dir, f"{ticker}_model.pth"))
    with open(os.path.join(out_dir, f"{ticker}_scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(out_dir, f"{ticker}_scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    print(f"Saved artifacts to {out_dir}")

    # обновляем метаданные о последнем обучении
    md = load_training_metadata(version)
    md[ticker] = datetime.today().strftime("%Y-%m-%d")
    md[f"{ticker}_model_version"] = version
    md[f"{cfg.data.ticker}_factory_key"] = cfg.model.name
    md[f"{ticker}_model_params"] = cfg.model.params
    save_training_metadata(md, version)
    print(f"Updated training metadata for {ticker} version {version}")


if __name__ == "__main__":
    main()
