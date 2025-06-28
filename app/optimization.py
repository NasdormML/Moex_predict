import ast
import inspect
import logging
from importlib import import_module

import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from app.models.factory import AVAILABLE

log = logging.getLogger(__name__)


def build_model_and_lr(
    model_name: str,
    model_cfg: dict,
    opt_cfg: DictConfig,
    trial: optuna.Trial,
    num_features: int = None,
    horizon: int = None,
):
    """
    Создаёт модель и learning rate для Optuna.
    Подставляет num_features и horizon в параметры build_model.
    """
    # Learning rate
    lr = trial.suggest_float("learning_rate", *opt_cfg.lr_range, log=True)

    # Подготовка параметров модели
    search_space = OmegaConf.to_container(opt_cfg.search_space, resolve=True)
    params = {}

    for key, base_val in model_cfg.items():
        if key in search_space:
            # Optuna подбор для гиперпараметров
            choices = search_space[key]
            if isinstance(choices, list) and key == "num_channels":
                choices_str = [str(lst) for lst in choices]
                sel = trial.suggest_categorical(key, tuple(choices_str))
                params[key] = ast.literal_eval(sel)
            else:
                params[key] = trial.suggest_categorical(
                    key, tuple(choices) if isinstance(choices, list) else choices
                )
        else:
            params[key] = base_val

    # Подставляем num_features и horizon
    if num_features is not None:
        params["num_features"] = num_features
    if horizon is not None:
        params["horizon"] = horizon
        if "output_dim" in params:
            params["output_dim"] = horizon

    if "seq_length" in params:
        params["seq_length"] = params.pop("seq_length")

    # Импорт модели и фильтрация параметров
    module_path = AVAILABLE.get(model_name)
    if not module_path:
        raise ValueError(f"Unknown model: {model_name}")
    module = import_module(module_path)
    build_fn = module.build_model

    sig = inspect.signature(build_fn)
    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}

    model = build_fn(**filtered_params)
    return model, lr


def get_scheduler_if(opt_cfg: DictConfig, optimizer):
    if opt_cfg.use_scheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=opt_cfg.scheduler_factor,
            patience=opt_cfg.scheduler_patience,
            mode="min",
        )
    return None


def train_one_epoch(model: torch.nn.Module, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model: torch.nn.Module, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def objective(trial: optuna.Trial, train_dl, val_dl, cfg: DictConfig, device):
    model_cfg = OmegaConf.to_container(cfg.model.params, resolve=True)
    # получаем число признаков из первого батча
    try:
        X0, Y0 = next(iter(train_dl))
        num_feat = X0.shape[-1]
    except Exception:
        raise RuntimeError("Не удалось определить num_features из train_dl")
    horizon = getattr(cfg.train, "horizon", None)

    model, lr = build_model_and_lr(
        cfg.model.name,
        model_cfg,
        cfg.optimization,
        trial,
        num_features=num_feat,
        horizon=horizon,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=cfg.optimization.weight_decay
    )
    scheduler = get_scheduler_if(cfg.optimization, optimizer)
    criterion = torch.nn.HuberLoss()

    for epoch in range(cfg.optimization.epochs_per_trial):
        train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss = evaluate(model, val_dl, criterion, device)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if scheduler:
            scheduler.step(val_loss)
    return val_loss


def optimize_model(train_dl, val_dl, cfg: DictConfig, device=None):
    sampler = optuna.samplers.TPESampler(seed=cfg.optimization.n_startup_trials)
    pruner = (
        optuna.pruners.HyperbandPruner(
            reduction_factor=cfg.optimization.sh_reduction_factor,
            min_early_stopping_rate=cfg.optimization.n_startup_trials,
        )
        if cfg.optimization.use_hyperband
        else optuna.pruners.MedianPruner()
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda t: objective(t, train_dl, val_dl, cfg, device),
        n_trials=cfg.optimization.n_trials,
        n_jobs=cfg.optimization.n_jobs,
    )
    return study.best_params, study
