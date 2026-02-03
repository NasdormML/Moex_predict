import ast
import inspect
import logging
from importlib import import_module

import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from app.models.factory import AVAILABLE

logger = logging.getLogger(__name__)


def build_model_and_lr(
    model_name: str,
    model_cfg: dict,
    opt_cfg: DictConfig,
    trial: optuna.Trial,
    num_features: int | None = None,
    horizon: int | None = None,
):
    """
    Create model and learning rate for Optuna trial.
    """
    # Learning rate
    lr = trial.suggest_float("learning_rate", *opt_cfg.lr_range, log=True)

    # Build parameters
    search_space = OmegaConf.to_container(opt_cfg.search_space, resolve=True)
    params = {}

    for key, base_val in model_cfg.items():
        if key in search_space:
            choices = search_space[key]
            if isinstance(choices, list) and key == "num_channels":
                # Handle list choices properly
                choices_str = [str(lst) for lst in choices]
                sel = trial.suggest_categorical(key, choices_str)
                params[key] = ast.literal_eval(sel)
            else:
                params[key] = trial.suggest_categorical(
                    key, tuple(choices) if isinstance(choices, list) else choices
                )
        else:
            params[key] = base_val

    # Inject dynamic parameters
    if num_features is not None:
        params["num_features"] = num_features
    if horizon is not None:
        params["horizon"] = horizon
        if "output_dim" in params:
            params["output_dim"] = horizon

    # Filter by signature
    module_path = AVAILABLE.get(model_name)
    if not module_path:
        raise ValueError(f"Unknown model: {model_name}")

    module = import_module(module_path)
    build_fn = module.build_model
    sig = inspect.signature(build_fn)

    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}

    model = build_fn(**filtered_params)
    return model, lr


def get_scheduler(cfg: DictConfig, optimizer):
    """Get learning rate scheduler if enabled."""
    if not cfg.use_scheduler:
        return None

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        mode="min",
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def evaluate(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> float:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def objective(
    trial: optuna.Trial,
    train_dl,
    val_dl,
    cfg: DictConfig,
    device: torch.device,
) -> float:
    """
    Optuna objective function.

    Returns validation loss for hyperparameter optimization.
    """
    # Get input dimensions
    try:
        X0, _ = next(iter(train_dl))
        num_feat = X0.shape[-1]
    except Exception as e:
        raise RuntimeError(f"Failed to determine num_features: {e}")

    horizon = getattr(cfg.train, "horizon", None)

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model.params, resolve=True)
    model, lr = build_model_and_lr(
        cfg.model.name,
        model_cfg,
        cfg.optimization,
        trial,
        num_features=num_feat,
        horizon=horizon,
    )
    model.to(device)

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.optimization.weight_decay,
    )
    scheduler = get_scheduler(cfg.optimization, optimizer)
    criterion = torch.nn.HuberLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = getattr(cfg.optimization, "early_stopping_patience", 10)

    # Training loop with early stopping
    for epoch in range(cfg.optimization.epochs_per_trial):
        _ = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss = evaluate(model, val_dl, criterion, device)

        # Report to Optuna
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    return best_val_loss


def optimize_model(
    train_dl,
    val_dl,
    cfg: DictConfig,
    device: torch.device | None = None,
):
    """
    Run hyperparameter optimization.

    Returns best parameters and study object.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure sampler and pruner
    sampler = optuna.samplers.TPESampler(
        seed=cfg.optimization.get("seed", 42),
        n_startup_trials=cfg.optimization.n_startup_trials,
    )

    if cfg.optimization.use_hyperband:
        pruner = optuna.pruners.HyperbandPruner(
            reduction_factor=cfg.optimization.sh_reduction_factor,
            min_early_stopping_rate=cfg.optimization.n_startup_trials,
        )
    else:
        pruner = optuna.pruners.MedianPruner()

    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # Optimize with exception handling for pruned trials
    def safe_objective(trial):
        try:
            return objective(trial, train_dl, val_dl, cfg, device)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise optuna.TrialPruned()

    study.optimize(
        safe_objective,
        n_trials=cfg.optimization.n_trials,
        n_jobs=cfg.optimization.n_jobs,
        show_progress_bar=True,
    )

    return study.best_params, study
