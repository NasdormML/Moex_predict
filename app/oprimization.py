import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from app.models import PricePredictionModel
from torch.utils.data import DataLoader

def objective(trial, train_loader, val_loader, seq_length, num_features, device):
    lstm_units = trial.suggest_int("lstm_units", 64, 256, step=32)
    fc_units = trial.suggest_int("fc_units", 32, 128, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    model = PricePredictionModel(seq_length, num_features, output_dim=1,
                                 lstm_units=lstm_units, fc_units=fc_units, dropout_rate=dropout_rate)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return avg_val_loss

def optimize_model(train_loader, val_loader, seq_length, num_features, n_trials=30, device=torch.device('cpu')):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, seq_length, num_features, device), n_trials=n_trials)
    return study.best_params, study
