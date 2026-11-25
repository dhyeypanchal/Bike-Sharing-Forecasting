import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import prepare_data, get_dataloaders
from model import build_model

SEED = 42


def set_seed(seed=SEED):
    """Fix all random sources (weight init, batch shuffling) so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(x_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(x_batch)

    return total_loss / len(loader.dataset)


def mae_in_bikes(model, dataset, y_scaler, device):
    """Mean absolute error in real bike counts (predictions un-scaled and clipped at 0).
    Reads the dataset tensors directly instead of a DataLoader, because starting a
    DataLoader draws a random number and would change the batch order of later epochs."""
    model.eval()
    scale, mean = float(y_scaler.scale_[0]), float(y_scaler.mean_[0])
    with torch.no_grad():
        preds = model(dataset.X.to(device)).cpu() * scale + mean
    actual = dataset.y * scale + mean
    return (preds.clamp(min=0) - actual).abs().mean().item()


def train_network(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=0.001,
    weight_decay=1e-4,
    patience=10,
    device='cpu',
    verbose=True,
    y_scaler=None
):
    """If y_scaler is given, train and validation MAE in bikes are also recorded every epoch."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }
    if y_scaler is not None:
        history['train_mae'] = []
        history['val_mae'] = []

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        if y_scaler is not None:
            history['train_mae'].append(mae_in_bikes(model, train_loader.dataset, y_scaler, device))
            history['val_mae'].append(mae_in_bikes(model, val_loader.dataset, y_scaler, device))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved_flag = "*"
        else:
            patience_counter += 1
            improved_flag = ""

        if verbose and (epoch % 5 == 0 or epoch == 1 or improved_flag == "*"):
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {improved_flag}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    # Restore weights that achieved the best validation score
    model.load_state_dict(best_weights)
    return model, history


if __name__ == '__main__':
    set_seed()
    data = prepare_data()
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=64)

    print("Training 5-layer default network for 15 quick epochs...")
    model = build_model('5_layer', input_dim=len(data['feature_cols']), dropout_rate=0.2)
    trained_model, hist = train_network(model, train_loader, val_loader, epochs=15, lr=0.001)

    print("Initial train loss:", round(hist['train_loss'][0], 4))
    print("Final train loss  :", round(hist['train_loss'][-1], 4))
    print("Best val loss     :", round(min(hist['val_loss']), 4))
