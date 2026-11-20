import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import prepare_data, get_dataloaders
from model import build_model, ARCHITECTURES
from train import train_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))


def predict_unscaled(model, loader, y_scaler, device='cpu'):
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            out = model(x_batch).squeeze().cpu().numpy()
            preds_scaled.extend(out if out.ndim > 0 else [out.item()])

    preds_scaled = np.array(preds_scaled)
    # Inverse transform to original physical bike counts
    preds_bikes = preds_scaled * y_scaler.scale_[0] + y_scaler.mean_[0]
    return np.clip(preds_bikes, 0, None)


def evaluate_on_loader(model, loader, y_scaler, y_true_raw, device='cpu'):
    preds = predict_unscaled(model, loader, y_scaler, device)
    mae = mean_absolute_error(y_true_raw, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true_raw, preds)))
    r2 = r2_score(y_true_raw, preds)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'preds': preds}


# Define the 10 systematic tuning configurations across 4 hyperparameter axes
TUNING_TRIALS = [
    # Axis 1: Hidden Layer Depth (3 vs 5 vs 7 layers)
    {'trial': 1, 'axis': 'Depth', 'arch': '3_layer',        'lr': 0.001, 'dropout': 0.2, 'note': '3 hidden layers'},
    {'trial': 2, 'axis': 'Depth', 'arch': '5_layer',        'lr': 0.001, 'dropout': 0.2, 'note': '5 hidden layers (baseline)'},
    {'trial': 3, 'axis': 'Depth', 'arch': '7_layer',        'lr': 0.001, 'dropout': 0.2, 'note': '7 hidden layers'},

    # Axis 2: Layer Width / Neurons (Narrow vs Medium vs Wide with 5 layers)
    {'trial': 4, 'axis': 'Width', 'arch': '5_layer_narrow', 'lr': 0.001, 'dropout': 0.2, 'note': 'Narrow (128 base)'},
    {'trial': 5, 'axis': 'Width', 'arch': '5_layer_wide',   'lr': 0.001, 'dropout': 0.2, 'note': 'Wide (512 base)'},

    # Axis 3: Learning Rate (Step size exploration with 5 layers)
    {'trial': 6, 'axis': 'Learning Rate', 'arch': '5_layer', 'lr': 0.01,   'dropout': 0.2, 'note': 'Fast lr (0.01)'},
    {'trial': 7, 'axis': 'Learning Rate', 'arch': '5_layer', 'lr': 0.0001, 'dropout': 0.2, 'note': 'Slow lr (0.0001)'},

    # Axis 4: Regularization / Dropout (0.0 to 0.3 with 5 layers)
    {'trial': 8, 'axis': 'Dropout', 'arch': '5_layer', 'lr': 0.001, 'dropout': 0.0, 'note': 'No dropout (p=0.0)'},
    {'trial': 9, 'axis': 'Dropout', 'arch': '5_layer', 'lr': 0.001, 'dropout': 0.1, 'note': 'Light dropout (p=0.1)'},
    {'trial': 10, 'axis': 'Dropout', 'arch': '5_layer', 'lr': 0.001, 'dropout': 0.3, 'note': 'Heavy dropout (p=0.3)'},
]


def run_tuning(epochs_per_trial=30, batch_size=64, device='cpu'):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("Assignment 2: Hyperparameter Tuning Experiments")
    print("=" * 70)

    data = prepare_data()
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=batch_size)

    y_val_raw = data['val_df']['cnt'].values
    y_scaler = data['y_scaler']

    results = []
    best_val_mae = float('inf')
    best_model_weights = None
    best_trial_config = None
    best_history = None

    print(f"\nRunning {len(TUNING_TRIALS)} controlled tuning trials across 4 axes...")
    print(f"{'Trial':<6} {'Axis':<14} {'Arch':<16} {'LR':<8} {'Drop':<6} {'Params':<8} {'Val Loss':<10} {'Val MAE':<10} {'Val R2'}")
    print("-" * 88)

    for cfg in TUNING_TRIALS:
        t_num = cfg['trial']
        arch = cfg['arch']
        lr = cfg['lr']
        dropout = cfg['dropout']
        axis = cfg['axis']

        # Instantiate model architecture
        model = build_model(arch=arch, input_dim=27, dropout_rate=dropout)
        n_params = model.count_parameters()
        n_layers = len(ARCHITECTURES[arch])

        # Train model
        trained_model, history = train_network(
            model,
            train_loader,
            val_loader,
            epochs=epochs_per_trial,
            lr=lr,
            patience=7,
            device=device,
            verbose=False
        )

        # Evaluate on validation set
        metrics = evaluate_on_loader(trained_model, val_loader, y_scaler, y_val_raw, device=device)
        best_loss = min(history['val_loss'])

        results.append({
            'Trial': t_num,
            'Axis': axis,
            'Architecture': arch,
            'Hidden_Layers': n_layers,
            'Layer_Dims': str(ARCHITECTURES[arch]),
            'Parameters': n_params,
            'Learning_Rate': lr,
            'Dropout': dropout,
            'Best_Val_Loss': round(best_loss, 4),
            'Val_MAE': round(metrics['MAE'], 2),
            'Val_RMSE': round(metrics['RMSE'], 2),
            'Val_R2': round(metrics['R2'], 4),
            'Epochs_Trained': len(history['val_loss']),
            'Description': cfg['note'],
        })

        is_best = ""
        if metrics['MAE'] < best_val_mae:
            best_val_mae = metrics['MAE']
            best_model_weights = trained_model.state_dict()
            best_trial_config = cfg
            best_history = history
            is_best = " (NEW BEST)"

        print(f"#{t_num:<5} {axis:<14} {arch:<16} {lr:<8} {dropout:<6} {n_params:<8} {best_loss:<10.4f} {metrics['MAE']:<10.2f} {metrics['R2']:<6.4f}{is_best}")

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, 'tuning_summary.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved full tuning summary to: {csv_path}")

    # Save champion model weights and its training history
    if best_model_weights is not None:
        weights_path = os.path.join(RESULTS_DIR, 'best_model.pt')
        torch.save(best_model_weights, weights_path)

        hist_df = pd.DataFrame(best_history)
        hist_df.to_csv(os.path.join(RESULTS_DIR, 'best_training_history.csv'), index=False)
        print(f"Saved champion model checkpoint to: {weights_path}")

    print("\n" + "=" * 70)
    print(f"Tuning Completed! Champion Config: Trial #{best_trial_config['trial']} ({best_trial_config['arch']}, lr={best_trial_config['lr']}, dropout={best_trial_config['dropout']})")
    print(f"Champion Validation Score: MAE = {best_val_mae:.2f} bikes")
    print("=" * 70)

    return results_df, best_trial_config


if __name__ == '__main__':
    run_tuning(epochs_per_trial=25)
