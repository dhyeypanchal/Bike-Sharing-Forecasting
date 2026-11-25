import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import prepare_data, get_dataloaders
from model import build_model
from tune import predict_unscaled

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
A1_RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'assignment1', 'results'))
A1_SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'assignment1', 'src'))


def load_champion_model(input_dim, device='cpu'):
    weights_path = os.path.join(RESULTS_DIR, 'best_model.pt')
    config_path = os.path.join(RESULTS_DIR, 'best_config.json')
    for path in (weights_path, config_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run tune.py first.")

    # Rebuild exactly the network that tune.py selected
    with open(config_path) as f:
        config = json.load(f)
    if config['input_dim'] != input_dim:
        raise ValueError(f"Champion was trained on {config['input_dim']} features, "
                         f"but the data now has {input_dim}. Re-run tune.py.")

    model = build_model(arch=config['arch'], input_dim=input_dim, dropout_rate=config['dropout'])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config


def dnn_label(config):
    return f"Tuned {len(config['hidden_dims'])}-layer DNN"


def plot_learning_curve(history_path, save_path, label):
    if not os.path.exists(history_path):
        return
    history = pd.read_csv(history_path)
    epochs = np.arange(1, len(history) + 1)
    # Epoch whose weights early stopping restored (lowest validation loss)
    best_epoch = int(history['val_loss'].idxmin()) + 1

    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax_loss.plot(epochs, history['train_loss'], label='Training loss', color='#2b5c8f', lw=2)
    ax_loss.plot(epochs, history['val_loss'], label='Validation loss', color='#d95f02', lw=2, linestyle='--')
    ax_loss.set_title('Loss (MSE on standardised demand)', fontsize=11)
    ax_loss.set_ylabel('Mean squared error (standardised)', fontsize=10)

    ax_mae.plot(epochs, history['train_mae'], label='Training MAE', color='#2b5c8f', lw=2)
    ax_mae.plot(epochs, history['val_mae'], label='Validation MAE', color='#d95f02', lw=2, linestyle='--')
    ax_mae.set_title('Accuracy: mean absolute error in bikes', fontsize=11)
    ax_mae.set_ylabel('MAE (bikes per hour)', fontsize=10)

    for ax in (ax_loss, ax_mae):
        ax.axvline(best_epoch, color='#555555', linestyle=':', lw=1.5,
                   label=f'Best epoch ({best_epoch}), weights restored')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle(f'Champion {label}: learning curves', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_hyperparameter_tuning(summary_path, save_path):
    if not os.path.exists(summary_path):
        return
    df = pd.read_csv(summary_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Axis 1: Depth
    depth_df = df[df['Axis'] == 'Depth'].copy()
    depth_labels = [f"{row['Hidden_Layers']} Layers\n({row['Architecture']})" for _, row in depth_df.iterrows()]
    axes[0, 0].bar(depth_labels, depth_df['Val_MAE'], color=['#7570b3', '#1b9e77', '#d95f02'], width=0.55)
    axes[0, 0].set_title('Ablation 1: Hidden Layer Depth', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Validation MAE (bikes)', fontsize=10)
    axes[0, 0].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(depth_df['Val_MAE']):
        axes[0, 0].text(i, v + 0.6, f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')

    # Axis 2: Width
    width_df = df[(df['Axis'] == 'Width') | (df['Trial'] == 2)].sort_values('Parameters').copy()
    width_labels = [f"{row['Architecture']}\n({row['Parameters']:,} params)" for _, row in width_df.iterrows()]
    axes[0, 1].bar(width_labels, width_df['Val_MAE'], color=['#e7298a', '#1b9e77', '#66a61e'], width=0.55)
    axes[0, 1].set_title('Ablation 2: Layer Width (5 Layers)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Validation MAE (bikes)', fontsize=10)
    axes[0, 1].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(width_df['Val_MAE']):
        axes[0, 1].text(i, v + 0.6, f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')

    # Axis 3: Learning Rate
    lr_df = df[(df['Axis'] == 'Learning Rate') | (df['Trial'] == 2)].sort_values('Learning_Rate').copy()
    lr_labels = [f"lr = {row['Learning_Rate']}" for _, row in lr_df.iterrows()]
    axes[1, 0].bar(lr_labels, lr_df['Val_MAE'], color=['#e6ab02', '#1b9e77', '#a6761d'], width=0.55)
    axes[1, 0].set_title('Ablation 3: Learning Rate Step Size', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Validation MAE (bikes)', fontsize=10)
    axes[1, 0].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(lr_df['Val_MAE']):
        axes[1, 0].text(i, v + 0.6, f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')

    # Axis 4: Dropout
    drop_df = df[(df['Axis'] == 'Dropout') | (df['Trial'] == 2)].sort_values('Dropout').copy()
    drop_labels = [f"p = {row['Dropout']}" for _, row in drop_df.iterrows()]
    colors = ['#1b9e77' if r['Dropout'] == 0.0 else '#386cb0' for _, r in drop_df.iterrows()]
    axes[1, 1].bar(drop_labels, drop_df['Val_MAE'], color=colors, width=0.55)
    axes[1, 1].set_title('Ablation 4: Dropout Regularization', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Validation MAE (bikes)', fontsize=10)
    axes[1, 1].grid(axis='y', linestyle=':', alpha=0.6)
    for i, v in enumerate(drop_df['Val_MAE']):
        axes[1, 1].text(i, v + 0.6, f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Hyperparameter Tuning Exploration Across 4 Axes (Validation Set)', fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tracking_week(test_df, y_true, dnn_preds, rf_preds, save_path, label):
    # Slice the first week of October 2012 (168 hours)
    week_mask = (test_df['timestamp'] >= '2012-10-01') & (test_df['timestamp'] <= '2012-10-07 23:59:59')
    week_df = test_df[week_mask].copy()

    if len(week_df) < 100:
        week_df = test_df.iloc[:168].copy()
        week_idx = np.arange(168)
    else:
        week_idx = np.where(week_mask)[0]

    y_actual = y_true[week_idx]
    y_dnn = dnn_preds[week_idx]
    y_rf = rf_preds[week_idx] if rf_preds is not None else None

    timestamps = week_df['timestamp']

    plt.figure(figsize=(13, 5))
    plt.plot(timestamps, y_actual, label='Actual Demand', color='#111111', lw=2, alpha=0.85)
    if y_rf is not None:
        plt.plot(timestamps, y_rf, label='Assignment 1: Random Forest', color='#e66101', lw=1.5, linestyle=':', alpha=0.8)
    plt.plot(timestamps, y_dnn, label=f'Assignment 2: {label}', color='#0571b0', lw=2, linestyle='-')

    plt.title('Test Set Forecast Tracking: October 1–7, 2012 (168 Hourly Steps)', fontsize=12, pad=12)
    plt.xlabel('Date and Hour', fontsize=11)
    plt.ylabel('Bike Count (cnt)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_by_hour(test_df, y_true, dnn_preds, rf_preds, save_path, label):
    df_err = pd.DataFrame({
        'hr': test_df['hr'].values,
        'DNN_Err': np.abs(y_true - dnn_preds),
    })
    if rf_preds is not None:
        df_err['RF_Err'] = np.abs(y_true - rf_preds)

    grouped = df_err.groupby('hr').mean().reset_index()

    plt.figure(figsize=(10, 5))
    if rf_preds is not None:
        plt.plot(grouped['hr'], grouped['RF_Err'], marker='s', label='Assignment 1: Random Forest', color='#e66101', lw=1.8)
    plt.plot(grouped['hr'], grouped['DNN_Err'], marker='o', label=f'Assignment 2: {label}', color='#0571b0', lw=2)

    plt.title('Mean Absolute Error by Hour of Day (Test Set)', fontsize=12, pad=12)
    plt.xlabel('Hour of Day (0–23)', fontsize=11)
    plt.ylabel('Mean Absolute Error (bikes)', fontsize=11)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("Assignment 2: Final Model Evaluation on Held-Out Test Set (3,476 Hours)")
    print("=" * 70)

    # 1. Load data and dataloaders
    data = prepare_data()
    _, _, test_loader = get_dataloaders(data, batch_size=64)
    y_test_raw = data['y_test_raw']
    test_df = data['test_df']
    y_scaler = data['y_scaler']

    # 2. Load Champion DNN and predict
    model, config = load_champion_model(input_dim=len(data['feature_cols']))
    label = dnn_label(config)
    print(f"\nLoaded champion: trial #{config['trial']} ({config['arch']} {config['hidden_dims']}, "
          f"lr={config['lr']}, dropout={config['dropout']}, val MAE {config['val_mae']})")
    predict_start = time.time()
    dnn_preds = predict_unscaled(model, test_loader, y_scaler)
    predict_seconds = time.time() - predict_start
    # Same definition as Assignment 1: training time (measured in tune.py) + test prediction time
    dnn_fit_predict_seconds = config['fit_seconds'] + predict_seconds

    # Compute DNN Test Metrics
    dnn_mae = mean_absolute_error(y_test_raw, dnn_preds)
    dnn_rmse = float(np.sqrt(mean_squared_error(y_test_raw, dnn_preds)))
    dnn_r2 = r2_score(y_test_raw, dnn_preds)

    print(f"\n{label} Test Set Results:")
    print(f"  MAE  : {dnn_mae:.2f} bikes")
    print(f"  RMSE : {dnn_rmse:.2f} bikes")
    print(f"  R2   : {dnn_r2:.4f}")

    # 3. Fit Random Forest from Assignment 1 for direct visual tracking comparison
    rf_preds = None
    try:
        sys.path.append(A1_SRC_DIR)
        from data_loader import load_data, chronological_split
        from models import build_random_forest
        a1_raw = load_data()
        a1_train, a1_test = chronological_split(a1_raw)
        rf_model = build_random_forest()
        rf_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        rf_model.fit(a1_train[rf_cols], a1_train['cnt'])
        rf_preds = np.clip(rf_model.predict(a1_test[rf_cols]), 0, None)
        print("Successfully generated A1 Random Forest test predictions for comparison.")
    except Exception as e:
        print(f"Note: Could not run A1 Random Forest directly ({e}). Proceeding with DNN metrics.")

    # 3b. Fairness check: the A1 random forest (same settings, untuned) given the SAME
    # engineered features as the DNN, trained on train + validation hours. Note the DNN
    # itself trains only on the train hours (validation is used for early stopping).
    feature_cols = data['feature_cols']
    rf_train_df = pd.concat([data['train_df'], data['val_df']], ignore_index=True)
    rf_start = time.time()
    rf_same = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_same.fit(rf_train_df[feature_cols], rf_train_df['cnt'])
    rf_same_preds = np.clip(rf_same.predict(test_df[feature_cols]), 0, None)
    rf_same_seconds = time.time() - rf_start
    rf_same_row = pd.DataFrame([{
        'Model': f'Random Forest (A1 settings, same {len(feature_cols)} features, train+val hours)',
        'MAE': round(mean_absolute_error(y_test_raw, rf_same_preds), 2),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_test_raw, rf_same_preds))), 2),
        'R2': round(r2_score(y_test_raw, rf_same_preds), 4),
        'Fit+predict time (s)': round(rf_same_seconds, 2),
    }])

    # 3c. Same random forest trained ONLY on the DNN's training hours (validation hours
    # excluded, as the DNN uses them only for early stopping): like-for-like comparison.
    rf_train_only_start = time.time()
    rf_train_only = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_train_only.fit(data['train_df'][feature_cols], data['train_df']['cnt'])
    rf_train_only_preds = np.clip(rf_train_only.predict(test_df[feature_cols]), 0, None)
    rf_train_only_seconds = time.time() - rf_train_only_start
    rf_train_only_row = pd.DataFrame([{
        'Model': f'Random Forest (A1 settings, same {len(feature_cols)} features, train hours only)',
        'MAE': round(mean_absolute_error(y_test_raw, rf_train_only_preds), 2),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_test_raw, rf_train_only_preds))), 2),
        'R2': round(r2_score(y_test_raw, rf_train_only_preds), 4),
        'Fit+predict time (s)': round(rf_train_only_seconds, 2),
    }])

    # 4. Load A1 holdout results and construct comprehensive comparison table
    a1_csv_path = os.path.join(A1_RESULTS_DIR, 'table2_holdout_results.csv')
    if os.path.exists(a1_csv_path):
        comparison_df = pd.read_csv(a1_csv_path)
    else:
        comparison_df = pd.DataFrame([
            {'Model': 'Seasonal naive', 'MAE': 65.01, 'RMSE': 112.18, 'R2': 0.7411, 'Fit+predict time (s)': 0.12},
            {'Model': 'Linear Regression', 'MAE': 98.69, 'RMSE': 133.82, 'R2': 0.6316, 'Fit+predict time (s)': 0.49},
            {'Model': 'SVR (RBF)', 'MAE': 75.53, 'RMSE': 127.09, 'R2': 0.6678, 'Fit+predict time (s)': 9.22},
            {'Model': 'Random Forest', 'MAE': 44.99, 'RMSE': 69.56, 'R2': 0.9005, 'Fit+predict time (s)': 0.82},
        ])

    dnn_row = pd.DataFrame([{
        'Model': f'{label} (Ours)',
        'MAE': round(dnn_mae, 2),
        'RMSE': round(dnn_rmse, 2),
        'R2': round(dnn_r2, 4),
        'Fit+predict time (s)': round(dnn_fit_predict_seconds, 2),
    }])
    full_comparison = pd.concat([comparison_df, rf_train_only_row, rf_same_row, dnn_row], ignore_index=True)

    comparison_out_path = os.path.join(RESULTS_DIR, 'model_comparison_dnn.csv')
    full_comparison.to_csv(comparison_out_path, index=False)
    print(f"\nFinal Comparative Test Evaluation (Saved to {comparison_out_path}):")
    print(full_comparison.to_string(index=False))

    # 5. Generate Publication Figures
    print("\nGenerating report figures...")
    plot_learning_curve(
        os.path.join(RESULTS_DIR, 'best_training_history.csv'),
        os.path.join(RESULTS_DIR, 'fig1_dnn_learning_curve.png'),
        label
    )
    print("  Saved fig1_dnn_learning_curve.png")

    plot_hyperparameter_tuning(
        os.path.join(RESULTS_DIR, 'tuning_summary.csv'),
        os.path.join(RESULTS_DIR, 'fig2_dnn_tuning_ablation.png')
    )
    print("  Saved fig2_dnn_tuning_ablation.png")

    plot_tracking_week(
        test_df,
        y_test_raw,
        dnn_preds,
        rf_preds,
        os.path.join(RESULTS_DIR, 'fig3_dnn_forecast_tracking_week.png'),
        label
    )
    print("  Saved fig3_dnn_forecast_tracking_week.png")

    plot_error_by_hour(
        test_df,
        y_test_raw,
        dnn_preds,
        rf_preds,
        os.path.join(RESULTS_DIR, 'fig4_dnn_error_by_hour.png'),
        label
    )
    print("  Saved fig4_dnn_error_by_hour.png")

    print("\n" + "=" * 70)
    print("Evaluation Complete! All figures and comparative metrics generated.")
    print("=" * 70)


if __name__ == '__main__':
    main()
