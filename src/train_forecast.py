import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import chronological_split, load_data
from eda import run_eda
from evaluate import compute_metrics, plot_cv_folds, plot_error_by_hour, plot_forecast_week
from models import MODEL_BUILDERS, fit_predict, seasonal_naive

# Suppress unknown categories warning for early CV folds
warnings.filterwarnings('ignore', message='Found unknown categories')

RESULTS_DIR = os.path.join('assignment1', 'results')
N_CV_SPLITS = 5
ALL_MODELS = ['Seasonal naive'] + list(MODEL_BUILDERS)


def evaluate_split(full_df, train_df, test_df):
    """Train each model on train_df and compute test metrics."""
    preds, metrics, timings = {}, {}, {}
    y_test = test_df['cnt'].values

    # 1. Seasonal naive baseline
    t0 = time.time()
    preds['Seasonal naive'], _ = seasonal_naive(full_df, test_df)
    timings['Seasonal naive'] = time.time() - t0

    # 2. Machine learning models
    for name, builder in MODEL_BUILDERS.items():
        t0 = time.time()
        _, preds[name] = fit_predict(builder, train_df, test_df)
        timings[name] = time.time() - t0

    # Compute metrics for all models
    for name in ALL_MODELS:
        metrics[name] = compute_metrics(y_test, preds[name])

    return metrics, preds, timings


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 64)
    print("ECE 9603/9063 - Assignment 1: Forecasting Pipeline")
    print("=" * 64)

    # 1. Load data & run EDA
    df = load_data("data/hour.csv")
    attr_table, profile = run_eda(df, RESULTS_DIR)
    print(f"\n[1/4] Loaded {len(df)} rows. Saved Figure 1 and Table 1.")

    # 2. Chronological 80/20 Hold-out evaluation
    train_df, test_df = chronological_split(df, train_ratio=0.8)
    print(f"\n[2/4] Hold-out split: Train={len(train_df)} hrs, Test={len(test_df)} hrs")

    ho_metrics, ho_preds, ho_times = evaluate_split(df, train_df, test_df)

    holdout_rows = []
    for m in ALL_MODELS:
        holdout_rows.append({
            'Model': m,
            'MAE': round(ho_metrics[m]['MAE'], 2),
            'RMSE': round(ho_metrics[m]['RMSE'], 2),
            'R2': round(ho_metrics[m]['R2'], 4),
            'Fit+predict time (s)': round(ho_times[m], 2),
        })
    holdout_table = pd.DataFrame(holdout_rows)
    holdout_table.to_csv(os.path.join(RESULTS_DIR, 'table2_holdout_results.csv'), index=False)
    print("\nHold-out Test Results (Table 2):")
    print(holdout_table.to_string(index=False))

    # 3. 5-Fold Time-Series Cross-Validation on Training Period
    print(f"\n[3/4] Running {N_CV_SPLITS}-fold expanding-window cross-validation on train set...")
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_rows, fold_rows = [], []

    for k, (tr_idx, va_idx) in enumerate(tscv.split(train_df), start=1):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        m, _, _ = evaluate_split(df, tr, va)

        unseen_months = sorted(set(va.mnth) - set(tr.mnth))
        fold_rows.append({
            'Fold': k,
            'Train hours': len(tr),
            'Months in validation unseen in training': ', '.join(str(x) for x in unseen_months) or 'none',
            'Train period': f"{tr.timestamp.min():%Y-%m-%d} to {tr.timestamp.max():%Y-%m-%d}",
            'Validation hours': len(va),
            'Validation period': f"{va.timestamp.min():%Y-%m-%d} to {va.timestamp.max():%Y-%m-%d}",
        })

        for name, scores in m.items():
            cv_rows.append({'Fold': k, 'Model': name, **scores})

        print(f"      Fold {k}: " + ", ".join(f"{n} MAE={scores['MAE']:.1f}" for n, scores in m.items()))

    cv_df = pd.DataFrame(cv_rows)
    cv_df.round(4).to_csv(os.path.join(RESULTS_DIR, 'cv_per_fold_results.csv'), index=False)
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(RESULTS_DIR, 'cv_fold_periods.csv'), index=False)

    cv_summary = cv_df.groupby('Model', sort=False)[['MAE', 'RMSE', 'R2']].agg(['mean', 'std'])
    cv_summary.columns = [f"{m} {stat}" for m, stat in cv_summary.columns]
    cv_summary.round(3).reset_index().to_csv(os.path.join(RESULTS_DIR, 'table3_cv_summary.csv'), index=False)

    # 4. Generate Figures
    print("\n[4/4] Generating report figures...")
    week_start = '2012-10-01'
    plot_forecast_week(test_df, ho_preds, os.path.join(RESULTS_DIR, 'fig2_forecast_week.png'), start=week_start)
    plot_error_by_hour(test_df, ho_preds, os.path.join(RESULTS_DIR, 'fig3_error_by_hour.png'))
    plot_cv_folds(cv_df, os.path.join(RESULTS_DIR, 'fig4_cv_fold_mae.png'))
    print("      Saved fig2_forecast_week.png, fig3_error_by_hour.png, and fig4_cv_fold_mae.png")

    print("\n" + "=" * 64)
    print("Forecasting Pipeline Completed Successfully!")
    print("=" * 64)


if __name__ == '__main__':
    main()
