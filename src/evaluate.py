import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fixed colors and line styles for each model
STYLES = {
    'Actual demand':     {'color': '#1a1a19', 'linestyle': '-',  'linewidth': 2.0},
    'Seasonal naive':    {'color': '#8a8a85', 'linestyle': ':',  'linewidth': 1.6},
    'Linear Regression': {'color': '#2a78d6', 'linestyle': '--', 'linewidth': 1.6},
    'SVR (RBF)':         {'color': '#eb6834', 'linestyle': '-.', 'linewidth': 1.6},
    'Random Forest':     {'color': '#1baf7a', 'linestyle': '-',  'linewidth': 1.6},
}

MARKERS = {
    'Seasonal naive': 'x',
    'Linear Regression': 's',
    'SVR (RBF)': '^',
    'Random Forest': 'o',
}


def _style_axes(ax):
    """Clean up plot borders and add light grid lines."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)
    ax.grid(axis='y', color='#e5e5e2', linewidth=0.8)
    ax.set_axisbelow(True)


def compute_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and R2 score."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': r2_score(y_true, y_pred),
    }


def plot_forecast_week(test_df, predictions, output_path, start='2012-10-01', days=7):
    """Plot actual demand vs model predictions for one full week (Mon-Sun)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    start = pd.Timestamp(start)
    mask = (test_df['timestamp'] >= start) & (test_df['timestamp'] < start + pd.Timedelta(days=days))
    ts = test_df.loc[mask, 'timestamp']

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(ts, test_df.loc[mask, 'cnt'], label='Actual demand', **STYLES['Actual demand'])
    
    for name, y_pred in predictions.items():
        if name in STYLES:
            ax.plot(ts, y_pred[mask.values], label=name, **STYLES[name])

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
    ax.set_ylabel('Bikes rented per hour', fontsize=10)
    ax.set_title(
        f'Actual vs. forecast hourly demand, test week {start:%d %b} - {start + pd.Timedelta(days=days - 1):%d %b %Y}',
        fontsize=11, loc='left'
    )
    _style_axes(ax)
    ax.legend(ncol=5, fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_error_by_hour(test_df, predictions, output_path):
    """Plot average absolute error across each hour of the day."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    hrs = test_df['hr'].values
    y = test_df['cnt'].values

    for name, y_pred in predictions.items():
        if name in STYLES:
            err = pd.Series(np.abs(y - y_pred)).groupby(hrs).mean()
            ax.plot(err.index, err.values, label=name, marker=MARKERS[name], markersize=5, **STYLES[name])

    ax.set_xticks(range(0, 24))
    ax.set_xlabel('Hour of day', fontsize=10)
    ax.set_ylabel('Mean absolute error (bikes)', fontsize=10)
    ax.set_title('Forecast error by hour of day (hold-out test set)', fontsize=11, loc='left')
    _style_axes(ax)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cv_folds(cv_df, output_path):
    """Bar chart comparing MAE of each model across CV folds."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    names = list(cv_df['Model'].unique())
    folds = sorted(cv_df['Fold'].unique())
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, name in enumerate(names):
        vals = cv_df[cv_df['Model'] == name].sort_values('Fold')['MAE'].values
        x = np.arange(len(folds)) + (i - (len(names) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.92, color=STYLES[name]['color'], label=name)

    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylabel('MAE (bikes)', fontsize=10)
    ax.set_title('Mean absolute error on each time-series cross-validation fold', fontsize=11, loc='left')
    _style_axes(ax)
    ax.set_ylim(0, cv_df['MAE'].max() * 1.2)
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    y_true = np.array([50, 100, 150, 200])
    y_pred = np.array([55, 90, 160, 195])
    metrics = compute_metrics(y_true, y_pred)
    print("Metrics test:", metrics)
