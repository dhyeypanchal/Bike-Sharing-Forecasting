"""
Assignment 1: Forecasting
Module: models.py
Purpose: Constructs forecasting pipelines for each model paradigm:
         0. Seasonal-Naive Baseline (Historical domain benchmark: t - 168h)
         1. Multivariate Linear Regression (OLS with One-Hot categoricals)
         2. Support Vector Regression (RBF with cyclical sin/cos coordinates & target scaling)
         3. Random Forest Regressor (Ensemble bagging on raw feature representations)

All models adhere to strict evaluation rules: transformations are fitted
strictly on training partitions within scikit-learn Pipelines, with zero
hyperparameter tuning in Assignment 1 (default configurations).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from data_loader import BINARY_COLS, CATEGORICAL_COLS, CONTINUOUS_COLS, FEATURE_COLS

RANDOM_STATE = 42

# Cycle period for periodic temporal features
CYCLE_PERIODS = {
    'hr': 24,       # 24 hours per day
    'mnth': 12,     # 12 months per year
    'weekday': 7    # 7 days per week
}


# ---------------------------------------------------------------------------
# Model 0: Seasonal-Naive Baseline (Domain Reference)
# ---------------------------------------------------------------------------
def seasonal_naive(full_df, test_df, lag_hours=168):
    """
    Predicts each hour using the observed count from the same hour one week
    (168 hours = 7 days * 24 hours) earlier.

    In real-world transit systems, weekly demand profiles repeat strongly.
    If the 168-hour lag observation is missing from the dataset, it falls back
    to the same hour 24 hours earlier, and finally to the most recent hour.

    Returns:
        preds (np.ndarray): Predicted counts for the test window.
        fallbacks (int): Number of times the week-earlier value was missing.
    """
    series = full_df.set_index('timestamp')['cnt']
    preds = []
    fallbacks = 0

    for ts in test_df['timestamp']:
        week_ago = ts - pd.Timedelta(hours=lag_hours)
        day_ago = ts - pd.Timedelta(hours=24)

        if week_ago in series.index:
            preds.append(series[week_ago])
        elif day_ago in series.index:
            preds.append(series[day_ago])
            fallbacks += 1
        else:
            preds.append(series[series.index < ts].iloc[-1])
            fallbacks += 1

    return np.array(preds, dtype=float), fallbacks


# ---------------------------------------------------------------------------
# Model 1: Multivariate Linear Regression (OLS with One-Hot Encoding)
# ---------------------------------------------------------------------------
def build_linear_regression():
    """
    Ordinary Least Squares (OLS) regression.
    
    Why One-Hot Encoding is required:
    Treating 'hr' as a single scalar (0 to 23) forces a single monotonic slope,
    making it impossible to capture both morning (8 AM) and evening (5 PM) peaks.
    One-Hot encoding allocates an independent indicator weight for each discrete hour,
    month, weekday, weather condition, and season.
    """
    preprocess = ColumnTransformer([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_COLS),
        ('scale', StandardScaler(), CONTINUOUS_COLS),
        ('binary', 'passthrough', BINARY_COLS),
    ])
    return Pipeline([
        ('preprocess', preprocess),
        ('model', LinearRegression())
    ])


# ---------------------------------------------------------------------------
# Model 2: Support Vector Regression (RBF Kernel with Cyclical Coordinates)
# ---------------------------------------------------------------------------
def _cyclical_encode(X):
    """
    Transforms periodic calendar attributes (hr, mnth, weekday) into
    continuous sine and cosine coordinate pairs on a unit circle:
        x_sin = sin(2 * pi * x / Period)
        x_cos = cos(2 * pi * x / Period)

    This resolves the Euclidean metric distortion in the RBF kernel,
    ensuring hour 23 (11 PM) and hour 0 (midnight) are adjacent in distance.
    """
    df = pd.DataFrame(X, columns=FEATURE_COLS).copy()

    for col, period in CYCLE_PERIODS.items():
        radians = 2 * np.pi * df[col] / period
        df[f'{col}_sin'] = np.sin(radians)
        df[f'{col}_cos'] = np.cos(radians)
        df = df.drop(columns=col)

    # One-hot encode nominal season (1: winter, 2: spring, 3: summer, 4: fall)
    for s in [1, 2, 3, 4]:
        df[f'season_{s}'] = (df['season'] == s).astype(float)
    df = df.drop(columns='season')

    return df.values.astype(float)


def build_svr():
    """
    Support Vector Regression with Radial Basis Function (RBF) kernel.
    
    Uses scikit-learn defaults (C=1.0, epsilon=0.1, gamma='scale').
    
    Why TransformedTargetRegressor is required:
    Default SVR hyperparameter scales assume a normalized target roughly in [-1, 1].
    On raw bicycle counts reaching ~1000, unscaled target errors cause slack penalties
    to blow up into millions, stalling optimization. TransformedTargetRegressor
    standardizes the target y for training and automatically converts predictions
    back to physical bicycle counts.
    """
    regressor_pipe = Pipeline([
        ('cyclical', FunctionTransformer(_cyclical_encode)),
        ('scale', StandardScaler()),
        ('model', SVR(kernel='rbf'))
    ])
    return TransformedTargetRegressor(
        regressor=regressor_pipe,
        transformer=StandardScaler()
    )


# ---------------------------------------------------------------------------
# Model 3: Random Forest Regressor (Ensemble Bagging)
# ---------------------------------------------------------------------------
def build_random_forest():
    """
    Ensemble of 100 bagged decision trees with default scikit-learn parameters.
    
    Tree algorithms recursively evaluate orthogonal decision thresholds,
    making them naturally scale-invariant and capable of capturing complex
    non-linear interactions (e.g. hr == 8 AND workingday == 1) directly on raw features.
    """
    return RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ---------------------------------------------------------------------------
# Model Dispatcher & Execution Helpers
# ---------------------------------------------------------------------------
MODEL_BUILDERS = {
    'Linear Regression': build_linear_regression,
    'SVR (RBF)': build_svr,
    'Random Forest': build_random_forest,
}


def fit_predict(builder, train_df, test_df, target_col='cnt'):
    """
    Fits an instantiated model pipeline on train_df and computes predictions on test_df.
    Outputs are clamped to a non-negative floor (np.clip(y_pred, 0, None)).
    """
    model = builder()
    model.fit(train_df[FEATURE_COLS], train_df[target_col].values)
    y_pred = model.predict(test_df[FEATURE_COLS])
    return model, np.clip(y_pred, 0, None)


def n_model_inputs(builder, train_df):
    """
    Determines the exact dimensionality of the feature space each model operates on
    after its own internal pipeline transformations.
    """
    model = builder()
    if isinstance(model, TransformedTargetRegressor):
        pre = Pipeline(model.regressor.steps[:-1])
    elif isinstance(model, Pipeline):
        pre = Pipeline(model.steps[:-1])
    else:
        return len(FEATURE_COLS)
    return pre.fit_transform(train_df[FEATURE_COLS]).shape[1]


if __name__ == "__main__":
    from data_loader import load_data, chronological_split

    df = load_data()
    train_df, test_df = chronological_split(df)

    print("=" * 60)
    print("Testing Model Pipelines Architecture:")
    for name, builder in MODEL_BUILDERS.items():
        n_features = n_model_inputs(builder, train_df)
        print(f"  {name:20s} -> Transformed input features: {n_features}")

    # Test fit_predict on a sample
    print("\nQuick verification fit (Linear Regression):")
    lr_model, preds = fit_predict(MODEL_BUILDERS['Linear Regression'], train_df, test_df)
    print(f"  Predictions generated: {len(preds)}, min: {preds.min():.2f}, max: {preds.max():.2f}")
    print("=" * 60)
