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

CYCLE_PERIODS = {
    'hr': 24,
    'mnth': 12,
    'weekday': 7,
}


# Seasonal naive baseline: same hour 1 week ago (t - 168)
def seasonal_naive(full_df, test_df, lag_hours=168):
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


# Model 1: Linear Regression (OLS with one-hot encoded categoricals)
def build_linear_regression():
    preprocess = ColumnTransformer([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_COLS),
        ('scale', StandardScaler(), CONTINUOUS_COLS),
        ('binary', 'passthrough', BINARY_COLS),
    ])
    return Pipeline([
        ('preprocess', preprocess),
        ('model', LinearRegression()),
    ])


# Model 2: SVR with cyclical sin/cos encoding
def _cyclical_encode(X):
    df = pd.DataFrame(X, columns=FEATURE_COLS).copy()

    for col, period in CYCLE_PERIODS.items():
        radians = 2 * np.pi * df[col] / period
        df[f'{col}_sin'] = np.sin(radians)
        df[f'{col}_cos'] = np.cos(radians)
        df = df.drop(columns=col)

    # one-hot season
    for s in [1, 2, 3, 4]:
        df[f'season_{s}'] = (df['season'] == s).astype(float)
    df = df.drop(columns='season')

    return df.values.astype(float)


def build_svr():
    pipe = Pipeline([
        ('cyclical', FunctionTransformer(_cyclical_encode)),
        ('scale', StandardScaler()),
        ('model', SVR(kernel='rbf')),
    ])
    # Target is scaled so default C=1.0 and epsilon=0.1 work properly
    return TransformedTargetRegressor(
        regressor=pipe,
        transformer=StandardScaler(),
    )


# Model 3: Random Forest (100 trees, default parameters)
def build_random_forest():
    return RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


MODEL_BUILDERS = {
    'Linear Regression': build_linear_regression,
    'SVR (RBF)': build_svr,
    'Random Forest': build_random_forest,
}


def fit_predict(builder, train_df, test_df, target_col='cnt'):
    model = builder()
    model.fit(train_df[FEATURE_COLS], train_df[target_col].values)
    y_pred = model.predict(test_df[FEATURE_COLS])
    return model, np.clip(y_pred, 0, None)


def n_model_inputs(builder, train_df):
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
    for name, builder in MODEL_BUILDERS.items():
        print(f"{name}: {n_model_inputs(builder, train_df)} inputs")
