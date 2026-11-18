import os
import numpy as np
import pandas as pd


def find_data_path(filename="hour.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    for _ in range(4):
        candidate = os.path.join(current, "data", filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        current = os.path.dirname(current)

    for candidate in [
        os.path.join("data", filename),
        os.path.join("Assignments", "data", filename),
        os.path.join("..", "data", filename),
    ]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(f"Cannot find dataset file: {filename}")


def load_raw_data(data_path=None):
    if data_path is None or not os.path.exists(data_path):
        data_path = find_data_path("hour.csv")

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def engineer_datetime_features(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # Historical demand lags (1h, 2h, 3h, 24h, 168h)
    df['lag_1h'] = df['cnt'].shift(1)
    df['lag_2h'] = df['cnt'].shift(2)
    df['lag_3h'] = df['cnt'].shift(3)
    df['lag_24h'] = df['cnt'].shift(24)
    df['lag_168h'] = df['cnt'].shift(168)

    # Rolling statistics
    rolling_6 = df['cnt'].shift(1).rolling(window=6, min_periods=1)
    df['rolling_mean_6h'] = rolling_6.mean()
    df['rolling_std_6h'] = rolling_6.std().fillna(0)

    rolling_24 = df['cnt'].shift(1).rolling(window=24, min_periods=1)
    df['rolling_mean_24h'] = rolling_24.mean()

    # Cyclical date/time encodings
    df['sin_hr'] = np.sin(2 * np.pi * df['hr'] / 24.0)
    df['cos_hr'] = np.cos(2 * np.pi * df['hr'] / 24.0)

    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7.0)

    df['sin_mnth'] = np.sin(2 * np.pi * df['mnth'] / 12.0)
    df['cos_mnth'] = np.cos(2 * np.pi * df['mnth'] / 12.0)

    day_of_year = df['timestamp'].dt.dayofyear
    df['sin_dayofyear'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_dayofyear'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Rush-hour and weekend flags
    is_morning_rush = (df['hr'].between(7, 9)) & (df['workingday'] == 1)
    is_evening_rush = (df['hr'].between(16, 19)) & (df['workingday'] == 1)
    df['is_rush_hour'] = (is_morning_rush | is_evening_rush).astype(float)
    df['is_weekend_midday'] = ((df['hr'].between(11, 15)) & (df['workingday'] == 0)).astype(float)

    # Fill initial warmup period
    lag_cols = [
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_6h', 'rolling_std_6h', 'rolling_mean_24h'
    ]
    df[lag_cols] = df[lag_cols].bfill()

    # Drop target leakage columns
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def get_feature_lists():
    raw_weather = ['temp', 'atemp', 'hum', 'windspeed', 'weathersit', 'season', 'yr', 'holiday', 'workingday']
    lags_and_rolling = [
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_6h', 'rolling_std_6h', 'rolling_mean_24h'
    ]
    cyclical_and_flags = [
        'sin_hr', 'cos_hr', 'sin_weekday', 'cos_weekday',
        'sin_mnth', 'cos_mnth', 'sin_dayofyear', 'cos_dayofyear',
        'is_rush_hour', 'is_weekend_midday'
    ]
    all_features = raw_weather + lags_and_rolling + cyclical_and_flags
    return {
        'all': all_features,
        'raw_weather': raw_weather,
        'lags_and_rolling': lags_and_rolling,
        'cyclical_and_flags': cyclical_and_flags,
    }


if __name__ == '__main__':
    raw = load_raw_data()
    featured = engineer_datetime_features(raw)
    flists = get_feature_lists()
    print("Features created:", len(flists['all']))
    print("Dataset shape:", featured.shape)
    print("Null values:", featured[flists['all']].isnull().sum().sum())
