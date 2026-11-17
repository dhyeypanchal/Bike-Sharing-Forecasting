import os
import numpy as np
import pandas as pd


def load_raw_data(data_path="data/hour.csv"):
    """Load hour.csv and build a continuous timestamp index."""
    if not os.path.exists(data_path):
        # Allow running from inside assignment2/ or project root
        alt_path = os.path.join("..", data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find dataset at {data_path}")

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def engineer_datetime_features(df):
    """
    Engineer domain-specific temporal features from dates and times:
    1. Target lag features (immediate, daily, and weekly rhythms)
    2. Rolling momentum statistics (short-term & daily trends)
    3. Cyclical trigonometric calendar encodings (hour, day of week, day of year)
    4. Commuter rush-hour and weekend interaction flags
    """
    df = df.copy()

    # Sort strictly by timestamp to preserve temporal causality
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ---------------------------------------------------------
    # 1. Historical Lag Features (Strictly backwards, no leakage)
    # ---------------------------------------------------------
    # Immediate momentum: past 1, 2, and 3 hours
    df['lag_1h'] = df['cnt'].shift(1)
    df['lag_2h'] = df['cnt'].shift(2)
    df['lag_3h'] = df['cnt'].shift(3)

    # Daily rhythm: exact same hour yesterday (24 hours prior)
    df['lag_24h'] = df['cnt'].shift(24)

    # Weekly rhythm: exact same hour last week (168 hours prior, A1 baseline)
    df['lag_168h'] = df['cnt'].shift(168)

    # ---------------------------------------------------------
    # 2. Rolling Momentum & Volatility (Shifted by 1 to prevent leakage)
    # ---------------------------------------------------------
    # 6-hour short-term rolling average & standard deviation
    rolling_6 = df['cnt'].shift(1).rolling(window=6, min_periods=1)
    df['rolling_mean_6h'] = rolling_6.mean()
    df['rolling_std_6h'] = rolling_6.std().fillna(0)

    # 24-hour daily rolling average
    rolling_24 = df['cnt'].shift(1).rolling(window=24, min_periods=1)
    df['rolling_mean_24h'] = rolling_24.mean()

    # ---------------------------------------------------------
    # 3. Cyclical Calendar Transformations (Continuous sin / cos)
    # ---------------------------------------------------------
    # Hour of day (24-hour cycle)
    df['sin_hr'] = np.sin(2 * np.pi * df['hr'] / 24.0)
    df['cos_hr'] = np.cos(2 * np.pi * df['hr'] / 24.0)

    # Day of week (7-day cycle)
    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7.0)

    # Month of year (12-month cycle)
    df['sin_mnth'] = np.sin(2 * np.pi * df['mnth'] / 12.0)
    df['cos_mnth'] = np.cos(2 * np.pi * df['mnth'] / 12.0)

    # Day of year (365.25-day seasonal cycle)
    day_of_year = df['timestamp'].dt.dayofyear
    df['sin_dayofyear'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_dayofyear'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # ---------------------------------------------------------
    # 4. Domain Interaction Flags
    # ---------------------------------------------------------
    # Commuter rush-hour indicators on working days (peaks at 8 AM and 5-6 PM)
    is_morning_rush = (df['hr'].between(7, 9)) & (df['workingday'] == 1)
    is_evening_rush = (df['hr'].between(16, 19)) & (df['workingday'] == 1)
    df['is_rush_hour'] = (is_morning_rush | is_evening_rush).astype(float)

    # Midday leisure curve on weekends
    df['is_weekend_midday'] = ((df['hr'].between(11, 15)) & (df['workingday'] == 0)).astype(float)

    # ---------------------------------------------------------
    # 5. Clean up Initial Warmup Rows (Lags need up to 168 hours)
    # ---------------------------------------------------------
    # Backfill the initial 168 warmup hours using forward propagation so no rows are lost,
    # or drop initial NaN rows. Backfilling using bfill ensures full continuity.
    lag_cols = [
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h', 'lag_168h',
        'rolling_mean_6h', 'rolling_std_6h', 'rolling_mean_24h'
    ]
    df[lag_cols] = df[lag_cols].bfill()

    # Drop target leakage columns and non-predictive identifiers from A1
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def get_feature_lists():
    """Returns categorized feature names for scaling and embedding."""
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
    print("=" * 60)
    print("Assignment 2: Feature Engineering Completed")
    print("=" * 60)
    print(f"Original features in Assignment 1 : 12 features")
    print(f"Engineered features in Assignment 2: {len(flists['all'])} features")
    print(f"Total dataset shape               : {featured.shape}")
    print("\nFeature breakdown:")
    for category, cols in flists.items():
        if category != 'all':
            print(f"  - {category} ({len(cols)}): {', '.join(cols[:4])}...")
    print(f"\nMissing values (NaNs): {featured[flists['all']].isnull().sum().sum()}")
