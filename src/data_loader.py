import os
import pandas as pd

# 12 predictive features used across models
CATEGORICAL_COLS = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
BINARY_COLS = ['yr', 'holiday', 'workingday']
CONTINUOUS_COLS = ['temp', 'atemp', 'hum', 'windspeed']
FEATURE_COLS = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
]
TARGET_COL = 'cnt'

# Excluded: instant (index), dteday (redundant), casual/registered (target leakage)
DROPPED_COLS = ['instant', 'dteday', 'casual', 'registered']


def load_data(data_path="data/hour.csv"):
    """Load hour.csv and create a continuous timestamp from dteday and hr."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def chronological_split(df, train_ratio=0.8):
    """First 80% hours for training, last 20% for testing (no shuffling)."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    df = load_data()
    train_df, test_df = chronological_split(df)
    print(f"Total: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
