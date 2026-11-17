import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import load_raw_data, engineer_datetime_features, get_feature_lists


class BikeDemandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(data_path="data/hour.csv", val_ratio=0.15, train_ratio=0.8):
    raw_df = load_raw_data(data_path)
    df = engineer_datetime_features(raw_df)

    feature_info = get_feature_lists()
    feature_cols = feature_info['all']
    target_col = 'cnt'

    # Chronological 80/20 train/test split
    split_idx = int(len(df) * train_ratio)
    train_full_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Validation split from training data
    val_split_idx = int(len(train_full_df) * (1 - val_ratio))
    train_df = train_full_df.iloc[:val_split_idx].copy().reset_index(drop=True)
    val_df = train_full_df.iloc[val_split_idx:].copy().reset_index(drop=True)

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_col].values

    X_val_raw = val_df[feature_cols].values
    y_val_raw = val_df[target_col].values

    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df[target_col].values

    # Fit scalers on train set only
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'y_test_raw': y_test_raw,
        'test_df': test_df,
        'feature_cols': feature_cols,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'train_df': train_df,
        'val_df': val_df,
    }


def get_dataloaders(data_dict, batch_size=64, shuffle_train=True):
    train_dataset = BikeDemandDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = BikeDemandDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = BikeDemandDataset(data_dict['X_test'], data_dict['y_test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    data = prepare_data()
    train_l, val_l, test_l = get_dataloaders(data, batch_size=64)
    print("Features:", len(data['feature_cols']))
    print("Train hours:", len(data['X_train']))
    print("Validation hours:", len(data['X_val']))
    print("Test hours:", len(data['X_test']))
