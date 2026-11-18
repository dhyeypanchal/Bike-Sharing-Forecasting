import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

df = load_data()

print("=== Dataset Overview ===")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\n=== Data Types and Null Count ===")
print(df.info())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Statistical Summary (Numeric Columns) ===")
summary = df.describe().T
print(summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

print("\n=== Categorical Value Counts ===")
for col in ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']:
    print(f"{col}: {sorted(df[col].unique())}")
