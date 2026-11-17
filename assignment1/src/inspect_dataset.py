import pandas as pd
import numpy as np

# Load dataset
data_path = "data/hour.csv"
df = pd.read_csv(data_path)

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
