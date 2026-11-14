# ECE 9603/9063: Data Analytics Foundations
## Assignment 1: Forecasting

**Western University — Faculty of Engineering**  
Department of Electrical and Computer Engineering  
**Dataset:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`data/hour.csv`)  

---

## 1. Project Overview
This repository contains the forecasting pipeline for Assignment 1. The goal is to forecast hourly bike rental demand (`cnt`) from temporal and weather attributes across distinct forecasting paradigms:
- **Baseline:** Seasonal Naive ($y_t = y_{t-168}$, same hour last week)
- **Model 1:** Multivariate Linear Regression (OLS with one-hot encoded categoricals)
- **Model 2:** Support Vector Regression (RBF kernel with cyclical $\sin/\cos$ coordinates and standardized target)
- **Model 3:** Random Forest Regressor (ensemble of 100 bagged trees on raw features)

### Group Responsibilities:
- **Dhyey Panchal:** Random Forest, Multivariate Linear Regression, Data Pipeline and EDA
- **Aseel Qureshi:** Support Vector Regression (RBF Kernel) and Seasonal Naive Baseline

---

## 2. Directory Structure

```text
├── data/
│   └── hour.csv                     # Raw UCI Bike Sharing Dataset (17,379 hours)
├── src/
│   ├── data_loader.py               # Ingestion, timestamp creation, chronological 80/20 split
│   ├── models.py                    # Pipeline definitions for all models
│   ├── evaluate.py                  # Evaluation metrics (MAE, RMSE, R²) and visualization functions
│   ├── eda.py                       # Exploratory Data Analysis & Table 1 / Figure 1 generation
│   ├── inspect_dataset.py           # Initial dataset sanity checks
│   └── train_forecast.py            # Master pipeline (hold-out + 5-fold expanding time-series CV)
├── assignment1/
│   └── results/                     # Generated tables and figures
│       ├── table1_attribute_summary.csv
│       ├── table2_holdout_results.csv
│       ├── table3_cv_summary.csv
│       ├── cv_fold_periods.csv
│       ├── cv_per_fold_results.csv
│       ├── fig1_hourly_demand_profile.png
│       ├── fig2_forecast_week.png
│       ├── fig3_error_by_hour.png
│       └── fig4_cv_fold_mae.png
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── requirements.txt                 # Project dependencies
├── .gitignore
└── README.md
```

---

## 3. Setup & Installation

Requirements: Python 3.9+ (tested on Python 3.11)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run

### A. Run Exploratory Data Analysis (EDA)
Computes summary statistics across attributes and plots the average hourly demand profile:
```bash
python src/eda.py
```
Outputs saved to `assignment1/results/`:
- `table1_attribute_summary.csv`
- `fig1_hourly_demand_profile.png`

### B. Run the Master Forecasting Pipeline
Executes the chronological 80/20 hold-out split, trains all models, runs 5-fold expanding-window cross-validation, and saves the comparison tables and figures:
```bash
python src/train_forecast.py
```
Outputs saved to `assignment1/results/`:
- `table2_holdout_results.csv` (hold-out performance)
- `table3_cv_summary.csv` (cross-validation mean & standard deviation)
- `fig2_forecast_week.png` (7-day actual vs forecast timeline)
- `fig3_error_by_hour.png` (error breakdown by hour of day)
- `fig4_cv_fold_mae.png` (MAE per CV fold)

---

## 5. Summary of Results

### Chronological Hold-Out Test (Last 20% Unseen Future, 3,476 Hours)

| Model | Paradigm | MAE (Bikes) | RMSE (Bikes) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Seasonal Naive** | Lag Benchmark ($t - 168$h) | 65.01 | 112.18 | 0.7411 |
| **Linear Regression** | Parametric (OLS + One-Hot) | 98.69 | 133.82 | 0.6316 |
| **SVR (RBF)** | Kernel Dual Space + Cyclical $\sin/\cos$ | 75.53 | 127.09 | 0.6678 |
| **Random Forest** | Non-Parametric Bagging Ensemble | **44.99** | **69.56** | **0.9005** |

### 5-Fold Expanding Time-Series Cross-Validation (Training Period)

| Model | MAE Mean (Std) | RMSE Mean (Std) | $R^2$ Mean (Std) |
| :--- | :---: | :---: | :---: |
| **Seasonal Naive** | 52.47 (6.85) | 88.71 (13.45) | 0.685 (0.103) |
| **Linear Regression** | 81.40 (19.03) | 111.61 (20.94) | 0.518 (0.117) |
| **SVR (RBF)** | 82.25 (27.46) | 127.92 (36.62) | 0.338 (0.385) |
| **Random Forest** | **52.29 (15.39)** | **80.20 (21.92)** | **0.737 (0.123)** |

---

## 6. Generative AI Statement (Code)

As required by the assignment guidelines, here is how we used AI during the coding portion:

When we started the project, we wrote the basic pipeline ourselves: loading the CSV, parsing the timestamps, dropping `casual` and `registered` to prevent leakage, setting up the 80/20 chronological split, and fitting standard baseline models (Linear Regression, SVR, and Random Forest). 

Our initial baseline results were quite poor for both linear regression ($R^2 \approx 0.31$) and SVR ($R^2 \approx 0.35$). Since the instructions mentioned not to tune hyperparameters in Assignment 1, we realized the issue came down to feature representation—treating cyclical attributes like hour of the day as plain linear numbers made it impossible for linear regression to capture rush-hour peaks, and it distorted distances in SVR.

We used an LLM as an advisory assistant to brainstorm feature representation techniques while keeping the default scikit-learn models. Specifically, it helped us with:
- Using `ColumnTransformer` and `OneHotEncoder` for linear regression so each hour and weekday gets its own weight rather than assuming a single straight slope.
- Converting cyclical features (`hr`, `mnth`, `weekday`) to $\sin/\cos$ coordinates so midnight and 11 PM stay adjacent in distance, and wrapping SVR in `TransformedTargetRegressor` so default parameters work on scaled targets.
- Setting up the expanding-window `TimeSeriesSplit` cross-validation on the training set.
- Structuring the seasonal-naive weekly benchmark with fallback logic for missing timestamps.

We took these ideas and manually wrote, connected, and debugged the scikit-learn Pipelines within our modular files (`data_loader.py`, `models.py`, `evaluate.py`, `train_forecast.py`). Both of us tested the code, verified the metrics, and made sure all figures and tables ran cleanly end-to-end.
