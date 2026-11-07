# ECE 9603/9063: Data Analytics Foundations
## Assignment 1: Forecasting

**Western University — Faculty of Engineering**  
Department of Electrical and Computer Engineering  
**Dataset:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`data/hour.csv`)

---

## 1. Project Overview
This repository contains the complete modular forecasting pipeline for Assignment 1. The objective is to forecast hourly bike rental demand (`cnt`) using temporal and environmental features across three distinct machine learning paradigms:
1. **Multivariate Linear Regression (OLS):** Parametric linear baseline.
2. **Support Vector Regression (SVR - RBF Kernel):** Non-parametric dual kernel-space optimization.
3. **Random Forest Regressor:** Non-parametric ensemble bootstrap aggregation (bagging).

---

## 2. Directory Structure

```text
├── data/
│   └── hour.csv                     # Raw UCI Bike Sharing Dataset (17,379 samples)
├── src/
│   ├── data_loader.py               # Feature selection, chronological 80/20 split, scaling
│   ├── models.py                    # Implementation of Linear Regression, SVR, Random Forest
│   ├── evaluate.py                  # Evaluation metrics (MAE, RMSE, R²) and visualization
│   ├── eda.py                       # Exploratory Data Analysis & descriptive statistics
│   └── train_forecast.py            # Master pipeline entry point
├── assignment1/
│   └── results/                     # Generated deliverables (tables & figures)
│       ├── dataset_summary_statistics.csv
│       ├── model_comparison_results.csv
│       ├── fig1_hourly_demand_profile.png
│       └── fig2_model_forecast_comparison.png
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── requirements.txt                 # Project dependencies
├── .gitignore
└── README.md
```

---

## 3. Environment Setup & Installation

### Requirements:
* Python 3.9+ (tested on Python 3.11)

Install dependencies using `pip`:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run the Code

### A. Run Exploratory Data Analysis (EDA)
Generates the descriptive statistics summary table and Figure 1 (commuter vs. leisure hourly profile):
```bash
python src/eda.py
```
Outputs saved to `assignment1/results/`:
* `dataset_summary_statistics.csv`
* `fig1_hourly_demand_profile.png`

### B. Run the Master Forecasting Pipeline
Executes the chronological 80/20 train-test split, feature standardization, trains all three algorithms, evaluates performance metrics (MAE, RMSE, $R^2$), and exports the comparison deliverables:
```bash
python src/train_forecast.py
```
Outputs saved to `assignment1/results/`:
* `model_comparison_results.csv`
* `fig2_model_forecast_comparison.png`

---

## 5. Summary of Results (Unseen 20% Chronological Test Set)

| Model / Algorithm | Paradigm | MAE (Bikes) | RMSE (Bikes) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Multivariate Linear Regression** | Parametric / Linear Hyperplane | 138.29 | 183.28 | 0.3090 |
| **Support Vector Regression (RBF)** | Non-Parametric / Kernel Dual Space | 111.82 | 177.04 | 0.3553 |
| **Random Forest Regressor** | Non-Parametric / Tree Bagging Ensemble | **45.02** | **69.62** | **0.9003** |
