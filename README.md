# ECE 9603/9063: Data Analytics Foundations
## Assignments 1 & 2: Forecasting & Deep Neural Networks

**Western University — Faculty of Engineering**  
Department of Electrical and Computer Engineering  
**Dataset:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`data/hour.csv`)  

---

## 1. Project Overview

This repository contains the complete forecasting and deep learning pipelines for ECE 9603/9063:
- **Assignment 1 (Forecasting Paradigms):** Evaluates baseline linear, kernel dual space, and tree ensemble models (Seasonal Naive, Multivariate Linear Regression, Support Vector Regression, Random Forest) on 12 static attributes across chronological 80/20 holdout and 5-fold expanding-window cross-validation.
- **Assignment 2 (Deep Neural Networks):** Expands the feature space to 27 engineered temporal lag, rolling momentum, and cyclical coordinates (zero lookahead leakage), and designs, trains, and systematically tunes a $\ge 5$-hidden-layer Deep Neural Network (DNN) with Batch Normalization and Dropout.

### Group Responsibilities & Work Distribution:

- **Dhyey Panchal:**
  - Led the Deep Neural Network architecture design and PyTorch implementation (`DeepBikeNet`), incorporating linear transformations, Batch Normalization, and ReLU activations.
  - Designed and implemented the temporal feature engineering pipeline (27 features: autoregressive demand lags, rolling window momentum, continuous cyclical coordinates, and commuter indicators with strict `.shift(1)` zero-leakage shifting).
  - Built the modular PyTorch training loop with mini-batch Adam optimization, gradient norm clipping (threshold 1.0), dynamic `ReduceLROnPlateau` learning rate scheduling, and early stopping.
  - Conceived and executed the systematic 10-trial hyperparameter ablation engine across Depth (3 vs 5 vs 7 layers), Width, Learning Rate, and Dropout.
  - Performed final test set evaluation on 3,476 held-out hours, generated comparative benchmarks against Assignment 1 models, and authored the Methodology and Results sections.

- **Aseel Qureshi:**
  - Conducted the raw dataset audit across all 17,379 hourly records from the UCI repository, verifying complete attribute completeness and zero missing values.
  - Set up and verified the chronological data partitioning (80% development, 15% validation, and 20% holdout test) and confirmed proper `StandardScaler` fitting on training data.
  - Researched alternative deep learning candidates (RNNs, LSTMs, and shallow MLPs), analyzing computational latency, parameter efficiency, and suitability for multi-scale seasonal jumps.
  - Evaluated baseline model performance across cross-validation folds, assisted with residual error analysis across the 24 hours of the day, and verified metric calculations.
  - Co-authored the Problem Description and Background sections, verified table formatting, and assisted with comparative holdout analysis.

---

## 2. Directory Structure

```text
├── data/
│   └── hour.csv                          # Raw UCI Bike Sharing Dataset (17,379 hours)
├── assignment1/                          # Assignment 1: Classical Forecasting
│   ├── src/
│   │   ├── data_loader.py                # Chronological 80/20 split and scaling
│   │   ├── models.py                     # Seasonal Naive, Linear Reg, SVR, Random Forest
│   │   ├── evaluate.py                   # Metrics & cross-validation routines
│   │   ├── eda.py                        # EDA & Table 1 / Figure 1 generation
│   │   ├── inspect_dataset.py            # Initial dataset audit
│   │   └── train_forecast.py             # Master pipeline (hold-out + 5-fold CV)
│   └── results/                          # Generated A1 figures and tables
│       ├── table1_attribute_summary.csv
│       ├── table2_holdout_results.csv
│       ├── table3_cv_summary.csv
│       ├── fig1_hourly_demand_profile.png
│       ├── fig2_forecast_week.png
│       ├── fig3_error_by_hour.png
│       └── fig4_cv_fold_mae.png
├── assignment2/                          # Assignment 2: Deep Neural Networks
│   ├── src/
│   │   ├── features.py                   # 27 lag, rolling, and cyclical features (zero-leakage)
│   │   ├── dataset.py                    # Scalers, 80/15/20 splits & PyTorch DataLoaders
│   │   ├── model.py                      # Configurable >= 5 hidden layer DeepBikeNet
│   │   ├── train.py                      # Training loop with Adam, ReduceLROnPlateau, Early Stopping
│   │   ├── tune.py                       # 10-trial hyperparameter ablation engine (depth, width, lr, drop)
│   │   └── evaluate_dnn.py               # Test set evaluation, comparative benchmarks & plots
│   └── results/                          # Generated A2 figures, checkpoints, and benchmark tables
│       ├── best_model.pt                 # Champion 5-layer model weights
│       ├── best_training_history.csv     # Training & validation loss history
│       ├── tuning_summary.csv            # 10-trial hyperparameter tuning summary
│       ├── model_comparison_dnn.csv      # Comparative benchmark against A1 models
│       ├── fig1_dnn_learning_curve.png   # Train vs. validation loss curve
│       ├── fig2_dnn_tuning_ablation.png  # 4-panel tuning ablation bar chart
│       ├── fig3_dnn_forecast_tracking_week.png # 7-day test forecast tracking
│       └── fig4_dnn_error_by_hour.png    # Residual error by hour of day
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── requirements.txt
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

### Assignment 1: Classical Forecasting Pipeline

1. **Run Exploratory Data Analysis (EDA):**
   ```bash
   python assignment1/src/eda.py
   ```
2. **Run Master Forecasting Pipeline:**
   ```bash
   python assignment1/src/train_forecast.py
   ```

### Assignment 2: Deep Neural Network Pipeline

1. **Verify Feature Engineering (27 features):**
   ```bash
   python assignment2/src/features.py
   ```
2. **Verify Dataset Splitting & PyTorch DataLoaders:**
   ```bash
   python assignment2/src/dataset.py
   ```
3. **Run 10-Trial Hyperparameter Tuning Ablation:**
   Explores Depth (3 vs 5 vs 7 layers), Width (Narrow vs Medium vs Wide), Learning Rate (0.01 vs 0.001 vs 0.0001), and Dropout (0.0 to 0.3):
   ```bash
   python assignment2/src/tune.py
   ```
4. **Evaluate Champion Model on Held-Out Test Set (3,476 Hours) & Generate Figures:**
   ```bash
   python assignment2/src/evaluate_dnn.py
   ```
   Outputs saved to `assignment2/results/`:
   - `model_comparison_dnn.csv` (holdout performance comparison)
   - `fig1_dnn_learning_curve.png` (training vs validation loss trajectory)
   - `fig2_dnn_tuning_ablation.png` (4-panel tuning comparison)
   - `fig3_dnn_forecast_tracking_week.png` (7-day forecast timeline)
   - `fig4_dnn_error_by_hour.png` (error breakdown across 24 hours)

---

## 5. Comparative Performance Summary

All models were evaluated on the exact same untouched 3,476-hour holdout test set (October 1, 2012 to December 31, 2012):

| Model | Paradigm / Architecture | Test MAE (bikes) | Test RMSE (bikes) | Test $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Parametric (OLS + One-Hot) | 98.69 | 133.82 | 0.6316 |
| **SVR (RBF)** | Kernel Dual Space + Cyclical $\sin/\cos$ | 75.53 | 127.09 | 0.6678 |
| **Seasonal Naive** | Lag Baseline ($t - 168$h) | 65.01 | 112.18 | 0.7411 |
| **Random Forest** | Bagging Ensemble (100 Trees, A1 Champion) | 44.99 | 69.56 | 0.9005 |
| **Tuned 5-Layer DNN (Ours)** | **Deep Neural Network (`[256, 128, 64, 32, 16]`)** | **31.41** | **47.65** | **0.9533** |

### Key Improvements:
- **30.2% error reduction** over Assignment 1's best model (Random Forest MAE dropped from 44.99 to 31.41 bikes).
- **51.7% error reduction** over the Seasonal Naive baseline.
- **$R^2$ exceeded 0.95**, capturing over 95.3% of all variance in hourly demand on unseen future data.

### 5-Fold Expanding Time-Series Cross-Validation (Training Period)

| Model | MAE Mean (Std) | RMSE Mean (Std) | $R^2$ Mean (Std) |
| :--- | :---: | :---: | :---: |
| **Seasonal Naive** | 52.47 (6.85) | 88.71 (13.45) | 0.685 (0.103) |
| **Linear Regression** | 81.40 (19.03) | 111.61 (20.94) | 0.518 (0.117) |
| **SVR (RBF)** | 82.25 (27.46) | 127.92 (36.62) | 0.338 (0.385) |
| **Random Forest** | **52.29 (15.39)** | **80.20 (21.92)** | **0.737 (0.123)** |

---

## 6. Generative AI Statement (Code)

In accordance with Western University's ECE 9603/9063 course policy, here is an honest description of our engineering journey, our individual contributions, and where we leveraged generative AI:

### Assignment 1 Engineering Progression:
- **Our Initial Work:** We began by writing the basic data loading script, parsing timestamps, dropping `casual` and `registered` to prevent leakage, implementing the 80/20 chronological split, and fitting standard scikit-learn models (Linear Regression, SVR, and Random Forest).
- **The Bottleneck We Encountered:** Our initial baseline results were quite poor for both Linear Regression ($R^2 \approx 0.31$) and SVR ($R^2 \approx 0.35$). Because hyperparameters could not be tuned in Assignment 1, we recognized that the root problem was feature representation: treating cyclical attributes like hour of the day (0–23) as a single linear slope distorted SVR distance calculations and prevented Linear Regression from capturing dual rush-hour commuter peaks.
- **Where We Used AI Assistance:** We consulted an LLM as an interactive technical assistant to explore feature representation ideas. Specifically, it helped us brainstorm:
  1. Using `ColumnTransformer` with `OneHotEncoder` for Linear Regression so each hour and weekday receives an independent weight.
  2. Converting cyclical variables (`hr`, `mnth`, `weekday`) into continuous trigonometric $\sin/\cos$ coordinates, and wrapping SVR in `TransformedTargetRegressor` so default parameters operate on normalized targets.
  3. Setting up the expanding-window `TimeSeriesSplit` cross-validation structure.
- **Our Follow-Up Effort:** We manually implemented, connected, and debugged these pipelines within our modular architecture (`data_loader.py`, `models.py`, `evaluate.py`, `train_forecast.py`), added fallback logic for missing timestamps in the Seasonal Naive baseline, and verified metrics across all 5 cross-validation folds.

### Assignment 2 Engineering Progression:
- **Our Initial Work:** For Assignment 2, we started by writing a basic 3-layer neural network in PyTorch. However, when trained only on raw weather and static time features, the network underfit and could not match our Assignment 1 Random Forest benchmark ($R^2 \approx 0.72$ vs $0.90$). We realized the network had no temporal memory—it was predicting blindly without knowing whether demand was surging or crashing over the preceding hours.
- **Where We Used AI Assistance:** We used an LLM as an advisory assistant to brainstorm ways to inject temporal dynamics and stabilize deeper architectures:
  1. Brainstorming time-series feature engineering techniques for feedforward neural networks (specifically recommending autoregressive demand lags, rolling window statistics, and domain rush-hour indicators).
  2. Architectural best practices for deep networks ($\ge 5$ layers), specifically validating the layer ordering: `Linear` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout`.
  3. PyTorch syntax assistance for gradient norm clipping (`torch.nn.utils.clip_grad_norm_`), configuring the dynamic `ReduceLROnPlateau` scheduler parameters, and structuring the 10-trial hyperparameter ablation loop.
- **Our Follow-Up Effort:**
  1. We wrote the feature engineering pipeline in `features.py`, ensuring strict zero lookahead leakage by writing explicit `.shift(1)` logic so target demand at hour $t$ is never accessible.
  2. We implemented `DeepBikeNet` in PyTorch, structured the custom PyTorch `Dataset` and `DataLoader` pipeline, and created narrow, medium, and wide configurations across 3, 5, and 7 hidden layers.
  3. We executed and monitored all 10 tuning trials, analyzed the resulting trade-offs (understanding why 7 layers degraded due to overparameterization and why dropout $p=0.0$ won with Batch Normalization), and wrote `evaluate_dnn.py` to compare our champion network against all Assignment 1 models.

Both group members tested, ran, and validated all scripts end-to-end. We fully understand, can justify, and can defend every architectural choice, mathematical formulation, and hyperparameter in our work. We wrote the entire accompanying report ourselves from our experimental findings.
