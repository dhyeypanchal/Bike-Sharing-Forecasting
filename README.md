# ECE 9603/9063: Data Analytics Foundations
## Assignments 1 & 2: Forecasting & Deep Neural Networks

**Western University — Faculty of Engineering**  
Department of Electrical and Computer Engineering  
**Dataset:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`data/hour.csv`)  

---

## 1. Project Overview

This repository contains the complete forecasting and deep learning pipelines for ECE 9603/9063:
- **Assignment 1 (Forecasting Paradigms):** Evaluates baseline linear, kernel dual space, and tree ensemble models (Seasonal Naive, Multivariate Linear Regression, Support Vector Regression, Random Forest) on 12 static attributes across chronological 80/20 holdout and 5-fold expanding-window cross-validation.
- **Assignment 2 (Deep Neural Networks):** Expands the feature space to 27 engineered features (timestamp-aligned demand lags, rolling statistics, cyclical date/time encodings and rush-hour flags) for a 1-hour-ahead forecast, and designs, trains, and systematically tunes a $\ge 5$-hidden-layer Deep Neural Network (DNN) with Batch Normalization and Dropout. Lags only use demand from earlier hours; the weather columns are the observed values, which strictly would only be available as a forecast at decision time.

### Group Responsibilities & Work Distribution:

- **Dhyey Panchal:**
  - Led the Deep Neural Network architecture design and PyTorch implementation (`DeepBikeNet`), incorporating linear transformations, Batch Normalization, and ReLU activations.
  - Designed and implemented the temporal feature engineering pipeline (27 features: autoregressive demand lags, rolling window momentum, continuous cyclical coordinates, and commuter indicators, using only demand from earlier hours).
  - Built the modular PyTorch training loop with mini-batch Adam optimization, gradient norm clipping (threshold 1.0), dynamic `ReduceLROnPlateau` learning rate scheduling, and early stopping.
  - Conceived and executed the systematic 10-trial hyperparameter ablation engine across Depth (3 vs 5 vs 7 layers), Width, Learning Rate, and Dropout.
  - Performed final test set evaluation on 3,476 held-out hours, generated comparative benchmarks against Assignment 1 models, and authored the Methodology and Results sections.

- **Aseel Qureshi:**
  - Conducted the raw dataset audit across all 17,379 hourly records from the UCI repository: no null values, but 165 of the 17,544 hours in 2011–2012 are absent from the file.
  - Reviewed the Assignment 2 pipeline and fixed lag features to be timestamp-aligned (previously row shifts, wrong after the 165 missing hours), removed a backfill leak in the first rows, added random seeds, longer early-stopping patience, saved champion configuration, measured training time, per-epoch MAE tracking, and a same-features Random Forest comparison.
  - Set up and verified the chronological data partitioning (11,680 train / 2,062 validation / 3,476 test hours, about 68/12/20) and confirmed proper `StandardScaler` fitting on training data.
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
│   │   ├── features.py                   # 27 timestamp-aligned lag, rolling, and cyclical features
│   │   ├── dataset.py                    # Scalers, 68/12/20 chronological split & PyTorch DataLoaders
│   │   ├── model.py                      # Configurable >= 5 hidden layer DeepBikeNet
│   │   ├── train.py                      # Training loop with Adam, ReduceLROnPlateau, Early Stopping
│   │   ├── tune.py                       # 10-trial hyperparameter ablation engine (depth, width, lr, drop)
│   │   └── evaluate_dnn.py               # Test set evaluation, comparative benchmarks & plots
│   └── results/                          # Generated A2 figures, checkpoints, and benchmark tables
│       ├── best_model.pt                 # Champion model weights
│       ├── best_config.json              # Champion configuration (read by evaluate_dnn.py)
│       ├── best_training_history.csv     # Per-epoch training/validation loss and MAE
│       ├── tuning_summary.csv            # 10-trial hyperparameter tuning summary
│       ├── model_comparison_dnn.csv      # Comparative benchmark against A1 models
│       ├── fig1_dnn_learning_curve.png   # Train vs. validation loss and MAE curves
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

Requirements: Python 3.9+ (tested on Python 3.12.3 with pandas 3.1.1, scikit-learn 1.9.1, PyTorch 2.15.0, CPU only)

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
   Explores Depth (3 vs 5 vs 7 layers), Width (Narrow vs Medium vs Wide), Learning Rate (0.01 vs 0.001 vs 0.0001), and Dropout (0.0 to 0.3), one factor at a time, selecting the champion by validation MAE. Each trial uses seed 42, Adam, batch size 64, weight decay 1e-4, at most 100 epochs with early stopping (patience 10). Takes about 6 minutes on a laptop CPU. Saves `tuning_summary.csv`, `best_model.pt`, `best_config.json` and `best_training_history.csv`:
   ```bash
   python assignment2/src/tune.py
   ```
4. **Evaluate Champion Model on Held-Out Test Set (3,476 Hours) & Generate Figures** (run `tune.py` first; the champion is rebuilt from `best_config.json`):
   ```bash
   python assignment2/src/evaluate_dnn.py
   ```
   Outputs saved to `assignment2/results/`:
   - `model_comparison_dnn.csv` (holdout comparison with Assignment 1 models and a Random Forest trained on the same 27 features)
   - `fig1_dnn_learning_curve.png` (training vs validation loss and MAE per epoch)
   - `fig2_dnn_tuning_ablation.png` (4-panel tuning comparison)
   - `fig3_dnn_forecast_tracking_week.png` (7-day forecast timeline)
   - `fig4_dnn_error_by_hour.png` (error breakdown across 24 hours)

---

## 5. Comparative Performance Summary

All models were evaluated on the exact same untouched 3,476-hour holdout test set (7 August 2012 12:00 to 31 December 2012 23:00):

| Model | Features / Architecture | Test MAE (bikes) | Test RMSE (bikes) | Test $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | 12 A1 attributes (OLS + One-Hot) | 98.69 | 133.82 | 0.6316 |
| **SVR (RBF)** | 12 A1 attributes, cyclical $\sin/\cos$ | 75.53 | 127.09 | 0.6678 |
| **Seasonal Naive** | Lag Baseline ($t - 168$h) | 65.01 | 112.18 | 0.7411 |
| **Random Forest** | 12 A1 attributes (100 trees, A1 champion) | 44.99 | 69.56 | 0.9005 |
| **Random Forest, same features (train hours only)** | Same 27 features and same 11,680 training hours as the DNN, A1 settings (untuned) | 38.22 | 62.56 | 0.9195 |
| **Random Forest, same features (train + val hours)** | Same 27 features, trained on 13,742 hours incl. the 2,062 validation hours, A1 settings (untuned) | **30.35** | 49.62 | 0.9494 |
| **Tuned DNN (Ours)** | 27 features, 5 hidden layers `[512, 256, 128, 64, 32]`, dropout 0.2 | 32.35 | **48.48** | **0.9517** |

### Key Findings:
- The tuned DNN reduces MAE by **28.1%** compared with Assignment 1's best model (44.99 → 32.35 bikes) and by **50.2%** compared with the Seasonal Naive baseline.
- About half of this gain comes from the new lag features and half from the network: with the same 27 features and training hours, the untuned Assignment 1 Random Forest improves from MAE 44.99 to 38.22 (15.0%), and the DNN improves further to 32.35 (15.4% lower than that forest).
- With the same features and the same training hours, the DNN beats the Random Forest (MAE 32.35 vs 38.22, RMSE 48.48 vs 62.56).
- When the Random Forest is also trained on the 2,062 most recent validation hours, it reaches MAE 30.35 (RMSE 49.62). Demand was growing, and trees cannot predict above levels seen in training, so the most recent hours matter more for the forest than for the network.
- The forecast is 1 hour ahead (it uses the previous hour's demand), and the weather inputs are observed values rather than forecasts.

### 5-Fold Expanding Time-Series Cross-Validation (Training Period)

| Model | MAE Mean (Std) | RMSE Mean (Std) | $R^2$ Mean (Std) |
| :--- | :---: | :---: | :---: |
| **Seasonal Naive** | 52.47 (6.85) | 88.71 (13.45) | 0.685 (0.103) |
| **Linear Regression** | 81.40 (19.03) | 111.61 (20.94) | 0.518 (0.117) |
| **SVR (RBF)** | 82.25 (27.46) | 127.92 (36.62) | 0.338 (0.385) |
| **Random Forest** | **52.29 (15.39)** | **80.20 (21.92)** | **0.737 (0.123)** |

---

## 6. Generative AI Statement (Code)

Generative AI was used to assist with parts of the code.

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
  1. We wrote the feature engineering pipeline in `features.py`, making sure target demand at hour $t$ is never used as an input.
  2. We implemented `DeepBikeNet` in PyTorch, structured the custom PyTorch `Dataset` and `DataLoader` pipeline, and created narrow, medium, and wide configurations across 3, 5, and 7 hidden layers.
  3. We executed and monitored all 10 tuning trials, analyzed the resulting trade-offs, and wrote `evaluate_dnn.py` to compare our champion network against all Assignment 1 models.
- **Review and fixes (AI-assisted):** During a code review we found that the lag features were row shifts rather than time shifts (wrong after the 165 missing hours), that backfilling leaked the target into the first rows, that trials were not seeded, and that the evaluation script hard-coded the champion and its fit time. An AI coding assistant helped implement the fixes (timestamp-aligned lags, seeds, early-stopping patience, saved champion configuration, measured timings, per-epoch MAE tracking, and the same-features Random Forest comparison). We checked each fix by re-running the pipeline; all numbers in this README come from that run.

Both group members tested, ran, and validated all scripts end-to-end. We fully understand, can justify, and can defend every architectural choice, mathematical formulation, and hyperparameter in our work. We wrote the entire accompanying report ourselves from our experimental findings.
