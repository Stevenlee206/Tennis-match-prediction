# Tennis Match Prediction

A Python-based ATP tennis match prediction pipeline with multiple model options, walk-forward validation, and hyperparameter tuning.

## Overview

This repository is centered on `src/execution/main.py`. It loads prepared tennis data, selects a model pipeline, trains and/or tunes the chosen model, and writes results to:

- `outputs/`
- `reports/figures/`

## Project structure

- `src/`: main source code.
  - `execution/`: pipeline orchestration and CLI entrypoint.
    - `main.py`: primary entrypoint; parses arguments, prepares data, selects and runs the model pipeline.
    - `parse_args.py`: defines all CLI arguments and validation rules.
    - `prepare_data.py`: loads and prepares train/test datasets.
    - `get_pipeline_runner.py`: maps model names and optimizer modes to pipeline functions.
    - `load_and_evaluate.py`: loads trained artifacts and evaluates final performance.
    - `model_registry.py`: contains the model-to-module routing table.
    - `pros_PC_main.py`: orchestrates the continual learning benchmark comparing Neural Network (NN) vs Predictive Coding (PCN) across multiple modes.
    - `pc-vs-dl_simulator/`: interactive browser-based web visualization comparing state inference and weight learning between PCN and Backpropagation (BP).
  - `models/`: model implementations and optimization pipelines.
    - `svm/`: SVM implementations for PyTorch and scikit-learn, including Optuna and SGD variants.
    - `rf/`: random forest pipelines and tuning code.
    - `xgboost/`: XGBoost pipeline variations.
    - `decisiontree/`: decision tree pipelines and search strategies.
    - `logistic_reg/`: logistic regression pipeline.
    - `Naive_Bayes/`: Naive Bayes pipeline.
    - `preco/`: predictive coding network implementations (PRECO package) and PC-specific tuning.
  - `preprocessing/`: data cleaning, encoding, feature engineering, and rating utilities.
    - `load_data_func/`: raw data loading helpers.
    - `feature_engineering_func/`: feature construction logic.
    - `feature_select/`: feature selection utilities.
    - `encoding/`: categorical encoding functions.
    - `data_cleaning/`: dataset cleaning steps.
- `data/`: raw and processed datasets.
  - `raw_data/`: original ATP match CSV files by year.
  - `processed/`: train/validation/test split files and final matrix files.
- `outputs/`: model artifacts, saved scalers, and output directories created by training runs.
- `reports/`: generated figures, plots, and report assets.

## Requirements

Install required packages from `requirements.txt`.

```powershell
cd "Tennis-match-prediction"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` currently includes:

- `torch`
- `scikit-learn`
- `scikit-tree`
- `pandas`
- `optuna`
- `sklearn-genetic-opt`
- `tqdm`

> Note: The code also imports `numpy`, `matplotlib`, `seaborn`, and `joblib`. If you encounter import errors, install them manually.

## Run the main script

From the repository root:

```powershell
python src\execution\main.py [options]
```

There is no compilation step; Python executes the code directly.

## Standalone model scripts

Some model implementations can also be run directly outside of the main pipeline.

### ResNet (Tabular ResNet)

The ResNet-style tabular model is implemented in `src/models/mlp_pytorch/model.py` and trained by `src/models/mlp_pytorch/train_dl.py`.

Run it from the repository root:

```powershell
python -m src.models.mlp_pytorch.train_dl
```

This script performs k-fold-like training, selects the best learning rate and weight decay, and finally evaluates on a quarantine test set.

### Predictive Coding & Continual Learning

Predictive Coding (PCN) model implementation is located in `src/models/preco/` (using the `PRECO` package structure), with the PyTorch-based wrapper `PredictiveCodingNetworkTorch`.

#### Run via main pipeline:
You can run the predictive coding model through the primary entry point:
```powershell
python -m src.execution.main --model preco --optimizer optuna --validation holdout [options]
```

#### Run the Continual Learning Benchmark:
The script `src/execution/pros_PC_main.py` runs a benchmark comparing Neural Network (NN) and Predictive Coding (PCN) under continual learning conditions (Static, Fine-tune, Retrain, Online pre-quential, and Ultimate Streaming modes):
```powershell
python src/execution/pros_PC_main.py [options]
```
**Options:**
- `--run_all`: Run all benchmark modes.
- `--run_static`: Run Static mode only.
- `--run_finetune`: Run Finetune mode only.
- `--run_retrain`: Run Retrain mode only.
- `--run_online`: Run Online streaming modes.
- `--weight_strategy <none|static>`: Sample weighting strategy.
- `--bootstrap_resamples <int>`: Number of bootstrap resamples for metric confidence intervals.

#### Visual Web Simulator:
An interactive simulator comparing state inference and weight learning between Backpropagation (BP/NN) and PCN is located at `src/execution/pc-vs-dl_simulator/index.html`.
You can open this file in any modern web browser to run and visualize the simulation.

## CLI argument reference

All CLI arguments are defined in `src/execution/parse_args.py`.

### Data split and validation

- `--test_size <float>`
  - Global test set ratio applied during data preparation.
  - Default: `0.10`
- `--val_size <float>`
  - Validation set ratio for the holdout strategy.
  - Default: `0.20`
- `--n_splits <int>`
  - Number of splits for walk-forward time-series cross-validation.
  - Default: `5`
- `--validation <holdout|walk_forward>`
  - Validation strategy.
  - `holdout`: separate train/validation split.
  - `walk_forward`: sequential time-series CV.
  - Default: `holdout`

### Model selection and feature engineering

- `--model <string>`
  - Selects the model pipeline.
  - Options: `svm`, `rf`, `pytorch_svm`, `tabnet`, `deepforest`, `xgboost`, `decisiontree`, `adaboost`, `logistic_regression`, `naive_bayes`, `preco`
  - Default: `svm`
- `--add_kmeans`
  - Enable KMeans-based feature augmentation.
  - Adds cluster distance features to the dataset.
- `--n_clusters <int>`
  - Number of clusters for KMeans augmentation.
  - Default: `5`
- `--add_pca`
  - Enable PCA feature reduction before model training.
- `--weight_strategy <string>`
  - Upset weighting strategy for sample weights.
  - Options: `none`, `static`, `magnitude`, `temporal`
  - Default: `none`
- `--upset_weight <float>`
  - Weight multiplier for upset samples when `--weight_strategy` is enabled.
  - Default: `1.5`

### Global optimizer and training hyperparameters

- `--optimizer <string>`
  - Hyperparameter tuning method or optimization mode.
  - Options: `optuna`, `pso`, `ga`, `grid`, `custom`, `random`
  - Default: `optuna`
- `--n_trials <int>`
  - Number of optimization trials for Optuna-like tuning.
  - Default: `30`
- `--epochs <int>`
  - Number of epochs for neural network or SGD training.
  - Default: `100`
- `--batch_size <int>`
  - Batch size for PyTorch training.
  - Default: `64`
- `--torch_opt <string>`
  - PyTorch optimizer choice.
  - Options: `adam`, `rmsprop`, `sgd`, `sgd_nesterov`
  - Default: `adam`
- `--torch_sched <string>`
  - Learning rate scheduler for PyTorch models.
  - Options: `constant`, `cosine`, `step`, `plateau`
  - Default: `cosine`

### SVM-specific options

- `--mode <string>`
  - SVM execution mode.
  - Options: `standard`, `sgd`
  - Default: `standard`
- `--kernel <string>`
  - SVM kernel function.
  - Options: `linear`, `poly`, `rbf`
  - Default: `linear`
- `--lr_schedule <string>`
  - Learning rate schedule for SGD SVM.
  - Options: `adaptive`, `optimal`, `invscaling`, `constant`
  - Default: `adaptive`
- `--c_min <float>`
  - Minimum C value for SVM hyperparameter search.
  - Default: `1e-3`
- `--c_max <float>`
  - Maximum C value for SVM hyperparameter search.
  - Default: `1e2`
- `--c_steps <int>`
  - Number of grid steps for SVM C when using grid search.
  - Default: `10`

> Important: `--mode sgd` is only valid when `--model svm` and `--kernel linear`.

### Random forest options

- `--rf_variant <string>`
  - Selects the RF variant pipeline.
  - Options: `rf`, `extra_trees`, `rrf`, `rotation_forest`, `oblique`, `weighted`
  - Default: `rf`
- `--rf_n_est_min <int>`
  - Minimum RF estimator count for tuning.
  - Default: `50`
- `--rf_n_est_max <int>`
  - Maximum RF estimator count for tuning.
  - Default: `500`
- `--rf_n_est_steps <int>`
  - Number of RF estimator values in the search grid.
  - Default: `5`
- `--rf_depth_min <int>`
  - Minimum tree depth for RF tuning.
  - Default: `5`
- `--rf_depth_max <int>`
  - Maximum tree depth for RF tuning.
  - Default: `50`
- `--rf_depth_steps <int>`
  - Number of tree depth values in the search grid.
  - Default: `5`

### XGBoost options

- `--custom_n_estimators <int>...`
  - Custom list of estimator counts for XGBoost.
- `--xgb_grid_n_estimators <int>...`
  - Grid values for number of XGBoost estimators.
  - Default: `[100, 300, 500, 800]`
- `--xgb_grid_max_depth <int>...`
  - Grid values for tree depth.
  - Default: `[3, 5, 7]`
- `--xgb_grid_lr <float>...`
  - Grid values for learning rate.
  - Default: `[0.01, 0.05, 0.1]`
- `--xgb_grid_colsample <float>...`
  - Grid values for colsample.
  - Default: `[0.8, 1.0]`

### Decision tree options

- `--dt_grid_max_depth <int>...`
  - Max depth values for decision tree grid search.
  - Default: `[3, 5, 7, 10]`
- `--dt_grid_min_split <int>...`
  - Minimum samples required to split nodes.
  - Default: `[2, 20, 50]`
- `--dt_grid_min_leaf <int>...`
  - Minimum samples required at leaf nodes.
  - Default: `[1, 10, 30]`
- `--dt_grid_criterion <string>...`
  - Impurity criteria for splitting.
  - Default: `['gini', 'entropy']`

### Logistic regression options

- `--log_reg_C <float>...`
  - Regularization strength values.
  - Default: `[0.01, 0.1, 1.0, 10.0]`
- `--log_reg_solver <string>...`
  - Solvers for logistic regression.
  - Default: `['lbfgs', 'liblinear']`
- `--log_reg_max_iter <int>...`
  - Maximum solver iterations.
  - Default: `[1000, 2000]`

### Naive Bayes options

- `--var_smoothing <float>...`
  - Smoothing values for GaussianNB.
  - Default: `[1e-9, 1e-7, 1e-5, 1e-3, 1e-1]`

### Optimization-specific tuning parameters

- `--particles <int>`
  - Number of particles for PSO tuning.
  - Default: `15`
- `--iterations <int>`
  - Number of PSO iterations.
  - Default: `20`
- `--population <int>`
  - Population size for GA tuning.
  - Default: `20`
- `--generations <int>`
  - Number of GA generations.
  - Default: `15`

## Example commands

### Holdout mode

```powershell
python -m src.execution.main --model svm --kernel linear --optimizer optuna --validation holdout --val_size 0.2 --add_pca --add_kmeans --n_clusters 5
```

### Walk-forward mode

```powershell
python -m src.execution.main --model pytorch_svm --optimizer optuna --validation walk_forward --n_splits 5 --torch_opt adam --torch_sched cosine --weight_strategy static --n_trials 30
```

## Output structure

- `outputs/`: saved models, preprocessing artifacts, and pipeline outputs.
- `reports/figures/`: generated charts, Optuna history, and performance visuals.

## Notes

- `src/execution/main.py` is the pipeline entrypoint.
- `src/execution/parse_args.py` defines the supported CLI arguments.
- `main.py` uses `get_pipeline_runner()` to dispatch the selected model.
- In `walk_forward` mode, the holdout test set is still evaluated at the end.
- If dependencies are missing, install missing packages manually.