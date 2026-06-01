import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, PredefinedSplit
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def plot_ga_results(evolved_estimator, reports_dir):
    """
    Extract the evolutionary history from GA and plot the convergence of the model across generations.
    """
    history = evolved_estimator.history
    if not history or "gen" not in history:
        print("[!] No evolutionary history data was found to plot the graph.")
        return

    generations = history["gen"]

    # Since scoring='neg_log_loss' (negative number), we multiply by -1 to convert it to standard Log Loss (positive).
    # The lower the Log Loss, the better the model.
    best_log_loss = [-x for x in history["fitness_max"]]  # fitness_max of a negative number is the value closest to 0
    mean_log_loss = [-x for x in history["fitness"]]  # mean log loss of population

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_log_loss, label='Best Log Loss (Outstanding Individual)',
             color='red', marker='s',
             markersize=8, linewidth=2)
    plt.plot(generations, mean_log_loss, label='Average Log Loss (Population)',
             color='teal', linestyle='--',
             marker='o', markersize=5)

    plt.title('Genetic Algorithm Evolution History\n(XGBoost Hyperparameter Tuning)',
              fontsize=14, fontweight='bold',pad=15)

    plt.xlabel('Generation ', fontsize=11)
    plt.ylabel('Log Loss', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = reports_dir / "xgboost_ga_tuning_history.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] The GA Evolution History chart has been saved at: {save_path.name}")


def run_xgboost_ga_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                            population=30, generations=40,validation="holdout",
                            n_splits=5, tscv_test_size=None, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Find model at {output_dir.name}. Skip training !")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n Hyperparameter tuning XGBoost by Genetic Algorithm (Pop: {population}, Gen: {generations})")

    # Init space and model
    clf = XGBClassifier(
        tree_method='hist',
        eval_metric='logloss',
        random_state=42
    )
    pipeline_steps = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clf)
    ])

    param_grid = {
        'model__n_estimators': Integer(100, 800),
        'model__max_depth': Integer(3, 8),
        'model__learning_rate': Continuous(0.01, 0.2, distribution='log-uniform'),
        'model__subsample': Continuous(0.6, 1.0),
        'model__colsample_bytree': Continuous(0.6, 1.0),
        'model__gamma': Continuous(0.0, 2.0)
    }

    if validation == "holdout" and X_val is not None:
        print("Holdout")
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        X_cv = X_train
        y_cv = y_train
        cv_strategy = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)

    # 2. Chạy Tiến hóa
    evolved_estimator = GASearchCV(
        estimator=pipeline_steps,
        cv=cv_strategy,
        scoring='neg_log_loss',
        param_grid=param_grid,
        population_size=population,
        generations=generations,
        tournament_size=3,
        elitism=True,
        mutation_probability=0.25,
        crossover_probability=0.75,
        verbose=True,
        n_jobs=-1
    )

    evolved_estimator.fit(X_cv, y_cv)

    raw_best_params = evolved_estimator.best_params_
    best_params = {k.replace('model__', ''): v for k, v in raw_best_params.items()}
    print(f"-> Best parameters from GA: {best_params}")

    # Plot
    plot_ga_results(evolved_estimator, reports_dir)

    print("\n Train final model on best params ...")
    if  X_val is not None:
        X_final_raw = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final_raw, y_final = X_train, y_train

    final_scaler = StandardScaler()
    X_final_proc = final_scaler.fit_transform(X_final_raw)
    final_clf = XGBClassifier(**best_params, tree_method='hist',
                              eval_metric='logloss',
                              random_state=42)

    final_clf.fit(X_final_proc, y_final)
    # Save result
    joblib.dump(final_clf, model_path)
    joblib.dump(final_scaler, scaler_path)
    config = {
        "model_type": "XGBoost_GA",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, final_scaler