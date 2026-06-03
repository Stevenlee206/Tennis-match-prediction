import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def plot_nb_grid_results(grid_search, reports_dir):
    """
    The dashboard shows the impact of var_smoothing on Log Loss.
    """
    cv_results = pd.DataFrame(grid_search.cv_results_)
    # Model prefix extraction__
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    cv_results.columns = [col.replace('param_model__', 'param_') for col in cv_results.columns]

    target_param = 'var_smoothing'
    filtered_df = cv_results.sort_values(f'param_{target_param}')

    x = filtered_df[f'param_{target_param}']
    y = -filtered_df['mean_test_score']

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', color='teal', linewidth=2, markersize=8)

    # Best Value
    best_x = best_params[target_param]
    best_y = y[x == best_x].values[0]
    plt.plot(best_x, best_y, marker='s', color='red', markersize=15, label='Best Value')

    plt.xscale('log')  # var_smoothing typically runs on a logarithmic scale.
    plt.title("Naive Bayes: Impact of 'var_smoothing'", fontsize=14, fontweight='bold')
    plt.xlabel('var_smoothing (log scale)', fontsize=12)
    plt.ylabel('Log Loss (Lower is better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_path = reports_dir / "nb_grid_tuning_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Dashboard tuning chart saved at: {save_path.name}")


def run_nb_grid_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                         nb_grid_var_smoothing=[1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
                         validation="holdout", n_splits=5, tscv_test_size=None, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "nb_model.joblib"
    scaler_path = output_dir / "nb_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Find model at {output_dir.name}. Skip training!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Start tuning Naive Bayes with Grid Search ---")

    pipeline_steps = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianNB())
    ])

    param_grid = {
        'model__var_smoothing': nb_grid_var_smoothing
    }

    if validation == "holdout" and X_val is not None:
        print("Holdout")
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        print("Walk-Forward")
        X_cv = X_train
        y_cv = y_train
        cv_strategy = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)

    grid_search = GridSearchCV(
        estimator=pipeline_steps,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='neg_log_loss',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_cv, y_cv)

    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    print(f"-> Best parameters from Grid Search: {best_params}")

    plot_nb_grid_results(grid_search, reports_dir)

    # Final training
    print("\n Training final model on best params...")
    X_final_raw = pd.concat([X_train, X_val]) if X_val is not None else X_train
    y_final = pd.concat([y_train, y_val]) if X_val is not None else y_train

    final_scaler = StandardScaler()
    X_final_proc = final_scaler.fit_transform(X_final_raw)

    final_clf = GaussianNB(**best_params)
    final_clf.fit(X_final_proc, y_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(final_scaler, scaler_path)

    with open(output_dir / "nb_config.json", "w") as f:
        json.dump({"model_type": "NB_GridSearch", "best_params": best_params}, f, indent=4)

    return final_clf, final_scaler