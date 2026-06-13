import json
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def plot_grid_results(grid_search, reports_dir):
    """
    Create a dashboard that displays the variations of independent parameters.
    Parameters that are not selected will be locked at the Best Parameters value.
    """
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    cv_results.columns = [col.replace('param_model__', 'param_') for col in cv_results.columns]
    tuned_params = list(best_params.keys())

    # The grid layout (Subplots) is calculated automatically based on the number of parameters.
    n_params = len(tuned_params)
    cols = 2
    rows = (n_params + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    # Ensure axes are always a one-dimensional array for easier iteration (even with only one parameter).
    axes = np.atleast_1d(axes).flatten()

    # Plot each chart
    for i, target_param in enumerate(tuned_params):
        ax = axes[i]

        # Create a filter: Lock all other parameters at the optimal level.
        query_parts = []
        for p, v in best_params.items():
            if p != target_param:
                if isinstance(v, str):
                    query_parts.append(f"param_{p} == '{v}'")
                else:
                    query_parts.append(f"param_{p} == {v}")

        if query_parts:
            query_str = " and ".join(query_parts)
            filtered_df = cv_results.query(query_str).copy()
        else:
            filtered_df = cv_results.copy()

        # Arrange the X-axis from smallest to largest so that the line doesn't break diagonally.
        filtered_df = filtered_df.sort_values(f'param_{target_param}')

        x = filtered_df[f'param_{target_param}']
        y = -filtered_df['mean_test_score']  # Change neg_log_loss to log_loss

        ax.plot(x, y, marker='o', color='teal', linewidth=2, markersize=8)

        # Decor each subplot
        ax.set_title(f"Impact of '{target_param}'\n(Others fixed at best values)", fontsize=11, pad=10)
        ax.set_xlabel(target_param, fontsize=10)
        ax.set_ylabel("Log Loss (Lower is better)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Best point
        best_x = best_params[target_param]
        best_y = y[x == best_x].values[0] if len(y[x == best_x]) > 0 else min(y)
        ax.plot(best_x, best_y, marker='s', color='red', markersize=15, label='Best Value')
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("XGBoost Hyperparameter Tuning Dashboard", fontsize=16, y=1.02, fontweight='bold')

    # Save plot
    save_path = reports_dir / "xgboost_grid_tuning_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] Dashboard tuning chart saved at : {save_path.name}")

def run_xgboost_grid_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                              xgb_grid_n_estimators=[100, 300, 500, 800],
                              xgb_grid_max_depth=[3, 5, 8],
                              xgb_grid_lr=[0.01, 0.05, 0.1],
                              xgb_grid_colsample=[0.6,0.8, 1.0],
                              xgb_grid_gamma=[0.0, 0.5, 1.0, 1.5, 2.0],
                              xgb_grid_subsample=[0.6, 0.8, 1.0],
                              validation="holdout", n_splits=5, tscv_test_size=None, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Find model at {output_dir.name}. Skip training!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Start tuning XGBoost with Grid Search ---")

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
        'model__n_estimators': xgb_grid_n_estimators,
        'model__max_depth': xgb_grid_max_depth,
        'model__learning_rate': xgb_grid_lr,
        'model__colsample_bytree': xgb_grid_colsample,
        'model__subsample': xgb_grid_subsample,
        'model__gamma': xgb_grid_gamma
    }

    if validation == "holdout" and X_val is not None:
        print("->Holdout")
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])

        # Create a split array: -1 (Train), 0 (Val)
        test_fold = np.concatenate([
            np.full(len(X_train), -1),
            np.full(len(X_val), 0)
        ])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        print("->Walk-Forward")
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
    raw_best_params = grid_search.best_params_
    best_params = {k.replace('model__', ''): v for k, v in raw_best_params.items()}
    print(f"-> Best parameters from Grid Search: {best_params}")

    if 'n_estimators' in best_params:
        plot_grid_results(grid_search, reports_dir)

    print("\n Training final model on best params...")
    if X_val is not None:
        X_final_raw = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final_raw, y_final = X_train, y_train

    final_scaler = StandardScaler()
    X_final_proc = final_scaler.fit_transform(X_final_raw)

    final_clf = XGBClassifier(**best_params, tree_method='hist', eval_metric='logloss', random_state=42)
    final_clf.fit(X_final_proc, y_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(final_scaler, scaler_path)

    config = {
        "model_type": "XGBoost_GridSearch",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, final_scaler