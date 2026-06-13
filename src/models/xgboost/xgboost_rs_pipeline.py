import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, PredefinedSplit
from scipy.stats import randint, uniform
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def plot_random_results(random_search, reports_dir):
    """
    Create a dashboard that displays the variation of each independent parameter in Random Search.
    Parameters that are not selected will be locked at the Best Parameters value.
    """
    cv_results = pd.DataFrame(random_search.cv_results_)
    best_params = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}
    cv_results.columns = [col.replace('param_model__', 'param_') for col in cv_results.columns]
    tuned_params = list(best_params.keys())

    n_params = len(tuned_params)
    cols = 3 if n_params >= 3 else 2
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, target_param in enumerate(tuned_params):
        ax = axes[i]
        x = cv_results[f'param_{target_param}'].astype(float) 
        y = -cv_results['mean_test_score']

        ax.scatter(x, y, alpha=0.6, color='teal', s=40)

        ax.set_title(f"Impact of '{target_param}'", fontsize=11, pad=10)
        ax.set_xlabel(target_param, fontsize=10)
        ax.set_ylabel("Log Loss (Lower is better)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        best_x = best_params[target_param]
        best_idx = random_search.best_index_
        best_y = y.iloc[best_idx]
        ax.scatter(best_x, best_y, color='red', s=150, marker='s', edgecolor='black', label='Best Value', zorder=5)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("XGBoost Random Search Tuning Dashboard", fontsize=16, y=1.02, fontweight='bold')

    save_path = reports_dir / "xgboost_random_tuning_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Dashboard tuning chart saved at: {save_path.name}")


def run_xgboost_random_pipeline(X_train, y_train, X_val, y_val,
                                output_dir, reports_dir,
                                n_trials=50,validation="holdout", n_splits=5,
                                tscv_test_size=None, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Find model at {output_dir.name}. Skip training!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Start tuning XGBoost with Random Search ({n_trials} trials) ---")

    clf = XGBClassifier(
        tree_method='hist',
        eval_metric='logloss',
        random_state=42
    )

    pipeline_steps = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clf)
    ])

    param_dist = {
        'model__n_estimators': randint(100, 800),
        'model__max_depth': randint(3, 8),
        'model__learning_rate': loguniform(0.01, 0.2),
        'model__subsample': uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4),
        'model__gamma': uniform(0, 2)
    }

    if validation == "holdout" and X_val is not None:
        # Hold out
        print("-> Holdout")
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        # Walkout
        print("-> Walk-Forward ")
        X_cv = X_train
        y_cv = y_train
        cv_strategy = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)

    random_search = RandomizedSearchCV(
        estimator=pipeline_steps,
        param_distributions=param_dist,
        n_iter=n_trials,
        cv=cv_strategy,
        scoring='neg_log_loss',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_cv, y_cv)

    best_params = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}
    print(f"-> Best parameters from Random Search: {best_params}")

    # Plot
    if len(best_params) > 0:
        plot_random_results(random_search, reports_dir)

    print("\n Training final model with both params...")
    if X_val is not None:
        X_final_raw = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final_raw, y_final = X_train, y_train

    final_scaler = StandardScaler()
    X_final_proc = final_scaler.fit_transform(X_final_raw)

    final_clf = XGBClassifier(**best_params, tree_method='hist',
                              eval_metric='logloss',random_state=42)
    final_clf.fit(X_final_proc, y_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(final_scaler, scaler_path)

    config = {
        "model_type": "XGBoost_RandomSearch",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, final_scaler