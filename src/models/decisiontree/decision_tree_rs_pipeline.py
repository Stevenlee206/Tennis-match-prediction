import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import randint


def plot_random_results(random_search, reports_dir):
    cv_results = pd.DataFrame(random_search.cv_results_)
    best_params = random_search.best_params_
    tuned_params = list(best_params.keys())

    n_params = len(tuned_params)
    cols = 2
    rows = (n_params + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, target_param in enumerate(tuned_params):
        ax = axes[i]

        try:
            x = cv_results[f'param_{target_param}'].astype(float)
        except ValueError:
            x = cv_results[f'param_{target_param}'].astype(str)

        y = -cv_results['mean_test_score']
        ax.scatter(x, y, alpha=0.6, color='teal', s=40)

        ax.set_title(f"Impact of '{target_param}'", fontsize=11, pad=10)
        ax.set_xlabel(target_param, fontsize=10)
        ax.set_ylabel("Log Loss (Lower is better)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        best_x = best_params[target_param]
        best_idx = random_search.best_index_
        best_y = y.iloc[best_idx]
        ax.scatter(best_x, best_y, color='red', s=150, marker='*', edgecolor='black', label='Best Value', zorder=5)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("Decision Tree Random Search Dashboard", fontsize=16, y=1.02, fontweight='bold')

    save_path = reports_dir / "dt_random_tuning_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_decision_tree_random_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=20, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "decisiontree_model.joblib"
    scaler_path = output_dir / "decisiontree_scaler.joblib"

    force_retrain = kwargs.get('force_retrain', False)
    if not force_retrain and model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu Decision Tree bằng Random Search ({n_trials} trials) ---")

    clf = DecisionTreeClassifier(random_state=42)
    param_dist = {
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 100),
        'min_samples_leaf': randint(1, 100),
        'criterion': ['gini', 'entropy']
    }

    if X_val is not None:
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        X_cv, y_cv = X_train, y_train
        cv_strategy = TimeSeriesSplit(n_splits=3)

    random_search = RandomizedSearchCV(
        estimator=clf, param_distributions=param_dist, n_iter=n_trials,
        cv=cv_strategy, scoring='neg_log_loss', verbose=1, n_jobs=-1, random_state=42
    )

    random_search.fit(X_cv, y_cv)
    best_params = random_search.best_params_
    print(f"-> Tham số tốt nhất: {best_params}")

    if len(best_params) > 0:
        plot_random_results(random_search, reports_dir)

    print("\nHuấn luyện final model...")
    X_final = pd.concat([X_train, X_val]) if X_val is not None else X_train
    y_final = pd.concat([y_train, y_val]) if X_val is not None else y_train

    final_clf = DecisionTreeClassifier(**best_params, random_state=42)
    final_clf.fit(X_final, y_final)

    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    with open(output_dir / "decisiontree_config.json", "w") as f:
        json.dump({"model_type": "DecisionTree_Random", "best_params": best_params}, f, indent=4)

    return final_clf, dummy_scaler