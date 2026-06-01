import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import FunctionTransformer


def plot_grid_results(grid_search, reports_dir):
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_
    tuned_params = list(best_params.keys())

    n_params = len(tuned_params)
    cols = 2
    rows = (n_params + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, target_param in enumerate(tuned_params):
        ax = axes[i]

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

        # Fix lỗi sort cho các cột chứa cả chuỗi và số
        try:
            filtered_df = filtered_df.sort_values(f'param_{target_param}')
        except TypeError:
            pass  # Bỏ qua sort nếu là categorical string (như 'gini', 'entropy')

        x = filtered_df[f'param_{target_param}'].astype(str)
        y = -filtered_df['mean_test_score']

        ax.plot(x, y, marker='o', color='teal', linewidth=2, markersize=8)

        ax.set_title(f"Impact of '{target_param}'\n(Others fixed at best values)", fontsize=11, pad=10)
        ax.set_xlabel(target_param, fontsize=10)
        ax.set_ylabel("Log Loss (Lower is better)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        best_x = str(best_params[target_param])
        best_y = y[x == best_x].values[0] if len(y[x == best_x]) > 0 else min(y)
        ax.plot(best_x, best_y, marker='*', color='red', markersize=15, label='Best Value')
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("Decision Tree Grid Search Dashboard", fontsize=16, y=1.02, fontweight='bold')

    save_path = reports_dir / "dt_grid_tuning_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu biểu đồ Dashboard tuning tại: {save_path.name}")


def run_decision_tree_grid_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "decisiontree_model.joblib"
    scaler_path = output_dir / "decisiontree_scaler.joblib"

    force_retrain = kwargs.get('force_retrain', False)
    if not force_retrain and model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu Decision Tree bằng Grid Search ---")

    clf = DecisionTreeClassifier(random_state=42)

    param_grid = {
        'max_depth': kwargs.get('dt_grid_max_depth', [3, 5, 7, 10]),
        'min_samples_split': kwargs.get('dt_grid_min_split', [2, 20, 50]),
        'min_samples_leaf': kwargs.get('dt_grid_min_leaf', [1, 10, 30]),
        'criterion': kwargs.get('dt_grid_criterion', ['gini', 'entropy'])
    }

    print(f"[*] Không gian tìm kiếm: {param_grid}")

    if X_val is not None:
        print("-> Sử dụng tập Validation cố định (Holdout)")
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        print("-> Sử dụng Time-Series CV 3-folds (Walk-Forward)")
        X_cv = X_train
        y_cv = y_train
        cv_strategy = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=clf, param_grid=param_grid, cv=cv_strategy,
        scoring='neg_log_loss', verbose=1, n_jobs=-1
    )

    grid_search.fit(X_cv, y_cv)
    best_params = grid_search.best_params_
    print(f"-> Tham số tốt nhất: {best_params}")

    if len(best_params) > 0:
        plot_grid_results(grid_search, reports_dir)

    print("\nHuấn luyện final model...")
    if X_val is not None:
        X_final, y_final = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    else:
        X_final, y_final = X_train, y_train

    final_clf = DecisionTreeClassifier(**best_params, random_state=42)
    final_clf.fit(X_final, y_final)

    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    with open(output_dir / "decisiontree_config.json", "w") as f:
        json.dump({"model_type": "DecisionTree_Grid", "best_params": best_params}, f, indent=4)

    return final_clf, dummy_scaler