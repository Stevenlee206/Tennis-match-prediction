import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import randint, uniform


def plot_random_results(random_search, reports_dir):
    """
    Tạo Dashboard hiển thị sự biến thiên của TỪNG tham số độc lập trong Random Search.
    Các tham số không được chọn sẽ bị khóa ở giá trị Best Params.
    """
    cv_results = pd.DataFrame(random_search.cv_results_)
    best_params = random_search.best_params_
    tuned_params = list(best_params.keys())

    n_params = len(tuned_params)
    cols = 3 if n_params >= 3 else 2
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, target_param in enumerate(tuned_params):
        ax = axes[i]

        # Với Random Search, không gian rải rác hơn, ta vẽ Scatter (chấm điểm) thay vì Line
        x = cv_results[f'param_{target_param}'].astype(float)  # Đảm bảo trục X là số
        y = -cv_results['mean_test_score']

        ax.scatter(x, y, alpha=0.6, color='teal', s=40)

        ax.set_title(f"Impact of '{target_param}'", fontsize=11, pad=10)
        ax.set_xlabel(target_param, fontsize=10)
        ax.set_ylabel("Log Loss (Lower is better)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Đánh dấu điểm Best
        best_x = best_params[target_param]
        # Tìm chỉ số index của dòng có best_params
        best_idx = random_search.best_index_
        best_y = y.iloc[best_idx]
        ax.scatter(best_x, best_y, color='red', s=150, marker='*', edgecolor='black', label='Best Value', zorder=5)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("XGBoost Random Search Tuning Dashboard", fontsize=16, y=1.02, fontweight='bold')

    save_path = reports_dir / "xgboost_random_tuning_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu biểu đồ Dashboard tuning tại: {save_path.name}")


def run_xgboost_random_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                                n_trials=20, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu XGBoost bằng Random Search ({n_trials} trials) ---")

    clf = XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42
    )

    param_dist = {
        'n_estimators': randint(100, 800),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.19),
        'subsample': uniform(0.6, 0.3),
        'colsample_bytree': uniform(0.6, 0.3),
        'gamma': uniform(0, 2)
    }

    # --- LUÂN CHUYỂN CHIẾN LƯỢC CROSS-VALIDATION ---
    if X_val is not None:
        # Hold out
        print("-> Sử dụng tập Validation cố định (Holdout)")
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        # Walkout
        print("-> Sử dụng Time-Series CV 3-folds (Walk-Forward)")
        X_cv = X_train
        y_cv = y_train
        cv_strategy = TimeSeriesSplit(n_splits=3)

    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=n_trials,
        cv=cv_strategy,
        scoring='neg_log_loss',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_cv, y_cv)

    best_params = random_search.best_params_
    print(f"-> Tham số tốt nhất từ Random Search: {best_params}")

    # Vẽ Dashboard báo cáo
    if len(best_params) > 0:
        plot_random_results(random_search, reports_dir)

    print("\nHuấn luyện final model với bộ tham số tốt nhất...")
    if X_val is not None:
        X_final = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final, y_final = X_train, y_train

    final_clf = XGBClassifier(**best_params, enable_categorical=True, tree_method='hist', eval_metric='logloss',
                              random_state=42)
    final_clf.fit(X_final, y_final)

    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    config = {
        "model_type": "XGBoost_RandomSearch",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, dummy_scaler