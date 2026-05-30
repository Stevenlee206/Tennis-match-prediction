import json
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import FunctionTransformer


def plot_grid_results(grid_search, reports_dir):
    """
    Tạo Dashboard hiển thị sự biến thiên của TỪNG tham số độc lập.
    Các tham số không được chọn sẽ bị khóa ở giá trị Best Params.
    """
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_
    tuned_params = list(best_params.keys())

    # 1. Tính toán bố cục lưới (Subplots) tự động dựa trên số lượng tham số
    n_params = len(tuned_params)
    cols = 2
    rows = (n_params + 1) // 2  # Làm tròn lên để đủ chỗ

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    # Đảm bảo axes luôn là mảng 1 chiều để dễ lặp (ngay cả khi chỉ có 1 tham số)
    axes = np.atleast_1d(axes).flatten()

    # 2. Vẽ từng biểu đồ
    for i, target_param in enumerate(tuned_params):
        ax = axes[i]

        # Tạo bộ lọc: Khóa tất cả các tham số KHÁC ở mức tốt nhất
        query_parts = []
        for p, v in best_params.items():
            if p != target_param:
                if isinstance(v, str):
                    query_parts.append(f"param_{p} == '{v}'")
                else:
                    query_parts.append(f"param_{p} == {v}")

        # Lọc dữ liệu
        if query_parts:
            query_str = " and ".join(query_parts)
            filtered_df = cv_results.query(query_str).copy()
        else:
            filtered_df = cv_results.copy()

        # Sắp xếp trục X từ bé đến lớn để đường line không bị gãy chéo
        filtered_df = filtered_df.sort_values(f'param_{target_param}')

        x = filtered_df[f'param_{target_param}']
        y = -filtered_df['mean_test_score']  # Chuyển neg_log_loss thành Log Loss dương

        # Vẽ line
        ax.plot(x, y, marker='o', color='teal', linewidth=2, markersize=8)

        # Trang trí từng Subplot
        ax.set_title(f"Impact of '{target_param}'\n(Others fixed at best values)", fontsize=11, pad=10)
        ax.set_xlabel(target_param, fontsize=10)
        ax.set_ylabel("Log Loss (Lower is better)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Đánh dấu điểm Best (điểm thấp nhất trên đồ thị này)
        best_x = best_params[target_param]
        best_y = y[x == best_x].values[0] if len(y[x == best_x]) > 0 else min(y)
        ax.plot(best_x, best_y, marker='*', color='red', markersize=15, label='Best Value')
        ax.legend()

    # 3. Dọn dẹp các ô trống (nếu số lượng tham số bị lẻ)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("XGBoost Hyperparameter Tuning Dashboard", fontsize=16, y=1.02, fontweight='bold')

    # 4. Lưu ảnh
    save_path = reports_dir / "xgboost_grid_tuning_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] Đã lưu biểu đồ Dashboard tuning tại: {save_path.name}")
def run_xgboost_grid_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                              xgb_grid_n_estimators=[100, 300, 500, 800],
                              xgb_grid_max_depth=[3, 5, 7],
                              xgb_grid_lr=[0.01, 0.05, 0.1],
                              xgb_grid_colsample=[0.8, 1.0],
                              **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu XGBoost bằng Grid Search ---")

    clf = XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42
    )

    # Lưới tham số (Có thể tinh chỉnh để tránh chạy quá lâu)
    param_grid = {
        'n_estimators': xgb_grid_n_estimators,
        'max_depth': xgb_grid_max_depth,
        'learning_rate': xgb_grid_lr,
        'colsample_bytree': xgb_grid_colsample
    }

    # --- LUÂN CHUYỂN CHIẾN LƯỢC CROSS-VALIDATION ---
    if X_val is not None:
        print("-> Sử dụng tập Validation cố định (Holdout)")
        # Ghép dữ liệu để đưa vào GridSearch
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])

        # Tạo mảng phân tách: -1 (Train), 0 (Val)
        test_fold = np.concatenate([
            np.full(len(X_train), -1),
            np.full(len(X_val), 0)
        ])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        print("-> Sử dụng Time-Series CV 3-folds (Walk-Forward)")
        X_cv = X_train
        y_cv = y_train
        cv_strategy = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv_strategy,  # Nhận chiến lược được gán ở trên
        scoring='neg_log_loss',
        verbose=1,
        n_jobs=-1
    )

    # Huấn luyện Grid Search trên tập dữ liệu tương ứng
    grid_search.fit(X_cv, y_cv)

    best_params = grid_search.best_params_
    print(f"-> Tham số tốt nhất từ Grid Search: {best_params}")

    if 'n_estimators' in best_params:
        plot_grid_results(grid_search, reports_dir)

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
        "model_type": "XGBoost_GridSearch",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, dummy_scaler