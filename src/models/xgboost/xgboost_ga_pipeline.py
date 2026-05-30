import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import FunctionTransformer
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous


def plot_ga_results(evolved_estimator, reports_dir):
    """
    Trích xuất lịch sử tiến hóa từ GA và vẽ biểu đồ sự hội tụ của mô hình qua từng thế hệ.
    """
    history = evolved_estimator.history
    if not history or "gen" not in history:
        print("[!] Không tìm thấy dữ liệu lịch sử tiến hóa để vẽ biểu đồ.")
        return

    generations = history["gen"]

    # Do scoring='neg_log_loss' (số âm), ta nhân -1 để chuyển về Log Loss chuẩn (dương)
    # Log loss càng thấp thì mô hình càng tốt
    best_log_loss = [-x for x in history["fitness_max"]]  # fitness_max của số âm là giá trị gần 0 nhất
    mean_log_loss = [-x for x in history["fitness"]]  # trung bình của quần thể

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_log_loss, label='Best Log Loss (Cá thể xuất sắc nhất)',
             color='red', marker='*',
             markersize=8, linewidth=2)
    plt.plot(generations, mean_log_loss, label='Average Log Loss (Trung bình quần thể)',
             color='teal', linestyle='--',
             marker='o', markersize=5)

    plt.title('Genetic Algorithm Evolution History\n(XGBoost Hyperparameter Tuning)', fontsize=14, fontweight='bold',
              pad=15)
    plt.xlabel('Generation (Thế hệ)', fontsize=11)
    plt.ylabel('Log Loss (Càng thấp càng tốt)', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = reports_dir / "xgboost_ga_tuning_history.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ Lịch sử tiến hóa GA tại: {save_path.name}")


def run_xgboost_ga_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                            population=30, generations=40, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n Hyperparameter tuning XGBoost by Genetic Algorithm (Pop: {population}, Gen: {generations})")

    # 1. Khởi tạo mô hình và không gian tìm kiếm
    clf = XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42
    )

    param_grid = {
        'n_estimators': Integer(100, 800),
        'max_depth': Integer(3, 8),
        'learning_rate': Continuous(0.01, 0.2, distribution='log-uniform'),
        'subsample': Continuous(0.6, 0.9),
        'colsample_bytree': Continuous(0.6, 0.9),
        'gamma': Continuous(0, 2)
    }

    # --- LUÂN CHUYỂN CHIẾN LƯỢC CROSS-VALIDATION ---
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

    # 2. Chạy Tiến hóa
    evolved_estimator = GASearchCV(
        estimator=clf,
        cv=cv_strategy,  # Nhận chiến lược được gán ở trên
        scoring='neg_log_loss',
        param_grid=param_grid,
        population_size=population,
        generations=generations,
        tournament_size=3,
        elitism=True,
        verbose=True,
        n_jobs=-1
    )

    evolved_estimator.fit(X_cv, y_cv)

    best_params = evolved_estimator.best_params_
    print(f"-> Tham số tốt nhất từ GA: {best_params}")

    # Vẽ biểu đồ tiến hóa
    plot_ga_results(evolved_estimator, reports_dir)

    # 3. Huấn luyện Final Model với dữ liệu tối đa (Train + Val)
    print("\nHuấn luyện final model với bộ tham số tốt nhất...")
    if X_val is not None:
        X_final = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final, y_final = X_train, y_train

    final_clf = XGBClassifier(**best_params, enable_categorical=True, tree_method='hist', eval_metric='logloss',
                              random_state=42)
    final_clf.fit(X_final, y_final)

    # 4. Dummy Scaler
    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    # 5. Lưu kết quả
    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    config = {
        "model_type": "XGBoost_GA",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, dummy_scaler