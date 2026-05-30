import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import FunctionTransformer


def plot_pso_results(gbest_history, mean_history, reports_dir):
    """
    Vẽ biểu đồ Lịch sử Hội tụ của bầy đàn PSO qua các vòng lặp.
    """
    iterations = range(len(gbest_history))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, gbest_history, color='red', marker='*', markersize=8, linewidth=2,
             label='Global Best (Log Loss cá thể tốt nhất)')
    plt.plot(iterations, mean_history, color='teal', linestyle='--', marker='o', markersize=5,
             label='Swarm Mean (Trung bình cả bầy)')

    plt.title('PSO Optimization History\n(XGBoost Hyperparameter Tuning)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Iteration (Vòng lặp)', fontsize=11)
    plt.ylabel('Log Loss (Càng thấp càng tốt)', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = reports_dir / "xgboost_pso_tuning_history.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ Lịch sử PSO tại: {save_path.name}")


def run_xgboost_pso_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                             particles=30, iterations=20, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu XGBoost bằng Custom PSO (Particles: {particles}, Iterations: {iterations}) ---")

    bounds = np.array([
        [100, 800],  # n_estimators
        [3, 8],  # max_depth
        [0.01, 0.2],  # learning_rate
        [0.6, 1.0],  # subsample
        [0.6, 1.0],  # colsample_bytree
        [0.0, 2.0]  # gamma
    ])
    dim = len(bounds)

    # --- ĐÃ SỬA: LUÂN CHUYỂN CHIẾN LƯỢC BÊN TRONG OBJECTIVE ---
    def objective(position):
        n_est = int(round(position[0]))
        max_d = int(round(position[1]))

        clf = XGBClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=position[2],
            subsample=position[3], colsample_bytree=position[4], gamma=position[5],
            enable_categorical=True, tree_method='hist', eval_metric='logloss', random_state=42
        )

        if X_val is not None:
            # 1. Chế độ Holdout
            clf.fit(X_train, y_train)
            preds = clf.predict_proba(X_val)
            return log_loss(y_val, preds)
        else:
            # 2. Chế độ Walk-Forward
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                clf.fit(X_tr, y_tr)
                preds = clf.predict_proba(X_v)
                scores.append(log_loss(y_v, preds))
            return np.mean(scores)

    # Khởi tạo bầy đàn (Swarm)
    np.random.seed(42)
    X_swarm = np.random.uniform(bounds[:, 0], bounds[:, 1], (particles, dim))
    V_swarm = np.random.uniform(-1, 1, (particles, dim))
    pbest = np.copy(X_swarm)

    print("[*] Đang đánh giá thế hệ bầy đàn khởi tạo...")
    pbest_scores = np.array([objective(p) for p in X_swarm])

    gbest_idx = np.argmin(pbest_scores)
    gbest = np.copy(pbest[gbest_idx])
    gbest_score = pbest_scores[gbest_idx]

    w, c1, c2 = 0.7, 1.5, 1.5

    # Lịch sử để vẽ biểu đồ (lưu lại trạng thái ban đầu)
    gbest_history = [gbest_score]
    mean_history = [np.mean(pbest_scores)]

    # Vòng lặp tối ưu PSO
    for i in range(iterations):
        current_iteration_scores = []
        for j in range(particles):
            r1, r2 = np.random.rand(2)
            V_swarm[j] = w * V_swarm[j] + c1 * r1 * (pbest[j] - X_swarm[j]) + c2 * r2 * (gbest - X_swarm[j])

            X_swarm[j] = X_swarm[j] + V_swarm[j]
            X_swarm[j] = np.clip(X_swarm[j], bounds[:, 0], bounds[:, 1])

            score = objective(X_swarm[j])
            current_iteration_scores.append(score)

            if score < pbest_scores[j]:
                pbest[j] = X_swarm[j]
                pbest_scores[j] = score
                if score < gbest_score:
                    gbest = np.copy(X_swarm[j])
                    gbest_score = score

        # Ghi nhận lịch sử sau mỗi vòng lặp
        gbest_history.append(gbest_score)
        mean_history.append(np.mean(current_iteration_scores))

        print(f"Iteration {i + 1}/{iterations} | Best LogLoss: {gbest_score:.4f}")

    best_params = {
        'n_estimators': int(round(gbest[0])),
        'max_depth': int(round(gbest[1])),
        'learning_rate': float(gbest[2]),
        'subsample': float(gbest[3]),
        'colsample_bytree': float(gbest[4]),
        'gamma': float(gbest[5])
    }

    print(f"-> Tham số tốt nhất từ PSO: {best_params}")

    # Vẽ biểu đồ bầy đàn
    plot_pso_results(gbest_history, mean_history, reports_dir)

    # ĐÃ HOÀN THIỆN: Huấn luyện mô hình cuối và lưu kết quả
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
        "model_type": "XGBoost_PSO",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, dummy_scaler