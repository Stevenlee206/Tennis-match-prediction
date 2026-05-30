import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import FunctionTransformer


def plot_optuna_results(study, reports_dir):
    """
    Trích xuất dữ liệu từ Optuna Study và vẽ biểu đồ Lịch sử Tối ưu.
    """
    df = study.trials_dataframe()
    # Chỉ lấy những trial thành công
    df = df[df['state'] == 'COMPLETE']

    if df.empty:
        print("[!] Không có dữ liệu trial hợp lệ để vẽ biểu đồ.")
        return

    trials = df['number']
    values = df['value']
    # cummin() giúp tạo một đường line giữ lại giá trị tốt nhất (thấp nhất) tính đến thời điểm hiện tại
    best_values = values.cummin()

    plt.figure(figsize=(10, 6))

    # Vẽ các chấm rải rác thể hiện từng Trial
    plt.scatter(trials, values, alpha=0.6, color='teal', label='Trial Value (Log Loss)')

    # Vẽ đường line thể hiện quá trình hội tụ
    plt.plot(trials, best_values, color='red', linewidth=2.5, label='Best Value (Hội tụ)')

    plt.title('Optuna Optimization History\n(XGBoost Hyperparameter Tuning)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Trial Number (Số vòng thử nghiệm)', fontsize=11)
    plt.ylabel('Log Loss (Càng thấp càng tốt)', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = reports_dir / "xgboost_optuna_tuning_history.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ Lịch sử Optuna tại: {save_path.name}")


def run_xgboost_optuna_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                                n_trials=30, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu XGBoost bằng Optuna ({n_trials} trials) ---")

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'enable_categorical': True,
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'random_state': 42
        }

        # --- LUÂN CHUYỂN CHIẾN LƯỢC CROSS-VALIDATION BÊN TRONG OPTUNA ---
        if X_val is not None:
            # 1. Chế độ Holdout (Tĩnh)
            model = XGBClassifier(**param)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            return log_loss(y_val, preds)

        else:
            # 2. Chế độ Walk-Forward (Time-Series CV)
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = XGBClassifier(**param)
                model.fit(X_tr, y_tr)

                preds = model.predict_proba(X_v)
                loss = log_loss(y_v, preds)
                cv_scores.append(loss)

            return np.mean(cv_scores)

    # Chạy Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"-> Tham số tốt nhất từ Optuna: {best_params}")

    # Vẽ biểu đồ tối ưu
    plot_optuna_results(study, reports_dir)

    # Huấn luyện mô hình cuối cùng
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

    # Lưu kết quả
    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    config = {
        "model_type": "XGBoost_Optuna",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, dummy_scaler