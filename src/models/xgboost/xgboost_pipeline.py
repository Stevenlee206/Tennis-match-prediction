import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import TimeSeriesSplit

# Normal tuning
def tune_n_estimators(X_train, y_train, X_val, y_val, n_range, lr, depth, reports_dir):
    """Find optimal num of tree Validation."""
    print("Tuning n_estimators")
    train_scores, val_scores = [], []

    if X_val is not None:
        # HOLDOUT
        print("-> Use fixed Validation (Holdout)")
        for n in n_range:
            clf = XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, eval_metric='logloss',
                                tree_method='hist', random_state=42)
            clf.fit(X_train, y_train)
            t_acc = accuracy_score(y_train, clf.predict(X_train))
            v_acc = accuracy_score(y_val, clf.predict(X_val))
            train_scores.append(t_acc)
            val_scores.append(v_acc)
            print(f"Estimators: {n:3} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
    else:
        # WALK-FORWARD (Time-Series CV)
        print("-> Use Time-Series CV 3-folds (Walk-Forward)")
        tscv = TimeSeriesSplit(n_splits=3)
        for n in n_range:
            clf = XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, eval_metric='logloss',
                                tree_method='hist', random_state=42)
            fold_t_acc, fold_v_acc = [], []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                clf.fit(X_tr, y_tr)
                fold_t_acc.append(accuracy_score(y_tr, clf.predict(X_tr)))
                fold_v_acc.append(accuracy_score(y_v, clf.predict(X_v)))

            t_acc, v_acc = np.mean(fold_t_acc), np.mean(fold_v_acc)
            train_scores.append(t_acc)
            val_scores.append(v_acc)
            print(f"Estimators: {n:3} | CV Train Acc: {t_acc:.4f} | CV Val Acc: {v_acc:.4f}")

    best_idx = np.argmax(val_scores)
    best_n = n_range[best_idx]
    best_val_acc = val_scores[best_idx]
    print(f"-> Best n_estimators: {best_n}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(n_range, train_scores, label='Train Accuracy', marker='o')
    plt.plot(n_range, val_scores, label='Validation Accuracy', marker='s')
    plt.title('XGBoost Tuning: Estimators vs Accuracy')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(reports_dir / "xgboost_tuning.png", dpi=300)
    plt.close()

    return best_n, best_val_acc

def save_xgboost_artifacts(output_dir, model, dummy_scaler, best_n, lr, depth, val_acc):
    """Save model, dummy scaler and config."""
    joblib.dump(model, output_dir / "xgboost_model.joblib")
    joblib.dump(dummy_scaler, output_dir / "xgboost_scaler.joblib")

    config = {
        "model_type": "xgboost",
        "best_n_estimators": int(best_n),
        "learning_rate": lr,
        "max_depth": depth,
        "val_accuracy": float(val_acc)
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)


def run_xgboost_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                         learning_rate=0.05, max_depth=2,n_estimators=None, **kwargs):
    if n_estimators is not None and len(n_estimators) > 0:
        n_estimators_range = n_estimators
        print(f"[*] Run n_estimators: {n_estimators_range}")
    else:
        n_estimators_range = [10, 30, 40, 50, 70, 100,
                              130, 160, 180, 200, 240,
                              250, 300, 340, 370, 400,
                              450, 480, 500,560, 600,
                              750,800,900,1000]
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] find model {output_dir.name}.Skip training!")
        return joblib.load(model_path), joblib.load(scaler_path)

    # Find best hyperparam
    best_n, best_val_acc = tune_n_estimators(
        X_train, y_train, X_val, y_val,
        n_estimators_range, learning_rate, max_depth, reports_dir
    )
    # Merge data and train final model
    print(f"\n Train final model with n_estimators={best_n}...")
    if X_val is not None:
        X_final = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final, y_final = X_train, y_train

    final_clf = XGBClassifier(
        n_estimators=best_n, learning_rate=learning_rate, max_depth=max_depth,
        eval_metric='logloss', tree_method='hist', random_state=42
    )
    final_clf.fit(X_final, y_final)

    # Tạo Dummy Scaler (Chỉ pass-through dữ liệu, không scale)
    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    # Save results
    save_xgboost_artifacts(output_dir, final_clf, dummy_scaler, best_n, learning_rate, max_depth, best_val_acc)

    return final_clf, dummy_scaler