import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# Normal tuning
def tune_n_estimators(X_train, y_train, X_val,
                      y_val, n_range, lr, depth, reports_dir,n_split):
    """
    Find optimal num of tree Validation based on Log Loss.
    """
    print("Tuning n_estimators (Metric: Log Loss)")
    train_scores, val_scores = [], []

    if X_val is not None:
        # Holdout
        print(" Holdout ")
        for n in n_range:
            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_train)
            X_v_scaled = fold_scaler.transform(X_val)

            clf = XGBClassifier(n_estimators=n, learning_rate=lr,
                                max_depth=depth, eval_metric='logloss',
                                tree_method='hist', random_state=42)
            clf.fit(X_tr_scaled, y_train)

            # Use predict_proba for log_loss
            t_loss = log_loss(y_train, clf.predict_proba(X_tr_scaled))
            v_loss = log_loss(y_val, clf.predict_proba(X_v_scaled))

            train_scores.append(t_loss)
            val_scores.append(v_loss)
            print(f"Estimators: {n:3} | Train LogLoss: {t_loss:.4f} | Val LogLoss: {v_loss:.4f}")
    else:
        # Walk forward
        print(" Walk-Forward ")
        tscv = TimeSeriesSplit(n_splits=n_split)
        for n in n_range:
            clf = XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, eval_metric='logloss',
                                tree_method='hist', random_state=42)
            fold_t_loss, fold_v_loss = [], []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                X_v = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                y_v = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

                fold_scaler = StandardScaler()
                X_tr_scaled = fold_scaler.fit_transform(X_tr)
                X_v_scaled = fold_scaler.transform(X_v)

                clf.fit(X_tr_scaled, y_tr)
                fold_t_loss.append(log_loss(y_tr, clf.predict_proba(X_tr_scaled)))
                fold_v_loss.append(log_loss(y_v, clf.predict_proba(X_v_scaled)))

            t_loss, v_loss = np.mean(fold_t_loss), np.mean(fold_v_loss)
            train_scores.append(t_loss)
            val_scores.append(v_loss)
            print(f"Estimators: {n:3} | CV Train LogLoss: {t_loss:.4f} | CV Val LogLoss: {v_loss:.4f}")

    best_idx = np.argmin(val_scores)
    best_n = n_range[best_idx]
    best_val_loss = val_scores[best_idx]
    print(f"-> Best n_estimators: {best_n}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(n_range, train_scores, label='Train Log Loss', marker='o')
    plt.plot(n_range, val_scores, label='Validation Log Loss', marker='s')
    plt.title('XGBoost Tuning: Estimators vs Log Loss')
    plt.xlabel('n_estimators')
    plt.ylabel('Log Loss (Lower is better)')
    plt.legend()
    plt.grid(True)
    plt.savefig(reports_dir / "xgboost_tuning.png", dpi=300)
    plt.close()

    return best_n, best_val_loss


def save_xgboost_artifacts(output_dir, model, scaler, best_n, lr, depth, val_loss):
    """
    Save model, scaler and config.
    """
    joblib.dump(model, output_dir / "xgboost_model.joblib")
    joblib.dump(scaler, output_dir / "xgboost_scaler.joblib")

    config = {
        "model_type": "xgboost",
        "best_n_estimators": int(best_n),
        "learning_rate": lr,
        "max_depth": depth,
        "val_log_loss": float(val_loss)
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)


def run_xgboost_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir,
                         learning_rate=0.05,
                         max_depth=2, n_estimators=None,
                         n_splits=5,**kwargs):
    if n_estimators is not None and len(n_estimators) > 0:
        n_estimators_range = n_estimators
        print(f"[*] Run n_estimators: {n_estimators_range}")
    else:
        n_estimators_range = [10, 30, 40, 50, 70, 100,
                              130, 160, 180, 200, 240,
                              250, 300, 340, 370, 400,
                              450, 480, 500, 560, 600,
                              750, 800, 900, 1000]

    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] find model {output_dir.name}. Skip training!")
        return joblib.load(model_path), joblib.load(scaler_path)



    # Find best hyperparam
    best_n, best_val_loss = tune_n_estimators(
        X_train, y_train, X_val, y_val,
        n_estimators_range, learning_rate, max_depth, reports_dir,n_splits
    )
    print(f"\n Train final model with n_estimators={best_n}...")
    if X_val is not None:
        X_final_raw = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final_raw, y_final = X_train, y_train

    # Rescale the pooled data for the final model.
    final_scaler = StandardScaler()
    X_final_proc = final_scaler.fit_transform(X_final_raw)

    final_clf = XGBClassifier(
        n_estimators=best_n, learning_rate=learning_rate, max_depth=max_depth,
        eval_metric='logloss', tree_method='hist', random_state=42
    )
    final_clf.fit(X_final_proc, y_final)

    # Save results
    save_xgboost_artifacts(output_dir, final_clf, final_scaler, best_n, learning_rate, max_depth, best_val_loss)

    return final_clf, final_scaler