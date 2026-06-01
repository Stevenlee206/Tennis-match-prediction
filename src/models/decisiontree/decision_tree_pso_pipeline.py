import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import FunctionTransformer


def plot_pso_results(gbest_history, mean_history, reports_dir):
    iterations = range(len(gbest_history))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, gbest_history, color='red', marker='*', markersize=8, linewidth=2, label='Global Best')
    plt.plot(iterations, mean_history, color='teal', linestyle='--', marker='o', markersize=5, label='Swarm Mean')
    plt.title('PSO Optimization History\n(Decision Tree Tuning)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Log Loss', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(reports_dir / "dt_pso_tuning_history.png", dpi=300)
    plt.close()


def run_decision_tree_pso_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, particles=20, iterations=15,
                                   **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "decisiontree_model.joblib"
    scaler_path = output_dir / "decisiontree_scaler.joblib"

    force_retrain = kwargs.get('force_retrain', False)
    if not force_retrain and model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu Decision Tree bằng Custom PSO ---")

    # Bounds: max_depth [3, 20], min_samples_split [2, 100], min_samples_leaf [1, 100], criterion [0, 1] (0:gini, 1:entropy)
    bounds = np.array([[3, 20], [2, 100], [1, 100], [0, 1.99]])
    dim = len(bounds)

    def objective(position):
        depth = int(round(position[0]))
        min_split = int(round(position[1]))
        min_leaf = int(round(position[2]))
        crit = 'gini' if position[3] < 1.0 else 'entropy'

        clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, min_samples_leaf=min_leaf,
                                     criterion=crit, random_state=42)

        if X_val is not None:
            clf.fit(X_train, y_train)
            return log_loss(y_val, clf.predict_proba(X_val))
        else:
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                clf.fit(X_tr, y_tr)
                scores.append(log_loss(y_v, clf.predict_proba(X_v)))
            return np.mean(scores)

    np.random.seed(42)
    X_swarm = np.random.uniform(bounds[:, 0], bounds[:, 1], (particles, dim))
    V_swarm = np.random.uniform(-1, 1, (particles, dim))
    pbest = np.copy(X_swarm)

    pbest_scores = np.array([objective(p) for p in X_swarm])
    gbest_idx = np.argmin(pbest_scores)
    gbest = np.copy(pbest[gbest_idx])
    gbest_score = pbest_scores[gbest_idx]

    w, c1, c2 = 0.7, 1.5, 1.5
    gbest_history = [gbest_score]
    mean_history = [np.mean(pbest_scores)]

    for i in range(iterations):
        current_scores = []
        for j in range(particles):
            r1, r2 = np.random.rand(2)
            V_swarm[j] = w * V_swarm[j] + c1 * r1 * (pbest[j] - X_swarm[j]) + c2 * r2 * (gbest - X_swarm[j])
            X_swarm[j] = np.clip(X_swarm[j] + V_swarm[j], bounds[:, 0], bounds[:, 1])
            score = objective(X_swarm[j])
            current_scores.append(score)

            if score < pbest_scores[j]:
                pbest[j] = X_swarm[j]
                pbest_scores[j] = score
                if score < gbest_score:
                    gbest = np.copy(X_swarm[j])
                    gbest_score = score

        gbest_history.append(gbest_score)
        mean_history.append(np.mean(current_scores))
        print(f"Iteration {i + 1}/{iterations} | Best LogLoss: {gbest_score:.4f}")

    best_params = {
        'max_depth': int(round(gbest[0])),
        'min_samples_split': int(round(gbest[1])),
        'min_samples_leaf': int(round(gbest[2])),
        'criterion': 'gini' if gbest[3] < 1.0 else 'entropy'
    }

    plot_pso_results(gbest_history, mean_history, reports_dir)

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
        json.dump({"model_type": "DecisionTree_PSO", "best_params": best_params}, f, indent=4)

    return final_clf, dummy_scaler