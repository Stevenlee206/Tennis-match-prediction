import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import FunctionTransformer
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical

def plot_ga_results(evolved_estimator, reports_dir):
    history = evolved_estimator.history
    if not history or "gen" not in history:
        return

    generations = history["gen"]
    best_log_loss = [-x for x in history["fitness_max"]]
    mean_log_loss = [-x for x in history["fitness"]]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_log_loss, label='Best Log Loss', color='red', marker='*', markersize=8, linewidth=2)
    plt.plot(generations, mean_log_loss, label='Average Log Loss', color='teal', linestyle='--', marker='o', markersize=5)

    plt.title('Genetic Algorithm Evolution History\n(Decision Tree Tuning)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Generation', fontsize=11)
    plt.ylabel('Log Loss', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = reports_dir / "dt_ga_tuning_history.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_decision_tree_ga_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, population=20, generations=15, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "decisiontree_model.joblib"
    scaler_path = output_dir / "decisiontree_scaler.joblib"

    force_retrain = kwargs.get('force_retrain', False)
    if not force_retrain and model_path.exists() and scaler_path.exists():
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu tối ưu Decision Tree bằng GA (Pop: {population}, Gen: {generations}) ---")

    clf = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': Integer(3, 20),
        'min_samples_split': Integer(2, 100),
        'min_samples_leaf': Integer(1, 100),
        'criterion': Categorical(['gini', 'entropy'])
    }

    if X_val is not None:
        X_cv = pd.concat([X_train, X_val])
        y_cv = pd.concat([y_train, y_val])
        test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
        cv_strategy = PredefinedSplit(test_fold)
    else:
        X_cv, y_cv = X_train, y_train
        cv_strategy = TimeSeriesSplit(n_splits=3)

    evolved_estimator = GASearchCV(
        estimator=clf, cv=cv_strategy, scoring='neg_log_loss', param_grid=param_grid,
        population_size=population, generations=generations, tournament_size=3, elitism=True, n_jobs=-1
    )

    evolved_estimator.fit(X_cv, y_cv)
    best_params = evolved_estimator.best_params_
    print(f"-> Tham số tốt nhất từ GA: {best_params}")

    plot_ga_results(evolved_estimator, reports_dir)

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
        json.dump({"model_type": "DecisionTree_GA", "best_params": best_params}, f, indent=4)

    return final_clf, dummy_scaler