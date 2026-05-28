import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.utils.paths import ensure_writable_path

class GAHyperparameterTuner2D:
    def __init__(self, pop_size=20, n_generations=15, mutation_rate=0.2):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        
    def optimize(self, evaluate_func, bounds):
        # bounds: np.array([[n_est_min, n_est_max], [depth_min, depth_max]]) -> [n_estimators bounds, max_depth bounds]
        population = np.zeros((self.pop_size, 2))
        for i in range(2):
            population[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], self.pop_size)
            
        gbest_position = None
        gbest_score = -np.inf
        history = []

        print(f"\n--- Starting 2D GA ({self.pop_size} Individuals, {self.n_generations} Generations) ---")
        
        for generation in range(self.n_generations):
            print(f"\n[ Gen {generation+1}/{self.n_generations} ]")
            fitness = np.zeros(self.pop_size)
            
            for j in range(self.pop_size):
                n_est = int(round(population[j, 0]))
                depth = int(round(population[j, 1]))
                
                print(f"  -> Ind {j+1:02d} [n_est={n_est:3d}, depth={depth:2d}]... ", end="", flush=True)
                fitness[j] = evaluate_func(n_est, depth)
                print(f"Acc: {fitness[j]:.4f}")
                
                if fitness[j] > gbest_score:
                    gbest_score = fitness[j]
                    gbest_position = population[j].copy()

            history.append(gbest_score)
            
            next_generation = np.zeros((self.pop_size, 2))
            next_generation[0] = population[np.argmax(fitness)] # Elitism
            
            for i in range(1, self.pop_size):
                # Tournament Selection
                idx1 = np.random.choice(self.pop_size, 3, replace=False)
                idx2 = np.random.choice(self.pop_size, 3, replace=False)
                p1 = population[idx1[np.argmax(fitness[idx1])]]
                p2 = population[idx2[np.argmax(fitness[idx2])]]
                
                # Blend Crossover
                beta = np.random.rand(2)
                child = p1 * beta + p2 * (1 - beta)
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    child[0] += np.random.normal(0, 20) # n_estimators mutation step
                    child[1] += np.random.normal(0, 5)  # max_depth mutation step
                    
                next_generation[i] = child
                
            # Clip bounds
            for k in range(2):
                population[:, k] = np.clip(next_generation[:, k], bounds[k, 0], bounds[k, 1])

        return int(round(gbest_position[0])), int(round(gbest_position[1])), gbest_score, history

def plot_ga_history(history, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(history)+1), y=history, marker="o", color='green')
    plt.title("Genetic Algorithm Convergence History")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(save_path / "ga_convergence.png", dpi=300)
    plt.close()

def run_rf_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, pop_size=20, n_generations=15, n_est_min=50, n_est_max=500, depth_min=5, depth_max=50):
    output_dir = ensure_writable_path(output_dir)
    reports_dir = ensure_writable_path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    def evaluate_rf(n_est, depth):
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        return accuracy_score(y_val, clf.predict(X_val_scaled))

    bounds = np.array([[n_est_min, n_est_max], [depth_min, depth_max]]) # Dynamic bounds
    ga = GAHyperparameterTuner2D(pop_size=pop_size, n_generations=n_generations)
    best_n, best_d, best_acc, history = ga.optimize(evaluate_rf, bounds)
    best_params = {"n_estimators": best_n, "max_depth": best_d}
    print(f"\nBest parameters: {best_params} with Accuracy: {best_acc:.4f}")
    final_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_scaled, y_train)
    config = {
        "model_type": "RandomForest",
        "optimizer": "GeneticAlgorithm",
        "ga_params": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "bounds": {"n_estimators": [n_est_min, n_est_max], "max_depth": [depth_min, depth_max]}
        },
        "best_params": {
            "n_estimators": best_n,
            "max_depth": best_d
        },
        "val_accuracy": best_acc,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "rf_config.json", "w") as f:
        json.dump(config, f, indent=4)
    joblib.dump(final_clf, output_dir / "rf_model.joblib")
    joblib.dump(scaler, output_dir / "rf_scaler.joblib")
    plot_ga_history(history, reports_dir)
    
    return final_clf, scaler