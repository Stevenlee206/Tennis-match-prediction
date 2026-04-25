import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

# ==========================================
# Custom Genetic Algorithm Optimizer
# ==========================================
class GAHyperparameterTuner:
    def __init__(self, pop_size=20, n_generations=20, mutation_rate=0.2, crossover_rate=0.8):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def _tournament_selection(self, population, fitness, k=3):
        """Selects the best individual from a random subset of size k."""
        selected_indices = np.random.choice(len(population), k, replace=False)
        best_idx = selected_indices[np.argmax(fitness[selected_indices])]
        return population[best_idx]
        
    def optimize(self, evaluate_func, bounds):
        min_bound, max_bound = bounds
        
        # Initialize random population in log space (e.g., -3 to 2 for C)
        population = np.random.uniform(min_bound, max_bound, self.pop_size)
        
        gbest_position = None
        gbest_score = -np.inf
        history = []

        print(f"\n--- Starting Genetic Algorithm ({self.pop_size} Individuals, {self.n_generations} Generations) ---")
        print(f"Total Model Trainings Required: {self.pop_size * self.n_generations}")
        
        for generation in range(self.n_generations):
            print(f"\n[ Generation {generation+1}/{self.n_generations} ]")
            fitness = np.zeros(self.pop_size)
            
            # Evaluate Fitness (Verbose)
            for j in range(self.pop_size):
                C_val = 10 ** population[j]
                print(f"  -> Evaluating Individual {j+1:02d}/{self.pop_size} [C = {C_val:10.5f}]... ", end="", flush=True)
                
                fitness[j] = evaluate_func(C_val)
                print(f"Acc: {fitness[j]:.4f}")
                
                # Update Global Best
                if fitness[j] > gbest_score:
                    gbest_score = fitness[j]
                    gbest_position = population[j]

            history.append(gbest_score)
            print(f">>> End of Gen {generation+1} | Current Best Acc: {gbest_score:.4f} <<<")

            # Breed next generation
            next_generation = np.zeros(self.pop_size)
            
            # Elitism: Automatically keep the absolute best individual from the current generation
            best_idx = np.argmax(fitness)
            next_generation[0] = population[best_idx]
            
            for i in range(1, self.pop_size):
                # 1. Selection
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # 2. Crossover (Blend Crossover for continuous variables)
                if np.random.rand() < self.crossover_rate:
                    beta = np.random.rand()
                    child = parent1 * beta + parent2 * (1 - beta)
                else:
                    child = parent1 # No crossover, clone parent 1
                    
                # 3. Mutation (Gaussian Noise)
                if np.random.rand() < self.mutation_rate:
                    # The scale determines how aggressive the mutation is
                    mutation_shift = np.random.normal(loc=0.0, scale=0.5)
                    child += mutation_shift
                    
                next_generation[i] = child
                
            # Update population and clip to strictly stay within bounds
            population = np.clip(next_generation, min_bound, max_bound)

        return 10 ** gbest_position, gbest_score, history

# ==========================================
# Plotting Utilities
# ==========================================
def plot_ga_history(history, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(history)+1), y=history, marker="o", color='green')
    plt.title("Genetic Algorithm Convergence History")
    plt.xlabel("Generation")
    plt.ylabel("Best Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "ga_convergence_history.png", dpi=300)
    plt.close()

def plot_feature_importance(clf, feature_names, save_path):
    importances = clf.coef_[0]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': importances,
        'Absolute_Importance': np.abs(importances)
    }).sort_values(by='Absolute_Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Coefficient', y='Feature', palette="vlag")
    plt.title("SVM Feature Importance (Linear Coefficients)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

# ==========================================
# Main Execution Pipeline
# ==========================================
def run_svm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, pop_size=20, n_generations=20, kernel="linear"):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "svm_model.joblib"
    scaler_path = output_dir / "svm_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training and inferring immediately!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def evaluate_svm(C):
        clf = SVC(C=C, kernel=kernel, max_iter=20000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        return accuracy_score(y_val, clf.predict(X_val_scaled))

    ga = GAHyperparameterTuner(pop_size=pop_size, n_generations=n_generations)
    best_c, best_val_acc, history = ga.optimize(evaluate_svm, bounds=(-3.0, 2.0))

    print(f"\nOptimization Complete.")
    print(f"Best parameter found (C): {best_c:.6f}")
    
    print(f"Training final model with optimal GA parameters ({kernel} kernel)...")
    final_clf = SVC(C=best_c, kernel=kernel, random_state=42)
    final_clf.fit(X_train_scaled, y_train)
    
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "C-SVM",
        "optimizer": "GA",
        "kernel": kernel,
        "best_C": best_c,
        "val_accuracy": best_val_acc,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "svm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_ga_history(history, reports_dir)
    
    if kernel == "linear":
        plot_feature_importance(final_clf, X_train.columns, reports_dir)
    else:
        print(f"[!] Skipping Feature Importance Plot: 'coef_' is not available for the {kernel.upper()} kernel.")
        
    return final_clf, scaler