import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

# ==========================================
# Custom Particle Swarm Optimizer
# ==========================================
class PSOHyperparameterTuner:
    def __init__(self, n_particles=20, n_iterations=30, w=0.5, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w       
        self.c1 = c1     
        self.c2 = c2     
        
    def optimize(self, evaluate_func, bounds):
        min_bound, max_bound = bounds
        
        positions = np.random.uniform(min_bound, max_bound, self.n_particles)
        velocities = np.random.uniform(-1, 1, self.n_particles)
        
        pbest_positions = positions.copy()
        pbest_scores = np.zeros(self.n_particles)
        
        gbest_position = None
        gbest_score = -np.inf
        history = [] 

        print(f"\n--- Starting PSO ({self.n_particles} particles, {self.n_iterations} iterations) ---")
        print(f"Total Model Trainings Required: {self.n_particles * self.n_iterations}")
        
        for i in range(self.n_iterations):
            print(f"\n[ Iteration {i+1}/{self.n_iterations} ]")
            scores = np.zeros(self.n_particles)
            
            # Evaluate fitness for the swarm (NOW VERBOSE)
            for j in range(self.n_particles):
                C_val = 10 ** positions[j]
                
                # Print exactly which particle is running so you know it's not frozen
                print(f"  -> Evaluating Particle {j+1:02d}/{self.n_particles} [C = {C_val:10.5f}]... ", end="", flush=True)
                
                # Run the SVM training
                scores[j] = evaluate_func(C_val)
                
                print(f"Acc: {scores[j]:.4f}")
                
                if scores[j] > pbest_scores[j]:
                    pbest_scores[j] = scores[j]
                    pbest_positions[j] = positions[j]
                    
                if scores[j] > gbest_score:
                    gbest_score = scores[j]
                    gbest_position = positions[j]

            history.append(gbest_score)
            print(f">>> End of Iteration {i+1} | Current Global Best Acc: {gbest_score:.4f} <<<")

            # Update velocities and positions
            r1 = np.random.uniform(0, 1, self.n_particles)
            r2 = np.random.uniform(0, 1, self.n_particles)
            
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (pbest_positions - positions) + 
                          self.c2 * r2 * (gbest_position - positions))
            
            positions = positions + velocities
            positions = np.clip(positions, min_bound, max_bound)

        return 10 ** gbest_position, gbest_score, history

# ==========================================
# Plotting Utilities
# ==========================================
def plot_pso_history(history, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(history)+1), y=history, marker="o", color='purple')
    plt.title("PSO Optimization Convergence History")
    plt.xlabel("Iteration")
    plt.ylabel("Best Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "pso_convergence_history.png", dpi=300)
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
def run_svm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_particles=15, n_iterations=20, kernel="linear"):
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

    pso = PSOHyperparameterTuner(n_particles=n_particles, n_iterations=n_iterations)
    best_c, best_val_acc, history = pso.optimize(evaluate_svm, bounds=(-3.0, 2.0))

    print(f"\nOptimization Complete.")
    print(f"Best parameter found (C): {best_c:.6f}")
    
    print(f"Training final model with optimal PSO parameters ({kernel} kernel)...")
    final_clf = SVC(C=best_c, kernel=kernel, random_state=42)
    final_clf.fit(X_train_scaled, y_train)
    
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "C-SVM",
        "optimizer": "PSO",
        "kernel": kernel,
        "best_C": best_c,
        "val_accuracy": best_val_acc,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "svm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_pso_history(history, reports_dir)
    
    if kernel == "linear":
        plot_feature_importance(final_clf, X_train.columns, reports_dir)
    else:
        print(f"[!] Skipping Feature Importance Plot: 'coef_' is not available for the {kernel.upper()} kernel.")
    
    return final_clf, scaler