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

class PSOHyperparameterTuner2D:
    def __init__(self, n_particles=15, n_iterations=20, w=0.5, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w, self.c1, self.c2 = w, c1, c2
        
    def optimize(self, evaluate_func, bounds):
        # bounds: np.array([[n_est_min, n_est_max], [depth_min, depth_max]])
        positions = np.zeros((self.n_particles, 2))
        velocities = np.zeros((self.n_particles, 2))
        
        for i in range(2):
            positions[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], self.n_particles)
            velocities[:, i] = np.random.uniform(-1, 1, self.n_particles)
            
        pbest_positions = positions.copy()
        pbest_scores = np.zeros(self.n_particles) - 1
        
        gbest_position = None
        gbest_score = -np.inf
        history = [] 

        print(f"\n--- Starting 2D PSO ({self.n_particles} Particles, {self.n_iterations} Iterations) ---")
        
        for i in range(self.n_iterations):
            print(f"\n[ Iteration {i+1}/{self.n_iterations} ]")
            
            for j in range(self.n_particles):
                n_est = int(round(positions[j, 0]))
                depth = int(round(positions[j, 1]))
                
                print(f"  -> Particle {j+1:02d} [n_est={n_est:3d}, depth={depth:2d}]... ", end="", flush=True)
                score = evaluate_func(n_est, depth)
                print(f"Acc: {score:.4f}")
                
                if score > pbest_scores[j]:
                    pbest_scores[j] = score
                    pbest_positions[j] = positions[j].copy()
                    
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = positions[j].copy()

            history.append(gbest_score)

            # Vectorized PSO Updates
            r1 = np.random.uniform(0, 1, (self.n_particles, 2))
            r2 = np.random.uniform(0, 1, (self.n_particles, 2))
            
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (pbest_positions - positions) + 
                          self.c2 * r2 * (gbest_position - positions))
            
            positions = positions + velocities
            for k in range(2):
                positions[:, k] = np.clip(positions[:, k], bounds[k, 0], bounds[k, 1])

        return int(round(gbest_position[0])), int(round(gbest_position[1])), gbest_score, history

def plot_pso_history(history, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(history)+1), y=history, marker="o", color='purple')
    plt.title("PSO Optimization Convergence History")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(save_path / "pso_convergence.png", dpi=300)
    plt.close()

def run_rf_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_particles=15, n_iterations=20, n_est_min=50, n_est_max=500, depth_min=5, depth_max=50):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
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
    pso = PSOHyperparameterTuner2D(n_particles=n_particles, n_iterations=n_iterations)
    best_n, best_d, best_acc, history = pso.optimize(evaluate_rf, bounds)
    
    final_clf = RandomForestClassifier(n_estimators=best_n, max_depth=best_d, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_scaled, y_train)
    config = {
        "model_type": "RandomForest",
        "optimizer": "PSO",
        "pso_params": {
            "n_particles": n_particles,
            "n_iterations": n_iterations,
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
    plot_pso_history(history, reports_dir)
    
    return final_clf, scaler