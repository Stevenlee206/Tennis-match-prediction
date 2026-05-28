import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import your existing preprocessing pipeline
from src.preprocessing.preprocessing import Preprocessing
from src.utils.paths import resolve_output_base

OUTPUT_DIR = resolve_output_base(Path.cwd()) / "reports" / "figures" / "legacy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_gmm_component_search(min_components=5, max_components=25):
    print(f"\n{'='*65}")
    print(f" AUTOMATED GMM COMPONENT SEARCH ({min_components} to {max_components})")
    print(f"{'='*65}")

    # --- Step 1: Data Preparation (Done once) ---
    print("-> Preprocessing and Splitting Data...")
    prep = Preprocessing()
    data = prep.run(train_ratio=0.90)
    
    if 'target' not in data.columns:
        raise ValueError("Error: 'target' column is missing from the preprocessed data.")

    X = data.drop(columns=['target', 'year'], errors='ignore')
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, shuffle=False 
    )
    
    if 'is_augmented' in X_test.columns:
        y_test = y_test[X_test['is_augmented'] == 0]
        X_test = X_test[X_test['is_augmented'] == 0].drop(columns=['is_augmented'])
        X_train = X_train.drop(columns=['is_augmented'], errors='ignore')

    # Scale once
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Step 2: Calculate Baseline (No GMM) ---
    print("\n-> Calculating Baseline Random Forest Accuracy...")
    base_clf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=25, 
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    base_clf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, base_clf.predict(X_test)) * 100
    print(f"   Baseline Accuracy: {baseline_acc:.2f}%\n")

    # --- Step 3: Iterate through Component Counts ---
    results = []
    
    for k in range(min_components, max_components + 1):
        print(f"Evaluating GMM with k={k} components...", end=" ")
        
        # Reset the feature sets to clean copies for each iteration
        X_train_aug = X_train.copy()
        X_test_aug = X_test.copy()

        # Train GMM
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42, n_init=5)
        gmm.fit(X_train_scaled)
        
        # Extract raw log-likelihoods (Mahalanobis variance)
        train_log_probs = gmm._estimate_log_prob(X_train_scaled)
        test_log_probs = gmm._estimate_log_prob(X_test_scaled)

        # Append features
        for i in range(k):
            col_name = f'gmm_ll_cluster_{i}'
            X_train_aug[col_name] = train_log_probs[:, i]
            X_test_aug[col_name] = test_log_probs[:, i]

        # Train Augmented Random Forest
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=25, 
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        clf.fit(X_train_aug, y_train)
        
        # Evaluate
        acc = accuracy_score(y_test, clf.predict(X_test_aug)) * 100
        results.append((k, acc))
        
        # Print diff from baseline
        diff = acc - baseline_acc
        trend = f"(+{diff:.2f}%)" if diff > 0 else f"({diff:.2f}%)"
        print(f"Accuracy: {acc:.2f}% {trend}")

    # --- Step 4: Generate the Line Graph ---
    print("\n-> Generating Accuracy Graph...")
    
    # Unpack results
    k_values = [r[0] for r in results]
    accuracies = [r[1] for r in results]

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plot the augmented accuracies
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8, color='#2c3e50', label='RF + GMM Log-Likelihoods')
    
    # Plot the baseline
    plt.axhline(y=baseline_acc, color='#e74c3c', linestyle='--', linewidth=2, label=f'Baseline RF ({baseline_acc:.2f}%)')
    
    # Aesthetics
    plt.title('Random Forest Accuracy vs. GMM Sub-Archetypes', fontsize=16, pad=15)
    plt.xlabel('Number of GMM Components (k)', fontsize=12)
    plt.ylabel('Test Set Accuracy (%)', fontsize=12)
    plt.xticks(range(min_components, max_components + 1))
    plt.legend(fontsize=12)
    
    # Annotate the peak
    best_k, best_acc = max(results, key=lambda x: x[1])
    plt.annotate(f'Peak: k={best_k}\n({best_acc:.2f}%)', 
                 xy=(best_k, best_acc), 
                 xytext=(best_k, best_acc + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gmm_components_search.png', dpi=300)
    print("-> Graph saved successfully as 'gmm_components_search.png'!")
    
    # Display the plot if running in an interactive environment (like a Jupyter notebook or specific IDEs)
    plt.show()

if __name__ == "__main__":
    # You can change the range here if you want to test even higher values!
    run_gmm_component_search(min_components=5, max_components=25)