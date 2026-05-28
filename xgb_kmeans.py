import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Import your existing preprocessing pipeline
from src.preprocessing.preprocessing import Preprocessing
from src.utils.paths import resolve_output_base

OUTPUT_DIR = resolve_output_base(Path.cwd()) / "reports" / "figures" / "legacy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_kmeans_cluster_search(min_clusters=3, max_clusters=25):
    print(f"\n{'='*65}")
    print(f" AUTOMATED K-MEANS CLUSTER SEARCH ({min_clusters} to {max_clusters})")
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

    # Scaling is absolutely mandatory for K-Means Euclidean distance calculation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Step 2: Calculate Baseline (No K-Means) ---
    print("\n-> Calculating Baseline XGBoost Accuracy...")
    base_clf = XGBClassifier(
        n_estimators=300, max_depth=15, random_state=42, 
        n_jobs=-1, eval_metric='logloss'
    )
    base_clf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, base_clf.predict(X_test)) * 100
    print(f"   Baseline Accuracy: {baseline_acc:.2f}%\n")

    # --- Step 3: Iterate through Cluster Counts ---
    results = []
    
    for k in range(min_clusters, max_clusters + 1):
        print(f"Evaluating K-Means with k={k} clusters...", end=" ")
        
        # Reset the feature sets to clean copies for each iteration
        X_train_aug = X_train.copy()
        X_test_aug = X_test.copy()

        # Train K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        
        # Extract continuous distances to all K centroids
        train_distances = kmeans.transform(X_train_scaled)
        test_distances = kmeans.transform(X_test_scaled)

        # Append features
        for i in range(k):
            col_name = f'kmeans_dist_cluster_{i}'
            X_train_aug[col_name] = train_distances[:, i]
            X_test_aug[col_name] = test_distances[:, i]

        # Train Augmented XGBoost
        clf = XGBClassifier(
            n_estimators=300, max_depth=15, random_state=42, 
            n_jobs=-1, eval_metric='logloss'
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

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plot the augmented accuracies
    ax.plot(k_values, accuracies, marker='s', linewidth=2, markersize=8, color='#2980b9', label='XGB + K-Means Centroid Distances')
    
    # Plot the baseline
    ax.axhline(y=baseline_acc, color='#e74c3c', linestyle='--', linewidth=2, label=f'Baseline XGB ({baseline_acc:.2f}%)')
    
    # Aesthetics
    ax.set_title('XGBoost Accuracy vs. K-Means Geometric Sub-Archetypes', fontsize=16, pad=15)
    ax.set_xlabel('Number of K-Means Clusters (k)', fontsize=12)
    ax.set_ylabel('Test Set Accuracy (%)', fontsize=12)
    ax.set_xticks(range(min_clusters, max_clusters + 1))
    ax.legend(fontsize=12)
    
    # Annotate the peak
    best_k, best_acc = max(results, key=lambda x: x[1])
    ax.annotate(f'Peak: k={best_k}\n({best_acc:.2f}%)', 
                 xy=(best_k, best_acc), 
                 xytext=(best_k, best_acc + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kmeans_cluster_search.png', dpi=300)
    print("-> Graph saved successfully as 'kmeans_cluster_search.png'!")

if __name__ == "__main__":
    run_kmeans_cluster_search(min_clusters=3, max_clusters=25)