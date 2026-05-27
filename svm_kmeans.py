import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Import your existing preprocessing pipeline
from src.preprocessing.preprocessing import Preprocessing

def run_svm_kmeans_search(min_clusters=3, max_clusters=25):
    print(f"\n{'='*65}")
    print(f" AUTOMATED K-MEANS + LINEAR SVM SEARCH ({min_clusters} to {max_clusters})")
    print(f"{'='*65}")

    # --- Step 1: Data Preparation ---
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

    # Scale the base features (Mandatory for both K-Means and SVM)
    base_scaler = StandardScaler()
    X_train_scaled = base_scaler.fit_transform(X_train)
    X_test_scaled = base_scaler.transform(X_test)

    # --- Step 2: Calculate Baseline (Pure Linear SVM) ---
    print("\n-> Calculating Baseline LinearSVC Accuracy...")
    # dual=False is highly recommended when n_samples > n_features, and max_iter prevents convergence warnings
    base_clf = LinearSVC(dual=False, random_state=42, max_iter=10000)
    base_clf.fit(X_train_scaled, y_train)
    baseline_acc = accuracy_score(y_test, base_clf.predict(X_test_scaled)) * 100
    print(f"   Baseline Accuracy: {baseline_acc:.2f}%\n")

    # --- Step 3: Iterate through Cluster Counts ---
    results = []
    
    for k in range(min_clusters, max_clusters + 1):
        print(f"Evaluating K-Means with k={k} clusters...", end=" ")
        
        # Train K-Means on the scaled training features
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        
        # Extract continuous distances to all K centroids
        train_distances = kmeans.transform(X_train_scaled)
        test_distances = kmeans.transform(X_test_scaled)

        # Scale the distances! SVMs are scale-sensitive, so we shouldn't pass raw Euclidean values.
        dist_scaler = StandardScaler()
        train_dists_scaled = dist_scaler.fit_transform(train_distances)
        test_dists_scaled = dist_scaler.transform(test_distances)

        # Horizontally stack the base scaled features with the new scaled distance features
        X_train_aug = np.hstack((X_train_scaled, train_dists_scaled))
        X_test_aug = np.hstack((X_test_scaled, test_dists_scaled))

        # Train Augmented Linear SVM
        clf = LinearSVC(dual=False, random_state=42, max_iter=10000)
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
    
    k_values = [r[0] for r in results]
    accuracies = [r[1] for r in results]

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plot the augmented accuracies
    plt.plot(k_values, accuracies, marker='D', linewidth=2, markersize=8, color='#8e44ad', label='LinearSVC + K-Means Centroid Distances')
    
    # Plot the baseline
    plt.axhline(y=baseline_acc, color='#e74c3c', linestyle='--', linewidth=2, label=f'Baseline LinearSVC ({baseline_acc:.2f}%)')
    
    # Aesthetics
    plt.title('Linear SVM Accuracy vs. K-Means Geometric Augmentation', fontsize=16, pad=15)
    plt.xlabel('Number of K-Means Clusters (k)', fontsize=12)
    plt.ylabel('Test Set Accuracy (%)', fontsize=12)
    plt.xticks(range(min_clusters, max_clusters + 1))
    plt.legend(fontsize=12)
    
    # Annotate the peak
    best_k, best_acc = max(results, key=lambda x: x[1])
    plt.annotate(f'Peak: k={best_k}\n({best_acc:.2f}%)', 
                 xy=(best_k, best_acc), 
                 xytext=(best_k, best_acc + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig('svm_kmeans_search.png', dpi=300)
    print("-> Graph saved successfully as 'svm_kmeans_search.png'!")
    
    plt.show()

if __name__ == "__main__":
    run_svm_kmeans_search(min_clusters=3, max_clusters=25)