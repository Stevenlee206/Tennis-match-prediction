import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import your existing preprocessing pipeline
from src.preprocessing.preprocessing import Preprocessing
from src.utils.paths import resolve_output_base

OUTPUT_DIR = resolve_output_base(Path.cwd()) / "reports" / "figures" / "legacy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_kmeans_silhouette_search(min_clusters=3, max_clusters=25, max_eval_samples=50000):
    print(f"\n{'='*65}")
    print(f" AUTOMATED K-MEANS SILHOUETTE SEARCH ({min_clusters} to {max_clusters})")
    print(f"{'='*65}")

    # --- Step 1: Data Preparation ---
    print("-> Preprocessing and Splitting Data...")
    prep = Preprocessing()
    data = prep.run(train_ratio=0.90)
    
    # Isolate clustering features from targets/metadata
    X = data.drop(columns=['target', 'year', 'is_augmented'], errors='ignore')
    
    # Standard 90/10 train/test split to isolate the 90% training block
    X_train, _, _, _ = train_test_split(
        X, data['target'] if 'target' in data.columns else None, 
        test_size=0.10, shuffle=False 
    )

    # Scaling is required for meaningful Euclidean distance space
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Handle sample size for silhouette calculation stability and speed
    num_samples = X_train_scaled.shape[0]
    if num_samples > max_eval_samples:
        print(f"-> Subsampling {max_eval_samples:,} out of {num_samples:,} training rows for efficient score evaluation...")
        np.random.seed(42)
        sample_indices = np.random.choice(num_samples, size=max_eval_samples, replace=False)
        X_eval = X_train_scaled[sample_indices]
    else:
        X_eval = X_train_scaled

    # --- Step 2: Iterate through Cluster Counts ---
    results = []
    
    for k in range(min_clusters, max_clusters + 1):
        print(f"Evaluating K-Means with k={k} clusters...", end=" ")
        
        # Fit K-Means on full 90% training data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        
        # Predict cluster identities for evaluation subset
        labels = kmeans.predict(X_eval)
        
        # Calculate overall mean Silhouette Coefficient
        score = silhouette_score(X_eval, labels, metric='euclidean', random_state=42)
        results.append((k, score))
        print(f"Silhouette Score: {score:.4f}")

    # --- Step 3: Generate the Silhouette Trend Graph ---
    print("\n-> Generating Silhouette Evaluation Graph...")
    k_values = [r[0] for r in results]
    scores = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.set_style("whitegrid")
    
    # Plot silhouette curve
    ax.plot(k_values, scores, marker='o', linewidth=2.5, markersize=8, color='#8e44ad', label='Mean Silhouette Coefficient')
    
    # Aesthetics
    ax.set_title('K-Means Cluster Optimization via Silhouette Analysis', fontsize=15, pad=15)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score (Higher is Better)', fontsize=12)
    ax.set_xticks(range(min_clusters, max_clusters + 1))
    ax.legend(fontsize=11)
    
    # Annotate the absolute peak structure
    best_k, best_score = max(results, key=lambda x: x[1])
    ax.annotate(f'Optimal Peak: k={best_k}\nScore: {best_score:.4f}', 
                 xy=(best_k, best_score), 
                 xytext=(best_k, best_score - (max(scores)-min(scores))*0.15),
                 arrowprops=dict(facecolor='#2c3e50', shrink=0.08, width=1.5, headwidth=7),
                 fontsize=10, ha='center', weight='bold')

    plt.tight_layout()
    output_filename = 'kmeans_silhouette_search_extended.png'
    plt.savefig(OUTPUT_DIR / output_filename, dpi=300)
    print(f"-> Graph saved successfully as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    # Adjust max_clusters down or up depending on structural target hypotheses
    run_kmeans_silhouette_search(min_clusters=26, max_clusters=40)