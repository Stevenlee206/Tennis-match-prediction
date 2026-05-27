import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import your existing preprocessing pipeline
from src.preprocessing.preprocessing import Preprocessing

def run_rf_dynamic_centroid_distances(use_distances=False, n_clusters=5):
    mode_name = f"WITH {n_clusters} Centroid Distances" if use_distances else "WITHOUT K-Means"
    print(f"\n{'='*55}")
    print(f" RUNNING RANDOM FOREST {mode_name}")
    print(f"{'='*55}")

    # --- Step 1: Preprocessing Data ---
    prep = Preprocessing()
    data = prep.run(train_ratio=0.90)
    
    if 'target' not in data.columns:
        raise ValueError("Error: 'target' column is missing from the preprocessed data.")

    X = data.drop(columns=['target', 'year'], errors='ignore')
    y = data['target']

    # --- Step 2: Splitting Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, shuffle=False 
    )
    
    if 'is_augmented' in X_test.columns:
        y_test = y_test[X_test['is_augmented'] == 0]
        X_test = X_test[X_test['is_augmented'] == 0].drop(columns=['is_augmented'])
        X_train = X_train.drop(columns=['is_augmented'], errors='ignore')

    X_train_final = X_train.copy()
    X_test_final = X_test.copy()

    # --- Step 3: Dynamic Centroid Distance Generator ---
    if use_distances:
        print(f"-> Calculating continuous distances to {n_clusters} distinct match archetypes...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(X_train_scaled)
        
        train_distances = kmeans.transform(X_train_scaled)
        test_distances = kmeans.transform(X_test_scaled)

        # Dynamically append a new distance column for however many clusters exist
        for i in range(n_clusters):
            col_name = f'dist_to_cluster_{i}'
            X_train_final[col_name] = train_distances[:, i]
            X_test_final[col_name] = test_distances[:, i]
        
        print(f"-> {n_clusters} new geometric features successfully appended!")
    else:
        print("-> Standard dataset (No geometric features added).")

    # --- Step 4: Training Random Forest ---
    clf = RandomForestClassifier(
        n_estimators=419,        
        max_depth=18,            
        min_samples_split=37,    
        max_features='log2',     
        random_state=42,         
        n_jobs=-1                
    )
    
    clf.fit(X_train_final, y_train)

    # --- Step 5: Evaluation ---
    y_pred = clf.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred) * 100
    
    print(f"\nFinal Test Accuracy ({mode_name}): {acc:.2f}%")
    print("-" * 55)
    
    # --- Feature Importance Analysis for ALL Clusters ---
    if use_distances:
        importances = clf.feature_importances_
        features = list(X_train_final.columns)
        sorted_indices = np.argsort(importances)[::-1]
        
        print("\nHow much did the Random Forest care about your new clusters?")
        for i in range(n_clusters):
            col_name = f'dist_to_cluster_{i}'
            idx = features.index(col_name)
            rank = np.where(sorted_indices == idx)[0][0] + 1
            print(f"  - {col_name}: Ranked #{rank} (Score: {importances[idx]:.4f})")

    return acc

if __name__ == "__main__":
    print("Initializing A/B Test: Random Forest vs. High-Resolution Augmented Random Forest...")
    
    # Set your desired cluster count here
    CLUSTER_COUNT = 13
    
    acc_baseline = run_rf_dynamic_centroid_distances(use_distances=False)
    acc_augmented = run_rf_dynamic_centroid_distances(use_distances=True, n_clusters=CLUSTER_COUNT)
    
    print("\n" + "#"*55)
    print(" FINAL RESULTS")
    print("#"*55)
    print(f"Accuracy Without Distances:    {acc_baseline:.2f}%")
    print(f"Accuracy With {CLUSTER_COUNT} Distances:      {acc_augmented:.2f}%")
    
    diff = acc_augmented - acc_baseline
    if diff > 0:
        print(f"\nVerdict: High-res geometry IMPROVED the Random Forest by +{diff:.2f}%!")
    elif diff < 0:
        print(f"\nVerdict: High-res geometry CONFUSED the Random Forest by {diff:.2f}%.")
    else:
        print("\nVerdict: A dead tie!")