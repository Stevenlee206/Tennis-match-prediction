import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import copy

def calculate_permutation_importance(model, X_test, y_test, feature_names, metric=accuracy_score, n_repeats=5):
    """
    Calculates Permutation Feature Importance for any model that supports predict_proba.
    
    Args:
        model: Trained model with `predict_proba` method.
        X_test: Test features (NumPy array).
        y_test: Test targets (NumPy array).
        feature_names: List of feature names.
        metric: Metric function to evaluate performance. Default is accuracy_score.
        n_repeats: Number of times to shuffle each feature.
        
    Returns:
        DataFrame containing mean importance and std for each feature.
    """
    # Baseline performance
    probs = model.predict_proba(X_test)
    if len(probs.shape) > 1 and probs.shape[1] > 1:
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = (probs >= 0.5).astype(int)
        
    baseline_score = metric(y_test, y_pred)
    
    importances = []
    
    for col_idx in range(X_test.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            # Shuffle the column
            np.random.shuffle(X_shuffled[:, col_idx])
            
            # Predict
            probs_shuf = model.predict_proba(X_shuffled)
            if len(probs_shuf.shape) > 1 and probs_shuf.shape[1] > 1:
                y_pred_shuf = np.argmax(probs_shuf, axis=1)
            else:
                y_pred_shuf = (probs_shuf >= 0.5).astype(int)
                
            score = metric(y_test, y_pred_shuf)
            # Importance is baseline - shuffled score
            # A positive value means the feature is important (shuffling degraded performance)
            scores.append(baseline_score - score)
            
        importances.append({
            'feature': feature_names[col_idx] if feature_names else f'Feature_{col_idx}',
            'importance_mean': np.mean(scores),
            'importance_std': np.std(scores)
        })
        
    df_imp = pd.DataFrame(importances).sort_values(by='importance_mean', ascending=False)
    return df_imp

def plot_feature_importance(df_imp, title="Feature Importance", save_path=None, top_n=15):
    """
    Plots the top N features based on permutation importance.
    """
    df_plot = df_imp.head(top_n).copy()
    df_plot = df_plot.sort_values(by='importance_mean', ascending=True) # Ascending for horizontal bar
    
    plt.figure(figsize=(10, 8))
    plt.barh(df_plot['feature'], df_plot['importance_mean'], xerr=df_plot['importance_std'], color='skyblue', edgecolor='black')
    plt.xlabel('Mean Accuracy Decrease')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
