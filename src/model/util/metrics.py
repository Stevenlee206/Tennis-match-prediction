from __future__ import annotations

from typing import Dict

import numpy as np


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
	from sklearn.metrics import (
		accuracy_score,
		average_precision_score,
		balanced_accuracy_score,
		brier_score_loss,
		f1_score,
		log_loss,
		precision_score,
		recall_score,
		roc_auc_score,
	)

	y_true = np.asarray(y_true).astype(int).reshape(-1)
	y_prob = np.asarray(y_prob).astype(float).reshape(-1)
	y_pred = (y_prob >= threshold).astype(int)

	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_true, y_prob)),
		"pr_auc": float(average_precision_score(y_true, y_prob)),
		"log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
		"brier": float(brier_score_loss(y_true, y_prob)),
	}

def evaluate_model_bias(y_true, y_pred, X_raw, dataset_name=""):
    """
    Calculates bias metrics and returns them as a dictionary for JSON logging.
    """
    from sklearn.metrics import classification_report, accuracy_score
    print("\n" + "="*50)
    print(f" MODEL BIAS & HEURISTIC ANALYSIS {dataset_name}")
    print("="*50)

    print("CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("-" * 50)

    metrics = {}

    # 1. Target Class Bias
    pred_p1_rate = np.mean(y_pred == 1) * 100
    metrics['class_1_prediction_rate'] = round(pred_p1_rate, 2)
    print(f"Class 1 Prediction Rate:   {pred_p1_rate:.2f}% (Ideal: ~50.0%)")

    # 2. Elo Analysis
    if 'elo_diff' in X_raw.columns:
        higher_elo_p1 = (X_raw['elo_diff'] > 0).astype(int)
        
        elo_reliance = np.mean(y_pred == higher_elo_p1) * 100
        elo_baseline_acc = np.mean(y_true == higher_elo_p1) * 100
        
        metrics['elo_reliance'] = round(elo_reliance, 2)
        metrics['elo_baseline_accuracy'] = round(elo_baseline_acc, 2)
        
        print(f"Elo Reliance (Safe Bet):   {elo_reliance:.2f}%")
        print(f"Blind Elo Baseline Acc:    {elo_baseline_acc:.2f}%")

        actual_upsets_mask = (y_true != higher_elo_p1)
        if actual_upsets_mask.sum() > 0:
            upset_acc = accuracy_score(y_true[actual_upsets_mask], y_pred[actual_upsets_mask]) * 100
            metrics['upset_prediction_accuracy'] = round(upset_acc, 2)
            print(f"Upset Prediction Accuracy: {upset_acc:.2f}% (Correctly guessing the underdog)")

    # 3. ATP Rank Analysis
    if 'rank_diff' in X_raw.columns:
        better_rank_p1 = (X_raw['rank_diff'] < 0).astype(int)
        rank_reliance = np.mean(y_pred == better_rank_p1) * 100
        metrics['rank_reliance'] = round(rank_reliance, 2)
        print(f"Rank Reliance Bias:        {rank_reliance:.2f}%")
        
    final_acc = accuracy_score(y_true, y_pred) * 100
    metrics['final_accuracy'] = round(final_acc, 2)
    print(f"\nFinal Set Accuracy:        {final_acc:.2f}%")
    print("==================================================\n")
    
    return metrics

def append_metrics_to_config(config_path, metrics):
    """Safely appends bias metrics to the existing JSON config file."""
    import json
    from pathlib import Path
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        config['bias_metrics'] = metrics
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[*] Bias metrics appended to {config_path.name}")
