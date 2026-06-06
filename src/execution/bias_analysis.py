import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, brier_score_loss
)

def evaluate_model_bias(y_true, y_pred, X_raw,y_prob=None):
    """
    Calculates bias metrics and returns them as a dictionary for JSON logging.
    """
    print("\n" + "=" * 50)
    print(" Model Eval & Bias analysis")
    print("=" * 50)

    print("CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("-" * 50)

    metrics = {}

    # Standard Metrics
    final_acc = accuracy_score(y_true, y_pred) * 100
    metrics['final_accuracy'] = round(final_acc, 2)
    metrics['precision'] = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
    metrics['recall'] = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
    metrics['f1_score'] = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
    metrics['matthews_corr_coef'] = round(matthews_corrcoef(y_true, y_pred), 4)

    print(f"Accuracy:                  {metrics['final_accuracy']:.2f}%")
    print(f"Precision:                 {metrics['precision']:.2f}%")
    print(f"Recall:                    {metrics['recall']:.2f}%")
    print(f"F1 Score:                  {metrics['f1_score']:.2f}%")
    print(f"Matthews Corr_coef (MCC):   {metrics['matthews_corr_coef']:.4f}")

    # Probabilistic Metrics (ROC-AUC & Brier) 
    if y_prob is not None:
        try:
            metrics['roc_auc'] = round(roc_auc_score(y_true, y_prob) * 100, 2)
            metrics['brier_score'] = round(brier_score_loss(y_true, y_prob), 4)
            print(f"ROC-AUC:                   {metrics['roc_auc']:.2f}%")
            print(f"Brier Score:               {metrics['brier_score']:.4f} (Ideal: closer to 0)")
        except Exception as e:
            print(f"Warning: Could not calculate Probabilistic Metrics: {str(e)}")
    else:
        print("ROC-AUC / Brier Score:     [Skipped - No predict_proba available]")

    print("-" * 50)

    # Target Class Bias
    pred_p1_rate = np.mean(y_pred == 1) * 100
    metrics['class_1_prediction_rate'] = round(pred_p1_rate, 2)
    print(f"Class 1 Prediction Rate:   {pred_p1_rate:.2f}% (Ideal: ~50.0%)")

    # Elo Analysis
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

    # ATP Rank Analysis
    if 'rank_diff' in X_raw.columns:
        better_rank_p1 = (X_raw['rank_diff'] < 0).astype(int)
        rank_reliance = np.mean(y_pred == better_rank_p1) * 100
        metrics['rank_reliance'] = round(rank_reliance, 2)
        print(f"Rank Reliance Bias:        {rank_reliance:.2f}%")


    return metrics


def append_metrics_to_config(config_path, metrics):
    """
    Safely appends bias metrics to the existing JSON config file.
    """
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        config['bias_metrics'] = metrics

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[*] Bias metrics appended to {config_path.name}")