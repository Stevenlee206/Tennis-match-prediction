import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

def evaluate_model_bias(y_true, y_pred, X_raw, dataset_name=""):
    """
    Calculates bias metrics and returns them as a dictionary.
    """
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

def evaluate_player_metrics(y_true, y_pred, df, selected_players):
    """
    Computes Acc, Recall, Precision specifically for target players.
    """
    print("\n--- PLAYER-SPECIFIC METRICS ---")
    results = {}
    
    for cat, p_list in selected_players.items():
        for player in p_list:
            # Find matches where this player was involved
            p_mask = (df['winner_name'] == player) | (df['loser_name'] == player)
            if not p_mask.any():
                print(f"{cat} | Player {player} not found in the evaluation dataset.")
                continue
                
            p_y_true = y_true[p_mask]
            p_y_pred = y_pred[p_mask]
            
            results[player] = {
                'Accuracy': round(accuracy_score(p_y_true, p_y_pred) * 100, 2),
                'Recall': round(recall_score(p_y_true, p_y_pred, zero_division=0) * 100, 2),
                'Precision': round(precision_score(p_y_true, p_y_pred, zero_division=0) * 100, 2),
                'Matches': p_mask.sum()
            }
            
            print(f"{cat:<10} | {player:<30} | Matches: {p_mask.sum():<3} | Acc: {results[player]['Accuracy']}% | Rec: {results[player]['Recall']}% | Prec: {results[player]['Precision']}%")
        
    return results
