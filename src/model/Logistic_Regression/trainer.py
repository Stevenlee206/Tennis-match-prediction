import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, precision_recall_curve, auc, 
    matthews_corrcoef, brier_score_loss
)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_prob)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)
    
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "MCC": mcc,
        "Brier-Score": brier
    }

def save_final_confusion_matrix(y_true, y_pred, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    display_labels = ['Player 2 Wins', 'Player 1 Wins']
    
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=display_labels,
        cmap=plt.cm.Blues,
        values_format='d'
    )
    model_title = "Logistic Regression" if "Logistic_Regression" in results_dir else "Naive Bayes"
    plt.title(f"{model_title} - Final Test Confusion Matrix")
    plt.xlabel("Predicted Match Winner")
    plt.ylabel("Actual Match Winner")

    output_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def time_series_cv(model,X, y, results_dir="results/Logistic_Regression"):
    # Do chronological walk-forward validation with fold 1 to 5 and then do a final test evaluation and then save it to a csv.
    os.makedirs(results_dir, exist_ok=True)
    n_samples = len(X)
    
    # Define chronological block sizes (chunks of 10%)
    block_size = int(n_samples * 0.10)
    
    fold_results = []
    
    # Volumetric chronological cv with 5 folds (40% train -> 10% val, then 50% train -> 10% val, then 60% train -> 10% val, then 70% train -> 10% val, then 80% train -> 10% val)
    for fold in range(1, 6):
        train_end = (fold + 3) * block_size
        val_end = train_end + block_size
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        
        # Train and validate
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        # Evaluate
        metrics = calculate_metrics(y_val, preds, probs)
        metrics["Phase"] = f"Fold {fold} (Train {int((fold+3)*10)}% -> Val 10%)"
        fold_results.append(metrics)
        print(f"Fold {fold} Complete.")

    # Final test evaluation (90% Train -> 10% Test)
    final_train_end = 9 * block_size
    X_train_final, y_train_final = X.iloc[:final_train_end], y.iloc[:final_train_end]
    X_test, y_test = X.iloc[final_train_end:], y.iloc[final_train_end:]
    
    model.fit(X_train_final, y_train_final)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    
    save_final_confusion_matrix(y_test, test_preds, results_dir)
    
    test_metrics = calculate_metrics(y_test, test_preds, test_probs)
    test_metrics["Phase"] = "FINAL TEST (Train 90% -> Test Final 10%)"
    fold_results.append(test_metrics)
    print("Final Test Phase Complete.")
    
    # Save results to CSV
    df_report = pd.DataFrame(fold_results)
    # Reorder columns
    columns = ["Phase"] + [col for col in df_report.columns if col != "Phase"]
    df_report = df_report[columns]
    
    output_path = os.path.join(results_dir, "chronological_evaluation_report.csv")
    df_report.to_csv(output_path, index=False)
    
    print(f"\nStrategic report saved to: {output_path}")
    return df_report