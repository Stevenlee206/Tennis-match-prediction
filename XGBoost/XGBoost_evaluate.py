import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef
)
from xgboost import XGBClassifier

class Evaluator:
    def num_estimator_validation(self, X_train, y_train, X_val, y_val, n_range):
        train_scores = []
        val_scores = []
        for n in n_range:
            clf = XGBClassifier(
                n_estimators=n,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
                eval_metric='logloss',
                enable_categorical=True,
                tree_method='hist'
            )
            clf.fit(X_train, y_train)
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            val_scores.append(accuracy_score(y_val, clf.predict(X_val)))
            print(f"Estimators: {n:3} | Train Acc: {train_scores[-1]:.4f} | Val Acc: {val_scores[-1]:.4f}")
        plt.figure(figsize=(10, 5))
        plt.plot(n_range, train_scores, label='Train Accuracy', marker='o')
        plt.plot(n_range, val_scores, label='Validation Accuracy', marker='s')
        plt.title('XGBoost Tuning: Estimators vs Accuracy')
        plt.xlabel('n_estimators')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
        best_n = n_range[np.argmax(val_scores)]
        return best_n

    def evaluate_model(self,y_true,y_pred,y_prob):
        acc=accuracy_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        loss = log_loss(y_true, y_prob)
        metrics = {
            'Precision': precision_score,
            'Recall': recall_score,
            'F1-Score': f1_score
        }
        results = []
        for name, func in metrics.items():
            micro = func(y_true, y_pred, average='micro')
            macro = func(y_true, y_pred, average='macro')
            results.append({'Metric': name, 'Micro': micro, 'Macro': macro})
        df_res = pd.DataFrame(results)
        print("\n" + "=" * 30)
        print("Test Result")
        print("=" * 30)
        print(f"Accuracy: {acc:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Log Loss: {loss:.4f}")
        print("-" * 30)
        print(df_res.to_string(index=False))
        auc = roc_auc_score(y_true, y_prob)
        print(f"\nROC-AUC Score: {auc:.4f}")
        return df_res





