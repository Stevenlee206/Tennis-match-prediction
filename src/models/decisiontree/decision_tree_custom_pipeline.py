import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer


def run_decision_tree_unlimited_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "decisiontree_model.joblib"
    scaler_path = output_dir / "decisiontree_scaler.joblib"

    force_retrain = kwargs.get('force_retrain', False)
    if not force_retrain and model_path.exists() and scaler_path.exists():
        print(f"\n[!] Find model at {output_dir.name}. Skip training!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Start training the Decision Tree (UNLIMITED DEPTH / NO PRUNING) ---")
    print("[!] WARNING: This model will grow leaves freely until it reaches 100% overfit. Use as a reference baseline..")

    X_final = pd.concat([X_train, X_val]) if X_val is not None else X_train
    y_final = pd.concat([y_train, y_val]) if X_val is not None else y_train

    final_clf = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    final_clf.fit(X_final, y_final)

    # Print the actual depth and number of leaves to visualize the tree's complexity
    print(f"[*] The tree has memorized the data with Maximum Depth: {final_clf.get_depth()}")
    print(f"[*] Total number of leaves created: {final_clf.get_n_leaves()}")

    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    config = {
        "model_type": "DecisionTree_Unlimited_Baseline",
        "best_params": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "tree_stats": {
            "depth": int(final_clf.get_depth()),
            "n_leaves": int(final_clf.get_n_leaves())
        }
    }

    with open(output_dir / "decisiontree_config.json", "w") as f:
        json.dump(config, f, indent=4)
    return final_clf, dummy_scaler