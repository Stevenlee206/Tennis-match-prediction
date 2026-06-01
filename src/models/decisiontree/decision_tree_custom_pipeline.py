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
        print(f"\n[!] Tìm thấy model tại {output_dir.name}. Bỏ qua huấn luyện!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Bắt đầu huấn luyện Decision Tree (UNLIMITED DEPTH / NO PRUNING) ---")
    print("[!] CẢNH BÁO: Mô hình này sẽ mọc lá tự do đến khi Overfit 100%. Dùng làm Baseline tham chiếu.")

    X_final = pd.concat([X_train, X_val]) if X_val is not None else X_train
    y_final = pd.concat([y_train, y_val]) if X_val is not None else y_train

    # Bí quyết nằm ở đây: max_depth=None, min_samples_split=2, min_samples_leaf=1
    final_clf = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    final_clf.fit(X_final, y_final)

    # In ra số lượng node và độ sâu thực tế của cây để bạn hình dung độ phức tạp
    print(f"[*] Cây đã học thuộc dữ liệu với Độ sâu cực đại (Depth): {final_clf.get_depth()}")
    print(f"[*] Tổng số lá (Leaves) được tạo ra: {final_clf.get_n_leaves()}")

    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    config = {
        "model_type": "DecisionTree_Unlimited_Baseline",
        "best_params": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "tree_stats": {
            "depth": int(final_clf.get_depth()),  # <-- Đã sửa
            "n_leaves": int(final_clf.get_n_leaves())  # <-- Đã sửa
        }
    }

    with open(output_dir / "decisiontree_config.json", "w") as f:
        json.dump(config, f, indent=4)
    return final_clf, dummy_scaler