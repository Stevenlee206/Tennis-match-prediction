import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from typing import Any


def calculate_feature_importances(model: Any,
                                  X_eval: np.ndarray | pd.DataFrame,
                                  y_eval: np.ndarray | pd.Series
                                  ) -> np.ndarray | None:
    """
    Tính toán và trả về mảng Feature Importance dựa trên đặc thù của từng loại mô hình.
    """
    print("\n" + "=" * 50)
    print(f"[*] ĐANG TÍNH TOÁN ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE)")
    print("=" * 50)

    try:
        # A. Các mô hình dạng Cây (XGBoost, Random Forest, Decision Tree, DeepForest)
        if hasattr(model, 'feature_importances_'):
            print("-> Sử dụng built-in feature_importances_ (Tree-based model)")
            return model.feature_importances_

        # B. Các mô hình Tuyến tính (SVM Linear)
        elif hasattr(model, 'coef_'):
            print("-> Sử dụng built-in coef_ (Linear model)")
            return np.abs(model.coef_[0])

        # C. Các mô hình Hộp đen (SVM RBF, TabNet) -> Dùng Permutation Importance
        else:
            print("-> Sử dụng Permutation Importance (Black-box model). Quá trình này có thể hơi lâu...")
            perm_importance = permutation_importance(model, X_eval, y_eval, n_repeats=5, random_state=42, n_jobs=-1)
            return perm_importance.importances_mean

    except Exception as e:
        print(f"⚠️ Không thể tính Feature Importance: {str(e)}")
        return None


def plot_interpretability(
        model: Any,
        X_eval: np.ndarray | pd.DataFrame,
        importances: np.ndarray | None,
        feature_names: list[str],
        out_dir: str | Path,
        model_name: str
) -> None:
    """
    Nhận mảng importances đã tính toán và vẽ các biểu đồ (Bar Chart & PDP).
    """
    if importances is None:
        print("⚠️ Bỏ qua quá trình vẽ biểu đồ do dữ liệu Feature Importance bị trống.")
        return

    out_dir = Path(out_dir)

    # Chuẩn bị DataFrame và sắp xếp
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    # Lấy top 15 features để vẽ
    top_n = min(15, len(feature_names))
    top_features_df = feat_imp_df.tail(top_n)

    # ==================================================
    # 1. VẼ BIỂU ĐỒ FEATURE IMPORTANCE
    # ==================================================
    plt.figure(figsize=(10, 8))
    plt.barh(top_features_df['Feature'], top_features_df['Importance'], color='teal', edgecolor='black')
    plt.title(f'Top {top_n} Feature Importance ({model_name.upper()})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fi_path = out_dir / f"{model_name}_feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu Feature Importance Plot tại: {fi_path.name}")

    # ==================================================
    # 2. VẼ PARTIAL DEPENDENCE PLOT (TOP 4 FEATURES)
    # ==================================================
    top_4_features = top_features_df['Feature'].tail(4).tolist()
    top_4_indices = [feature_names.index(feat) for feat in top_4_features]

    print(f"-> Đang vẽ Partial Dependence Plot cho Top 4: {top_4_features}...")

    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        display = PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X_eval,
            features=top_4_indices,
            feature_names=feature_names,
            kind='average',
            ax=ax,
            grid_resolution=30
        )
        fig.suptitle(f'Partial Dependence Plots ({model_name.upper()})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        pdp_path = out_dir / f"{model_name}_partial_dependence.png"
        plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Đã lưu Partial Dependence Plot tại: {pdp_path.name}")

    except Exception as e:
        print(f"⚠️ Không thể vẽ PDP cho mô hình {model_name}: {str(e)}")
        print("Gợi ý: Một số mô hình không tương thích sẵn với hàm PDP của Sklearn.")