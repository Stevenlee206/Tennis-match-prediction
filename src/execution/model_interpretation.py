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
    Calculate and return a Feature Importance array based on the specifics of each model type.
    """
    print("\n" + "=" * 50)
    print(f"[*] Calculating Feature Importances")
    print("=" * 50)

    try:
        # Tree-like models (XGBoost, Random Forest, Decision Tree, DeepForest)
        if hasattr(model, 'feature_importances_'):
            print("-> Use built-in feature_importances_ (Tree-based model)")
            return model.feature_importances_

        # Linear Models (SVM Linear)
        elif hasattr(model, 'coef_'):
            print("-> Use the built-in Coef_ (Linear model)")
            return np.abs(model.coef_[0])

        # Black Box Models (SVM RBF, TabNet) -> Use Permutation Importance
        else:
            print("-> Use Permutation Importance (Black-box model). This process might take a little while....")
            perm_importance = permutation_importance(model, X_eval, y_eval, n_repeats=5, random_state=42, n_jobs=-1)
            return perm_importance.importances_mean

    except Exception as e:
        print(f"Feature Importance cannot be calculated : {str(e)}")
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
    Get the calculated importance array and plot the charts (Bar Chart & PDP).
    """
    if importances is None:
        print("Skip the charting process because the Feature Importance data is empty.")
        return

    out_dir = Path(out_dir)

    # Prepare and organize the DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    # Get the top 15 features to draw.
    top_n = min(15, len(feature_names))
    top_features_df = feat_imp_df.tail(top_n)

    # Draw feature importance chart
    plt.figure(figsize=(10, 8))
    plt.barh(top_features_df['Feature'], top_features_df['Importance'], color='teal', edgecolor='black')
    plt.title(f'Top {top_n} Feature Importance ({model_name.upper()})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fi_path = out_dir / f"{model_name}_feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"[*] Feature Importance Plot has been saved at : {fi_path.name}")

    # PARTIAL DEPENDENCE PLOT (TOP 15 FEATURES)
    top_15_features = top_features_df['Feature'].tail(15).tolist()
    top_15_indices = [feature_names.index(feat) for feat in top_15_features]

    print(f"-> Currently drawing the Partial Dependence Plot for the Top 15: {top_15_features}...")

    try:
        fig, ax = plt.subplots(figsize=(20,18))
        display = PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X_eval,
            features=top_15_indices,
            feature_names=feature_names,
            kind='average',
            ax=ax,
            grid_resolution=30
        )
        fig.suptitle(f'Partial Dependence Plots ({model_name.upper()})', fontsize=20, fontweight='bold', y=0.96)
        for i, axis in enumerate(display.axes_.ravel()):
            if axis is not None and i < len(top_15_features):
                axis.set_title(top_15_features[i], fontsize=13, fontweight='bold', color='teal',pad=10)
                axis.set_xlabel('')
        plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.6, wspace=0.3)
        pdp_path = out_dir / f"{model_name}_partial_dependence.png"
        plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Partial Dependence Plot has been saved at: {pdp_path.name}")

    except Exception as e:
        print(f" Unable to draw PDP for the model {model_name}: {str(e)}")
        print(" Some models are not readily compatible with Sklearn's PDP function.")