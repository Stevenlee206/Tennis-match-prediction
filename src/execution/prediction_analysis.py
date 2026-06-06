import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


# Testing the hypothesis of class bias
# Does the model have a biased tendency, always predicting player 1 (the favorite) to win?
def plot_prediction_summary(y_true: np.ndarray | pd.Series,
                            y_pred: np.ndarray | pd.Series,
                            out_dir: str | Path,
                            model_name: str
                            ) -> None:
    """
    Draw the Confusion Matrix and the Win/Loss Rate Distribution to see if the model is biased.
    """
    out_dir = Path(out_dir)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predict Lose (0)', 'Predict Win (1)'],
                yticklabels=['Actually Lost (0)', 'Actually Wins (1)'])
    axes[0].set_title(f'Confusion Matrix ({model_name.upper()})', fontweight='bold', pad=15)

    # Predicted vs. Actual Distribution
    actual_counts = pd.Series(y_true_arr).value_counts(normalize=True).sort_index() * 100
    pred_counts = pd.Series(y_pred_arr).value_counts(normalize=True).sort_index() * 100

    bar_width = 0.35
    x = np.arange(2)

    axes[1].bar(x - bar_width / 2, actual_counts, bar_width, label='Actual', color='gray', alpha=0.7)
    axes[1].bar(x + bar_width / 2, pred_counts, bar_width, label='Predicted', color='teal')

    axes[1].set_title('Class Distribution', fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Player 1 Lose (0)', 'Player 1 Win (1)'])
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend()

    plt.tight_layout()
    save_path = out_dir / f"{model_name}_prediction_summary.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] The Predicted Distribution chart has been saved at: {save_path.name}")


# Confidence hypothesis testing
# When the model is wrong, its probability is usually hovering around 50-55%?
def plot_confidence_analysis(   y_true: np.ndarray | pd.Series,
                                y_prob: np.ndarray,
                                out_dir: str | Path,
                                model_name: str
                            ) -> None:
    """
    Draw a probability distribution (Confidence) separating the Correct and Incorrect guesses.
    """
    out_dir = Path(out_dir)
    y_true_arr = np.array(y_true)

    # Calculate your confidence level (Confidence: Distance from 0.5)
    # Example: Prob = 0.9 -> Conf = 90%. Prob = 0.1 -> Conf = 90% (confident in predicting a loss)
    confidence = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    y_pred_arr = (y_prob >= 0.5).astype(int)

    # True/False Classification
    is_correct = (y_pred_arr == y_true_arr)
    df_conf = pd.DataFrame({
        'Confidence': confidence,
        'Result': ['True prediction' if c else 'False prediction' for c in is_correct]
    })

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_conf, x='Confidence', hue='Result', fill=True,
                palette={'True prediction': 'teal', 'False prediction': 'crimson'},
                alpha=0.4, linewidth=2)

    plt.title(f'Confidence Level Distribution - {model_name.upper()}', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Probability (From 0.5 [Undecided] to 1.0 [Absolutely])')
    plt.ylabel('Density')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.xlim(0.5, 1.0)

    save_path = out_dir / f"{model_name}_confidence_analysis.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] The Confidence Analysis chart has been saved at: {save_path.name}")


# FUNCTION 3: HYPOTHESIS TESTING BASED ON SPECIFIC FEATURES (FEATURE ERROR HYPOTHESIS)
# Does the model frequently make incorrect predictions when the Elo ratings of two individuals are too close?
def plot_error_by_feature(
        X_raw: pd.DataFrame,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        feature_name: str,
        out_dir: str | Path,
        model_name: str,
        verbose: bool = True
        ) -> None:
    """
    Visualize the differences of a specific Feature (e.g., elo_diff) between the Correct Guess and Incorrect Guess groups.
    """
    if feature_name not in X_raw.columns:
        print(f"Skip the analysis {feature_name}: This column was not found in the data.")
        return

    out_dir = Path(out_dir)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    is_correct = (y_pred_arr == y_true_arr)

    df_feature = pd.DataFrame({
        feature_name: X_raw[feature_name].values,
        'Prediction_Status': ['True' if c else 'False' for c in is_correct]
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot chart (see outliers)
    sns.boxplot(data=df_feature, x='Prediction_Status', y=feature_name,
                palette={'True': 'lightseagreen', 'False': 'lightcoral'}, ax=axes[0])
    axes[0].set_title(f'Distribution of {feature_name} based on prediction', fontweight='bold')

    # KDE Chart (Density to see patterns or breaks at which values)
    sns.kdeplot(data=df_feature, x=feature_name, hue='Prediction_Status', fill=True,
                palette={'True': 'teal', 'False': 'crimson'}, ax=axes[1], alpha=0.3)
    axes[1].set_title(f'Density {feature_name} where the model makes a wrong prediction.', fontweight='bold')

    plt.tight_layout()
    save_path = out_dir / f"{model_name}_error_analysis_{feature_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    if verbose:
        print(f"[*] The error analysis chart based on {feature_name} has been saved at : {save_path.name}")


# LOOP TO ANALYZE ALL FEATURES

def plot_all_features_errors(
        X_raw: pd.DataFrame,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        out_dir: str | Path,
        model_name: str
) -> None:
    out_dir = Path(out_dir)

    # Important features
    keep_list = [
        'elo_diff', 'elo_hard_diff', 'fatigue_diff', 'h2h_advantage_diff'
    ]

    # Create destination folders for important charts.
    final_dir = out_dir.parent / "final_research_figures"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"-> Start the filtering process: Draw only {len(keep_list)} important features...")

    for feature_name in keep_list:
        if feature_name in X_raw.columns:
            if pd.api.types.is_numeric_dtype(X_raw[feature_name]):
                # Draw directly into the Final folder.
                plot_error_by_feature(
                    X_raw=X_raw,
                    y_true=y_true,
                    y_pred=y_pred,
                    feature_name=feature_name,
                    out_dir=final_dir,
                    model_name=model_name,
                    verbose=True
                )
        else:
            print(f" Warning: Features '{feature_name}' does not exist in the data.")

    print(f"[*] Finished! Only {len(keep_list)} required chart has been exported at: {final_dir.name}/")