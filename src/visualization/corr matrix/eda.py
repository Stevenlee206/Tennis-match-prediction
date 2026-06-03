import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Directory Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}. Run preprocessing first.")
    print(f"Loading data from {DATA_PATH}...")
    data = pd.read_csv(DATA_PATH)
    return data

# Correlation Matrix
def plot_correlation_matrix(df, target_col='target'):
    print("Generating Correlation Matrix...")
    
    # Compute the correlation matrix
    corr = df.corr()
    
    # Generate a mask for the upper triangle to make it readable
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(14, 12))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5},
        annot=False # Set to True if you want the actual numbers, but it gets messy with many features
    )
    
    plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_matrix.png", dpi=300)
    plt.close()
    
    # Print the features most highly correlated with the target
    target_corr = corr[target_col].sort_values(ascending=False)
    print("\nTop 5 Positively Correlated Features with Target:")
    print(target_corr.head(6)[1:]) # Skip the target itself
    print("\nTop 5 Negatively Correlated Features with Target:")
    print(target_corr.tail(5))


if __name__ == "__main__":
    print(" Exploratory Data Analysis (EDA)")
    # load_data_func
    df = load_data()
    # Separate features and target
    if 'target' not in df.columns:
        raise ValueError("Target column not found in dataset.")
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Plot Correlation Matrix
    plot_correlation_matrix(df)
