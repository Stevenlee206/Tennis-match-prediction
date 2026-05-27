import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# 1. Correlation Matrix
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

# ==========================================
# 2. Principal Component Analysis (PCA)
# ==========================================
def plot_pca_analysis(X_scaled, y):
    print("\nRunning PCA Analysis...")
    
    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # ------------------------------------------
    # Plot 1: Cumulative Explained Variance
    # ------------------------------------------
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle='-', alpha=0.5, label='90% Variance Threshold')
    plt.title('PCA: Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pca_explained_variance.png", dpi=300)
    plt.close()
    
    # Find how many components are needed for 90% variance
    n_components_90 = np.argmax(explained_variance >= 0.90) + 1
    print(f"Components needed to explain 90% of variance: {n_components_90} out of {X_scaled.shape[1]}")

    # ------------------------------------------
    # Plot 2: 2D PCA Scatter Plot (Class Separation)
    # ------------------------------------------
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
    pca_df['Target'] = y.values
    
    plt.figure(figsize=(10, 8))
    # Using a sample to prevent massive over-plotting if dataset is huge
    sample_df = pca_df.sample(n=min(10000, len(pca_df)), random_state=42)
    
    sns.scatterplot(
        x='PC1', y='PC2', hue='Target', data=sample_df, 
        palette=['#1f77b4', '#d62728'], alpha=0.5, s=15
    )
    
    var_1, var_2 = pca_2d.explained_variance_ratio_ * 100
    plt.title('2D PCA Class Separation (Target 1 vs 0)')
    plt.xlabel(f'Principal Component 1 ({var_1:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({var_2:.1f}% Variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pca_2d_scatter.png", dpi=300)
    plt.close()
    print(f"PCA Plots saved to {REPORTS_DIR}")


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
    
    # PCA Analysis requires strictly scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    plot_pca_analysis(X_scaled, y)
    print("\nEDA Pipeline Complete.")