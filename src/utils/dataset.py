import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class TennisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_loaders(df, target_col='target', batch_size=64, n_splits=5, val_ratio=0.1, scaler=None):
    cols_to_drop = [target_col]
    if 'is_augmented' in df.columns:
        cols_to_drop.append('is_augmented')
        
    # Separate original and augmented rows.
    # Since they are perfectly interleaved, original matches are at even indices, duplicates at odd indices.
    df_original = df[df['is_augmented'] == 0].reset_index(drop=True)
    df_augmented = df[df['is_augmented'] == 1].reset_index(drop=True)
    
    X_orig = df_original.drop(columns=cols_to_drop).values
    y_orig = df_original[target_col].values
    
    X_aug = df_augmented.drop(columns=cols_to_drop).values
    y_aug = df_augmented[target_col].values
    
    N_orig = len(df_original)

    # Calculate test size and validation size based on original matches size
    val_size = int(N_orig * val_ratio)
    test_size = val_size
    train_size = N_orig - (n_splits + 1) * val_size

    # Split the quarantine test set from the end (original matches only)
    X_test_raw = X_orig[-test_size:]
    y_test = y_orig[-test_size:]

    # Scale the full train+val set and the test set using a scaler fit on full train+val (including augmented rows)
    X_train_val_orig = X_orig[:-test_size]
    y_train_val_orig = y_orig[:-test_size]
    
    X_train_val_aug = X_aug[:-test_size]
    y_train_val_aug = y_aug[:-test_size]
    
    # Combine original and augmented for final train_val training
    X_train_val_combined = np.empty((2 * len(X_train_val_orig), X_orig.shape[1]), dtype=X_orig.dtype)
    X_train_val_combined[0::2] = X_train_val_orig
    X_train_val_combined[1::2] = X_train_val_aug
    
    y_train_val_combined = np.empty(2 * len(y_train_val_orig), dtype=y_orig.dtype)
    y_train_val_combined[0::2] = y_train_val_orig
    y_train_val_combined[1::2] = y_train_val_aug
    
    if scaler is not None:
        scaler_full = scaler
        X_train_val_scaled = scaler_full.transform(X_train_val_combined)
        X_test = scaler_full.transform(X_test_raw)
    else:
        scaler_full = StandardScaler()
        X_train_val_scaled = scaler_full.fit_transform(X_train_val_combined)
        X_test = scaler_full.transform(X_test_raw)

    test_loader = DataLoader(TennisDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    train_val_loader = DataLoader(TennisDataset(X_train_val_scaled, y_train_val_combined), batch_size=batch_size, shuffle=True)

    folds = []
    for k in range(n_splits):
        # Calculate sliding window indices on original matches
        train_start = k * val_size
        train_end = train_size + k * val_size
        val_start = train_end
        val_end = val_start + val_size

        X_train_orig = X_orig[train_start:train_end]
        y_train_orig = y_orig[train_start:train_end]
        
        X_train_aug = X_aug[train_start:train_end]
        y_train_aug = y_aug[train_start:train_end]
        
        # Combine original and augmented training fold
        X_train_combined = np.empty((2 * len(X_train_orig), X_orig.shape[1]), dtype=X_orig.dtype)
        X_train_combined[0::2] = X_train_orig
        X_train_combined[1::2] = X_train_aug
        
        y_train_combined = np.empty(2 * len(y_train_orig), dtype=y_orig.dtype)
        y_train_combined[0::2] = y_train_orig
        y_train_combined[1::2] = y_train_aug

        # Validation set (original matches only, no leakage)
        X_val_raw = X_orig[val_start:val_end]
        y_val = y_orig[val_start:val_end]

        if scaler is not None:
            X_train = scaler.transform(X_train_combined)
            X_val = scaler.transform(X_val_raw)
        else:
            scaler_fold = StandardScaler()
            X_train = scaler_fold.fit_transform(X_train_combined)
            X_val = scaler_fold.transform(X_val_raw)

        train_loader = DataLoader(TennisDataset(X_train, y_train_combined), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TennisDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        folds.append((train_loader, val_loader))

    return folds, test_loader, X_orig.shape[1], train_val_loader, scaler_full


print(torch.cuda.is_available())
print(torch.version.cuda)