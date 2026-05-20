import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TennisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_loaders(df, target_col='target', batch_size=64, n_splits=5, val_ratio=0.1):
    cols_to_drop = [target_col]
    if 'is_augmented' in df.columns:
        cols_to_drop.append('is_augmented')
    X = df.drop(columns=cols_to_drop).values
    y = df[target_col].values
    N = len(df)

    # Calculate test size and validation size to be approximately equal
    val_size = int(N * val_ratio)
    test_size = val_size
    train_size = N - (n_splits + 1) * val_size

    # Split the quarantine test set from the end
    X_test_raw = X[-test_size:]
    y_test = y[-test_size:]

    # Scale the full train+val set and the test set using a scaler fit on full train+val
    X_train_val_raw = X[:-test_size]
    y_train_val = y[:-test_size]
    
    scaler_full = StandardScaler()
    X_train_val_scaled = scaler_full.fit_transform(X_train_val_raw)
    X_test = scaler_full.transform(X_test_raw)

    test_loader = DataLoader(TennisDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    train_val_loader = DataLoader(TennisDataset(X_train_val_scaled, y_train_val), batch_size=batch_size, shuffle=True)

    folds = []
    for k in range(n_splits):
        # Calculate sliding window indices
        train_start = k * val_size
        train_end = train_size + k * val_size
        val_start = train_end
        val_end = val_start + val_size

        X_train_raw = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_val_raw = X[val_start:val_end]
        y_val = y[val_start:val_end]

        scaler_fold = StandardScaler()
        X_train = scaler_fold.fit_transform(X_train_raw)
        X_val = scaler_fold.transform(X_val_raw)

        train_loader = DataLoader(TennisDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TennisDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        folds.append((train_loader, val_loader))

    return folds, test_loader, X.shape[1], train_val_loader


print(torch.cuda.is_available())
print(torch.version.cuda)