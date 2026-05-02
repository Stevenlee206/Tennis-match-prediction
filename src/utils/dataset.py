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


def prepare_loaders(df, target_col='target', batch_size=64):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Chia Train / (Val + Test)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Chia Val / Test (50/50 của 20% còn lại)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    train_loader = DataLoader(TennisDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TennisDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TennisDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, X.shape[1]


print(torch.cuda.is_available())
print(torch.version.cuda)