import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.models.model import TennisNet
from src.preprocessing.preprocessing import Preprocessing
import torch
import numpy as np
import torch.nn.functional as F
from utils.dataset import prepare_loaders
from utils.visualizer import plot_learning_curves, plot_hyperparameter_heatmap


def train():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            # Blackwell GPU compatibility check (sm_100+)
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 10:
                raise RuntimeError("Incompatible CUDA capability")
            # Run a dummy GEMM to verify cuBLAS
            x = torch.zeros(64, 64).cuda()
            l = torch.nn.Linear(64, 64).cuda()
            l(x)
            device = torch.device("cuda")
        except Exception:
            print("CUDA is available but not compatible with this PyTorch build. Falling back to CPU.")
            device = torch.device("cpu")

    print(f"Bắt đầu huấn luyện trên: {device}")

    # 1. Load Data
    preprocessor = Preprocessing()
    df = preprocessor.run()
    folds, test_loader, input_dim, train_val_loader = prepare_loaders(df)
    
    learning_rates = [4.6e-3]
    weight_decays = [2.2e-5]

    best_acc = 0
    best_lr = None
    best_wd = None
    best_history = None
    results_matrix = np.zeros((len(learning_rates), len(weight_decays)))

    for i, lr in enumerate(learning_rates):
        for j, wd in enumerate(weight_decays):
            print(f"Testing LR: {lr:.1e}, WD: {wd:.1e}")

            epochs = 50
            fold_losses = np.zeros((len(folds), epochs))
            fold_accs = np.zeros((len(folds), epochs))

            for fold_idx, (train_loader, val_loader) in enumerate(folds):
                # QUAN TRỌNG: Khởi tạo model mới cho mỗi fold
                model = TennisNet(input_dim).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        logits = model(X_batch)
                        loss = F.cross_entropy(logits, y_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    # Validation
                    model.eval()
                    correct = 0
                    with torch.no_grad():
                        for X_val, y_val in val_loader:
                            X_val, y_val = X_val.to(device), y_val.to(device)
                            preds = model(X_val).argmax(dim=1)
                            correct += (preds == y_val).sum().item()

                    acc = correct / len(val_loader.dataset)
                    avg_loss = epoch_loss / len(train_loader)
                    fold_losses[fold_idx, epoch] = avg_loss
                    fold_accs[fold_idx, epoch] = acc
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        print(f"    Fold {fold_idx + 1} | Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

            # Tính trung bình các fold cho mỗi epoch
            avg_train_loss = fold_losses.mean(axis=0).tolist()
            avg_val_acc = fold_accs.mean(axis=0).tolist()

            mean_final_acc = avg_val_acc[-1]
            results_matrix[i, j] = mean_final_acc
            print(f"  Mean CV Accuracy: {mean_final_acc:.4f}")

            # Lưu kết quả tốt nhất
            if mean_final_acc > best_acc:
                best_acc = mean_final_acc
                best_lr = lr
                best_wd = wd
                best_history = {'train_loss': avg_train_loss, 'val_acc': avg_val_acc}

    plot_hyperparameter_heatmap(results_matrix, learning_rates, weight_decays)
    plot_learning_curves(best_history)

    print(f"\nBest CV Accuracy: {best_acc:.4f} (LR: {best_lr:.1e}, WD: {best_wd:.1e})")
    
    # 2. Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu Train + Val
    print(f"\nHuấn luyện mô hình cuối cùng với LR: {best_lr:.1e}, WD: {best_wd:.1e} trên toàn bộ dữ liệu Train + Val...")
    final_model = TennisNet(input_dim).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
    epochs = 20
    
    for epoch in range(epochs):
        final_model.train()
        for X_batch, y_batch in train_val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            final_optimizer.zero_grad()
            logits = final_model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            final_optimizer.step()
            
    # 3. Đánh giá trên quarantine test set
    evaluate_test(final_model, test_loader, device)


def evaluate_test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            preds = model(X_test).argmax(dim=1)
            correct += (preds == y_test).sum().item()
    print(f"Final Test Accuracy: {correct / len(test_loader.dataset):.4f}")


if __name__ == "__main__":
    train()