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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Bắt đầu huấn luyện trên: {device}")

    # 1. Load Data
    preprocessor = Preprocessing()
    df = preprocessor.run()
    train_loader, val_loader, test_loader, input_dim = prepare_loaders(df)
    # learning_rates = [2.2e-3]
    # weight_decays = [7.7e-5]
    learning_rates = np.logspace(-4, 1, num=2)

    weight_decays = np.logspace(-5, -1, num=2)

    best_acc = 0
    best_model_state = None
    best_history = None
    results_matrix = np.zeros((len(learning_rates), len(weight_decays)))

    for i, lr in enumerate(learning_rates):
        for j, wd in enumerate(weight_decays):
            print(f"Testing LR: {lr:.1e}, WD: {wd:.1e}")

            # QUAN TRỌNG: Khởi tạo model mới cho mỗi cặp thông số
            model = TennisNet(input_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            history = {'train_loss': [], 'val_acc': []}
            epochs = 20

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
                history['train_loss'].append(epoch_loss / len(train_loader))
                history['val_acc'].append(acc)

            # Lưu kết quả tốt nhất
            results_matrix[i, j] = history['val_acc'][-1]
            if history['val_acc'][-1] > best_acc:
                best_acc = history['val_acc'][-1]
                best_model_state = model.state_dict()
                best_history = history
    plot_hyperparameter_heatmap(results_matrix, learning_rates, weight_decays)
    plot_learning_curves(best_history)

    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    evaluate_test(best_model_state, test_loader, input_dim, device)


def evaluate_test(model_state, test_loader, input_dim, device):
    model = TennisNet(input_dim).to(device)
    model.load_state_dict(model_state)
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