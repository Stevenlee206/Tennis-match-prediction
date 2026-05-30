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
import joblib

# Resolve project root and models directory dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_dir = os.path.join(project_root, 'models')


def train(model_path=None, epochs=5, skip_cv=False):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Bắt đầu huấn luyện trên: {device}")

    # 1. Load Data
    n_splits = 5
    val_ratio = 0.1
    train_ratio = 1.0 - (n_splits + 1) * val_ratio

    preprocessor = Preprocessing()
    df = preprocessor.run(train_ratio=train_ratio)
    
    # Load features list and scaler if loading a model
    loaded_scaler = None
    feature_cols = None
    
    if model_path is not None:
        if os.path.exists(model_path):
            model_dir = os.path.dirname(model_path)
            features_json_path = os.path.join(model_dir, 'features.json')
            scaler_joblib_path = os.path.join(model_dir, 'scaler.joblib')
            
            # Load features if exists
            if os.path.exists(features_json_path):
                print(f"Đang tải danh sách đặc trưng từ: {features_json_path}")
                import json
                with open(features_json_path, 'r', encoding='utf-8') as f:
                    feature_cols = json.load(f)
                    
                # Align df features with feature_cols
                df_copy = df.copy()
                for col in feature_cols:
                    if col not in df_copy.columns:
                        df_copy[col] = 0.0
                
                # Keep only the features in feature_cols plus target and is_augmented
                cols_to_keep = feature_cols.copy()
                if 'target' in df.columns:
                    cols_to_keep.append('target')
                if 'is_augmented' in df.columns:
                    cols_to_keep.append('is_augmented')
                df = df_copy[cols_to_keep]
            else:
                print("Cảnh báo: Không tìm thấy features.json của model cũ, sẽ sử dụng đặc trưng của lần chạy này.")
                
            # Load scaler if exists
            if os.path.exists(scaler_joblib_path):
                print(f"Đang tải bộ chuẩn hóa từ: {scaler_joblib_path}")
                loaded_scaler = joblib.load(scaler_joblib_path)
            else:
                print("Cảnh báo: Không tìm thấy scaler.joblib của model cũ, sẽ fit scaler mới.")
        else:
            print(f"Đường dẫn file model được truyền vào ({model_path}) không tồn tại. Sẽ huấn luyện từ đầu.")

    folds, test_loader, input_dim, train_val_loader, scaler_full = prepare_loaders(
        df, n_splits=n_splits, val_ratio=val_ratio, scaler=loaded_scaler
    )
    
    # Save the feature columns to models/features.json
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ['target', 'is_augmented']]
    
    import json
    os.makedirs(models_dir, exist_ok=True)
    features_json_path = os.path.join(models_dir, 'features.json')
    with open(features_json_path, 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu danh sách {len(feature_cols)} đặc trưng tại {features_json_path}")
    
    best_lr = 4.6e-3
    best_wd = 2.2e-5

    if not skip_cv:
        # Tự động tìm kiếm trên lưới 10x10
        learning_rates = np.logspace(-4, -2, 2).tolist()
        weight_decays = np.logspace(-6, -3, 2).tolist()

        best_acc = 0
        best_history = None
        results_matrix = np.zeros((len(learning_rates), len(weight_decays)))

        for i, lr in enumerate(learning_rates):
            for j, wd in enumerate(weight_decays):
                print(f"Testing LR: {lr:.1e}, WD: {wd:.1e}")

                fold_losses = np.zeros((len(folds), epochs))
                fold_accs = np.zeros((len(folds), epochs))

                for fold_idx, (train_loader, val_loader) in enumerate(folds):
                    # QUAN TRỌNG: Khởi tạo model mới cho mỗi fold
                    model = TennisNet(input_dim).to(device)
                    if model_path is not None and os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
    else:
        print("\nBỏ qua Cross Validation theo cấu hình.")
    
    # 2. Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu Train + Val
    print(f"\nHuấn luyện mô hình cuối cùng với LR: {best_lr:.1e}, WD: {best_wd:.1e} trên toàn bộ dữ liệu Train + Val...")
    final_model = TennisNet(input_dim).to(device)
    if model_path is not None and os.path.exists(model_path):
        print(f"Tải trọng số mô hình từ {model_path} vào final model")
        final_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
    
    os.makedirs(models_dir, exist_ok=True)
    if loaded_scaler is not None:
        scaler_full = loaded_scaler
    scaler_save_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler_full, scaler_save_path)
    print(f"Đã lưu bộ chuẩn hóa tại {scaler_save_path}")

    best_loss = float('inf')
    
    for epoch in range(epochs):
        final_model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            final_optimizer.zero_grad()
            logits = final_model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            final_optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_val_loader)
        
        # Save best model based on training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(final_model.state_dict(), os.path.join(models_dir, 'final_model_best.pth'))
            
        # Periodical save every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(models_dir, f'final_model_epoch_{epoch + 1}.pth')
            torch.save(final_model.state_dict(), checkpoint_path)
            print(f"    Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Saved checkpoint.")
        elif (epoch + 1) == 1:
            print(f"    Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f}")
            
    # Save last epoch model
    torch.save(final_model.state_dict(), os.path.join(models_dir, 'final_model_last.pth'))
    print(f"Đã lưu trọng số mô hình cuối cùng (last, best) tại thư mục {models_dir}")
            
    # 3. Lưu báo cáo kết quả report.md
    report_content = f"""# Báo Cáo Kết Quả Huấn Luyện (Tennis Match Prediction)

## Siêu tham số tối ưu tìm được qua Grid Search
- **Learning Rate (LR):** {best_lr:.2e}
- **Weight Decay (WD):** {best_wd:.2e}
- **Độ chính xác Cross-Validation tốt nhất (Best CV Accuracy):** {best_acc:.4f}

## Biểu đồ trực quan hóa

### Biểu đồ phân tích lưới siêu tham số (Hyperparameter Grid Search Heatmap)
![Hyperparameter Heatmap](hyperparameter_heatmap.png)

### Biểu đồ đường cong học tập (Convergence Curves - Loss vs. Accuracy)
![Learning Curves](learning_curves.png)
"""
    report_path = os.path.join(project_root, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Đã lưu báo cáo kết quả tại: {report_path}")



if __name__ == "__main__":
    train()