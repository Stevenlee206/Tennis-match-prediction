import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import pandas as pd
import numpy as np
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
models_dir = os.path.join(project_root, 'models')

from src.models.model import TennisNet
from src.preprocessing.preprocessing import Preprocessing


def predict(df: pd.DataFrame, model_path: str, scaler_path: str, batch_size: int = 64, device: str = 'cpu'):
    """
    Dự đoán kết quả trận đấu từ dữ liệu đã tiền xử lý.
    
    df: DataFrame chứa các đặc trưng đã qua xử lý (có thể chứa target và is_augmented, chúng sẽ tự động bị bỏ qua).
    return: (probabilities_p1_win, predicted_classes)
    """
    from torch.utils.data import DataLoader
    from utils.dataset import TennisDataset
    import json

    # Load danh sách đặc trưng đã khớp khi train
    features_path = os.path.join(os.path.dirname(model_path), 'features.json')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Không tìm thấy danh sách đặc trưng tại: {features_path}")
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)

    # Đảm bảo tất cả các cột đặc trưng mong muốn tồn tại trong df
    df_copy = df.copy()
    for col in feature_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0.0

    # Lấy đúng danh sách đặc trưng theo thứ tự chuẩn
    X = df_copy[feature_cols].values

    # Load scaler và chuẩn hóa đặc trưng
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Không tìm thấy bộ chuẩn hóa tại: {scaler_path}")
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # Khởi tạo dummy target cho Dataset loader
    dummy_y = np.zeros(len(X))
    dataset = TennisDataset(X_scaled, dummy_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load mô hình
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy trọng số mô hình tại: {model_path}")
    
    model = TennisNet(input_dim=X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Xác suất Player 1 thắng
            preds = logits.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_probs), np.array(all_preds)


def evaluate_model(model_path: str, scaler_path: str, test_data_path: str = None, 
                   output_csv_path: str = None, device: str = 'cpu'):
    """
    Đánh giá mô hình trên tập test (quarantine test set mặc định hoặc file CSV tùy chỉnh),
    in ra Accuracy, Confusion Matrix, Classification Report và lưu kết quả.
    """
    if output_csv_path is None:
        output_csv_path = os.path.join(models_dir, 'test_predictions.csv')

    import json
    from utils.dataset import prepare_loaders, TennisDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("\n" + "="*50)
    print("      ĐÁNH GIÁ MÔ HÌNH DỰ ĐOÁN TENNIS (INFERENCE)")
    print("="*50)
    print(f"Model path: {model_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Device: {device}")

    # 1. Load feature columns list
    model_dir = os.path.dirname(model_path)
    features_path = os.path.join(model_dir, 'features.json')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Không tìm thấy file danh sách đặc trưng features.json tại {features_path}")
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    print(f"Đã tải danh sách {len(feature_cols)} đặc trưng.")

    # 2. Load Scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Không tìm thấy file scaler.joblib tại {scaler_path}")
    scaler = joblib.load(scaler_path)

    # 3. Load/Prepare Test Data
    if test_data_path is None:
        print("\n[Dữ liệu] Không truyền --test_data_path. Tự động chuẩn bị tập test quarantine (10% cuối của dataset)...")
        n_splits = 5
        val_ratio = 0.1
        train_ratio = 1.0 - (n_splits + 1) * val_ratio # 0.4
        
        prep = Preprocessing()
        df = prep.run(train_ratio=train_ratio)
        
        # Align features
        df_copy = df.copy()
        for col in feature_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0.0
                
        cols_to_keep = feature_cols.copy()
        if 'target' in df.columns:
            cols_to_keep.append('target')
        if 'is_augmented' in df.columns:
            cols_to_keep.append('is_augmented')
        df = df_copy[cols_to_keep]
        
        _, test_loader, input_dim, _, _ = prepare_loaders(
            df, n_splits=n_splits, val_ratio=val_ratio, scaler=scaler
        )
        
        # Extract features and targets from test_loader
        X_test_list = []
        y_test_list = []
        for X_b, y_b in test_loader:
            X_test_list.append(X_b.numpy())
            y_test_list.append(y_b.numpy())
        X_test_scaled = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
    else:
        print(f"\n[Dữ liệu] Đang đọc từ file test chỉ định: {test_data_path}")
        df_test = pd.read_csv(test_data_path)
        
        # Align features
        df_copy = df_test.copy()
        for col in feature_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0.0
                
        X_test_raw = df_copy[feature_cols].values
        X_test_scaled = scaler.transform(X_test_raw)
        
        if 'target' in df_test.columns:
            y_test = df_test['target'].values
        else:
            y_test = None
            print("Lưu ý: Không tìm thấy cột 'target' trong file test. Sẽ chỉ thực hiện dự đoán.")

    # 4. Load Model and Predict
    input_dim = X_test_scaled.shape[1]
    model = TennisNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    dummy_y = np.zeros(len(X_test_scaled)) if y_test is None else y_test
    dataset = TennisDataset(X_test_scaled, dummy_y)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    probs = np.array(all_probs)
    preds = np.array(all_preds)

    # 5. Evaluate Metrics
    if y_test is not None:
        acc = accuracy_score(y_test, preds)
        print("\n" + "="*20 + " KẾT QUẢ ĐÁNH GIÁ " + "="*20)
        print(f"Tổng số mẫu đánh giá: {len(y_test)}")
        print(f"Độ chính xác (Accuracy): {acc:.4f} ({acc * 100:.2f}%)")
        
        print("\nMa trận nhầm lẫn (Confusion Matrix):")
        cm = confusion_matrix(y_test, preds)
        print(cm)
        
        print("\nBáo cáo chi tiết (Classification Report):")
        report = classification_report(y_test, preds, target_names=["Player 2 Win", "Player 1 Win"])
        print(report)
        print("=" * 58)
    else:
        print(f"\nĐã hoàn thành dự đoán trên {len(preds)} mẫu.")

    # 6. Save Predictions
    output_df = pd.DataFrame({
        'Prob_Player1_Win': probs,
        'Pred_Winner_Label': preds,
        'Pred_Winner': np.where(preds == 1, 'Player 1', 'Player 2')
    })
    
    if y_test is not None:
        output_df['True_Winner_Label'] = y_test
        output_df['True_Winner'] = np.where(y_test == 1, 'Player 1', 'Player 2')
        output_df['Correct'] = output_df['Pred_Winner'] == output_df['True_Winner']

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n[Kết quả] Đã lưu kết quả dự đoán chi tiết tại: {output_csv_path}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_model(
        model_path=os.path.join(models_dir, 'final_model_best.pth'),
        scaler_path=os.path.join(models_dir, 'scaler.joblib'),
        test_data_path=None,
        output_csv_path=os.path.join(models_dir, 'test_predictions.csv'),
        device=device
    )
