import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing.preprocessing import Preprocessing
from utils.dataset import prepare_loaders
from src.models.knn_scratch import KNNFromScratch
from src.utils.knn_evaluator import KNNEvaluator

# --- CẤU HÌNH ---
K_FIND_RANGE = list(range(5, 100, 10))  # Thử các giá trị K lẻ từ 5 đến 45: [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
SCRATCH_SUBSET_SIZE = 100            # Kích thước tập con để kiểm thử mô hình Scratch (vì Scratch chạy chậm)

def extract_numpy_data(loader):
    """Trích xuất dữ liệu numpy từ PyTorch DataLoader."""
    X_list = []
    y_list = []
    for X_batch, y_batch in loader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def train_knn():
    print("\n[=== BẮT ĐẦU HUẤN LUYỆN & TỐI ƯU HÓA KNN ===]\n")

    # Định nghĩa thư mục kết quả tuyệt đối để luôn lưu đúng thư mục dự án
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    result_dir = os.path.join(project_root, 'result')

    # 1. Load và tiền xử lý dữ liệu
    print("[*] Đang tải và tiền xử lý dữ liệu...")
    preprocessor = Preprocessing()
    df = preprocessor.run()
    folds, test_loader, input_dim, train_val_loader = prepare_loaders(df)
    
    # Trích xuất dữ liệu tập test và tập train+val toàn bộ
    X_test, y_test = extract_numpy_data(test_loader)
    X_train_val, y_train_val = extract_numpy_data(train_val_loader)
    
    print(f"Tổng số đặc trưng đầu vào (input_dim): {input_dim}")
    print(f"Kích thước tập huấn luyện + val: {X_train_val.shape}")
    print(f"Kích thước tập test: {X_test.shape}")

    # 2. Tìm giá trị K tối ưu bằng Cross-Validation (sử dụng Sklearn KNN để tối ưu tốc độ chạy)
    print(f"\n[*] Đang chạy Cross-Validation tìm K tối ưu (sử dụng Sklearn) trong khoảng: {K_FIND_RANGE}")
    
    n_splits = len(folds)
    cv_accuracies = np.zeros((n_splits, len(K_FIND_RANGE)))
    
    start_cv_time = time.time()
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        print(f"  --- Fold {fold_idx + 1}/{n_splits} ---")
        X_tr, y_tr = extract_numpy_data(train_loader)
        X_va, y_va = extract_numpy_data(val_loader)
        
        for k_idx, k in enumerate(K_FIND_RANGE):
            # Dùng Sklearn cho CV để chạy siêu nhanh (chỉ mất ~0.1s mỗi fold)
            knn_cv = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn_cv.fit(X_tr, y_tr)
            y_pred = knn_cv.predict(X_va)
            acc = np.mean(y_pred == y_va)
            cv_accuracies[fold_idx, k_idx] = acc
            print(f"    K = {k:02d} | Val Acc: {acc:.4f}")
            
    cv_time = time.time() - start_cv_time
    print(f"[+] Hoàn thành Cross-Validation trong {cv_time:.2f}s (Cực nhanh nhờ Sklearn!)")
    
    # Tính trung bình độ chính xác và tỷ lệ lỗi qua các folds
    mean_accuracies = cv_accuracies.mean(axis=0)
    error_rates = 1.0 - mean_accuracies
    
    best_idx = np.argmax(mean_accuracies)
    optimal_k = K_FIND_RANGE[best_idx]
    best_cv_acc = mean_accuracies[best_idx]
    print(f"[+] K tối ưu tìm được: {optimal_k} (Mean CV Accuracy: {best_cv_acc:.4f})")

    # 3. Huấn luyện các mô hình cuối cùng trên toàn bộ dữ liệu Train + Val
    print(f"\n[*] Đang huấn luyện các mô hình cuối cùng với K={optimal_k}...")
    
    # 3.1. Sklearn KNN - Huấn luyện và dự đoán trên TOÀN BỘ tập test
    print("[*] Chạy Sklearn KNN trên toàn bộ tập Test...")
    knn_sklearn = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
    knn_sklearn.fit(X_train_val, y_train_val)
    
    start_sklearn_full = time.time()
    y_pred_sklearn_full = knn_sklearn.predict(X_test)
    time_sklearn_full = time.time() - start_sklearn_full
    print(f"  [Sklearn - Full] Thời gian dự đoán trên toàn bộ tập Test: {time_sklearn_full:.4f}s")

    # 3.2. So sánh song song trên TẬP CON (Subset) của tập Test
    sub_size = min(SCRATCH_SUBSET_SIZE, len(X_test))
    X_test_sub = X_test[:sub_size]
    y_test_sub = y_test[:sub_size]
    print(f"\n[*] Tiến hành chạy so sánh song song trên tập con gồm {sub_size} mẫu...")

    # Chạy KNN From Scratch trên tập con
    knn_scratch = KNNFromScratch(k=optimal_k)
    knn_scratch.fit(X_train_val, y_train_val)
    
    start_scratch_sub = time.time()
    y_pred_scratch_sub = knn_scratch.predict(X_test_sub)
    time_scratch_sub = time.time() - start_scratch_sub
    print(f"  [Scratch - Subset] Thời gian dự đoán trên {sub_size} mẫu: {time_scratch_sub:.4f}s")

    # Chạy Sklearn KNN trên tập con (để đo thời gian chính xác)
    start_sklearn_sub = time.time()
    y_pred_sklearn_sub = knn_sklearn.predict(X_test_sub)
    time_sklearn_sub = time.time() - start_sklearn_sub
    print(f"  [Sklearn - Subset] Thời gian dự đoán trên {sub_size} mẫu: {time_sklearn_sub:.4f}s")

    # Tính toán ước tính thời gian chạy của Scratch trên toàn bộ tập test
    estimated_scratch_full_time = time_scratch_sub * (len(X_test) / sub_size)
    print(f"  --> Ước tính nếu chạy Scratch trên TOÀN BỘ tập Test sẽ mất khoảng: {estimated_scratch_full_time:.2f}s (Sklearn chỉ mất {time_sklearn_full:.4f}s)")

    # 4. Đánh giá, Trực quan hóa và Lưu báo cáo
    evaluator = KNNEvaluator()
    os.makedirs(result_dir, exist_ok=True)
    
    # Vẽ các biểu đồ
    evaluator.plot_k_optimization(
        K_FIND_RANGE, 
        error_rates, 
        optimal_k, 
        save_path=os.path.join(result_dir, 'knn_k_optimization.png')
    )
    evaluator.plot_confusion_matrix(
        y_test, 
        y_pred_sklearn_full, 
        title=f"Confusion Matrix: Sklearn KNN (K={optimal_k})", 
        save_path=os.path.join(result_dir, 'knn_confusion_matrix_sklearn.png')
    )
    
    # So sánh hiệu suất
    report_text, acc_sklearn_full = evaluator.compare_performances(
        y_test, y_pred_sklearn_full, 
        y_test_sub, y_pred_scratch_sub, y_pred_sklearn_sub
    )
    
    # Thêm chi tiết thời gian và gợi ý Hyperparameters tuning vào báo cáo
    time_report = """
========================================
BÁO CÁO THỜI GIAN CHẠY (RUNTIME REPORT)
========================================
- Tổng thời gian Cross-Validation (11 giá trị K, 5 Folds): {cv_time:.2f}s
- Thời gian dự đoán của Sklearn trên toàn bộ tập Test ({total_test} mẫu): {time_sklearn_full:.4f}s
- Thời gian dự đoán của Sklearn trên tập con ({sub_size} mẫu): {time_sklearn_sub:.4f}s
- Thời gian dự đoán của Scratch trên tập con ({sub_size} mẫu): {time_scratch_sub:.4f}s
- Tốc độ vượt trội của Sklearn trên tập con: gấp {speedup:.1f} lần so với Scratch
- Ước lượng thời gian nếu chạy Scratch trên toàn bộ tập Test: {est_scratch:.2f}s
""".format(
        cv_time=cv_time,
        total_test=len(X_test),
        sub_size=sub_size,
        time_sklearn_full=time_sklearn_full,
        time_sklearn_sub=time_sklearn_sub,
        time_scratch_sub=time_scratch_sub,
        speedup=time_scratch_sub / max(time_sklearn_sub, 1e-6),
        est_scratch=estimated_scratch_full_time
    )
    
    tuning_suggestions = """
========================================
GỢI Ý TỐI ƯU HÓA SIÊU THAM SỐ (HYPERPARAMETER TUNING)
========================================
1. Số láng giềng (K):
   - Đã được tìm kiếm tự động bằng phương pháp Cross-Validation (K tối ưu hiện tại = {optimal_k}).
   - Có thể mở rộng khoảng tìm kiếm hoặc thử nghiệm các bước nhảy nhỏ hơn nếu cần (ví dụ: các số lẻ từ 1 đến 100).

2. Khoảng cách (Distance Metrics):
   - Thử nghiệm các hàm tính khoảng cách khác trong Sklearn KNN như:
     * 'manhattan' (L1 distance): Phù hợp khi các đặc trưng có phân phối thưa hoặc nhiều nhiễu.
     * 'chebyshev' (L_infinity distance): Chỉ xét chênh lệch lớn nhất giữa các chiều đặc trưng.
     * 'minkowski' với tham số p khác nhau (p=3, p=4...).

3. Trọng số đóng góp (Weights):
   - 'uniform': Tất cả các điểm láng giềng có trọng số bỏ phiếu ngang nhau (hiện tại đang dùng).
   - 'distance': Các láng giềng gần hơn sẽ có tiếng nói/trọng số lớn hơn khi dự đoán.

4. Chuẩn hóa đặc trưng (Feature Scaling):
   - Hiện tại đang sử dụng StandardScaler. Đối với KNN, phân bố đặc trưng ảnh hưởng cực kỳ lớn. 
   - Có thể thử nghiệm MinMaxScaler hoặc RobustScaler (nếu dữ liệu chứa nhiều Outliers).

5. Giảm chiều dữ liệu (Dimensionality Reduction):
   - Sử dụng PCA (Principal Component Analysis) để giữ lại 95% phương sai (giống như trong file `mlp_pytorch_optuna.py`).
   - Việc giảm chiều giúp giảm hiện tượng "curse of dimensionality" của KNN và tăng tốc độ dự đoán của KNN Scratch đáng kể.
""".format(optimal_k=optimal_k)

    full_report = report_text + "\n" + time_report + "\n" + tuning_suggestions
    
    report_file_path = os.path.join(result_dir, 'knn_report.txt')
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
        
    print(f"\n[+] Báo cáo huấn luyện chi tiết đã được ghi tại: {report_file_path}")
    print("\n[=== KẾT THÚC HUÂN LUYỆN KNN ===]\n")

if __name__ == "__main__":
    train_knn()
