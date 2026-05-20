# Hướng Dẫn Chi Tiết: Mô Hình Predictive Coding (PC)

Tài liệu này giải thích chi tiết về cách mô hình Predictive Coding (PC) được lập trình và tích hợp vào pipeline của project, các hướng đi để gỡ lỗi và phân tích cải thiện hiệu năng, cũng như các câu lệnh terminal để khởi chạy.

---

## 1. Sơ đồ File & Chức năng

Codebase của Predictive Coding được chia thành các phần chính:

- **`src/model/Predictive_Coding/pc_layer.py`**:
  Định nghĩa một layer đơn của mạng lưới PC. Chứa ma trận trọng số `W`, ma trận bias `b`, cùng các hàm kích hoạt (activation) và đạo hàm của nó (derivative) phục vụ cho dự đoán forward và tính gradient.
- **`src/model/Predictive_Coding/pc_network.py`**:
  Trái tim của mô hình PC. Quản lý mạng nơ-ron nhiều lớp.
    - **`infer_hidden_states`**: Khác với Backprop thông thường, PC lặp lại quá trình giảm năng lượng (Energy) ngay trong local từng batch (Inference) dựa vào Sai số của lớp hiện tại.
    - **`train_on_batch`**: Cập nhật trọng số của `W` và `b` sau khi states hội tụ. Hỗ trợ hệ số phạt `sample_weights` cho mục đích ưu tiên Upset.
    - **`get_state_dict` / `load_state_dict`**: Lưu và tải trọng số theo mô hình từ điển (giống PyTorch) giúp Checkpointing.

- **`src/models/pc/pc_optuna.py`**:
  Scaffold cho Pipeline ML. Tương tự như vòng lặp PyTorch nhưng viết riêng cho NumPy.
    - **`train_pc_model_loop`**: Iterator qua các Epoch. Nó có cơ chế ngầm hiểu **Best Epoch Checkpointing**, lưu lại `best_state` nếu Validation Accuracy tăng lên.
    - **`objective`**: Thể hiện Optuna search space cho PC (`learning_rate`, `inference_steps`, `depth`, `width`, `epochs`). Quản lý chẻ nhánh `holdout` hoặc `walk_forward` (TimeSeriesSplit).
    - **`plot_feature_importance_permutation`**: Xáo trộn từng cột feature để đánh giá đâu là cột quan trọng nhất ảnh hưởng đến Accuracy.

- **`main.py`**:
  Nơi kết nối tất cả. Xóa bỏ dòng trạng thái `is_augmented` để chống Data Leakage, điều hướng ghi kết quả vào `/outputs/predictive_coding/` và `/reports/figures/predictive_coding/`. Tính toán Elo Bias (thiên kiến tỷ lệ cược).

---

## 2. Hướng Giải Quyết Khi Mô Hình Gặp Vấn Đề

Dưới đây là một số mẹo phân tích và chỉnh sửa nếu PC kết xuất ra một số bất thường:

### 2.1. Hiện tượng Overfitting (Quá khớp)

Dấu hiệu: Training energy giảm liên tục, nhưng Validation Accuracy lại giảm hoặc đi ngang ở mức thấp.
**Các hướng khắc phục:**

1. **Tinh chỉnh không gian Optuna ngắn lại**: Hãy thử giới hạn `--epochs` tối đa thấp xuống (ví dụ 20-40 thay vì 100), hoặc giới hạn `inference_steps` xuống dưới 15. Cập nhật mã nguồn ở `pc_optuna.py` khối lệnh `objective`.
2. **Thêm Penalty vào Hàm Energy**: Hiện tại hàm cập nhật Weights trong `train_on_batch` chưa có L1/L2. Đưa thêm một hệ số decay cho `layer.W`:
    ```python
    # Trong pc_network.py -> train_on_batch
    weight_decay = 1e-4
    layer.W = (layer.W + lr * grad_W - weight_decay * layer.W).astype(np.float32)
    ```
3. **Giảm độ lớn Width / Depth**: Lưới PC quá sâu mà dữ liệu ít thì sẽ overfit. Giữ Optuna Depth ở `[1, 2]`.

### 2.2. Hiện tượng Underfitting (Chưa đủ khớp)

Dấu hiệu: Trọng số không thay đổi nhiều, Loss/Energy không giảm, Accuracy lẹt đẹt mãi ở 50 - 64% (Random guessing dựa trên base rates).
**Các hướng khắc phục:**

1. **Learning Rate & Inference**: Nếu `learning_rate` hoặc `inference_lr` quá thấp, energy không chạy xuống được. Optuna hiện đang để chặn Log-scale khá thoải mái nhưng có thể bạn sẽ cần nới trần `learning_rate` lên `0.2` nếu cần.
2. **Kích hoạt hàm phi tuyến (Activation)**: Đổi `hidden_activation` từ `tanh` sang `relu`.
3. **Thêm Normalize / PCA**: Mô hình NumPy nhạy cảm với scale dữ liệu. Mặc định `pc_optuna.py` ĐÃ áp dụng `StandardScaler`. Hãy thử truyền flag `--add_pca` khi gọi lệnh để nén feature nếu các features đang dư thừa nhiễu.

### 2.3. Model dự đoán nghiêng hẳn về Cửa trên (Elo Reliance cao)

Nếu bạn mở JSON `bias_metrics` lên và thấy `elo_reliance` = 85%+ và `upset_prediction_accuracy` cực thấp:

1. Tăng hệ số của **Upset Weight**: Khai báo cờ `--weight_strategy magnitude --upset_weight 3.0` để ép mô hình phải tập trung dự đoán đúng cửa dưới (underdogs).
2. Xóa bỏ một số GA/Matchup Feature quá phụ thuộc vào ELO bằng cách check theo biểu đồ Permutation Feature Importance.

---

## 3. Các Lệnh Chạy Thực Tế (CLI Commands)

Đây là các lệnh để chạy Pipeline từ đầu đến cuối một cách mượt mà nhất. Đảm bảo bạn đang đứng ở thư mục gốc của project (có chứa `main.py`).

### TH 1: Chạy Thử Nhanh (Fast Trial - Chỉ 1 lượt tối ưu)

Chỉ chạy 1 lượt, số epoch thấp để debug xem file code có trơn tru không, không tốn thời gian tune quá nhiều. Phù hợp nếu bạn muốn khóa cứng 1 bộ hyperparameter để test workflow.

```powershell
python main.py --model predictive_coding --optimizer optuna --validation holdout --n_trials 1 --epochs 20
```

### TH 2: Tối ưu Toàn Diện (Hyperparameter Tuning Chuẩn)

Dành cho việc Training chính thức để tìm thông số tốt nhất. Nó sẽ sử dụng **Walk-Forward Validation** với Optuna để cắt TimeSeries, loại bỏ Leakage và chạy 15 Trials để săn cấu hình ẩn số. Quá trình có thể tốn từ 10-20 phút. Mạng kết hợp phạt Upset theo cấp độ chênh lệch Elo:

```powershell
python main.py --model predictive_coding --optimizer optuna --validation walk_forward --n_trials 15 --epochs 50 --weight_strategy magnitude --upset_weight 2.0
```

### TH 3: Tối ưu với Kích thước Dữ liệu thu gọn (Thêm PCA)

Kết hợp giảm chiều dữ liệu nhiễu bằng PCA trước khi đưa vào PC. Phù hợp khi bạn thêm đến hơn 400 features. Holdout split.

```powershell
python main.py --model predictive_coding --optimizer optuna --validation holdout --n_trials 10 --add_pca
```

---

_Mọi trọng số, Checkpoint, cấu hình tốt nhất sẽ được dump dưới hệ quy chiếu `/outputs/predictive_coding/` và các biểu đồ Importance / Loss Plot sẽ văng ra tại nhánh `/reports/figures/predictive_coding/`._
