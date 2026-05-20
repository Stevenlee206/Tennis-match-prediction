(Update May 20, 2026) User chọn triển khai Phương án A: tích hợp Predictive Coding vào root main.py như một --model mới.

## Plan: Predictive Coding Tuning + Walk-forward CV

Mục tiêu: nâng cấp pipeline training cho Predictive Coding (PC) để chạy đúng chuẩn time-series: split 80% train / 20% test theo thời gian cho giai đoạn 2014–2024 (giống root main.py), dùng walk-forward cross validation (TimeSeriesSplit) bên trong tập train để so sánh hyperparameter (Optuna + Grid), log chi tiết theo epoch (train/val), chọn best epoch, lưu trọng số + config rõ ràng, trực quan hóa so sánh trial, thêm thống kê feature importance + thời gian, và thêm loss/weight để “thưởng” các trận upset (Elo thấp thắng).

**Bối cảnh code hiện tại (đã khảo sát)**

- PC hiện chạy tại `src/model/Predictive_Coding/main.py` gọi `PCTrainer.fit()`.
- `PCTrainer` (numpy) chỉ có split theo năm (2014–2023 train, 2024 test), log bằng `print`, không checkpoint.
- Preprocessing (`src/preprocessing/preprocessing.py`) đã sort theo `tourney_date` và có logic chống leakage dựa trên ngưỡng 0.8 (fit một số bước trên phần đầu 80%).
- Repo đã có pattern walk-forward + Optuna + sample-weight theo upset ở các pipeline SVM/RF: `src/models/svm/svm_sklearn_optuna.py::generate_sample_weights`.

---

**Giải thích `is_augmented`**

- `is_augmented` được tạo trong `src/preprocessing/target_encoding.py::create_target(augment=True)`.
- Mỗi match được nhân đôi và “đổi góc nhìn” player1/player2 (qua sign), nhằm tăng dữ liệu training.
- Best practice trong repo hiện tại: dùng augmented cho training (tăng robustness), nhưng đánh giá Val/Test chỉ trên `is_augmented==0` để tránh “đánh giá trên mẫu tổng hợp/nhân bản”.

---

**Steps**

### Phase 1 — Chuẩn hóa data split & fold generator (time-series)

1. Tạo hàm dataset-level để lấy dataframe đã preprocess rồi lọc đúng years 2014–2024.
    - Dựa trên `src/model/util/dataset.py::load_preprocessed_df()` và cột `year`.
2. Thêm logic split 80/20 theo thời gian (shuffle=False) ở tầng dataframe.
    - Đề xuất API: `split_chronological(df, test_size=0.2, keep_cols=[...]) -> trainval_df, test_df`.
    - Test set: lọc `is_augmented==0` nếu tồn tại.
    - Trainval: có thể giữ cả augmented; nhưng drop cột `is_augmented` khỏi features.
3. Tạo walk-forward CV generator dựa trên `TimeSeriesSplit`.
    - Đề xuất API: `iter_time_series_folds(trainval_df, n_splits=5) -> (fold_id, train_df, val_df)`.
    - Mỗi fold fit preprocessor (scaler/onehot) trên `train_df` rồi transform `val_df`.

### Phase 2 — Nâng cấp PCTrainer: train/val theo epoch + checkpoint + sample weighting

4. Refactor `PCTrainer` để hỗ trợ train/val explicit (không tự split theo năm).
    - Thêm constructor nhận trực tiếp `train_df`, `val_df`, `test_df` hoặc nhận `X_train, y_train, X_val, y_val`.
    - Tái sử dụng `build_feature_preprocessor()` trong `src/model/util/dataset.py` để fit trên train và transform val/test.
5. Logging theo epoch:
    - Trong mỗi epoch, log:
        - train: `energy` (loss proxy) + metrics (accuracy, precision, f1, log_loss, …) qua `binary_classification_metrics`.
        - val: metrics tương tự.
        - thời gian: train_epoch_time, eval_time.
    - Ghi ra file `metrics.csv` (hoặc JSONL) theo format nhất quán: `epoch, split, energy, accuracy, ... , seconds`.
6. Best-epoch selection:
    - Track `best_epoch` theo metric bạn chọn (đã chốt: Accuracy maximize).
    - Lưu checkpoint “best so far” (weights) khi val accuracy tăng.
7. Lưu model weights + preprocessor:
    - Vì PC là numpy, thêm cơ chế `state_dict` trong `PredictiveCodingNetwork`:
        - `get_state_dict()` trả về list weights/bias.
        - `load_state_dict(state)` để restore.
    - Save weights bằng `np.savez` (hoặc pickle) + save preprocessor bằng `joblib.dump`.
    - Save config JSON gồm: hyperparams, best_epoch, val metrics, thời gian train, seed, feature_names.
8. Upset reward (Elo thấp thắng):
    - Dùng logic giống `generate_sample_weights()` (SVM/RF) dựa trên `elo_diff` và nhãn `target`.
    - Thêm `sample_weight` vào `train_on_batch(x, y, sample_weight=None)`.
    - Áp trọng số vào lỗi output layer (tương đương weighted energy): nhân error output bởi `sqrt(w)` để energy đóng góp $w\cdot e^2$.
    - Cho phép chọn `weight_strategy` (`none|static|magnitude|temporal`) và `upset_weight` (float).
    - Log thêm metric `upset_accuracy` (accuracy trên subset upset) để kiểm chứng vấn đề bias Elo.

### Phase 3 — Hyperparameter search: Optuna + Grid (walk-forward inner-CV)

9. Tạo module tuning cho PC (tương tự style `src/models/*_optuna.py`):
    - `run_pc_pipeline(..., optimizer="optuna"|"grid", validation="walk_forward"|"holdout")`.
    - Objective = mean(val_accuracy_best_epoch) qua các folds của TimeSeriesSplit.
10. Optuna:

- Dùng sqlite storage như SVM (`sqlite:///...optuna.db`).
- Dùng `trial.report()` theo fold hoặc theo epoch (gợi ý report theo fold để giảm overhead) + pruner.

11. Grid search:

- Cho grid nhỏ (vì PC training tốn thời gian) và vẫn đánh giá bằng walk-forward.

12. Refit per hyperparam config (để “có model tốt nhất cho từng cài đặt”):

- Sau khi score CV xong, train lại trên toàn bộ trainval và dùng một validation tail (vd last 10% trainval) để pick best epoch và lưu artifact.
- Lưu artifact theo `trial_id` để so sánh trực tiếp.

### Phase 4 — Báo cáo, so sánh, visualization, feature importance, timing

13. Tổng hợp kết quả trials:

- Ghi `trials.csv` gồm: trial_id, params, cv_mean_acc, refit_best_acc, best_epoch, train_seconds, upset_acc.

14. Visualization:

- Plot `trial vs accuracy` (Optuna history) + có thể plot scatter một vài hyperparam quan trọng.
- Với grid: heatmap cho (depth/width) hoặc (lr/inference_steps) nếu phù hợp.

15. Feature importance:

- Vì PC không có importance native, dùng permutation importance trên holdout test:
    - baseline test accuracy;
    - shuffle từng feature (theo `preprocessor.get_feature_names_out()`), đo drop accuracy;
    - lưu `feature_importance.csv` + barplot top-k.

16. Timing stats:

- Log tổng thời gian theo: trial, fold, epoch; và aggregate (mean/median) cho train vs eval.

### Phase 5 — Tích hợp entrypoint (khuyến nghị)

17. Tích hợp vào root `main.py`:

- Thêm `--model predictive_coding` (hoặc `pc`).
- Reuse flags đã có: `--optimizer optuna|grid`, `--validation holdout|walk_forward`, `--weight_strategy`, `--upset_weight`, `--n_trials`, `--epochs`, ...
- Route output giống các model khác: `outputs/predictive_coding/...` và `reports/figures/predictive_coding/...`.

18. Giữ `src/model/Predictive_Coding/main.py` như “demo script” hoặc chuyển thành wrapper gọi pipeline mới.

---

**Gợi ý hyperparameters nên tune cho Predictive Coding**

- Architecture
    - `hidden_sizes` (depth/width): depth 1–4, width theo {8, 16, 32, 64, 128, 256}; ví dụ: (64,), (128,), (64,32), (128,64), (128,64,32).
    - `hidden_activation`: {tanh, relu}.
    - `output_activation`: giữ sigmoid (binary) (hoặc thử identity nếu muốn output logits, nhưng cần đổi metrics/predict).
    - `weight_scale` của PCLayer (nếu expose): {0.5, 1.0, 1.5}.
- Learning / Inference
    - `learning_rate`: log-uniform [1e-4, 1e-1].
    - `inference_lr`: log-uniform [1e-3, 1.0].
    - `inference_steps`: int [5, 50].
- Training
    - `epochs`: int [20, 200] (Optuna), hoặc grid {30, 50, 80, 120}.
    - `batch_size`: {64, 128, 256, 512}.
    - `threshold`: {0.45, 0.5, 0.55} (nếu muốn tune theo acc).
- Upset reward
    - `weight_strategy`: none/static/magnitude/temporal.
    - `upset_weight`: float [1.0, 5.0].

---

**Relevant files**

- `src/model/Predictive_Coding/trainer.py` — refactor fit loop: train/val logging theo epoch, best-epoch checkpoint, sample_weight.
- `src/model/Predictive_Coding/pc_network.py` — thêm `get_state_dict/load_state_dict`, hỗ trợ weighted output error.
- `src/model/util/dataset.py` — thêm split 80/20 chronological + fold generator + transform per fold.
- `src/preprocessing/preprocessing.py` và `src/preprocessing/target_encoding.py` — hiểu nguồn `is_augmented`, đảm bảo split tương thích.
- `src/models/svm/svm_sklearn_optuna.py` — tham khảo `generate_sample_weights` và TimeSeriesSplit.
- `main.py` — tham khảo cách route output/reports, validation flags.

---

**Verification**

1. Smoke-run PC giữ nguyên (không tuning): chạy entry mới với `--validation holdout` để tạo artifacts và log epoch.
2. Walk-forward CV: chạy `--validation walk_forward --optimizer grid` với grid nhỏ (2–4 configs) để kiểm tra folds + logging.
3. Optuna: chạy `--optimizer optuna --n_trials 5` để kiểm tra study db + best trial save.
4. Kiểm tra upset weighting:
    - so sánh `upset_accuracy` trước/sau khi bật `weight_strategy`.
5. Feature importance: confirm tạo `feature_importance.csv` + figure trên holdout test.

---

**Decisions**

- Split: 80/20 theo thời gian (chronological) giống root `main.py`.
- Metric chọn best epoch/hyperparam: Accuracy.
- Augmented: dùng cho train; mọi evaluation (val/test và val trong từng fold) chỉ dùng `is_augmented==0` để tránh leakage kiểu “cùng một trận ở train và val dưới góc nhìn hoán đổi”.

**Further Considerations**

1. Số folds cho walk-forward: mặc định 5 (theo repo). Có thể giảm 3 nếu training PC chậm.
2. Nếu cần “theo năm” thay vì theo index, có thể thay TimeSeriesSplit bằng year-based folds, nhưng sẽ lệch với logic leakage-control đang dùng train_split_idx=0.8 trong preprocessing.
