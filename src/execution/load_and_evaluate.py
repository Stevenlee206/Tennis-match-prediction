import json
import joblib
import numpy as np
from pathlib import Path
from src.execution.bias_analysis import evaluate_model_bias, append_metrics_to_config
from src.execution.model_interpretation import calculate_feature_importances, plot_interpretability

def _get_file_names(args):
    """Determine the model, scaler, and configuration file names based on the algorithm.."""
    if args.model == "svm":
        if args.mode == "sgd":
            return "svm_sgd_model.joblib", "svm_sgd_scaler.joblib", "svm_sgd_config.json"
        return f"{args.kernel}_model.joblib", f"{args.kernel}_scaler.joblib", f"{args.kernel}_config.json"

    elif args.model == "pytorch_svm":
        return "svm_pytorch_model.pth", "svm_pytorch_scaler.joblib", "svm_pytorch_config.json"

    elif args.model == "deepforest":
        return "deepforest_model.joblib", "deepforest_scaler.joblib", "deepforest_config.json"

    elif args.model == "xgboost":
        return "xgboost_model.joblib", "xgboost_scaler.joblib", "xgboost_config.json"

    elif args.model == "decisiontree":
        return "decisiontree_model.joblib", "decisiontree_scaler.joblib", "decisiontree_config.json"

    elif args.model == "logistic_regression":
        return "log_reg_model.joblib","log_reg_scaler.joblib","log_reg_config.json"

    elif args.model == "naive_bayes":

        return "nb_model.joblib", "nb_scaler.joblib", "nb_config.json"
    else:  # Random Forest variants
        return f"{args.rf_variant}_model.joblib", f"{args.rf_variant}_scaler.joblib", f"{args.rf_variant}_config.json"



def _get_feature_prefix(args):
    """Xác định tiền tố tên file cho PCA và KMeans."""
    if args.model in ["pytorch_svm", "pytorch_mlp", "tabnet", "deepforest"]:
        return args.model.replace('pytorch_', '') + '_pytorch' if 'pytorch' in args.model else args.model
    elif args.model == "svm":
        return "svm_sgd" if args.mode == "sgd" else args.kernel
    else:
        return args.rf_variant


def load_and_evaluate_model(args, X_eval, y_eval, out_dir):
    """
    Load toàn bộ pipeline (Scaler -> PCA/KMeans -> Model), thực hiện dự đoán,
    đánh giá thiên kiến (bias) và lưu vào file cấu hình JSON.
    """
    out_dir = Path(out_dir)
    print(f"\n[*] Đang đánh giá mô hình từ thư mục: {out_dir}")

    # 1. Xác định tên file
    model_name, scaler_name, config_name = _get_file_names(args)
    model_path = out_dir / model_name
    scaler_path = out_dir / scaler_name
    config_path = out_dir / config_name

    # Kiểm tra tồn tại (bỏ qua đuôi .zip cho TabNet khi check path nếu cần)
    check_model_path = Path(str(model_path).replace('.zip', '') + '.zip') if args.model == "tabnet" else model_path

    if not (check_model_path.exists() and scaler_path.exists()):
        print(f"Lỗi: Không tìm thấy model hoặc scaler tại {out_dir}. Bỏ qua đánh giá.")
        return

    # Sao chép dữ liệu gốc để giữ lại làm bằng chứng tính Bias
    X_eval_raw = X_eval.copy()

    # 2. Load Scaler & Transform
    scaler = joblib.load(scaler_path)
    X_eval_scaled = scaler.transform(X_eval)

    # 3. Load PCA & Transform
    prefix = _get_feature_prefix(args)
    if getattr(args, 'add_pca', False):
        pca_path = out_dir / f"{prefix}_pca.joblib"
        if pca_path.exists():
            pca = joblib.load(pca_path)
            X_eval_scaled = pca.transform(X_eval_scaled)
        else:
            print(f"Warning: PCA is enabled but not found {pca_path.name}")

    # 3.1 Load KMeans & Transform
    if getattr(args, 'add_kmeans', False):
        kmeans_path = out_dir / f"{prefix}_kmeans.joblib"
        if kmeans_path.exists():
            kmeans = joblib.load(kmeans_path)
            v_distances = kmeans.transform(X_eval_scaled)
            X_eval_scaled = np.hstack((X_eval_scaled, v_distances))
        else:
            print(f" Warning: KMeans is enabled but not found {kmeans_path.name}")

    # 4. Phân luồng Load Model & Predict
    y_pred = []

    if args.model == "pytorch_svm":
        # Import Lazy (Chỉ load PyTorch khi cần)
        import torch
        from src.models.svm.svm_pytorch_optuna import PyTorchLinearSVM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PyTorchLinearSVM(X_eval_scaled.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            tensor_X = torch.FloatTensor(X_eval_scaled).to(device)
            preds_raw = model(tensor_X)
            y_pred = (preds_raw > 0).cpu().numpy().astype(int)

    else:
        # Load các model hệ Sklearn / TabNet / DeepForest
        model = joblib.load(model_path)
        y_pred = model.predict(X_eval_scaled)

    # 5. Tính toán Bias & Đánh giá
    bias_metrics = evaluate_model_bias(y_eval.values, y_pred, X_eval_raw)

    # 6. Ghi kết quả vào file JSON
    if config_path.exists():
        append_metrics_to_config(config_path, bias_metrics)
    else:
        print(f"Không tìm thấy {config_name} để ghi kết quả đánh giá.")

    # 7.1. Tái tạo lại danh sách Feature Names từ dữ liệu thô
    feature_names = list(X_eval_raw.columns)
    # Xử lý đổi tên nếu có dùng PCA (giảm chiều)
    if getattr(args, 'add_pca', False) and pca_path.exists():
        feature_names = [f"PCA_{i}" for i in range(X_eval_scaled.shape[1])]

        # Xử lý nối thêm tên nếu có dùng KMeans (thêm cụm)
    if getattr(args, 'add_kmeans', False) and kmeans_path.exists():
        n_clusters = getattr(args, 'n_clusters', 5)
        cluster_names = [f"KMeans_Dist_{i}" for i in range(n_clusters)]
        if not getattr(args, 'add_pca', False):
            feature_names.extend(cluster_names)

        # 7.2. Kích hoạt tính toán và vẽ biểu đồ
    if args.model not in ["pytorch_svm", "pytorch_mlp"]:
        # Gọi hàm 1: Tính toán độ quan trọng
        importances = calculate_feature_importances(
            model=model,
            X_eval=X_eval_scaled,
            y_eval=y_eval.values
        )

        # Gọi hàm 2: Truyền kết quả vào để vẽ
        plot_interpretability(
            model=model,
            X_eval=X_eval_scaled,
            importances=importances,
            feature_names=feature_names,
            out_dir=out_dir,
            model_name=args.model
        )
    else:
        print(f"\n[*] Bỏ qua Feature Importance/PDP cho {args.model}.")

     # 8. ERROR ANALYSIS & HYPOTHESIS TESTING
    print("\n" + "=" * 50)
    print(f"[*] ĐANG PHÂN TÍCH LỖI MÔ HÌNH (ERROR ANALYSIS)")
    print("=" * 50)

    from src.execution.prediction_analysis import (
        plot_prediction_summary,
        plot_confidence_analysis,
        plot_all_features_errors
    )

    # 8.1. Kiểm tra Giả thuyết Phân bổ nhãn
    plot_prediction_summary(y_eval.values, y_pred, out_dir, args.model)

    # 8.2. Kiểm tra Giả thuyết Độ tự tin (Yêu cầu model có hàm predict_proba)
    if hasattr(model, "predict_proba"):
        try:
        # Lấy xác suất của class 1 (Player 1 thắng)
            y_prob = model.predict_proba(X_eval_scaled)[:, 1]
            plot_confidence_analysis(y_eval.values, y_prob, out_dir, args.model)
        except Exception as e:
            print(f"Không thể vẽ biểu đồ Confidence: {str(e)}")
    else:
        print(f"-> Bỏ qua biểu đồ Confidence do {args.model} không hỗ trợ predict_proba.")

        # 8.3. Kiểm tra Giả thuyết Lỗi do chỉ số quá sát nhau (Dùng elo_diff)
        # Bạn có thể thay đổi 'elo_diff' thành bất kỳ feature nào bạn muốn test (vd: 'days_since_last_match_diff')
    plot_all_features_errors(X_eval_raw, y_eval.values, y_pred, out_dir, args.model)