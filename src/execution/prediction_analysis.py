import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from typing import Any


# HÀM 1: KIỂM TRA GIẢ THUYẾT VỀ SỰ THIÊN LỆCH NHÃN (CLASS BIAS HYPOTHESIS)
# Giả thuyết: Mô hình có xu hướng an toàn, luôn dự đoán tay vợt 1 (cửa trên) thắng?
def plot_prediction_summary( y_true: np.ndarray | pd.Series,
                            y_pred: np.ndarray | pd.Series,
                            out_dir: str | Path,
                            model_name: str
                            ) -> None:
    """
    Vẽ Confusion Matrix và Biểu đồ phân phối tỷ lệ Thắng/Thua để xem model có bị lệch (bias) không.
    """
    out_dir = Path(out_dir)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Dự đoán Thua (0)', 'Dự đoán Thắng (1)'],
                yticklabels=['Thực tế Thua (0)', 'Thực tế Thắng (1)'])
    axes[0].set_title(f'Confusion Matrix ({model_name.upper()})', fontweight='bold', pad=15)

    # 2. Phân phối Dự đoán vs Thực tế
    actual_counts = pd.Series(y_true_arr).value_counts(normalize=True).sort_index() * 100
    pred_counts = pd.Series(y_pred_arr).value_counts(normalize=True).sort_index() * 100

    bar_width = 0.35
    x = np.arange(2)

    axes[1].bar(x - bar_width / 2, actual_counts, bar_width, label='Thực tế', color='gray', alpha=0.7)
    axes[1].bar(x + bar_width / 2, pred_counts, bar_width, label='Mô hình Dự đoán', color='teal')

    axes[1].set_title('Tỷ lệ Phân bổ Lớp (Class Distribution %)', fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Player 1 Thua (0)', 'Player 1 Thắng (1)'])
    axes[1].set_ylabel('Phần trăm (%)')
    axes[1].legend()

    plt.tight_layout()
    save_path = out_dir / f"{model_name}_prediction_summary.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ Phân phối dự đoán tại: {save_path.name}")


# HÀM 2: KIỂM TRA GIẢ THUYẾT VỀ MỨC ĐỘ TỰ TIN (CONFIDENCE HYPOTHESIS)
# Giả thuyết: Khi mô hình đoán sai, xác suất (probability) của nó thường lấp lửng ở mức 50-55%?
def plot_confidence_analysis(   y_true: np.ndarray | pd.Series,
                                y_prob: np.ndarray,
                                out_dir: str | Path,
                                model_name: str
                            ) -> None:
    """
    Vẽ phân phối xác suất dự đoán (Confidence) tách biệt giữa nhóm Đoán Đúng và Đoán Sai.
    """
    out_dir = Path(out_dir)
    y_true_arr = np.array(y_true)

    # Tính mức độ tự tin (Confidence: Khoảng cách từ mức 0.5)
    # Ví dụ: Prob = 0.9 -> Conf = 90%. Prob = 0.1 -> Conf = 90% (tự tin đoán thua)
    confidence = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    y_pred_arr = (y_prob >= 0.5).astype(int)

    # Phân loại Đúng/Sai
    is_correct = (y_pred_arr == y_true_arr)

    df_conf = pd.DataFrame({
        'Confidence': confidence,
        'Result': ['Đoán Đúng' if c else 'Đoán Sai' for c in is_correct]
    })

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_conf, x='Confidence', hue='Result', fill=True,
                palette={'Đoán Đúng': 'teal', 'Đoán Sai': 'crimson'},
                alpha=0.4, linewidth=2)

    plt.title(f'Phân phối Mức độ Tự tin (Confidence) - {model_name.upper()}', fontsize=14, fontweight='bold')
    plt.xlabel('Xác suất Tự tin (Từ 0.5 [Lưỡng lự] đến 1.0 [Tuyệt đối])')
    plt.ylabel('Mật độ (Density)')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.xlim(0.5, 1.0)

    save_path = out_dir / f"{model_name}_confidence_analysis.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ Phân tích độ tự tin tại: {save_path.name}")


# HÀM 3: KIỂM TRA GIẢ THUYẾT THEO ĐẶC TRƯNG CỤ THỂ (FEATURE ERROR HYPOTHESIS)
# Giả thuyết: Mô hình thường xuyên đoán sai khi Elo của 2 người quá sát nhau?
def plot_error_by_feature(
        X_raw: pd.DataFrame,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        feature_name: str,
        out_dir: str | Path,
        model_name: str,
        verbose: bool = True
        ) -> None:
    """
    Trực quan hóa sự khác biệt của một Feature cụ thể (vd: elo_diff) giữa nhóm Đoán Đúng và Đoán Sai.
    """
    if feature_name not in X_raw.columns:
        print(f"⚠️ Bỏ qua phân tích {feature_name}: Không tìm thấy cột này trong dữ liệu.")
        return

    out_dir = Path(out_dir)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    is_correct = (y_pred_arr == y_true_arr)

    df_feature = pd.DataFrame({
        feature_name: X_raw[feature_name].values,
        'Prediction_Status': ['Chính xác' if c else 'Sai lầm' for c in is_correct]
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Biểu đồ Boxplot (xem các điểm ngoại lai)
    sns.boxplot(data=df_feature, x='Prediction_Status', y=feature_name,
                palette={'Chính xác': 'lightseagreen', 'Sai lầm': 'lightcoral'}, ax=axes[0])
    axes[0].set_title(f'Phân bổ {feature_name} theo Kết quả đoán', fontweight='bold')

    # Biểu đồ KDE (Mật độ để xem mô hình hay gãy ở khoảng giá trị nào)
    sns.kdeplot(data=df_feature, x=feature_name, hue='Prediction_Status', fill=True,
                palette={'Chính xác': 'teal', 'Sai lầm': 'crimson'}, ax=axes[1], alpha=0.3)
    axes[1].set_title(f'Mật độ {feature_name} nơi Mô hình đoán sai', fontweight='bold')

    plt.tight_layout()
    save_path = out_dir / f"{model_name}_error_analysis_{feature_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    if verbose:
        print(f"[*] Đã lưu biểu đồ Phân tích lỗi theo {feature_name} tại: {save_path.name}")


# ==============================================================================
# HÀM 4 (MỚI): VÒNG LẶP PHÂN TÍCH TẤT CẢ ĐẶC TRƯNG
# ==============================================================================
import shutil  # Thêm thư viện này ở đầu file


def plot_all_features_errors(
        X_raw: pd.DataFrame,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        out_dir: str | Path,
        model_name: str
) -> None:
    out_dir = Path(out_dir)

    # 1. Định nghĩa "Danh sách Vàng" (Chỉ vẽ những cái này)
    keep_list = [
        'elo_diff', 'elo_hard_diff', 'fatigue_diff', 'h2h_advantage_diff'
    ]

    # 2. Tạo thư mục đích cho biểu đồ quan trọng
    final_dir = out_dir.parent / "final_research_figures"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"-> Bắt đầu quá trình lọc: Chỉ vẽ {len(keep_list)} đặc trưng quan trọng...")

    # 3. Chỉ lặp qua những cột có trong danh sách Vàng
    for feature_name in keep_list:
        if feature_name in X_raw.columns:
            if pd.api.types.is_numeric_dtype(X_raw[feature_name]):
                # Vẽ trực tiếp vào thư mục Final
                plot_error_by_feature(
                    X_raw=X_raw,
                    y_true=y_true,
                    y_pred=y_pred,
                    feature_name=feature_name,
                    out_dir=final_dir,
                    model_name=model_name,
                    verbose=True
                )
        else:
            print(f"⚠️ Cảnh báo: Đặc trưng '{feature_name}' không tồn tại trong dữ liệu.")

    print(f"[*] Xong! Chỉ {len(keep_list)} biểu đồ cần thiết đã được xuất tại: {final_dir.name}/")