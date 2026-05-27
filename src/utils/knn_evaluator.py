import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class KNNEvaluator:
    @staticmethod
    def compare_performances(y_true_full, y_pred_sklearn_full, y_true_sub, y_pred_scratch_sub, y_pred_sklearn_sub):
        """In báo cáo so sánh hiệu suất giữa Sklearn (toàn bộ & subset) và Scratch (subset)."""
        acc_sklearn_full = accuracy_score(y_true_full, y_pred_sklearn_full)
        acc_sklearn_sub = accuracy_score(y_true_sub, y_pred_sklearn_sub)
        acc_scratch_sub = accuracy_score(y_true_sub, y_pred_scratch_sub)
        
        report_str = ""
        report_str += "\n" + "="*55 + "\n"
        report_str += f"BÁO CÁO SO SÁNH HIỆU SUẤT KNN (K tối ưu)\n"
        report_str += "="*55 + "\n"
        report_str += f"{'Mô hình':<30} | {'Tập dữ liệu':<15} | {'Accuracy':<10}\n"
        report_str += "-" * 62 + "\n"
        report_str += f"{'Sklearn KNN (Đầy đủ)':<30} | {'Full Test':<15} | {acc_sklearn_full:.4f}\n"
        report_str += f"{'Sklearn KNN (Tập con)':<30} | {'Subset 100':<15} | {acc_sklearn_sub:.4f}\n"
        report_str += f"{'KNN From Scratch (Tập con)':<30} | {'Subset 100':<15} | {acc_scratch_sub:.4f}\n"
        report_str += "="*55 + "\n"

        # So sánh xem dự đoán trên tập con của 2 mô hình có trùng khớp hoàn toàn không
        matches = (y_pred_scratch_sub == y_pred_sklearn_sub).sum()
        match_percentage = (matches / len(y_true_sub)) * 100
        report_str += f"\n[Kiểm thử đối chiếu] Mức độ trùng khớp dự đoán giữa Scratch và Sklearn trên tập con: {match_percentage:.1f}% ({matches}/{len(y_true_sub)} mẫu)\n"

        report_str += "\n[Chi tiết] Báo cáo chi tiết Sklearn KNN trên Full Test:\n"
        report_str += classification_report(y_true_full, y_pred_sklearn_full)
        
        print(report_str)
        return report_str, acc_sklearn_full

    @staticmethod
    def plot_k_optimization(k_values, error_rates_scratch, optimal_k, save_path):
        """Vẽ biểu đồ tối ưu hóa K (Elbow Method) và lưu lại."""
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, error_rates_scratch, color='blue', linestyle='dashed', 
                 marker='o', markerfacecolor='red', markersize=8, label='Validation Error Rate')
        
        plt.axvline(x=optimal_k, color='green', linestyle='-', label=f'Optimal K={optimal_k}')
        
        plt.title('Tối ưu hóa K: Tỷ lệ lỗi CV vs Giá trị K')
        plt.xlabel('Giá trị K (Số láng giềng)')
        plt.ylabel('Tỷ lệ lỗi (Error Rate)')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Đã lưu biểu đồ tối ưu hóa K tại: {save_path}")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title, save_path):
        """Vẽ confusion matrix và lưu lại."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Dự đoán (Predicted)')
        plt.ylabel('Thực tế (Actual)')
        plt.title(title)
        plt.xticks([0.5, 1.5], ['Thua (0)', 'Thắng (1)'])
        plt.yticks([0.5, 1.5], ['Thua (0)', 'Thắng (1)'])
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Đã lưu biểu đồ Confusion Matrix tại: {save_path}")
