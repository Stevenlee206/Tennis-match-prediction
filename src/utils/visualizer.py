import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def _show_or_save_plot(filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Mô hình lưu biểu đồ tại: {filename}")
    plt.close()

def plot_learning_curves(history):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history['train_loss'], color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(history['val_acc'], color='tab:blue', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Convergence Speed (Loss vs Accuracy)')
    _show_or_save_plot('learning_curves.png')

def plot_hyperparameter_heatmap(results, lr_list, wd_list):
    plt.figure(figsize=(12, 8))
    sns.heatmap(results, annot=True, fmt=".4f",
                xticklabels=[f"{x:.1e}" for x in wd_list],
                yticklabels=[f"{x:.1e}" for x in lr_list])
    plt.xlabel('Weight Decay')
    plt.ylabel('Learning Rate')
    plt.title('Hyperparameter Grid Search (Validation Accuracy)')
    _show_or_save_plot('hyperparameter_heatmap.png')