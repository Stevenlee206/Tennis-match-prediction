import os
import sys
import numpy as np
import pandas as pd
import json
import datetime
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.continual_learning_benchmark.data_splitter import BenchmarkDataPipeline, get_benchmark_splits
from src.continual_learning_benchmark.player_categorizer import categorize_players
from src.continual_learning_benchmark.evaluator import evaluate_model_bias, evaluate_player_metrics
from src.continual_learning_benchmark.models_setup import tune_hyperparameters_tscv, train_model_full
from src.continual_learning_benchmark.feature_importance import calculate_permutation_importance, plot_feature_importance
from src.continual_learning_benchmark.utils import TeeLogger

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def plot_learning_curves(history, title, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Energy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Acc', color='blue')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_metrics_bar(all_results, save_dir):
    """
    Plots a bar chart comparing accuracy across models and modes.
    """
    for model_type, modes in all_results.items():
        modes_list = list(modes.keys())
        accs = [modes[m]['bias_metrics']['final_accuracy'] for m in modes_list]
        
        plt.figure(figsize=(8, 6))
        plt.bar(modes_list, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(modes_list)])
        plt.ylim(0, 100)
        for i, v in enumerate(accs):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
        plt.title(f'{model_type.upper()} Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.savefig(os.path.join(save_dir, f"{model_type}_accuracy_comparison.png"), dpi=300)
        plt.close()

def prep_xy(df):
    df_clean = df.copy()
    raw = df.copy()
    drop_cols = ['target', 'year', 'is_augmented', 'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc', 'score', 'tourney_date', 'winner_elo', 'loser_elo']
    X = df_clean.drop(columns=drop_cols, errors='ignore').fillna(0)
    features = X.columns.tolist()
    X = X.values
    y = df_clean['target'].values if 'target' in df_clean.columns else np.zeros(len(df_clean))
    return X, y, raw, features

def run_test_benchmark():
    # Setup Logging & Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "outputs", "test_benchmark", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "run.log")
    sys.stdout = TeeLogger(log_file)
    
    print(f"==================================================")
    print(f" TEST BENCHMARK STARTED (1 Trial, 1 Epoch)")
    print(f" Run Directory: {run_dir}")
    print(f"==================================================\n")

    # 1. Data Processing
    print("--- 1. Loading & Processing Data ---")
    pipeline = BenchmarkDataPipeline()
    data, train_split_idx, pre_2025_count = pipeline.run()
    
    print("--- 2. Identifying Target Players ---")
    selected_players, _ = categorize_players(data, num_players_per_category=3)
    
    print("--- 3. Splitting Benchmark Data ---")
    D_Train_Base, D_Mean, D_Test, D_Holdout = get_benchmark_splits(data, train_split_idx, pre_2025_count)
    
    X_base, y_base, raw_base, feature_names = prep_xy(D_Train_Base)
    X_mean, y_mean, raw_mean, _ = prep_xy(D_Mean)
    X_test, y_test, raw_test, _ = prep_xy(D_Test)
    
    input_dim = X_base.shape[1]
    
    # Scale Data
    scaler = StandardScaler()
    X_base = scaler.fit_transform(X_base)
    X_mean = scaler.transform(X_mean)
    X_test = scaler.transform(X_test)
    
    X_base_mean = np.vstack([X_base, X_mean])
    y_base_mean = np.concatenate([y_base, y_mean])

    models_to_run = ["nn", "pcn"]
    modes = ["static", "finetune", "retrain"]
    all_results = {}

    for model_type in models_to_run:
        all_results[model_type] = {}
        print(f"\n====================== Evaluating Model: {model_type.upper()} ======================")
        model_out_dir = os.path.join(run_dir, model_type)
        os.makedirs(model_out_dir, exist_ok=True)
        
        static_weights_path = os.path.join(weights_dir, f"{model_type}_static.pt" if model_type == "nn" else f"{model_type}_static.npz")
        
        for mode in modes:
            print(f"\n[MODE: {mode.upper()}]")
            mode_dir = os.path.join(model_out_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            if mode == "static":
                X_train_pool, y_train_pool = X_base, y_base
                base_weights = None
            elif mode == "finetune":
                X_train_pool, y_train_pool = X_mean, y_mean
                base_weights = static_weights_path
            elif mode == "retrain":
                X_train_pool, y_train_pool = X_base_mean, y_base_mean
                base_weights = None
                
            # 1. Tune Hyperparameters - strictly 1 trial, 1 epoch
            best_params = tune_hyperparameters_tscv(
                model_type, input_dim, X_train_pool, y_train_pool, 
                n_splits=3, # Fewer splits to speed up test
                n_trials=1, # 1 Trial
                epochs=1,   # 1 Epoch
                patience=1,
                base_weights_path=base_weights
            )
            
            # Save params
            with open(os.path.join(mode_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=4)
                
            # 2. Final Training on Full Pool - strictly 1 epoch
            model, history = train_model_full(
                model_type, input_dim, X_train_pool, y_train_pool, 
                best_params, 
                epochs=1, # 1 Epoch
                batch_size=256, 
                base_weights_path=base_weights
            )
            
            # Plot Learning Curves
            plot_learning_curves(history, title=f'{model_type.upper()} {mode.capitalize()} Learning Curves', save_path=os.path.join(mode_dir, 'learning_curves.png'))
            
            # Save Final Weights (to ensure finetune can load it)
            weights_save_path = os.path.join(weights_dir, f"{model_type}_{mode}.pt" if model_type == "nn" else f"{model_type}_{mode}.npz")
            if model_type == "nn":
                torch.save(model.get_state_dict(), weights_save_path)
            elif model_type == "pcn":
                np.savez(weights_save_path, **model.get_state_dict())
                
            # 3. Evaluate
            probs = model.predict_proba(X_test)
            y_pred = (probs >= 0.5).astype(int)
            
            bias_m = evaluate_model_bias(y_test, y_pred, raw_test, dataset_name=f"{mode.upper()} {model_type.upper()}")
            player_m = evaluate_player_metrics(y_test, y_pred, raw_test, selected_players)
            
            # 4. Feature Importance - 1 repeat for speed
            print(f"Calculating Fast Feature Importance...")
            df_imp = calculate_permutation_importance(model, X_test, y_test, feature_names, n_repeats=1)
            df_imp.to_csv(os.path.join(mode_dir, 'feature_importance.csv'), index=False)
            plot_feature_importance(df_imp, title=f'{model_type.upper()} {mode.capitalize()} Feature Importance', save_path=os.path.join(mode_dir, 'feature_importance.png'))
            
            all_results[model_type][mode] = {
                "bias_metrics": bias_m,
                "player_metrics": player_m
            }
            print(f"--> Test completed for {model_type.upper()} {mode.upper()}")

    # Finalize
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        
    print(f"\nGenerating Final Visualizations...")
    plot_final_metrics_bar(all_results, run_dir)
    print(f"All outputs saved to: {run_dir}")
    
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal

    print("\n[SUCCESS] Test pipeline finished successfully without errors.")

if __name__ == "__main__":
    run_test_benchmark()
