import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import datetime
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.continual_learning_benchmark.data_splitter import BenchmarkDataPipeline, get_benchmark_splits
from src.continual_learning_benchmark.player_categorizer import categorize_players
from src.continual_learning_benchmark.evaluator import evaluate_model_bias, evaluate_player_metrics
from src.continual_learning_benchmark.models_setup import (
    tune_hyperparameters_tscv, 
    find_optimal_epochs_tscv, 
    train_model_full, 
    train_online_stream
)
from src.continual_learning_benchmark.feature_importance import calculate_permutation_importance, plot_feature_importance
from src.continual_learning_benchmark.utils import TeeLogger

###
TRIALS = 20 # 20
EPOCHS = 100 # 100
PLAYERS = 5
PATIENCE = 10
###

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
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

def plot_streaming_accuracy(rolling_history, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_history['cumulative_accuracy'], label='Cumulative Accuracy')
    ra = pd.Series(rolling_history['rolling_accuracy'])
    plt.plot(ra.rolling(10, min_periods=1).mean(), label='Rolling Batch Acc (EMA 10)', alpha=0.7)
    plt.title(title)
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_final_metrics_bar(all_results, save_dir):
    for model_type, modes in all_results.items():
        modes_list = list(modes.keys())
        accs = [modes[m]['bias_metrics']['final_accuracy'] for m in modes_list if 'bias_metrics' in modes[m]]
        
        plt.figure(figsize=(10, 6))
        plt.bar(modes_list, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(modes_list)])
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

def run_benchmark():
    parser = argparse.ArgumentParser(description="Continual Learning Benchmark")
    parser.add_argument("--run_all", action="store_true", help="Run all modes")
    parser.add_argument("--run_static", action="store_true", help="Run Static mode")
    parser.add_argument("--run_finetune", action="store_true", help="Run Finetune mode")
    parser.add_argument("--run_retrain", action="store_true", help="Run Retrain mode")
    parser.add_argument("--run_online", action="store_true", help="Run Online and Ultimate Streaming modes")
    args = parser.parse_args()

    if not (args.run_all or args.run_static or args.run_finetune or args.run_retrain or args.run_online):
        args.run_all = True

    modes_to_run = []
    if args.run_all or args.run_static: modes_to_run.append("static")
    if args.run_all or args.run_finetune: modes_to_run.append("finetune")
    if args.run_all or args.run_retrain: modes_to_run.append("retrain")
    if args.run_all or args.run_online: 
        modes_to_run.append("online")
        modes_to_run.append("ultimate_streaming")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "outputs", "optuna_benchmark", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "run.log")
    sys.stdout = TeeLogger(log_file)
    
    print(f"==================================================")
    print(f" CONTINUAL LEARNING BENCHMARK V2 STARTED")
    print(f" Modes: {modes_to_run}")
    print(f" Run Directory: {run_dir}")
    print(f"==================================================\n")

    print("--- 1. Loading & Processing Data ---")
    pipeline = BenchmarkDataPipeline()
    data, train_split_idx, pre_2025_count = pipeline.run()
    
    print("--- 2. Identifying Target Players ---")
    selected_players, _ = categorize_players(data, num_players_per_category=PLAYERS)
    
    print("--- 3. Splitting Benchmark Data ---")
    D_Train_Base, D_Mean, D_Test, D_Holdout = get_benchmark_splits(data, train_split_idx, pre_2025_count)
    
    X_base, y_base, raw_base, feature_names = prep_xy(D_Train_Base)
    X_mean, y_mean, raw_mean, _ = prep_xy(D_Mean)
    X_test, y_test, raw_test, _ = prep_xy(D_Test)
    
    input_dim = X_base.shape[1]
    
    scaler = StandardScaler()
    X_base = scaler.fit_transform(X_base)
    X_mean = scaler.transform(X_mean)
    X_test = scaler.transform(X_test)
    X_base_mean = np.vstack([X_base, X_mean])
    y_base_mean = np.concatenate([y_base, y_mean])

    models_to_run = ["nn", "pcn"]
    all_results = {}

    for model_type in models_to_run:
        all_results[model_type] = {}
        print(f"\n====================== Evaluating Model: {model_type.upper()} ======================")
        model_out_dir = os.path.join(run_dir, model_type)
        os.makedirs(model_out_dir, exist_ok=True)
        
        static_weights_path = os.path.join(weights_dir, f"{model_type}_static.pt" if model_type == "nn" else f"{model_type}_static.npz")
        online_weights_path = os.path.join(weights_dir, f"{model_type}_online.pt" if model_type == "nn" else f"{model_type}_online.npz")
        
        # We must tune on static first to get best_params
        print(f"\n[PHASE: TUNING {model_type.upper()}]")
        best_params = tune_hyperparameters_tscv(
            model_type, input_dim, X_base, y_base, 
            n_splits=5, n_trials=TRIALS, max_epochs=EPOCHS, patience=PATIENCE
        )
        
        with open(os.path.join(model_out_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=4)
            
        for mode in modes_to_run:
            print(f"\n[MODE: {mode.upper()}]")
            mode_dir = os.path.join(model_out_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            if mode == "static":
                X_train_pool, y_train_pool = X_base, y_base
                base_weights = None
                opt_epochs = best_params.get("optimal_epochs", 50)
                model, history = train_model_full(model_type, input_dim, X_train_pool, y_train_pool, best_params, epochs=opt_epochs, batch_size=64, base_weights_path=base_weights)
                plot_learning_curves(history, f'{model_type.upper()} STATIC Learning Curves', os.path.join(mode_dir, 'learning_curves.png'))
                
                # Save
                if model_type == "nn": torch.save(model.get_state_dict(), static_weights_path)
                elif model_type == "pcn": np.savez(static_weights_path, **model.get_state_dict())
                
                probs = model.predict_proba(X_test)
                
            elif mode == "finetune":
                X_train_pool, y_train_pool = X_mean, y_mean
                base_weights = static_weights_path
                # TSCV Early Stopping to find optimal epochs on Finetune pool
                opt_epochs = find_optimal_epochs_tscv(model_type, input_dim, X_train_pool, y_train_pool, best_params, n_splits=5, max_epochs=EPOCHS, patience=PATIENCE, base_weights_path=base_weights)
                model, history = train_model_full(model_type, input_dim, X_train_pool, y_train_pool, best_params, epochs=opt_epochs, batch_size=64, base_weights_path=base_weights)
                plot_learning_curves(history, f'{model_type.upper()} FINETUNE Learning Curves', os.path.join(mode_dir, 'learning_curves.png'))
                probs = model.predict_proba(X_test)
                
            elif mode == "retrain":
                X_train_pool, y_train_pool = X_base_mean, y_base_mean
                base_weights = None
                opt_epochs = find_optimal_epochs_tscv(model_type, input_dim, X_train_pool, y_train_pool, best_params, n_splits=5, max_epochs=EPOCHS, patience=PATIENCE, base_weights_path=base_weights)
                model, history = train_model_full(model_type, input_dim, X_train_pool, y_train_pool, best_params, epochs=opt_epochs, batch_size=64, base_weights_path=base_weights)
                plot_learning_curves(history, f'{model_type.upper()} RETRAIN Learning Curves', os.path.join(mode_dir, 'learning_curves.png'))
                probs = model.predict_proba(X_test)
                
            elif mode == "online":
                base_weights = static_weights_path
                # Train and Test interleaved
                model, rolling_history = train_online_stream(model_type, input_dim, X_mean, y_mean, best_params, batch_size=50, base_weights_path=base_weights)
                plot_streaming_accuracy(rolling_history, f'{model_type.upper()} ONLINE Streaming Acc', os.path.join(mode_dir, 'streaming_acc.png'))
                
                # Save online weights
                if model_type == "nn": torch.save(model.get_state_dict(), online_weights_path)
                elif model_type == "pcn": np.savez(online_weights_path, **model.get_state_dict())
                
                # Evaluate frozen weights on X_test for fair comparison
                probs = model.predict_proba(X_test)
                
            elif mode == "ultimate_streaming":
                base_weights = online_weights_path
                # Stream through X_test
                model, rolling_history = train_online_stream(model_type, input_dim, X_test, y_test, best_params, batch_size=50, base_weights_path=base_weights)
                plot_streaming_accuracy(rolling_history, f'{model_type.upper()} ULTIMATE Streaming Acc (Test Set)', os.path.join(mode_dir, 'streaming_acc.png'))
                
                # We extract the predicted labels from the rolling history
                probs = np.array(rolling_history['rolling_accuracy']) # This doesn't matter, we only care about metrics. 
                # Actually, wait, evaluate_model_bias expects probabilities to compute y_pred.
                # Ultimate streaming doesn't use static probs on X_test. We must intercept the predictions.
                # So we can't just pass `probs`.
                # We will just evaluate differently or mock the probs for the ultimate streaming case.
                
                # Since we already evaluated it in train_online_stream, let's just log the final acc.
                all_results[model_type][mode] = {
                    "bias_metrics": {"final_accuracy": rolling_history['cumulative_accuracy'][-1] * 100},
                    "player_metrics": {}
                }
                print(f"Ultimate Streaming X_test Cumulative Accuracy: {rolling_history['cumulative_accuracy'][-1]:.4f}")
                continue # Skip standard evaluation since it's already evaluated during stream
                
            # Standard Evaluation
            y_pred = (probs >= 0.5).astype(int)
            bias_m = evaluate_model_bias(y_test, y_pred, raw_test, dataset_name=f"{mode.upper()} {model_type.upper()}")
            player_m = evaluate_player_metrics(y_test, y_pred, raw_test, selected_players)
            
            print(f"Calculating Permutation Feature Importance for {model_type.upper()} {mode.upper()}...")
            df_imp = calculate_permutation_importance(model, X_test, y_test, feature_names, n_repeats=3)
            df_imp.to_csv(os.path.join(mode_dir, 'feature_importance.csv'), index=False)
            plot_feature_importance(df_imp, title=f'{model_type.upper()} {mode.capitalize()} Feature Importance', save_path=os.path.join(mode_dir, 'feature_importance.png'))
            
            all_results[model_type][mode] = {
                "bias_metrics": bias_m,
                "player_metrics": player_m
            }
            
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        
    print(f"\nGenerating Final Visualizations...")
    plot_final_metrics_bar(all_results, run_dir)
    print(f"All outputs saved to: {run_dir}")
    
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    run_benchmark()
