import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.continual_learning_benchmark.data_splitter import BenchmarkDataPipeline
from src.continual_learning_benchmark.player_categorizer import categorize_players
from src.continual_learning_benchmark.evaluator import evaluate_model_bias, evaluate_player_metrics
from src.continual_learning_benchmark.models_setup import get_nn_model, get_pcn_model, train_nn_full
from src.continual_learning_benchmark.utils import TeeLogger, plot_benchmark_results
import datetime
import json
import torch
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def train_pcn_full(wrapper, X: np.ndarray, y: np.ndarray, epochs=5, batch_size=64):
    """
    Helper to train PCN from scratch over a full dataset.
    """
    print(f"Training PCN from scratch for {epochs} epochs on {len(X)} samples...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        energies = []
        for i in range(0, len(X), batch_size):
            X_b = X_shuf[i:i+batch_size]
            y_b = y_shuf[i:i+batch_size]
            energy = wrapper.model.train_on_batch(X_b, y_b)
            energies.append(energy)
            
        print(f" Epoch {epoch+1}/{epochs} - Energy: {np.mean(energies):.4f}")
    return wrapper

def main():
    parser = argparse.ArgumentParser(description="Continual Learning Benchmark")
    parser.add_argument("--run_static", action="store_true", help="Run Static mode")
    parser.add_argument("--run_retrain", action="store_true", help="Run Retrain mode")
    parser.add_argument("--run_online", action="store_true", help="Run Online mode")
    parser.add_argument("--run_all", action="store_true", help="Run all modes")
    parser.add_argument("--model", type=str, choices=["nn", "pcn", "both"], default="nn", help="Model to evaluate")
    parser.add_argument("--pcn_dir", type=str, default="", help="Directory containing PCN config and weights")
    parser.add_argument("--nn_weights", type=str, default="", help="Optional NN pre-trained weights")
    
    args = parser.parse_args()
    
    if args.run_all:
        args.run_static = args.run_retrain = args.run_online = True
        
    if not (args.run_static or args.run_retrain or args.run_online):
        print("No mode selected. Provide at least one of: --run_static, --run_retrain, --run_online, or --run_all")
        return

    # Setup Logging & Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "outputs", "benchmark", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "run.log")
    sys.stdout = TeeLogger(log_file)
    
    print(f"==================================================")
    print(f" CONTINUAL LEARNING BENCHMARK STARTED")
    print(f" Run Directory: {run_dir}")
    print(f"==================================================\n")

    all_results = {}

    print("--- 1. Loading & Processing Data ---")
    pipeline = BenchmarkDataPipeline()
    data, train_split_idx, pre_2024_count = pipeline.run()
    
    print("--- 2. Identifying Target Players ---")
    selected_players, _ = categorize_players(data, num_players_per_category=3)
    
    print("--- 3. Splitting Benchmark Data ---")
    from src.continual_learning_benchmark.data_splitter import get_benchmark_splits
    D_Base, D_Stream, D_Test, D_Holdout = get_benchmark_splits(data, train_split_idx, pre_2024_count, selected_players)
    
    # Separate features and metadata
    def prep_xy(df):
        df_clean = df.copy()
        raw = df.copy()
        
        # Remove metadata columns
        drop_cols = ['target', 'year', 'is_augmented', 'winner_name', 'loser_name', 'winner_ioc', 'loser_ioc', 'score', 'tourney_date', 'winner_elo', 'loser_elo']
        X = df_clean.drop(columns=drop_cols, errors='ignore').fillna(0).values
        y = df_clean['target'].values if 'target' in df_clean.columns else np.zeros(len(df_clean))
        return X, y, raw
        
    X_base, y_base, raw_base = prep_xy(D_Base)
    X_stream, y_stream, raw_stream = prep_xy(D_Stream)
    X_test, y_test, raw_test = prep_xy(D_Test)
    
    input_dim = X_base.shape[1]
    
    models_to_run = ["nn", "pcn"] if args.model == "both" else [args.model]
    
    pc_model_path = os.path.join(args.pcn_dir, "pc_model.npz") if args.pcn_dir else ""
    pc_config_path = os.path.join(args.pcn_dir, "pc_config.json") if args.pcn_dir else ""
    
    # Scale Data
    # In a real setup, scale based on D_Base only.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_base = scaler.fit_transform(X_base)
    X_stream = scaler.transform(X_stream)
    X_test = scaler.transform(X_test)

    # Prepare merged sets for Retrain
    X_base_stream = np.vstack([X_base, X_stream])
    y_base_stream = np.concatenate([y_base, y_stream])
    
    for model_type in models_to_run:
        all_results[model_type] = {}
        print(f"\n====================== Evaluating Model: {model_type.upper()} ======================")
        
        if args.run_static:
            print("\n[MODE: STATIC]")
            if model_type == "nn":
                model = get_nn_model(input_dim, mode="static", external_weights_path=args.nn_weights)
                if not args.nn_weights:
                    model = train_nn_full(model, X_base, y_base, epochs=3)
                
            elif model_type == "pcn":
                model = get_pcn_model(input_dim, mode="static", model_path=pc_model_path, config_path=pc_config_path)
            
            y_pred = (model.predict_proba(X_test) >= 0.5).astype(int)
            bias_m = evaluate_model_bias(y_test, y_pred, raw_test, dataset_name=f"Static {model_type.upper()}")
            player_m = evaluate_player_metrics(y_test, y_pred, raw_test, selected_players)
            
            all_results[model_type]["static"] = {
                "bias_metrics": bias_m,
                "player_metrics": player_m
            }
            
        if args.run_retrain:
            print("\n[MODE: RETRAIN]")
            if model_type == "nn":
                model = get_nn_model(input_dim, mode="retrain", external_weights_path=None) # Always from scratch
                model = train_nn_full(model, X_base_stream, y_base_stream, epochs=3)
            elif model_type == "pcn":
                model = get_pcn_model(input_dim, mode="retrain", model_path="", config_path=pc_config_path)
                model = train_pcn_full(model, X_base_stream, y_base_stream, epochs=3)
                
            y_pred = (model.predict_proba(X_test) >= 0.5).astype(int)
            bias_m = evaluate_model_bias(y_test, y_pred, raw_test, dataset_name=f"Retrain {model_type.upper()}")
            player_m = evaluate_player_metrics(y_test, y_pred, raw_test, selected_players)
            
            all_results[model_type]["retrain"] = {
                "bias_metrics": bias_m,
                "player_metrics": player_m
            }
            
            # Save weights
            if model_type == "nn":
                torch.save(model.model.state_dict(), os.path.join(weights_dir, f"{model_type}_retrain_weights.pt"))
            elif model_type == "pcn":
                np.savez(os.path.join(weights_dir, f"{model_type}_retrain_weights.npz"), **model.model.get_state_dict())
            
        if args.run_online:
            print("\n[MODE: ONLINE]")
            # Initialize with base model
            if model_type == "nn":
                model = get_nn_model(input_dim, mode="online", external_weights_path=args.nn_weights)
                if not args.nn_weights:
                    model = train_nn_full(model, X_base, y_base, epochs=3)
            elif model_type == "pcn":
                model = get_pcn_model(input_dim, mode="online", model_path=pc_model_path, config_path=pc_config_path)
                
            # Online Learning on D_Stream
            print(f"Streaming {len(X_stream)} samples for continuous updates...")
            for i in range(len(X_stream)):
                # Update sequentially
                model.online_train_step(X_stream[i:i+1], y_stream[i:i+1])
                if i % 1000 == 0 and i > 0:
                    print(f" Processed {i}/{len(X_stream)} stream samples...")
                    
            # Predict on D_Test
            y_pred = (model.predict_proba(X_test) >= 0.5).astype(int)
            bias_m = evaluate_model_bias(y_test, y_pred, raw_test, dataset_name=f"Online {model_type.upper()}")
            player_m = evaluate_player_metrics(y_test, y_pred, raw_test, selected_players)
            
            all_results[model_type]["online"] = {
                "bias_metrics": bias_m,
                "player_metrics": player_m
            }
            
            # Save weights
            if model_type == "nn":
                torch.save(model.model.state_dict(), os.path.join(weights_dir, f"{model_type}_online_weights.pt"))
            elif model_type == "pcn":
                np.savez(os.path.join(weights_dir, f"{model_type}_online_weights.npz"), **model.model.get_state_dict())

    # Finalize: Save Metrics & Visualize
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
        
    print(f"\nGenerating Visualizations...")
    plot_benchmark_results(all_results, run_dir)
    print(f"All outputs saved to: {run_dir}")
    
    # Restore stdout
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()
