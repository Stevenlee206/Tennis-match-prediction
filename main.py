import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import Preprocessing
import joblib
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

def evaluate_model_bias(y_true, y_pred, X_raw, dataset_name=""):
    """
    Calculates bias metrics and returns them as a dictionary for JSON logging.
    """
    print("\n" + "="*50)
    print(f" MODEL BIAS & HEURISTIC ANALYSIS {dataset_name}")
    print("="*50)

    print("CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("-" * 50)

    metrics = {}

    # 1. Target Class Bias
    pred_p1_rate = np.mean(y_pred == 1) * 100
    metrics['class_1_prediction_rate'] = round(pred_p1_rate, 2)
    print(f"Class 1 Prediction Rate:   {pred_p1_rate:.2f}% (Ideal: ~50.0%)")

    # 2. Elo Analysis
    if 'elo_diff' in X_raw.columns:
        higher_elo_p1 = (X_raw['elo_diff'] > 0).astype(int)
        
        elo_reliance = np.mean(y_pred == higher_elo_p1) * 100
        elo_baseline_acc = np.mean(y_true == higher_elo_p1) * 100
        
        metrics['elo_reliance'] = round(elo_reliance, 2)
        metrics['elo_baseline_accuracy'] = round(elo_baseline_acc, 2)
        
        print(f"Elo Reliance (Safe Bet):   {elo_reliance:.2f}%")
        print(f"Blind Elo Baseline Acc:    {elo_baseline_acc:.2f}%")

        actual_upsets_mask = (y_true != higher_elo_p1)
        if actual_upsets_mask.sum() > 0:
            upset_acc = accuracy_score(y_true[actual_upsets_mask], y_pred[actual_upsets_mask]) * 100
            metrics['upset_prediction_accuracy'] = round(upset_acc, 2)
            print(f"Upset Prediction Accuracy: {upset_acc:.2f}% (Correctly guessing the underdog)")

    # 3. ATP Rank Analysis
    if 'rank_diff' in X_raw.columns:
        better_rank_p1 = (X_raw['rank_diff'] < 0).astype(int)
        rank_reliance = np.mean(y_pred == better_rank_p1) * 100
        metrics['rank_reliance'] = round(rank_reliance, 2)
        print(f"Rank Reliance Bias:        {rank_reliance:.2f}%")
        
    final_acc = accuracy_score(y_true, y_pred) * 100
    metrics['final_accuracy'] = round(final_acc, 2)
    print(f"\nFinal Set Accuracy:        {final_acc:.2f}%")
    print("==================================================\n")
    
    return metrics

def append_metrics_to_config(config_path, metrics):
    """Safely appends bias metrics to the existing JSON config file."""
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        config['bias_metrics'] = metrics
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[*] Bias metrics appended to {config_path.name}")

def main():
    parser = argparse.ArgumentParser(description="ATP Tennis Match Prediction Pipeline")
    
    # --- ADD MODEL FLAG ---
    parser.add_argument("--model", type=str, choices=["svm", "rf", "pytorch_svm", "tabnet", "deepforest", "pytorch_mlp", "predictive_coding"], default="svm", help="Algorithm to use")    
    # --- ADDED PYTORCH FLAGS ---
    parser.add_argument("--torch_opt", type=str, choices=["adam", "rmsprop", "sgd", "sgd_nesterov"], default="adam", help="Optimizer for PyTorch SVM")
    parser.add_argument("--torch_sched", type=str, choices=["constant", "cosine", "step", "plateau"], default="cosine", help="Learning rate scheduler for PyTorch SVM")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for PyTorch DataLoaders")
    parser.add_argument("--rf_variant", type=str, choices=["rf", "extra_trees", "rrf", "rotation_forest", "oblique", "weighted"], default="rf", help="Specific RF variant (for Optuna)")
    parser.add_argument("--add_pca", action="store_true", help="Apply PCA for dimensionality reduction and denoising before passing to model")
    parser.add_argument("--mode", type=str, choices=["standard", "sgd"], default="standard")
    parser.add_argument("--optimizer", type=str, choices=["optuna", "pso", "ga", "grid"], default="optuna", help="Optimizer to tune hyperparameters")
    parser.add_argument("--validation", type=str, choices=["holdout", "walk_forward"], default="holdout")
    parser.add_argument("--kernel", type=str, choices=["linear", "poly", "rbf"], default="linear", help="SVM Kernel to use (Ignored if model=rf)")
    parser.add_argument("--lr_schedule", type=str, choices=["adaptive", "optimal", "invscaling", "constant"], default="adaptive", help="Learning rate schedule for SGD mode.")
    
    # Optimizer Params
    parser.add_argument("--test_size", type=float, default=0.10, help="Global test set ratio")
    parser.add_argument("--val_size", type=float, default=0.20, help="Validation set ratio (for holdout)")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of TimeSeries CV splits for walk-forward")
    parser.add_argument("--n_trials", type=int, default=30, help="Used for Optuna")
    parser.add_argument("--particles", type=int, default=15, help="Used for PSO")
    parser.add_argument("--iterations", type=int, default=20, help="Used for PSO")
    parser.add_argument("--population", type=int, default=20, help="Used for GA")
    parser.add_argument("--generations", type=int, default=15, help="Used for GA")
    parser.add_argument("--epochs", type=int, default=100, help="Used for SGD / PyTorch / TabNet mode")
    
    # Grid Search Params (SVM)
    parser.add_argument("--c_min", type=float, default=1e-3, help="Min C (SVM)")
    parser.add_argument("--c_max", type=float, default=1e2, help="Max C (SVM)")
    parser.add_argument("--c_steps", type=int, default=10, help="Grid steps (SVM)")

    # Hyperparameter Search Bounds (RF)
    parser.add_argument("--rf_n_est_min", type=int, default=50, help="Min n_estimators (RF)")
    parser.add_argument("--rf_n_est_max", type=int, default=500, help="Max n_estimators (RF)")
    parser.add_argument("--rf_n_est_steps", type=int, default=5, help="Grid steps for n_estimators (RF Grid Search)")
    parser.add_argument("--rf_depth_min", type=int, default=5, help="Min max_depth (RF)")
    parser.add_argument("--rf_depth_max", type=int, default=50, help="Max max_depth (RF)")
    parser.add_argument("--rf_depth_steps", type=int, default=5, help="Grid steps for max_depth (RF Grid Search)")
    # Hyperparameter Search Bounds (Deep Forest)
    parser.add_argument("--df_max_layers_min", type=int, default=2, help="Min max_layers (Deep Forest)")
    parser.add_argument("--df_max_layers_max", type=int, default=8, help="Max max_layers (Deep Forest)")
    parser.add_argument("--df_n_trees_min", type=int, default=50, help="Min n_trees per estimator (Deep Forest)")
    parser.add_argument("--df_n_trees_max", type=int, default=200, help="Max n_trees per estimator (Deep Forest)")
    parser.add_argument("--df_depth_min", type=int, default=5, help="Min max_depth (Deep Forest)")
    parser.add_argument("--df_depth_max", type=int, default=30, help="Max max_depth (Deep Forest)")
    # Weights Strategy
    parser.add_argument("--weight_strategy", type=str, choices=["none", "static", "magnitude", "temporal"], default="none", help="Which sample weighting schedule to use.")
    parser.add_argument("--upset_weight", type=float, default=2.0, help="The base penalty multiplier for upsets (Used if strategy is not 'none').")
    parser.add_argument("--config", type=str, default="", help="Path to JSON config file to override arguments")
    
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only evaluate on the test set.")
    parser.add_argument("--weights_dir", type=str, default="", help="Directory containing the pre-trained weights for evaluation.")
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    # Safety Checks
    if args.mode == "sgd" and args.model != "svm":
        raise ValueError("Error: SGD mode is currently only implemented for SVM.")
    if args.mode == "sgd" and args.kernel != "linear":
        raise ValueError("Error: SGD mode only supports the 'linear' kernel.")

    print("==============================================")
    print(f" ATP Tennis Prediction")
    print(f" Model: {args.model.upper()} | Mode: {args.mode.upper()} | Optimizer: {args.optimizer.upper()} | Val: {args.validation.upper()}")
    print("==============================================\n")

    print("--- Step 1: Preprocessing Data ---")
    precomputed_folds = None
    if args.validation == "walk_forward" and args.model == "predictive_coding":
        print(f"Pre-computing {args.n_splits} folds to prevent Data Leakage...")
        train_ratios = [0.4, 0.5, 0.6, 0.7, 0.8]
        val_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
        precomputed_folds = []
        for t_ratio, v_ratio in zip(train_ratios, val_ratios):
            prep_fold = Preprocessing()
            data_fold = prep_fold.run(train_ratio=t_ratio)
            idx_train = int(len(data_fold) * t_ratio)
            idx_val = int(len(data_fold) * v_ratio)
            
            train_df = data_fold.iloc[:idx_train].copy()
            val_df = data_fold.iloc[idx_train:idx_val].copy()
            
            precomputed_folds.append((train_df, val_df))
        
        train_ratio_final = 1.0 - args.test_size
    else:
        train_ratio_final = (1.0 - args.test_size) * (1.0 - args.val_size) if args.validation == "holdout" else (1.0 - args.test_size)
        
    prep = Preprocessing()
    data = prep.run(train_ratio=train_ratio_final)
    
    if 'target' not in data.columns:
        raise ValueError("Error: 'target' missing.")

    # ==========================================
    # GLOBAL TEST SET EXTRACTION (QUARANTINE)
    # ==========================================
    X_full = data.drop(columns=['target', 'year'], errors='ignore')
    y_full = data['target']

    X_train_val_pool, X_test, y_train_val_pool, y_test = train_test_split(
        X_full, y_full, test_size=args.test_size, shuffle=False
    )
    
    # --- CLEANUP GLOBAL TEST SET ---
    if 'is_augmented' in X_test.columns:
        y_test = y_test[X_test['is_augmented'] == 0]
        X_test = X_test[X_test['is_augmented'] == 0]
    
    # Drop string columns before evaluation
    X_test = X_test.drop(columns=['is_augmented', 'winner_name', 'loser_name', 'match_id'], errors='ignore')
        
    print(f"\nGlobal Splits -> Modeling Pool: {len(X_train_val_pool)} | Quarantined Test: {len(X_test)}")
    BASE_DIR = Path(__file__).resolve().parent
    
    # --- DYNAMIC PATH ROUTING ---
    if args.model == "svm":
        model_subpath = f"sklearn/svm/{args.kernel}"
    elif args.model == "rf":
        model_subpath = f"sklearn/rf/{args.rf_variant}"
    elif args.model == "pytorch_svm":
        model_subpath = "pytorch_svm" # Sibling to sklearn
    elif args.model == "pytorch_mlp":
        model_subpath = "pytorch_mlp" # <--- ADD THIS
    elif args.model == "tabnet":
        model_subpath = "tabnet"      # Sibling to sklearn
    elif args.model == "deepforest":
        model_subpath = "deepforest"
    elif args.model == "predictive_coding":
        model_subpath = "predictive_coding"
    
    run_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    BASE_OUT = BASE_DIR / "outputs" / model_subpath / args.mode / args.optimizer / args.validation / run_timestamp
    BASE_REP = BASE_DIR / "reports" / "figures" / model_subpath / args.mode / args.optimizer / args.validation / run_timestamp

    # --- DYNAMIC IMPORTS ---
    if args.model == "pytorch_svm":
        from src.models.svm.svm_pytorch_optuna import run_pytorch_pipeline as run_pipeline
    elif args.model == "pytorch_mlp": # <--- ADD THIS
        from src.models.mlp.mlp_pytorch_optuna import run_pytorch_mlp_pipeline as run_pipeline
    elif args.model == "deepforest":
        from src.models.rf.deepforest_optuna import run_deepforest_pipeline as run_pipeline
    elif args.model == "predictive_coding":
        from src.models.pc.pc_optuna import run_pc_pipeline as run_pipeline
    elif args.model == "svm":
        if args.optimizer == "pso":
            if args.mode == "sgd": raise NotImplementedError("PSO is for 'standard' mode SVM.")
            from src.models.svm.svm_sklearn_pso import run_svm_pipeline as run_pipeline
        elif args.optimizer == "ga":
            if args.mode == "sgd": raise NotImplementedError("GA is for 'standard' mode SVM.")
            from src.models.svm.svm_sklearn_ga import run_svm_pipeline as run_pipeline
        elif args.optimizer == "grid":
            if args.mode == "sgd": raise NotImplementedError("Grid Search is for 'standard' mode SVM.")
            from src.models.svm.svm_sklearn_grid import run_svm_pipeline as run_pipeline
        else: # Optuna
            if args.mode == "standard":
                from src.models.svm.svm_sklearn_optuna import run_svm_pipeline as run_pipeline
            else:
                from src.models.svm.svm_sklearn_SGD import run_svm_pipeline as run_pipeline
    elif args.model == "rf":
        if args.optimizer == "pso":
            from src.models.rf.rf_sklearn_pso import run_rf_pipeline as run_pipeline
        elif args.optimizer == "ga":
            from src.models.rf.rf_sklearn_ga import run_rf_pipeline as run_pipeline
        elif args.optimizer == "grid":
            from src.models.rf.rf_sklearn_grid import run_rf_pipeline as run_pipeline
        else: # Optuna
            from src.models.rf.rf_sklearn_optuna import run_rf_pipeline as run_pipeline

    # ==========================================
    # ROUTE A: STANDARD HOLDOUT VALIDATION
    # ==========================================
    if args.validation == "holdout":
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val_pool, y_train_val_pool, test_size=0.2, shuffle=False
        )

        # --- PURGE AUGMENTED DATA FROM EVALUATION ---
        if 'is_augmented' in X_train.columns:
            # 1. Filter Validation Set strictly to real historical matches
            y_val = y_val[X_val['is_augmented'] == 0]
            X_val = X_val[X_val['is_augmented'] == 0]

            # 2. Drop the flag so models don't train on it (unless it's predictive_coding, we drop inside pipeline to keep val chunks clean)
            if args.model != "predictive_coding":
                X_train = X_train.drop(columns=['is_augmented'])
                X_val = X_val.drop(columns=['is_augmented'])

        print(f"Holdout Splits -> Train: {len(X_train)} | Val: {len(X_val)}")
        
        # --- DYNAMIC PIPELINE EXECUTION ---
        if args.model == "svm":
            if args.optimizer == "pso":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_particles=args.particles, n_iterations=args.iterations, kernel=args.kernel)
            elif args.optimizer == "ga":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, pop_size=args.population, n_generations=args.generations, kernel=args.kernel)
            elif args.optimizer == "grid":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, c_min=args.c_min, c_max=args.c_max, c_steps=args.c_steps, kernel=args.kernel)
            else: # Optuna
                if args.mode == "standard":
                    run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                                 n_trials=args.n_trials, kernel=args.kernel, c_min=args.c_min, c_max=args.c_max,                         
                                 add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight)
                elif args.mode == "sgd":
                    run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                                 n_trials=args.n_trials, n_epochs=args.epochs, kernel="linear", c_min=args.c_min, c_max=args.c_max,                        
                                 add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight, lr_schedule=args.lr_schedule)
        
        elif args.model == "pytorch_svm":
            run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                         n_trials=args.n_trials, epochs=args.epochs, batch_size=args.batch_size, c_min=args.c_min, c_max=args.c_max,                         
                         add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         torch_opt=args.torch_opt, torch_sched=args.torch_sched)
        elif args.model == "pytorch_mlp":
            run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                         n_trials=args.n_trials, epochs=args.epochs, batch_size=args.batch_size, 
                         add_pca=args.add_pca, validation=args.validation, 
                         weight_strategy=args.weight_strategy, upset_weight=args.upset_weight)             
        elif args.model == "tabnet":
            run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                         n_trials=args.n_trials, epochs=args.epochs, batch_size=args.batch_size, add_pca=args.add_pca, validation=args.validation)
                         
        elif args.model == "deepforest":
            run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                         n_trials=args.n_trials, add_pca=args.add_pca, validation=args.validation, 
                         weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         max_layers_min=args.df_max_layers_min, max_layers_max=args.df_max_layers_max,
                         n_trees_min=args.df_n_trees_min, n_trees_max=args.df_n_trees_max,
                         max_depth_min=args.df_depth_min, max_depth_max=args.df_depth_max)

        elif args.model == "predictive_coding" and not args.eval_only:
            run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                         n_trials=args.n_trials, validation=args.validation, optimizer=args.optimizer,
                         weight_strategy=args.weight_strategy, upset_weight=args.upset_weight)

        elif args.model == "rf":
            if args.optimizer == "pso":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_particles=args.particles, n_iterations=args.iterations, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)
            elif args.optimizer == "ga":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, pop_size=args.population, n_generations=args.generations, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)
            elif args.optimizer == "grid":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, n_est_steps=args.rf_n_est_steps, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max, depth_steps=args.rf_depth_steps)
            else: # Optuna
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, 
                             n_trials=args.n_trials, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, 
                             depth_min=args.rf_depth_min, depth_max=args.rf_depth_max, variant=args.rf_variant,
                             add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight)

        print("\n--- Training and Tuning Complete ---")
        
        # ==========================================
        # RUN BIAS EVALUATION ON THE VAL SET
        # ==========================================
        print("Loading saved model to evaluate bias on Validation Set...")
        
        if args.model == "svm":
            if args.mode == "sgd":
                model_name, scaler_name, config_name = "svm_sgd_model.joblib", "svm_sgd_scaler.joblib", "svm_sgd_config.json"
            else:
                model_name, scaler_name, config_name = f"{args.kernel}_model.joblib", f"{args.kernel}_scaler.joblib", f"{args.kernel}_config.json"
        elif args.model == "pytorch_svm":
            model_name, scaler_name, config_name = "svm_pytorch_model.pth", "svm_pytorch_scaler.joblib", "svm_pytorch_config.json"
        elif args.model == "pytorch_mlp":
            model_name, scaler_name, config_name = "mlp_pytorch_model.pth", "mlp_pytorch_scaler.joblib", "mlp_pytorch_config.json"
        elif args.model == "deepforest":
            model_name, scaler_name, config_name = "deepforest_model.joblib", "deepforest_scaler.joblib", "deepforest_config.json"
        elif args.model == "predictive_coding":
            model_name, scaler_name, config_name = "pc_model.pt", "pc_scaler.joblib", "pc_config.json"
        else:
            model_name, scaler_name, config_name = f"{args.rf_variant}_model.joblib", f"{args.rf_variant}_scaler.joblib", f"{args.rf_variant}_config.json"
            
        model_path = Path(args.weights_dir) / model_name if args.weights_dir else BASE_OUT / model_name
        scaler_path = Path(args.weights_dir) / scaler_name if args.weights_dir else BASE_OUT / scaler_name
        config_path = Path(args.weights_dir) / config_name if args.weights_dir else BASE_OUT / config_name
        
        check_path = Path(str(model_path)) if args.model != "tabnet" else Path(str(model_path).replace('.zip', '') + '.zip')

        if check_path.exists() and scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_val_clean = X_val.drop(columns=['is_augmented', 'match_id', 'winner_name', 'loser_name'], errors='ignore')
            X_val_scaled = scaler.transform(X_val_clean)
            
            # Handle PCA if enabled
            if hasattr(args, 'add_pca') and args.add_pca:
                if args.model in ["pytorch_svm", "pytorch_mlp", "tabnet", "deepforest"]:
                    # Flips 'pytorch_svm' -> 'svm_pytorch' and 'pytorch_mlp' -> 'mlp_pytorch'
                    prefix = args.model.replace('pytorch_', '') + '_pytorch' if 'pytorch' in args.model else args.model
                    pca_name = f"{prefix}_pca.joblib"
                elif args.model == "predictive_coding":
                    pca_name = "pc_pca.joblib"
                elif args.model == "svm":
                    if args.mode == "sgd":
                        pca_name = "svm_sgd_pca.joblib"
                    else:
                        pca_name = f"{args.kernel}_pca.joblib"
                else:
                    pca_name = f"{args.rf_variant}_pca.joblib"
                    
                pca_path = BASE_OUT / pca_name
                if pca_path.exists():
                    pca = joblib.load(pca_path)
                    X_val_scaled = pca.transform(X_val_scaled)
                else:
                    print(f"⚠️ Warning: PCA enabled but {pca_name} not found at {pca_path}!")
            
            # Properly Route Model Loading & Prediction
            if args.model == "pytorch_svm":
                import torch
                from src.models.svm.svm_pytorch_optuna import PyTorchLinearSVM
                device = "cuda" if torch.cuda.is_available() else "cpu"
                best_model = PyTorchLinearSVM(X_val_scaled.shape[1]).to(device)
                best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                best_model.eval()
                with torch.no_grad():
                    preds_raw = best_model(torch.FloatTensor(X_val_scaled).to(device))
                    y_pred_val = (preds_raw > 0).cpu().numpy().astype(int)  
            elif args.model == "pytorch_mlp": # <--- ADD THIS BLOCK
                import torch
                from src.models.mlp.mlp_pytorch_optuna import DynamicMLP
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                best_model = DynamicMLP(X_val_scaled.shape[1], cfg['hidden_layers'], cfg['best_params']['dropout'], cfg['best_params']['activation']).to(device)
                best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                best_model.eval()
                with torch.no_grad():
                    preds_raw = best_model(torch.FloatTensor(X_val_scaled).to(device))
                    # MLP outputs logits. Sigmoid > 0.5 gets the binary class
                    y_pred_val = (torch.sigmoid(preds_raw) > 0.5).cpu().numpy().astype(int).flatten()              
            elif args.model == "predictive_coding":
                import torch
                from src.models.predictive_coding.pc_network_torch import PredictiveCodingNetworkTorch
                from src.models.predictive_coding.pc_network import PCNetworkConfig
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                
                pc_cfg = PCNetworkConfig(**cfg.get('best_params', {}))
                pc_cfg.output_activation = "identity"
                
                # layer sizes include in and out
                layer_sizes = [X_val_scaled.shape[1], *cfg['model_params']['hidden_sizes'], 1]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                best_model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=pc_cfg, device=device)
                
                # Load weights
                state = dict(np.load(model_path))
                best_model.load_state_dict(state)
                probs = best_model.predict_proba_torch(X_val_scaled).detach().cpu().numpy()
                y_pred_val = (probs >= 0.5).astype(int)
            else: # SKLearn / DeepForest models
                best_model = joblib.load(model_path)
                y_pred_val = best_model.predict(X_val_scaled)
            
            # Evaluate and Append to JSON
            bias_metrics = evaluate_model_bias(y_val.values, y_pred_val, X_val)
            append_metrics_to_config(config_path, bias_metrics)
        else:
            print(f"Could not find saved models in {BASE_OUT} for evaluation.")
    
    # ==========================================
    # ROUTE B: WALK-FORWARD VALIDATION (Global TSCV)
    # ==========================================
    elif args.validation == "walk_forward":
        print("\n" + "="*50)
        print(" RUNNING GLOBAL TIME-SERIES CROSS VALIDATION")
        print("="*50)

        # Prepare the entire dataset
        X_all = data.drop(columns=['target', 'year'], errors='ignore')
        y_all = data['target']
        
        # --- PREPARE POOL FOR TRAINING ---
        # We hide the flag from the model, but keep the rows for balanced training folds
        if 'is_augmented' in X_train_val_pool.columns:
            if args.model != "predictive_coding":
                X_train_val_pool = X_train_val_pool.drop(columns=['is_augmented'])
            
        print(f"Total Rows for TSCV: {len(X_all)}")
        
        global_out = BASE_OUT / f"global_tscv_{args.model}"
        global_rep = BASE_REP / f"global_tscv_{args.model}"

        # --- DYNAMIC PIPELINE EXECUTION ---
        if args.model == "svm":
            if args.optimizer == "optuna":
                if args.mode == "standard":
                    run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                                 n_trials=args.n_trials, kernel=args.kernel, c_min=args.c_min, c_max=args.c_max,                         
                                 add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight)
                elif args.mode == "sgd":
                    run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                                 n_trials=args.n_trials, n_epochs=args.epochs, kernel="linear", c_min=args.c_min, c_max=args.c_max,                        
                                 add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight, lr_schedule=args.lr_schedule)
        
        elif args.model == "pytorch_svm":
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                         n_trials=args.n_trials, epochs=args.epochs, batch_size=args.batch_size, c_min=args.c_min, c_max=args.c_max,                         
                         add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         torch_opt=args.torch_opt, torch_sched=args.torch_sched)                         
        elif args.model == "deepforest":
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                         n_trials=args.n_trials, add_pca=args.add_pca, validation=args.validation, 
                         weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         max_layers_min=args.df_max_layers_min, max_layers_max=args.df_max_layers_max,
                         n_trees_min=args.df_n_trees_min, n_trees_max=args.df_n_trees_max,
                         max_depth_min=args.df_depth_min, max_depth_max=args.df_depth_max)

        elif args.model == "predictive_coding" and not args.eval_only:
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                         n_trials=args.n_trials, validation=args.validation, optimizer=args.optimizer,
                         weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         precomputed_folds=precomputed_folds)

        elif args.model == "rf" and not args.eval_only:
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                          n_trials=args.n_trials, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, 
                          depth_min=args.rf_depth_min, depth_max=args.rf_depth_max, variant=args.rf_variant,
                          add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight)       

        # ==========================================
        # RUN BIAS EVALUATION ON LATEST MATCHES OR UNSEEN 10% TEST
        # ==========================================
        if args.model == "predictive_coding":
            print("\n" + "="*50)
            print(" FINAL EVALUATION ON UNSEEN TEST SET (90-10)")
            print("="*50)
            model_name, scaler_name, config_name = "pc_model.pt", "pc_scaler.joblib", "pc_config.json"
            model_path = Path(args.weights_dir) / model_name if args.weights_dir else global_out / model_name
            scaler_path = Path(args.weights_dir) / scaler_name if args.weights_dir else global_out / scaler_name
            config_path = Path(args.weights_dir) / config_name if args.weights_dir else global_out / config_name
            
            if model_path.exists() and scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X_test_scaled = scaler.transform(X_test)
                
                import torch
                from src.models.predictive_coding.pc_network_torch import PredictiveCodingNetworkTorch
                from src.models.predictive_coding.pc_network import PCNetworkConfig
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                
                pc_cfg = PCNetworkConfig(**cfg.get('best_params', {}))
                pc_cfg.output_activation = "identity"
                layer_sizes = [X_test_scaled.shape[1], *cfg['model_params']['hidden_sizes'], 1]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                best_model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=pc_cfg, device=device)
                
                if str(model_path).endswith('.npz'):
                    state = dict(np.load(model_path))
                else:
                    state = torch.load(model_path, map_location=device, weights_only=True)
                best_model.load_state_dict(state)
                probs = best_model.predict_proba_torch(X_test_scaled).detach().cpu().numpy()
                y_pred_test = (probs >= 0.5).astype(int)
                
                bias_metrics = evaluate_model_bias(y_test.values, y_pred_test, X_test, dataset_name="(UNSEEN TEST SET)")
                
                from src.models.utils.metrics import binary_classification_metrics
                final_test_metrics = binary_classification_metrics(y_test.values, probs)
                print("\nFINAL METRICS ON UNSEEN TEST SET:")
                for k, v in final_test_metrics.items():
                    print(f" - {k:>16}: {v:.4f}")
                
                bias_metrics.update(final_test_metrics)
                append_metrics_to_config(config_path, {"final_test_eval": bias_metrics})
            else:
                print(f"Could not find saved PC models in {global_out} for evaluation.")
            
            return

        print("Loading saved model to evaluate bias on the most recent chronological Validation chunk...")
        
        if args.model == "svm":
            if args.mode == "sgd":
                model_name, scaler_name, config_name = "svm_sgd_model.joblib", "svm_sgd_scaler.joblib", "svm_sgd_config.json"
            else:
                model_name, scaler_name, config_name = f"{args.kernel}_model.joblib", f"{args.kernel}_scaler.joblib", f"{args.kernel}_config.json"
        elif args.model == "pytorch_svm":
            model_name, scaler_name, config_name = "svm_pytorch_model.pth", "svm_pytorch_scaler.joblib", "svm_pytorch_config.json"
        elif args.model == "deepforest":
            model_name, scaler_name, config_name = "deepforest_model.joblib", "deepforest_scaler.joblib", "deepforest_config.json"
        elif args.model == "predictive_coding":
            model_name, scaler_name, config_name = "pc_model.pt", "pc_scaler.joblib", "pc_config.json"
        else:
            model_name, scaler_name, config_name = f"{args.rf_variant}_model.joblib", f"{args.rf_variant}_scaler.joblib", f"{args.rf_variant}_config.json"
            
        model_path = Path(args.weights_dir) / model_name if args.weights_dir else global_out / model_name
        scaler_path = Path(args.weights_dir) / scaler_name if args.weights_dir else global_out / scaler_name
        config_path = Path(args.weights_dir) / config_name if args.weights_dir else global_out / config_name
        
        check_path = Path(str(model_path)) if args.model != "tabnet" else Path(str(model_path).replace('.zip', '') + '.zip')

        if check_path.exists() and scaler_path.exists():
            scaler = joblib.load(scaler_path)
            
            # Take the last 10% of data (chronologically most recent) as the VAL set
            split_idx = int(len(X_all) * 0.9)
            X_val_chunk = X_all.iloc[split_idx:].copy()
            y_val_chunk = y_all.iloc[split_idx:].copy()
            
            # --- PURGE AUGMENTED DATA FROM EVALUATION ---
            if 'is_augmented' in X_val_chunk.columns:
                y_val_chunk = y_val_chunk[X_val_chunk['is_augmented'] == 0]
                X_val_chunk = X_val_chunk[X_val_chunk['is_augmented'] == 0]
                
            # Drop string columns before evaluation
            X_val_chunk = X_val_chunk.drop(columns=['is_augmented', 'winner_name', 'loser_name', 'match_id'], errors='ignore')
            
            X_val_scaled = scaler.transform(X_val_chunk)
            
            if hasattr(args, 'add_pca') and args.add_pca:
                if args.model in ["pytorch_svm", "tabnet", "deepforest"]:
                    pca_name = f"{args.model.replace('pytorch_svm', 'svm_pytorch')}_pca.joblib"
                elif args.model == "predictive_coding":
                    pca_name = "pc_pca.joblib"
                elif args.model == "svm":
                    pca_name = f"{args.kernel}_pca.joblib"
                else:
                    pca_name = f"{args.rf_variant}_pca.joblib"
                    
                pca_path = global_out / pca_name
                if pca_path.exists():
                    pca = joblib.load(pca_path)
                    X_val_scaled = pca.transform(X_val_scaled)
            
            # Properly Route Model Loading & Prediction
            if args.model == "pytorch_svm":
                import torch
                from src.models.svm.svm_pytorch_optuna import PyTorchLinearSVM
                device = "cuda" if torch.cuda.is_available() else "cpu"
                best_model = PyTorchLinearSVM(X_val_scaled.shape[1]).to(device)
                best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                best_model.eval()
                with torch.no_grad():
                    preds_raw = best_model(torch.FloatTensor(X_val_scaled).to(device))
                    y_pred_val = (preds_raw > 0).cpu().numpy().astype(int)
            elif args.model == "predictive_coding":
                import torch
                from src.models.predictive_coding.pc_network_torch import PredictiveCodingNetworkTorch
                from src.models.predictive_coding.pc_network import PCNetworkConfig
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                
                pc_cfg = PCNetworkConfig(**cfg.get('best_params', {}))
                pc_cfg.output_activation = "identity"
                
                # layer sizes include in and out
                layer_sizes = [X_val_scaled.shape[1], *cfg['model_params']['hidden_sizes'], 1]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                best_model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=pc_cfg, device=device)
                
                # Load weights
                if str(model_path).endswith('.npz'):
                    state = dict(np.load(model_path))
                else:
                    state = torch.load(model_path, map_location=device, weights_only=True)
                best_model.load_state_dict(state)
                probs = best_model.predict_proba_torch(X_val_scaled).detach().cpu().numpy()
                y_pred_val = (probs >= 0.5).astype(int)
            else: # SKLearn / DeepForest models
                best_model = joblib.load(model_path)
                y_pred_val = best_model.predict(X_val_scaled)
            
            # Evaluate and append
            bias_metrics = evaluate_model_bias(y_val_chunk.values, y_pred_val, X_val_chunk, dataset_name="(VALIDATION SET)")
            append_metrics_to_config(config_path, bias_metrics)
        else:
            print(f"Could not find saved models in {global_out} for evaluation.")

if __name__ == "__main__":
    main()