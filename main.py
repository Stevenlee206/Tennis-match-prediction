import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import Preprocessing
import joblib
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_bias(y_true, y_pred, X_raw):
    """
    Calculates bias metrics and returns them as a dictionary for JSON logging.
    """
    print("\n" + "="*50)
    print(" MODEL BIAS & HEURISTIC ANALYSIS")
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
    
    # --- ADDED SPLIT & TSCV CONFIG FLAGS ---
    parser.add_argument("--test_size", type=float, default=0.10, help="Global test set ratio")
    parser.add_argument("--val_size", type=float, default=0.20, help="Validation set ratio (for holdout)")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of TimeSeries CV splits for walk-forward")
    
    parser.add_argument("--model", type=str, choices=["svm", "rf", "pytorch_svm", "tabnet", "deepforest", "pytorch_mlp"], default="svm", help="Algorithm to use")    
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
    
    # Optimizer Params (Remaining params unchanged)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--particles", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--c_min", type=float, default=1e-3)
    parser.add_argument("--c_max", type=float, default=1e2)
    parser.add_argument("--c_steps", type=int, default=10)
    parser.add_argument("--rf_n_est_min", type=int, default=50)
    parser.add_argument("--rf_n_est_max", type=int, default=500)
    parser.add_argument("--rf_n_est_steps", type=int, default=5)
    parser.add_argument("--rf_depth_min", type=int, default=5)
    parser.add_argument("--rf_depth_max", type=int, default=50)
    parser.add_argument("--rf_depth_steps", type=int, default=5)
    parser.add_argument("--df_max_layers_min", type=int, default=2)
    parser.add_argument("--df_max_layers_max", type=int, default=10)
    parser.add_argument("--df_n_trees_min", type=int, default=50)
    parser.add_argument("--df_n_trees_max", type=int, default=200)
    parser.add_argument("--df_depth_min", type=int, default=5)
    parser.add_argument("--df_depth_max", type=int, default=30)
    parser.add_argument("--weight_strategy", type=str, choices=["none", "static", "magnitude", "temporal"], default="none")
    parser.add_argument("--upset_weight", type=float, default=1.5)
    args = parser.parse_args()

    # Safety Checks
    if args.mode == "sgd" and args.model != "svm":
        raise ValueError("Error: SGD mode is currently only implemented for SVM.")
    if args.mode == "sgd" and args.kernel != "linear":
        raise ValueError("Error: SGD mode only supports the 'linear' kernel.")

    print("==============================================")
    print(f" ATP Tennis Prediction")
    print(f" Model: {args.model.upper()} | Mode: {args.mode.upper()} | Optimizer: {args.optimizer.upper()} | Val: {args.validation.upper()}")
    print("==============================================\n")

    # ==========================================
    # DYNAMIC TRAIN RATIO CALCULATION
    # ==========================================
    if args.validation == "holdout":
        train_ratio = (1.0 - args.test_size) * (1.0 - args.val_size)
    else:
        # To prevent leakage into the first TSCV validation fold, imputation must only fit on the INITIAL training window.
        # Initial window = Total dataset - Test Set - (n_splits * test_size equivalent folds)
        train_ratio = (1.0 - args.test_size) - (args.n_splits * args.test_size)
        
        # Safety bound if args cause ratio to collapse
        if train_ratio <= 0.1:
            print("⚠️ Warning: Walk-forward folds are large. Adjusting initial train window safely.")
            train_ratio = (1.0 - args.test_size) / 2.0

    print("--- Step 1: Preprocessing Data ---")
    prep = Preprocessing()
    data = prep.run(train_ratio=train_ratio)  # <--- PASSED DYNAMICALLY
    
    if 'target' not in data.columns:
        raise ValueError("Error: 'target' missing.")

    # ==========================================
    # GLOBAL TEST SET EXTRACTION
    # ==========================================
    X_full = data.drop(columns=['target', 'year'], errors='ignore')
    y_full = data['target']

    X_train_val_pool, X_test, y_train_val_pool, y_test = train_test_split(
        X_full, y_full, test_size=args.test_size, shuffle=False # <--- USING ARG
    )
    
    if 'is_augmented' in X_test.columns:
        y_test = y_test[X_test['is_augmented'] == 0]
        X_test = X_test[X_test['is_augmented'] == 0].drop(columns=['is_augmented'])
        
    print(f"\nGlobal Splits -> Modeling Pool: {len(X_train_val_pool)} | Quarantined Test: {len(X_test)}")
    
    # Calculate the precise integer length of the test set to pass to TSCV down the line
    tscv_test_size = len(X_test)
    
    BASE_DIR = Path(__file__).resolve().parent
    
    # [Path Routing & Imports remain exactly the same as your code]
    if args.model == "svm":
        model_subpath = f"sklearn/svm/{args.kernel}"
    elif args.model == "rf":
        model_subpath = f"sklearn/rf/{args.rf_variant}"
    elif args.model == "pytorch_svm":
        model_subpath = "pytorch_svm"
    elif args.model == "pytorch_mlp":
        model_subpath = "pytorch_mlp"
    elif args.model == "tabnet":
        model_subpath = "tabnet"
    elif args.model == "deepforest":
        model_subpath = "deepforest"
    
    BASE_OUT = BASE_DIR / "outputs" / model_subpath / args.mode / args.optimizer / args.validation
    BASE_REP = BASE_DIR / "reports" / "figures" / model_subpath / args.mode / args.optimizer / args.validation

    # --- DYNAMIC IMPORTS ---
    if args.model == "pytorch_svm":
        from src.models.svm.svm_pytorch_optuna import run_pytorch_pipeline as run_pipeline
    elif args.model == "pytorch_mlp":
        from src.models.mlp.mlp_pytorch_optuna import run_pytorch_mlp_pipeline as run_pipeline
    elif args.model == "deepforest":
        from src.models.rf.deepforest_optuna import run_deepforest_pipeline as run_pipeline
    elif args.model == "svm":
        if args.optimizer == "pso":
            from src.models.svm.svm_sklearn_pso import run_svm_pipeline as run_pipeline
        elif args.optimizer == "ga":
            from src.models.svm.svm_sklearn_ga import run_svm_pipeline as run_pipeline
        elif args.optimizer == "grid":
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
            X_train_val_pool, y_train_val_pool, test_size=args.val_size, shuffle=False # <--- USING ARG
        )

        if 'is_augmented' in X_train.columns:
            y_val = y_val[X_val['is_augmented'] == 0]
            X_val = X_val[X_val['is_augmented'] == 0]
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
        else:
            model_name, scaler_name, config_name = f"{args.rf_variant}_model.joblib", f"{args.rf_variant}_scaler.joblib", f"{args.rf_variant}_config.json"
            
        model_path = BASE_OUT / model_name
        scaler_path = BASE_OUT / scaler_name
        config_path = BASE_OUT / config_name
        
        check_path = Path(str(model_path)) if args.model != "tabnet" else Path(str(model_path).replace('.zip', '') + '.zip')

        if check_path.exists() and scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X_val_scaled = scaler.transform(X_val)
            
            # Handle PCA if enabled
            if hasattr(args, 'add_pca') and args.add_pca:
                if args.model in ["pytorch_svm", "pytorch_mlp", "tabnet", "deepforest"]:
                    # Flips 'pytorch_svm' -> 'svm_pytorch' and 'pytorch_mlp' -> 'mlp_pytorch'
                    prefix = args.model.replace('pytorch_', '') + '_pytorch' if 'pytorch' in args.model else args.model
                    pca_name = f"{prefix}_pca.joblib"
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
            elif args.model == "pytorch_mlp":
                import torch
                from src.models.mlp.mlp_pytorch_optuna import TimeSeriesTennisNet, TimeSeriesTennisDataset
                from torch.utils.data import DataLoader
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    
                best_model = TimeSeriesTennisNet(X_val_scaled.shape[1], cfg['hidden_dim']).to(device)
                best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                best_model.eval()
                
                # We MUST use the custom Dataset to generate the 3D rolling windows
                val_dataset = TimeSeriesTennisDataset(X_val_scaled, y_val.values, window_size=5)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                
                y_pred_val = []
                
                with torch.no_grad():
                    for batch_X, batch_y, _ in val_loader:
                        batch_X = batch_X.to(device)
                        preds_raw = best_model(batch_X)
                        # MLP outputs logits. Sigmoid > 0.5 gets the binary class
                        preds_binary = (torch.sigmoid(preds_raw) > 0.5).cpu().numpy().astype(int).flatten()
                        y_pred_val.extend(preds_binary)
                        
                y_pred_val = np.array(y_pred_val)
                
                # Because window_size=5, the dataset drops the first 4 matches. 
                # We must truncate the evaluation data to match!
                y_val = y_val.iloc[4:]
                X_val = X_val.iloc[4:]
            else: # SKLearn / DeepForest models
                best_model = joblib.load(model_path)
                y_pred_val = best_model.predict(X_val_scaled)
            
            # Evaluate and Append to JSON
            bias_metrics = evaluate_model_bias(y_val.values, y_pred_val, X_val)
            append_metrics_to_config(config_path, bias_metrics)
        else:
            print(f"Could not find saved models in {BASE_OUT} for evaluation.")
    
    elif args.validation == "walk_forward":
        print("\n" + "="*50)
        print(f" RUNNING GLOBAL TIME-SERIES CV ({args.n_splits} SPLITS)")
        print(f" Target Test Size per Fold: {tscv_test_size}")
        print("="*50)

        if 'is_augmented' in X_train_val_pool.columns:
            X_train_val_pool = X_train_val_pool.drop(columns=['is_augmented'])
            
        global_out = BASE_OUT / f"global_tscv_{args.model}"
        global_rep = BASE_REP / f"global_tscv_{args.model}"

        # --- DYNAMIC PIPELINE EXECUTION (Added tscv_test_size and n_splits keyword args) ---
        if args.model == "svm":
            if args.optimizer == "optuna":
                if args.mode == "standard":
                    run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                                 n_trials=args.n_trials, kernel=args.kernel, c_min=args.c_min, c_max=args.c_max,                         
                                 add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                                 n_splits=args.n_splits, tscv_test_size=tscv_test_size) # <-- NEW
                elif args.mode == "sgd":
                    run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                                 n_trials=args.n_trials, n_epochs=args.epochs, kernel="linear", c_min=args.c_min, c_max=args.c_max,                        
                                 add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight, lr_schedule=args.lr_schedule,
                                 n_splits=args.n_splits, tscv_test_size=tscv_test_size) # <-- NEW
        
        elif args.model == "pytorch_svm":
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                         n_trials=args.n_trials, epochs=args.epochs, batch_size=args.batch_size, c_min=args.c_min, c_max=args.c_max,                         
                         add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         torch_opt=args.torch_opt, torch_sched=args.torch_sched,
                         n_splits=args.n_splits, tscv_test_size=tscv_test_size) # <-- NEW
                         
        elif args.model == "deepforest":
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                         n_trials=args.n_trials, add_pca=args.add_pca, validation=args.validation, 
                         weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         max_layers_min=args.df_max_layers_min, max_layers_max=args.df_max_layers_max,
                         n_trees_min=args.df_n_trees_min, n_trees_max=args.df_n_trees_max,
                         max_depth_min=args.df_depth_min, max_depth_max=args.df_depth_max,
                         n_splits=args.n_splits, tscv_test_size=tscv_test_size) # <-- NEW
                         
        elif args.model == "pytorch_mlp":
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                         n_trials=args.n_trials, epochs=args.epochs, batch_size=args.batch_size, 
                         add_pca=args.add_pca, validation=args.validation, 
                         weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                         n_splits=args.n_splits, tscv_test_size=tscv_test_size) # <-- NEW
                         
        elif args.model == "tabnet":
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                         n_trials=args.n_trials, epochs=args.epochs, batch_size=args.batch_size, 
                         add_pca=args.add_pca, validation=args.validation,
                         n_splits=args.n_splits, tscv_test_size=tscv_test_size) # <-- NEW
                         
        elif args.model == "rf":
            run_pipeline(X_train_val_pool, y_train_val_pool, None, None, global_out, global_rep, 
                          n_trials=args.n_trials, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, 
                          depth_min=args.rf_depth_min, depth_max=args.rf_depth_max, variant=args.rf_variant,
                          add_pca=args.add_pca, validation=args.validation, weight_strategy=args.weight_strategy, upset_weight=args.upset_weight,
                          n_splits=args.n_splits, tscv_test_size=tscv_test_size) # <-- NEW

        # ==========================================
        # RUN BIAS EVALUATION ON QUARANTINED CHUNK 7
        # ==========================================
        print("\nLoading saved model to evaluate bias on the completely unseen Quarantined test chunk...")
        
        # FIXED: Added pytorch_mlp and tabnet routing
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
        elif args.model == "tabnet":
            model_name, scaler_name, config_name = "tabnet_model.zip", "tabnet_scaler.joblib", "tabnet_config.json"
        else:
            model_name, scaler_name, config_name = f"{args.rf_variant}_model.joblib", f"{args.rf_variant}_scaler.joblib", f"{args.rf_variant}_config.json"
            
        model_path = global_out / model_name
        scaler_path = global_out / scaler_name
        config_path = global_out / config_name
        
        check_path = Path(str(model_path)) if args.model != "tabnet" else Path(str(model_path).replace('.zip', '') + '.zip')

        if check_path.exists() and scaler_path.exists():
            scaler = joblib.load(scaler_path)
            
            # Use the rigorously quarantined global test set (Chunk 7)
            X_test_eval = X_test.copy()
            y_test_eval = y_test.copy()
            
            X_test_scaled = scaler.transform(X_test_eval)
            
            if hasattr(args, 'add_pca') and args.add_pca:
                if args.model in ["pytorch_svm", "pytorch_mlp", "tabnet", "deepforest"]:
                    prefix = args.model.replace('pytorch_', '') + '_pytorch' if 'pytorch' in args.model else args.model
                    pca_name = f"{prefix}_pca.joblib"
                elif args.model == "svm":
                    pca_name = "svm_sgd_pca.joblib" if args.mode == "sgd" else f"{args.kernel}_pca.joblib"
                else:
                    pca_name = f"{args.rf_variant}_pca.joblib"
                    
                pca_path = global_out / pca_name
                if pca_path.exists():
                    pca = joblib.load(pca_path)
                    X_test_scaled = pca.transform(X_test_scaled) 
            
            # FIXED: Properly Route Model Loading & Prediction for ALL models
            if args.model == "pytorch_svm":
                import torch
                from src.models.svm.svm_pytorch_optuna import PyTorchLinearSVM
                device = "cuda" if torch.cuda.is_available() else "cpu"
                best_model = PyTorchLinearSVM(X_test_scaled.shape[1]).to(device) 
                best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                best_model.eval()
                with torch.no_grad():
                    preds_raw = best_model(torch.FloatTensor(X_test_scaled).to(device)) 
                    y_pred_eval = (preds_raw > 0).cpu().numpy().astype(int)
                    
            elif args.model == "pytorch_mlp":
                import torch
                from src.models.mlp.mlp_pytorch_optuna import TimeSeriesTennisNet, TimeSeriesTennisDataset
                from torch.utils.data import DataLoader
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    
                best_model = TimeSeriesTennisNet(X_test_scaled.shape[1], cfg['hidden_dim']).to(device)
                best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                best_model.eval()
                
                test_dataset = TimeSeriesTennisDataset(X_test_scaled, y_test_eval.values, window_size=5)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                y_pred_eval = []
                with torch.no_grad():
                    for batch_X, batch_y, _ in test_loader:
                        batch_X = batch_X.to(device)
                        preds_raw = best_model(batch_X)
                        preds_binary = (torch.sigmoid(preds_raw) > 0.5).cpu().numpy().astype(int).flatten()
                        y_pred_eval.extend(preds_binary)
                        
                y_pred_eval = np.array(y_pred_eval)
                
                # Truncate evaluation data to match dropped window matches
                y_test_eval = y_test_eval.iloc[4:]
                X_test_eval = X_test_eval.iloc[4:]
                
            else: # SKLearn / DeepForest / TabNet
                best_model = joblib.load(model_path)
                y_pred_eval = best_model.predict(X_test_scaled) 
            
            # Evaluate and append
            bias_metrics = evaluate_model_bias(y_test_eval.values, y_pred_eval, X_test_eval) 
            append_metrics_to_config(config_path, bias_metrics)
        else:
            print(f"Could not find saved models in {global_out} for evaluation.")

if __name__ == "__main__":
    main()