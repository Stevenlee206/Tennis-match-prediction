import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocessing import Preprocessing

def main():
    parser = argparse.ArgumentParser(description="ATP Tennis Match Prediction Pipeline")
    
    # --- ADD MODEL FLAG ---
    parser.add_argument("--model", type=str, choices=["svm", "rf"], default="svm", help="Algorithm to use")
    
    parser.add_argument("--mode", type=str, choices=["standard", "sgd"], default="standard")
    parser.add_argument("--optimizer", type=str, choices=["optuna", "pso", "ga", "grid"], default="optuna", help="Optimizer to tune hyperparameters")
    parser.add_argument("--validation", type=str, choices=["holdout", "walk_forward"], default="holdout")
    parser.add_argument("--kernel", type=str, choices=["linear", "poly", "rbf"], default="linear", help="SVM Kernel to use (Ignored if model=rf)")
    
    # Optimizer Params
    parser.add_argument("--n_trials", type=int, default=30, help="Used for Optuna")
    parser.add_argument("--particles", type=int, default=15, help="Used for PSO")
    parser.add_argument("--iterations", type=int, default=20, help="Used for PSO")
    parser.add_argument("--population", type=int, default=20, help="Used for GA")
    parser.add_argument("--generations", type=int, default=15, help="Used for GA")
    parser.add_argument("--epochs", type=int, default=100, help="Used for SGD mode")
    
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

    print("--- Step 1: Preprocessing Data ---")
    prep = Preprocessing()
    data = prep.run()
    
    if 'target' not in data.columns:
        raise ValueError("Error: 'target' missing.")

    BASE_DIR = Path(__file__).resolve().parent
    
    # --- DYNAMIC PATH ROUTING ---
    model_subpath = f"svm/{args.kernel}" if args.model == "svm" else "rf"
    
    BASE_OUT = BASE_DIR / "outputs" / "sklearn" / model_subpath / args.mode / args.optimizer / args.validation
    BASE_REP = BASE_DIR / "reports" / "figures" / "sklearn" / model_subpath / args.mode / args.optimizer / args.validation

    # --- DYNAMIC IMPORTS ---
    if args.model == "svm":
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
                from src.models.svm.svm_sklearn import run_svm_pipeline as run_pipeline
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
        X = data.drop(columns=['target', 'year'], errors='ignore')
        y = data['target']

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.166, shuffle=False)

        print(f"\nSplits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # --- DYNAMIC PIPELINE EXECUTION ---
        if args.model == "svm":
            if args.optimizer == "pso":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_particles=args.particles, n_iterations=args.iterations, kernel=args.kernel)
            elif args.optimizer == "ga":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, pop_size=args.population, n_generations=args.generations, kernel=args.kernel)
            elif args.optimizer == "grid":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, c_min=args.c_min, c_max=args.c_max, c_steps=args.c_steps, kernel=args.kernel)
            else:
                if args.mode == "standard":
                    run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_trials=args.n_trials, kernel=args.kernel)
                else:
                    run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_trials=args.n_trials, n_epochs=args.epochs, kernel=args.kernel)
        
        elif args.model == "rf":
            if args.optimizer == "pso":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_particles=args.particles, n_iterations=args.iterations, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)
            elif args.optimizer == "ga":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, pop_size=args.population, n_generations=args.generations, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)
            elif args.optimizer == "grid":
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, n_est_steps=args.rf_n_est_steps, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max, depth_steps=args.rf_depth_steps)
            else:
                run_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_trials=args.n_trials, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)

        print("\n--- Training and Tuning Complete ---")
    
    # ==========================================
    # ROUTE B: WALK-FORWARD VALIDATION
    # ==========================================
    elif args.validation == "walk_forward":
        all_years = sorted(data['year'].unique())
        MIN_TRAIN_YEARS = 5

        for test_idx in range(MIN_TRAIN_YEARS + 1, len(all_years)):
            test_year = all_years[test_idx]
            val_year = all_years[test_idx - 1]
            train_years = all_years[:test_idx - 1]

            print(f"\n" + "="*50)
            print(f" FOLD: Tuning model for Test Year {test_year}")
            print("="*50)

            fold_out = BASE_OUT / f"fold_{test_year}"
            fold_rep = BASE_REP / f"fold_{test_year}"

            train_df = data[data['year'].isin(train_years)]
            val_df = data[data['year'] == val_year]

            X_train = train_df.drop(columns=['target', 'year'], errors='ignore')
            y_train = train_df['target']
            X_val = val_df.drop(columns=['target', 'year'], errors='ignore')
            y_val = val_df['target']

            if args.model == "svm":
                if args.optimizer == "pso":
                    run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_particles=args.particles, n_iterations=args.iterations, kernel=args.kernel)
                elif args.optimizer == "ga":
                    run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, pop_size=args.population, n_generations=args.generations, kernel=args.kernel)
                elif args.optimizer == "grid":
                    run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, c_min=args.c_min, c_max=args.c_max, c_steps=args.c_steps, kernel=args.kernel)
                else:
                    if args.mode == "standard":
                        run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_trials=args.n_trials, kernel=args.kernel)
                    else:
                        run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_trials=args.n_trials, n_epochs=args.epochs, kernel=args.kernel)

            elif args.model == "rf":
                if args.optimizer == "pso":
                    run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_particles=args.particles, n_iterations=args.iterations, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)
                elif args.optimizer == "ga":
                    run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, pop_size=args.population, n_generations=args.generations, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)
                elif args.optimizer == "grid":
                    run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, n_est_steps=args.rf_n_est_steps, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max, depth_steps=args.rf_depth_steps)
                else:
                    run_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_trials=args.n_trials, n_est_min=args.rf_n_est_min, n_est_max=args.rf_n_est_max, depth_min=args.rf_depth_min, depth_max=args.rf_depth_max)

        print("\n" + "="*50)
        print(" WALK-FORWARD TRAINING & TUNING COMPLETE")
        print("="*50)

if __name__ == "__main__":
    main()