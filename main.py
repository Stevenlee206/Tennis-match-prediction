import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.preprocessing import Preprocessing

def main():
    parser = argparse.ArgumentParser(description="ATP Tennis Match Prediction Pipeline")
    parser.add_argument("--mode", type=str, choices=["standard", "sgd"], default="standard")
    parser.add_argument("--optimizer", type=str, choices=["optuna", "pso", "ga"], default="optuna", help="Optimizer to tune hyperparameters")
    parser.add_argument("--validation", type=str, choices=["holdout", "walk_forward"], default="holdout")
    parser.add_argument("--kernel", type=str, choices=["linear", "poly", "rbf"], default="linear", help="SVM Kernel to use")
    
    # Optimizer Params
    parser.add_argument("--n_trials", type=int, default=30, help="Used for Optuna")
    parser.add_argument("--particles", type=int, default=15, help="Used for PSO")
    parser.add_argument("--iterations", type=int, default=20, help="Used for PSO")
    parser.add_argument("--population", type=int, default=20, help="Used for GA (Population Size)")
    parser.add_argument("--generations", type=int, default=15, help="Used for GA (Number of Generations)")
    parser.add_argument("--epochs", type=int, default=100, help="Used for SGD mode")
    args = parser.parse_args()

    # Safety check: SGDClassifier inherently uses a linear boundary
    if args.mode == "sgd" and args.kernel != "linear":
        raise ValueError("Error: SGD mode only supports the 'linear' kernel. Please change --kernel or switch to --mode standard.")

    print("==============================================")
    print(f" ATP Tennis Prediction")
    print(f" Mode: {args.mode.upper()} | Kernel: {args.kernel.upper()} | Optimizer: {args.optimizer.upper()} | Val: {args.validation.upper()}")
    print("==============================================\n")

    print("--- Step 1: Preprocessing Data ---")
    prep = Preprocessing()
    data = prep.run()
    
    if 'target' not in data.columns:
        raise ValueError("Error: 'target' missing.")

    # Injected args.kernel into the path hierarchy
    BASE_OUT = Path("outputs/sklearn/svm") / args.kernel / args.mode / args.optimizer / args.validation
    BASE_REP = Path("reports/figures/sklearn/svm") / args.kernel / args.mode / args.optimizer / args.validation

    # Dynamically import the chosen pipeline
    if args.optimizer == "pso":
        if args.mode == "sgd":
            raise NotImplementedError("PSO is currently only implemented for 'standard' mode SVM.")
        from src.models.svm.svm_sklearn_pso import run_svm_pipeline
    elif args.optimizer == "ga":
        if args.mode == "sgd":
            raise NotImplementedError("GA is currently only implemented for 'standard' mode SVM.")
        from src.models.svm.svm_sklearn_ga import run_svm_pipeline
    else: # Optuna
        if args.mode == "standard":
            from src.models.svm.svm_sklearn import run_svm_pipeline
        elif args.mode == "sgd":
            from src.models.svm.svm_sklearn_SGD import run_svm_pipeline

    # ==========================================
    # ROUTE A: STANDARD HOLDOUT VALIDATION
    # ==========================================
    if args.validation == "holdout":
        X = data.drop(columns=['target', 'year'], errors='ignore')
        y = data['target']

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.166, shuffle=False)

        print(f"\nSplits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        if args.optimizer == "pso":
            model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_particles=args.particles, n_iterations=args.iterations, kernel=args.kernel)
        elif args.optimizer == "ga":
            model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, pop_size=args.population, n_generations=args.generations, kernel=args.kernel)
        else:
            if args.mode == "standard":
                model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_trials=args.n_trials, kernel=args.kernel)
            else:
                model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, BASE_OUT, BASE_REP, n_trials=args.n_trials, n_epochs=args.epochs, kernel=args.kernel)

        print("\n--- Final Evaluation on Test Set ---")
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        print(f"\n>>> Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f} <<<")
        print(classification_report(y_test, y_pred))

    # ==========================================
    # ROUTE B: WALK-FORWARD VALIDATION
    # ==========================================
    elif args.validation == "walk_forward":
        all_years = sorted(data['year'].unique())
        MIN_TRAIN_YEARS = 5
        fold_results = []

        for test_idx in range(MIN_TRAIN_YEARS + 1, len(all_years)):
            test_year = all_years[test_idx]
            val_year = all_years[test_idx - 1]
            train_years = all_years[:test_idx - 1]

            print(f"\n" + "="*50)
            print(f" FOLD: Test Year {test_year}")
            print("="*50)

            fold_out = BASE_OUT / f"fold_{test_year}"
            fold_rep = BASE_REP / f"fold_{test_year}"

            train_df = data[data['year'].isin(train_years)]
            val_df = data[data['year'] == val_year]
            test_df = data[data['year'] == test_year]

            X_train = train_df.drop(columns=['target', 'year'], errors='ignore')
            y_train = train_df['target']
            
            X_val = val_df.drop(columns=['target', 'year'], errors='ignore')
            y_val = val_df['target']
            
            X_test = test_df.drop(columns=['target', 'year'], errors='ignore')
            y_test = test_df['target']

            if args.optimizer == "pso":
                model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_particles=args.particles, n_iterations=args.iterations, kernel=args.kernel)
            elif args.optimizer == "ga":
                model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, pop_size=args.population, n_generations=args.generations, kernel=args.kernel)
            else:
                if args.mode == "standard":
                    model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_trials=args.n_trials, kernel=args.kernel)
                else:
                    model, scaler = run_svm_pipeline(X_train, y_train, X_val, y_val, fold_out, fold_rep, n_trials=args.n_trials, n_epochs=args.epochs, kernel=args.kernel)

            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)

            test_acc = accuracy_score(y_test, y_pred)
            print(f">>> Fold Test Accuracy for {test_year}: {test_acc:.4f} <<<")
            
            fold_results.append({
                'test_year': test_year,
                'matches': len(y_test),
                'accuracy': test_acc
            })

        print("\n" + "="*50)
        print(" WALK-FORWARD VALIDATION COMPLETE")
        print("="*50)
        
        total_matches = sum(res['matches'] for res in fold_results)
        weighted_acc_sum = sum(res['accuracy'] * res['matches'] for res in fold_results)
        print(f"OVERALL WALK-FORWARD ACCURACY: {(weighted_acc_sum / total_matches):.4f}")

if __name__ == "__main__":
    main()