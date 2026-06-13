import os
import sys
import joblib
import numpy as np
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.execution.bias_analysis import evaluate_model_bias, append_metrics_to_config
from src.execution.model_interpretation import calculate_feature_importances, plot_interpretability

def _get_file_names(args):
    """Determine the model, scaler, and configuration file names based on the algorithm.."""
    if args.model == "svm":
        if args.mode == "sgd":
            return "svm_sgd_model.joblib", "svm_sgd_scaler.joblib", "svm_sgd_config.json"
        return f"{args.kernel}_model.joblib", f"{args.kernel}_scaler.joblib", f"{args.kernel}_config.json"

    elif args.model == "pytorch_svm":
        return "svm_pytorch_model.pth", "svm_pytorch_scaler.joblib", "svm_pytorch_config.json"

    elif args.model == "deepforest":
        return "deepforest_model.joblib", "deepforest_scaler.joblib", "deepforest_config.json"

    elif args.model == "xgboost":
        return "xgboost_model.joblib", "xgboost_scaler.joblib", "xgboost_config.json"

    elif args.model == "decisiontree":
        return "decisiontree_model.joblib", "decisiontree_scaler.joblib", "decisiontree_config.json"

    elif args.model == "logistic_regression":
        return "log_reg_model.joblib","log_reg_scaler.joblib","log_reg_config.json"

    elif args.model == "naive_bayes":

        return "nb_model.joblib", "nb_scaler.joblib", "nb_config.json"
    else:  # Random Forest variants
        return f"{args.rf_variant}_model.joblib", f"{args.rf_variant}_scaler.joblib", f"{args.rf_variant}_config.json"



def _get_feature_prefix(args):
    """
    Define the filename prefix for PCA and K-Means.
    """
    if args.model in ["pytorch_svm", "pytorch_mlp", "tabnet", "deepforest"]:
        return args.model.replace('pytorch_', '') + '_pytorch' if 'pytorch' in args.model else args.model
    elif args.model == "svm":
        return "svm_sgd" if args.mode == "sgd" else args.kernel
    else:
        return args.rf_variant


def load_and_evaluate_model(args, X_eval, y_eval, out_dir):
    """
    Load the entire pipeline (Scaler -> PCA/KMeans -> Model), perform predictions,
    assess bias, and save to a JSON configuration file.
    """
    out_dir = Path(out_dir)
    print(f"\n[*] Evaluating the model from the directory : {out_dir}")

    #  Identify the file name.
    model_name, scaler_name, config_name = _get_file_names(args)
    model_path = out_dir / model_name
    scaler_path = out_dir / scaler_name
    config_path = out_dir / config_name

    # Check for existence (ignore the .zip extension for TabNet when checking the path if necessary)
    check_model_path = Path(str(model_path).replace('.zip', '') + '.zip') if args.model == "tabnet" else model_path

    if not (check_model_path.exists() and scaler_path.exists()):
        print(f" Error: Model or scaler not found at {out_dir}. Skip evaluation.")
        return

    # Copy the original data to retain as proof for calculating Bias
    X_eval_raw = X_eval.copy()

    # Load Scaler & Transform
    scaler = joblib.load(scaler_path)
    X_eval_scaled = scaler.transform(X_eval)

    # Load PCA & Transform
    prefix = _get_feature_prefix(args)
    if getattr(args, 'add_pca', False):
        pca_path = out_dir / f"{prefix}_pca.joblib"
        if pca_path.exists():
            pca = joblib.load(pca_path)
            X_eval_scaled = pca.transform(X_eval_scaled)
        else:
            print(f"Warning: PCA is enabled but not found {pca_path.name}")

    # Load K-Means & Transform
    if getattr(args, 'add_kmeans', False):
        kmeans_path = out_dir / f"{prefix}_kmeans.joblib"
        if kmeans_path.exists():
            kmeans = joblib.load(kmeans_path)
            v_distances = kmeans.transform(X_eval_scaled)
            X_eval_scaled = np.hstack((X_eval_scaled, v_distances))
        else:
            print(f" Warning: KMeans is enabled but not found {kmeans_path.name}")

    # Load Model & Prediction Threading
    y_prob = None
    if args.model == "pytorch_svm":
        import torch
        from src.models.svm.svm_pytorch_optuna import PyTorchLinearSVM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PyTorchLinearSVM(X_eval_scaled.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            tensor_X = torch.FloatTensor(X_eval_scaled).to(device)
            preds_raw = model(tensor_X)
            y_pred = (preds_raw > 0).cpu().numpy().astype(int).flatten()
            y_prob = torch.sigmoid(preds_raw).cpu().numpy().flatten()
    else:
        # Load Sklearn / TabNet / DeepForest system models
        model = joblib.load(model_path)
        y_pred = model.predict(X_eval_scaled)
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_eval_scaled)[:, 1]
            except Exception:
                pass

    # Bias Calculation & Rating
    bias_metrics = evaluate_model_bias(y_eval.values, y_pred, X_eval_raw,y_prob)

    # Write the results to a JSON file.
    if config_path.exists():
        append_metrics_to_config(config_path, bias_metrics)
    else:
        print(f" Not found {config_name} to record the evaluation results.")

    # Reconstructing the Feature Names list from raw data
    feature_names = list(X_eval_raw.columns)
    # Handle renaming if PCA (dimensional reduction) is used.
    if getattr(args, 'add_pca', False) and pca_path.exists():
        feature_names = [f"PCA_{i}" for i in range(X_eval_scaled.shape[1])]

        # Handle name appending if using KMeans (adding clusters).
    if getattr(args, 'add_kmeans', False) and kmeans_path.exists():
        n_clusters = getattr(args, 'n_clusters', 5)
        cluster_names = [f"KMeans_Dist_{i}" for i in range(n_clusters)]
        if not getattr(args, 'add_pca', False):
            feature_names.extend(cluster_names)

        # Activating calculations and plotting graphs
    if args.model not in ["pytorch_svm", "pytorch_mlp"]:
        # calculate feature importance
        importances = calculate_feature_importances(
            model=model,
            X_eval=X_eval_scaled,
            y_eval=y_eval.values
        )

        # plot
        plot_interpretability(
            model=model,
            X_eval=X_eval_scaled,
            importances=importances,
            feature_names=feature_names,
            out_dir=out_dir,
            model_name=args.model
        )
    else:
        print(f"\n[*] Skip Feature Importance/PDP for {args.model}.")

     # ERROR ANALYSIS & HYPOTHESIS TESTING
    print("\n" + "=" * 50)
    print(f"[*] ERROR ANALYSIS")
    print("=" * 50)

    from src.execution.prediction_analysis import (
        plot_prediction_summary,
        plot_confidence_analysis,
        plot_all_features_errors
    )

    # Testing the Label Allocation Hypothesis
    plot_prediction_summary(y_eval.values, y_pred, out_dir, args.model)

    # Test the Confidence Hypothesis (Requires the model to have a predict_proba function)
    if hasattr(model, "predict_proba"):
        try:
        # Calculate the probability of class 1 (Player 1 wins).
            y_prob = model.predict_proba(X_eval_scaled)[:, 1]
            plot_confidence_analysis(y_eval.values, y_prob, out_dir, args.model)
        except Exception as e:
            print(f" Unable to draw Confidence chart : {str(e)}")
    else:
        print(f"-> Ignore the Confidence chart because {args.model} predict_proba is not supported..")

        # Testing the Hypothesis of Errors Due to Overly Close Indexes (Using elo_diff)
    plot_all_features_errors(X_eval_raw, y_eval.values, y_pred, out_dir, args.model)