import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ATP Tennis Match Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Validation & Split Configuration
    split_group = parser.add_argument_group("Data Split & Validation Config")
    split_group.add_argument("--test_size", type=float, default=0.10, help="Global test set ratio")
    split_group.add_argument("--val_size", type=float, default=0.20, help="Validation set ratio (for holdout)")
    split_group.add_argument("--n_splits", type=int, default=5, help="Number of TimeSeries CV splits for walk-forward")
    split_group.add_argument("--validation", type=str, choices=["holdout", "walk_forward"], default="holdout",help="Validation strategy")

    # Models & feature engineering (k-means,pca) Config
    model_group = parser.add_argument_group("Core Model & Feature Engineering")
    model_group.add_argument("--model", type=str,
                             choices=["svm", "rf", "pytorch_svm", "tabnet", "deepforest", "pytorch_mlp"], default="svm",
                             help="Algorithm to use")
    model_group.add_argument("--add_kmeans", action="store_true",
                             help="Apply KMeans clustering to add spatial features")
    model_group.add_argument("--n_clusters", type=int, default=5, help="Number of clusters if using --add_kmeans")
    model_group.add_argument("--add_pca", action="store_true",
                             help="Apply PCA for dimensionality reduction and denoising")
    model_group.add_argument("--weight_strategy", type=str, choices=["none", "static", "magnitude", "temporal"],
                             default="none", help="Sample weighting strategy")
    model_group.add_argument("--upset_weight", type=float, default=1.5, help="Weight multiplier for upset matches")

    # Deep learning opt and params configuration
    opt_group = parser.add_argument_group("Global Optimizer & DL Params")
    opt_group.add_argument("--optimizer", type=str, choices=["optuna", "pso", "ga", "grid"], default="optuna",
                           help="Optimizer to tune hyperparameters")
    opt_group.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    opt_group.add_argument("--epochs", type=int, default=100, help="Epochs for Neural Networks / SGD")
    opt_group.add_argument("--batch_size", type=int, default=64, help="Batch size for PyTorch DataLoaders")
    opt_group.add_argument("--torch_opt", type=str, choices=["adam", "rmsprop", "sgd", "sgd_nesterov"], default="adam",
                           help="Optimizer for PyTorch models")
    opt_group.add_argument("--torch_sched", type=str, choices=["constant", "cosine", "step", "plateau"],
                           default="cosine", help="LR scheduler for PyTorch models")

    # hyperparameter tuning configurations

    # PSO
    opt_group.add_argument("--particles", type=int, default=15, help="Particles for PSO")
    opt_group.add_argument("--iterations", type=int, default=20, help="Iterations for PSO")

    # GA
    opt_group.add_argument("--population", type=int, default=20, help="Population size for GA")
    opt_group.add_argument("--generations", type=int, default=15, help="Generations for GA")

    # SVM configuration
    svm_group = parser.add_argument_group("SVM Specific Params")
    svm_group.add_argument("--mode", type=str, choices=["standard", "sgd"], default="standard",
                           help="SVM calculation mode")
    svm_group.add_argument("--kernel", type=str, choices=["linear", "poly", "rbf"], default="linear", help="SVM Kernel")
    svm_group.add_argument("--lr_schedule", type=str, choices=["adaptive", "optimal", "invscaling", "constant"],
                           default="adaptive", help="LR schedule for SGD mode")
    svm_group.add_argument("--c_min", type=float, default=1e-3, help="Min bound for C parameter")
    svm_group.add_argument("--c_max", type=float, default=1e2, help="Max bound for C parameter")
    svm_group.add_argument("--c_steps", type=int, default=10, help="Steps for GridSearch C")

    # RF config
    rf_group = parser.add_argument_group("Random Forest Specific Params")
    rf_group.add_argument("--rf_variant", type=str,
                          choices=["rf", "extra_trees", "rrf", "rotation_forest", "oblique", "weighted"], default="rf",
                          help="Specific RF variant")
    rf_group.add_argument("--rf_n_est_min", type=int, default=50)
    rf_group.add_argument("--rf_n_est_max", type=int, default=500)
    rf_group.add_argument("--rf_n_est_steps", type=int, default=5)
    rf_group.add_argument("--rf_depth_min", type=int, default=5)
    rf_group.add_argument("--rf_depth_max", type=int, default=50)
    rf_group.add_argument("--rf_depth_steps", type=int, default=5)

    # DF config
    df_group = parser.add_argument_group("DeepForest Specific Params")
    df_group.add_argument("--df_max_layers_min", type=int, default=2)
    df_group.add_argument("--df_max_layers_max", type=int, default=10)
    df_group.add_argument("--df_n_trees_min", type=int, default=50)
    df_group.add_argument("--df_n_trees_max", type=int, default=200)
    df_group.add_argument("--df_depth_min", type=int, default=5)
    df_group.add_argument("--df_depth_max", type=int, default=30)

    args = parser.parse_args()


    if args.mode == "sgd" and args.model != "svm":
        raise ValueError("Error: SGD mode is currently only implemented for SVM (--model svm).")

    if args.mode == "sgd" and args.kernel != "linear":
        raise ValueError("Error: SGD mode only supports the 'linear' kernel (--kernel linear).")

    return args