import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.execution.parse_args import parse_arguments
from src.execution.prepare_data import prepare_data
from src.execution.get_pipeline_runner import get_pipeline_runner
from pathlib import Path
from src.execution.load_and_evaluate import load_and_evaluate_model
import inspect
from sklearn.model_selection import train_test_split


def split_holdout(X_train_val, y_train_val, args):
    """
    Further divide the Modeling Pool into separate Train and Validation sets for the Holdout strategy.
    Simultaneously, remove augmented data from the Validation set.
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_size, shuffle=False
    )

    # The validation set used for evaluation during training should not contain virtual data.
    if 'is_augmented' in X_train.columns:
        y_val = y_val[X_val['is_augmented'] == 0]
        X_val = X_val[X_val['is_augmented'] == 0].drop(columns=['is_augmented'])
        X_train = X_train.drop(columns=['is_augmented'])

    return X_train, X_val, y_train, y_val

def get_output_directories(args):
    BASE_DIR = Path(__file__).resolve().parent
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
    elif args.model == "xgboost":
        model_subpath = "xgboost"
    elif args.model == 'decisiontree':
        model_subpath = 'decisiontree'
    elif args.model == 'logistic_regression':
        model_subpath = 'logistic_regression'
    elif args.model == 'naive_bayes':
        model_subpath = 'nb'

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    out_dir = PROJECT_ROOT / "outputs" / model_subpath / args.mode / args.optimizer / args.validation
    rep_dir = PROJECT_ROOT / "reports" / "figures" / model_subpath / args.mode / args.optimizer / args.validation

    if args.validation == "walk_forward":
        out_dir = out_dir / f"global_tscv_{args.model}"
        rep_dir = rep_dir / f"global_tscv_{args.model}"

    return out_dir, rep_dir
def main():
    # Init Config
    args = parse_arguments()
    print(f" ATP Tennis Prediction Pipeline")
    print(
        f" Model: {args.model.upper()} | Mode: {args.mode.upper()} | Optimizer: {args.optimizer.upper()} | Val: {args.validation.upper()}")
    # Prepare data
    X_train_val, X_test, y_train_val, y_test = prepare_data(args)

    # Algorithm Routing & Directory
    run_pipeline = get_pipeline_runner(args.model, args.optimizer, args.mode)
    out_dir, rep_dir = get_output_directories(args)

    # Gather common parameters (Extract kwargs from argparse)
    pipeline_kwargs = vars(args).copy()
    pipeline_kwargs['tscv_test_size'] = len(X_test) if args.validation == "walk_forward" else None
    sig = inspect.signature(run_pipeline)
    valid_kwargs = {k: v for k, v in pipeline_kwargs.items() if k in sig.parameters}
    # Implement Training & Assessment
    print("\n Training Model ")
    if args.validation == "holdout":
        X_train, X_val, y_train, y_val = split_holdout(X_train_val, y_train_val, args)
        print(f"Holdout Splits -> Train: {len(X_train)} | Val: {len(X_val)}")

        # Run the pipeline (pass parameters automatically using kwargs)
        run_pipeline(X_train, y_train, X_val, y_val, out_dir, rep_dir, **valid_kwargs)
        load_and_evaluate_model(args, X_test, y_test, out_dir)

    elif args.validation == "walk_forward":
        if 'is_augmented' in X_train_val.columns:
            X_train_val = X_train_val.drop(columns=['is_augmented'])

        print(f"Time-Series CV Splits -> Modelling Pool: {len(X_train_val)}")

        run_pipeline(X_train_val, y_train_val, None, None, out_dir, rep_dir, **valid_kwargs)

        # Assessment based on the Quarantine Test (Last chunk never used)
        load_and_evaluate_model(args, X_test, y_test, out_dir)

    print("\n--- Xong! ---")


if __name__ == "__main__":
    main()