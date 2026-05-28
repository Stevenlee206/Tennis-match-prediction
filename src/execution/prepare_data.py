from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import Preprocessing


def prepare_data(args):
    """
    Args:
        args: Đối tượng cấu hình từ argparse chứa test_size, val_size, n_splits, validation_mode

    Returns:
        X_train_val (DataFrame).
        X_test (DataFrame).
        y_train_val (Series).
        y_test (Series).
    """
    if args.validation == "holdout":
        train_ratio = (1.0 - args.test_size) * (1.0 - args.val_size)
    else:
        # To prevent leakage into the first TSCV validation fold, imputation must only fit on the INITIAL training window.
        # Initial window = Total dataset - Test Set - (n_splits * test_size equivalent folds)
        train_ratio = (1.0 - args.test_size) - (args.n_splits * args.test_size)

        # Safety bound if args cause ratio to collapse
        if train_ratio <= 0.1:
            print("Warning: Walk-forward folds are large. Adjusting initial train window safely.")
            print("Config safe ratio")
            train_ratio = (1.0 - args.test_size) / 2.0

    print("---Preprocessing Data ---")
    prep = Preprocessing()
    data = prep.run(train_ratio=train_ratio)

    if 'target' not in data.columns:
        raise ValueError("Error: 'target' missing.")
    # Global test set extraction
    X_full = data.drop(columns=['target', 'year'], errors='ignore')
    y_full = data['target']
    X_train_val_pool, X_test, y_train_val, y_test = train_test_split(
        X_full, y_full, test_size=args.test_size, shuffle=False
    )


    # LỌC DỮ LIỆU TĂNG CƯỜNG KHỎI TẬP TEST (DE-AUGMENTATION)
    if 'is_augmented' in X_test.columns:
        y_test = y_test[X_test['is_augmented'] == 0]
        X_test = X_test[X_test['is_augmented'] == 0].drop(columns=['is_augmented'])

    print(f"\nGlobal Splits -> Modeling Pool: {len(X_train_val_pool)} | Quarantined Test: {len(X_test)}")

    return X_train_val_pool, X_test, y_train_val, y_test