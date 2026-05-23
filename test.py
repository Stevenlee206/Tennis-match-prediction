import argparse
import json
import joblib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import os
import contextlib
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Import từ project
from main import evaluate_model_bias
from src.preprocessing.preprocessing import Preprocessing
from src.models.svm.svm_sklearn_optuna import generate_sample_weights
from src.model.Predictive_Coding.pc_network import PCNetworkConfig
from src.model.Predictive_Coding.pc_network_torch import PredictiveCodingNetworkTorch
from src.models.pc.pc_optuna import train_pc_model_loop

def test_predictive_coding(train_dir):
    """
    Load best hyperparameters từ training folder, train lại model trên 80% pool,
    evaluate trên 20% quarantine test set, lưu output_test folder.
    """
    train_dir = Path(train_dir)
    
    # Auto-detect parent nếu user truyền file path
    if train_dir.is_file():
        train_dir = train_dir.parent
    
    config_path = train_dir / "pc_config.json"
    
    if not config_path.exists():
        print(f"Not found error: {train_dir} not contain pc_config.json")
        return
    
    print("\n" + "="*70)
    print(" PREDICTIVE CODING: RETRAIN + TEST ON UNSEEN 20% DATA")
    print("="*70)
    
    # Load config với best_params
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    best_params = cfg.get('best_params', {})
    model_params = cfg.get('model_params', {})
    
    print(f"\n[*] Loading best hyperparameters from: {train_dir.name}")
    print(f"    Learning Rate: {best_params.get('learning_rate'):.6f}")
    print(f"    Inference LR: {best_params.get('inference_lr'):.6f}")
    print(f"    Inference Steps: {best_params.get('inference_steps')}")
    print(f"    Hidden Activation: {best_params.get('hidden_activation')}")
    print(f"    Epochs: {model_params.get('epochs')}, Batch Size: {model_params.get('batch_size')}")
    print(f"\n[*] Network Architecture Configuration:")
    print(f"    - Depth (# Hidden Layers): {model_params.get('depth')}")
    print(f"    - Width (neurons per layer): {model_params.get('width')}")
    print(f"    - Hidden Sizes: {[model_params.get('width')] * model_params.get('depth')}")
    
    # === PREPARE DATA (80/20 SPLIT) ===
    print("\n" + "="*70)
    print(" STEP 1: PREPARE DATA (80% TRAIN + 20% TEST)")
    print("="*70)
    
    with open(os.devnull, 'w', encoding='utf-8') as f, contextlib.redirect_stdout(f):
        prep = Preprocessing()
        data = prep.run()
    
    X_full = data.drop(columns=['target', 'year'], errors='ignore')
    y_full = data['target']
    
    # Split 80/20 (không shuffle - keep temporal order)
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_full, y_full, test_size=0.20, shuffle=False
    )
    
    # Remove augmented data từ test
    if 'is_augmented' in X_test.columns:
        y_test = y_test[X_test['is_augmented'] == 0]
        X_test = X_test[X_test['is_augmented'] == 0].drop(columns=['is_augmented'])
    
    # Remove augmented từ training pool
    if 'is_augmented' in X_train_pool.columns:
        X_train_pool = X_train_pool.drop(columns=['is_augmented'])
    
    print(f"[*] Training Pool: {len(X_train_pool)} matches (80%)")
    print(f"[*] Test Set: {len(X_test)} matches (20%, quarantined)")
    
    # === PREPROCESS TRAIN POOL ===
    print("\n" + "="*70)
    print(" STEP 2: PREPROCESS & SCALE DATA")
    print("="*70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pool).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    y_train_np = y_train_pool.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.float32)
    
    print(f"[*] StandardScaler fit on training pool")
    print(f"[*] Feature dimension: {X_train_scaled.shape[1]}")
    
    # === BUILD MODEL ===
    print("\n" + "="*70)
    print(" STEP 3: INITIALIZE MODEL WITH BEST HYPERPARAMETERS")
    print("="*70)
    
    pc_cfg = PCNetworkConfig(
        learning_rate=best_params['learning_rate'],
        inference_lr=best_params['inference_lr'],
        inference_steps=best_params['inference_steps'],
        hidden_activation=best_params['hidden_activation'],
        random_seed=42
    )
    
    depth = model_params['depth']
    width = model_params['width']
    hidden_sizes = [width] * depth
    layer_sizes = [X_train_scaled.shape[1], *hidden_sizes, 1]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=pc_cfg, device=device)
    
    print(f"[*] Full Layer Architecture: {layer_sizes}")
    print(f"    Input Layer: {layer_sizes[0]} neurons")
    print(f"    Hidden Layers: {len(hidden_sizes)} layers × {width} neurons each")
    print(f"    Output Layer: {layer_sizes[-1]} neuron (binary)")
    print(f"[*] Device: {device.upper()}")
    
    # === TRAIN ===
    print("\n" + "="*70)
    print(" STEP 4: TRAIN ON 80% POOL")
    print("="*70)
    
    # Generate sample weights nếu cần
    weight_strategy = cfg.get('weight_strategy', 'none')
    upset_weight = cfg.get('upset_weight', 1.0)
    weights = generate_sample_weights(X_train_pool, y_train_pool, weight_strategy, upset_weight)
    
    epochs = model_params['epochs']
    batch_size = model_params['batch_size']
    
    t_start = time.perf_counter()
    best_train_acc, best_epoch, best_state, metrics_hist, train_time = train_pc_model_loop(
        model, X_train_scaled, y_train_np, X_test_scaled, y_test_np, 
        weights, max_epochs=epochs, batch_size=batch_size, verbose=True
    )
    t_total = time.perf_counter() - t_start
    
    # Load best state
    model.load_state_dict(best_state)
    
    print(f"[✓] Training complete in {t_total:.1f}s")
    
    # === EVALUATE ON TEST SET ===
    print("\n" + "="*70)
    print(" STEP 5: FINAL EVALUATION ON 20% TEST SET")
    print("="*70)
    
    probs = model.predict_proba(X_test_scaled)
    y_pred = (probs >= 0.5).astype(int)
    
    # Calculate detailed metrics using binary_classification_metrics
    from src.model.util.metrics import binary_classification_metrics
    test_metrics = binary_classification_metrics(y_test_np, probs)
    
    print("\nTEST METRICS SUMMARY:")
    for k, v in test_metrics.items():
        print(f" - {k:>20}: {v:.4f}")
    
    # Capture bias metrics from evaluate_model_bias
    import io
    f_log = io.StringIO()
    with contextlib.redirect_stdout(f_log):
        bias_metrics = evaluate_model_bias(y_test_np, y_pred, X_test)
    
    log_content = f_log.getvalue()
    print(log_content)
    
    # Combine all metrics
    metrics = {**test_metrics, **bias_metrics}
    
    # Add per-class precision/recall/f1 for JSON
    metrics['precision_class_0'] = round(precision_score(y_test_np, y_pred, pos_label=0, zero_division=0), 4)
    metrics['recall_class_0'] = round(recall_score(y_test_np, y_pred, pos_label=0, zero_division=0), 4)
    metrics['f1_class_0'] = round(f1_score(y_test_np, y_pred, pos_label=0, zero_division=0), 4)
    
    metrics['precision_class_1'] = round(precision_score(y_test_np, y_pred, pos_label=1, zero_division=0), 4)
    metrics['recall_class_1'] = round(recall_score(y_test_np, y_pred, pos_label=1, zero_division=0), 4)
    metrics['f1_class_1'] = round(f1_score(y_test_np, y_pred, pos_label=1, zero_division=0), 4)
    
    # === CONFIGURATION VERIFICATION ===
    print("\n" + "="*70)
    print(" CONFIGURATION VERIFICATION")
    print("="*70)
    print("\n[✓] HYPERPARAMETERS (Best Config):")
    print(f"    - Learning Rate: {best_params['learning_rate']:.6f}")
    print(f"    - Inference Learning Rate: {best_params['inference_lr']:.6f}")
    print(f"    - Inference Steps: {best_params['inference_steps']}")
    print(f"    - Hidden Activation: {best_params['hidden_activation']}")
    
    print("\n[✓] NETWORK STRUCTURE (Best Config):")
    print(f"    - Depth (Hidden Layers): {depth}")
    print(f"    - Width (Neurons/Layer): {width}")
    print(f"    - Full Architecture: {layer_sizes}")
    print(f"    - Total Parameters: ~{sum((layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]) for i in range(len(layer_sizes)-1)):,}")
    
    print("\n[✓] TRAINING CONFIG:")
    print(f"    - Epochs: {epochs}")
    print(f"    - Batch Size: {batch_size}")
    print(f"    - Weight Strategy: {weight_strategy}")
    print(f"    - Upset Weight: {upset_weight}")
    
    # === SAVE OUTPUT_TEST ===
    print("\n" + "="*70)
    print(" STEP 6: SAVE RESULTS TO output_test/")
    print("="*70)
    
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    output_test = train_dir.parent / f"output_test_{timestamp}"
    output_test.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_test_path = output_test / "pc_model_TEST.npz"
    best_state_save = {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v 
                       for k, v in best_state.items()}
    np.savez(model_test_path, **best_state_save)
    
    # Save scaler
    scaler_test_path = output_test / "pc_scaler_TEST.joblib"
    joblib.dump(scaler, scaler_test_path)
    
    # Save metrics
    metrics_test_path = output_test / "test_metrics.json"
    with open(metrics_test_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save config
    config_test_path = output_test / "test_config.json"
    test_config = {
        "model_type": "PredictiveCoding_TEST",
        "source_train_dir": str(train_dir),
        "best_params": best_params,
        "model_params": model_params,
        "architecture": {
            "input_dim": X_train_scaled.shape[1],
            "hidden_layers": depth,
            "width_per_layer": width,
            "hidden_sizes": hidden_sizes,
            "full_layer_sizes": layer_sizes,
            "output_dim": 1
        },
        "test_metrics": metrics,
        "training_info": {
            "epochs_trained": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "training_time_seconds": train_time,
            "total_time_seconds": t_total
        },
        "data_info": {
            "train_pool_size": len(X_train_pool),
            "test_size": len(X_test),
            "n_features": X_train_scaled.shape[1]
        }
    }
    with open(config_test_path, 'w') as f:
        json.dump(test_config, f, indent=4)
    
    # Save log
    log_test_path = output_test / "test_log.txt"
    with open(log_test_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PREDICTIVE CODING - FINAL TEST EVALUATION\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Training: {train_dir.name}\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION VERIFICATION:\n")
        f.write("-" * 70 + "\n")
        f.write("HYPERPARAMETERS (from Best Config):\n")
        f.write(f"  - Learning Rate: {best_params['learning_rate']:.6f}\n")
        f.write(f"  - Inference LR: {best_params['inference_lr']:.6f}\n")
        f.write(f"  - Inference Steps: {best_params['inference_steps']}\n")
        f.write(f"  - Hidden Activation: {best_params['hidden_activation']}\n\n")
        
        f.write("NETWORK ARCHITECTURE (from Best Config):\n")
        f.write(f"  - Depth (Hidden Layers): {depth}\n")
        f.write(f"  - Width (Neurons/Layer): {width}\n")
        f.write(f"  - Full Layer Sizes: {layer_sizes}\n")
        f.write(f"  - Input Features: {X_train_scaled.shape[1]}\n")
        f.write(f"  - Output: Binary (1 neuron)\n\n")
        
        f.write("TRAINING CONFIG:\n")
        f.write(f"  - Epochs: {epochs}\n")
        f.write(f"  - Batch Size: {batch_size}\n")
        f.write(f"  - Weight Strategy: {weight_strategy}\n")
        f.write(f"  - Best Epoch Reached: {best_epoch}\n")
        f.write(f"  - Training Time: {train_time:.1f}s\n\n")
        
        f.write("TEST METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(log_content)
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n[✓] Model weights saved:     {model_test_path.name}")
    print(f"[✓] Scaler saved:           {scaler_test_path.name}")
    print(f"[✓] Metrics saved:          {metrics_test_path.name}")
    print(f"[✓] Config saved:           {config_test_path.name}")
    print(f"[✓] Log saved:              {log_test_path.name}")
    print(f"\n[✓] All outputs in: {output_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain + Test Predictive Coding Model on Unseen 20% Test Data"
    )
    parser.add_argument(
        "--dir", 
        type=str, 
        required=True,
        help="Path to training output folder (e.g., outputs/predictive_coding/standard/optuna/holdout/20_05_...)"
    )
    
    args = parser.parse_args()
    test_predictive_coding(Path(args.dir))