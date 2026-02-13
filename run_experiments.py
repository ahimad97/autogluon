#!/usr/bin/env python3
"""
CORAL Loss Experiments: Walkability Score Prediction

This script runs systematic experiments comparing different image backbones, 
tabular encoders, and fusion mechanisms using CORAL ordinal regression loss
for predicting walkability scores (1-5) from street-view images and user demographics.

Data: ~24K training / ~6K test samples with:
  - Image: Street-view photographs (image_path)
  - Tabular: age, gender, childhood_country, childhood_area, disability, walking_frequency, residence_type
  - Label: rating (ordinal 1-5)

Usage:
    python run_experiments.py                    # Run default experiments
    python run_experiments.py --full-data        # Use full dataset
    python run_experiments.py --cpu              # Force CPU (no MPS)
    python run_experiments.py --experiments coral_resnet18 coral_resnet50  # Run specific experiments
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime

# Mock pynvml to prevent NVML errors on macOS (before any other imports!)
class MockPynvml:
    """Mock pynvml module to prevent NVML errors on systems without NVIDIA GPUs."""
    def nvmlInit(self):
        raise RuntimeError("NVML not available")
    def __getattr__(self, name):
        def _mock(*args, **kwargs):
            return None
        return _mock

sys.modules['pynvml'] = MockPynvml()

# Disable GPU stats monitoring to avoid NVML errors on macOS
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["PL_DEV_DEBUG"] = "0"

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score, confusion_matrix

# Set PyTorch default dtype to float32 (MPS doesn't support float64)
import torch
torch.set_default_dtype(torch.float32)

from autogluon.multimodal import MultiModalPredictor

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
TRAIN_CSV = r'/Users/ahmademami/Library/CloudStorage/OneDrive-UNSW/TERM 1/Github/aglone/train_data.csv'
TEST_CSV  = r'/Users/ahmademami/Library/CloudStorage/OneDrive-UNSW/TERM 1/Github/aglone/test_data.csv'
IMG_DIR   = r'/Users/ahmademami/Library/CloudStorage/OneDrive-UNSW/TERM 1/Github/aglone/images'

# Data columns
LABEL_COL = "rating"
IMAGE_COL = "image_path"
DROP_COLS = ["response_id"]  # Non-feature columns

# Training defaults (can be overridden by CLI args)
MAX_EPOCHS = 10
TIME_LIMIT = 1800  # 30 minutes per experiment
TRAIN_SAMPLE = 500  # Set to None for full training
TEST_SAMPLE = 200   # Set to None for full evaluation

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log(msg, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def to_abs(path: str, base: str) -> str:
    """Convert relative path to absolute."""
    p = str(path).strip()
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base, p))


def add_abs_image_paths(df, image_col, base_dir):
    """Add absolute paths to image column."""
    df = df.copy()
    df[image_col] = df[image_col].astype(str).map(lambda p: to_abs(p, base_dir))
    return df


def compute_ordinal_metrics(y_true, y_pred):
    """Compute all ordinal-relevant metrics."""
    # Convert to numpy arrays and ensure float32 (MPS doesn't support float64)
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    
    acc = accuracy_score(y_true, y_pred)
    within_one = np.mean(np.abs(y_true - y_pred) <= 1)
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return {
        "accuracy": float(acc),
        "within_one_accuracy": float(within_one),
        "mae": float(mae),
        "quadratic_kappa": float(qwk),
    }


def print_metrics(metrics, name=""):
    """Pretty print metrics."""
    print(f"\n{'='*70}")
    if name:
        print(f"  RESULTS: {name}")
        print(f"{'='*70}")
    print(f"  Exact Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Within-One Accuracy:  {metrics['within_one_accuracy']:.4f} ({metrics['within_one_accuracy']*100:.2f}%)")
    print(f"  MAE:                  {metrics['mae']:.4f}")
    print(f"  Quadratic Kappa:      {metrics['quadratic_kappa']:.4f}")
    if 'time_sec' in metrics:
        print(f"  Training Time:        {metrics['time_sec']:.1f}s ({metrics['time_sec']/60:.1f} min)")
    print(f"{'='*70}\n")


def print_confusion_matrix(y_true, y_pred, classes):
    """Print confusion matrix analysis."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(
        cm, 
        index=[f"True {c}" for c in classes], 
        columns=[f"Pred {c}" for c in classes]
    )
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Error distribution
    errors = y_pred - y_true
    print(f"\nError Distribution (Predicted - True):")
    error_counts = pd.Series(errors).value_counts().sort_index()
    max_count = max(error_counts)
    for err, count in error_counts.items():
        pct = count / len(errors) * 100
        bar = "█" * int(count / max_count * 40)
        print(f"  {err:+2d}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    print(f"{'Class':>6} {'Count':>6} {'Accuracy':>9} {'Within-1':>9} {'MAE':>8}")
    print("-" * 50)
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == c).mean()
            class_w1 = (np.abs(y_pred[mask] - c) <= 1).mean()
            class_mae = np.abs(y_pred[mask] - c).mean()
            print(f"{c:>6} {mask.sum():>6} {class_acc:>9.3f} {class_w1:>9.3f} {class_mae:>8.3f}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(train_sample=None, test_sample=None):
    """Load and prepare training and test data."""
    log("Loading data...")
    
    train_df = pd.read_csv(TRAIN_CSV, index_col=0)
    test_df = pd.read_csv(TEST_CSV, index_col=0)
    
    # Drop non-feature columns
    train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])
    
    # Sample if requested
    if train_sample and train_sample < len(train_df):
        log(f"Sampling {train_sample} training examples (full dataset: {len(train_df)})")
        train_df = train_df.sample(n=train_sample, random_state=42).reset_index(drop=True)
    else:
        log(f"Using full training set: {len(train_df)} samples")
    
    if test_sample and test_sample < len(test_df):
        log(f"Sampling {test_sample} test examples (full dataset: {len(test_df)})")
        test_df = test_df.sample(n=test_sample, random_state=42).reset_index(drop=True)
    else:
        log(f"Using full test set: {len(test_df)} samples")
    
    # Add absolute image paths
    train_df = add_abs_image_paths(train_df, IMAGE_COL, IMG_DIR)
    test_df = add_abs_image_paths(test_df, IMAGE_COL, IMG_DIR)
    
    # Verify images exist
    missing_train = (~train_df[IMAGE_COL].map(os.path.exists)).sum()
    missing_test = (~test_df[IMAGE_COL].map(os.path.exists)).sum()
    
    if missing_train > 0:
        log(f"WARNING: {missing_train}/{len(train_df)} training images not found", "WARN")
    if missing_test > 0:
        log(f"WARNING: {missing_test}/{len(test_df)} test images not found", "WARN")
    
    # Data summary
    log(f"Data loaded successfully")
    log(f"  Train: {len(train_df)} samples")
    log(f"  Test:  {len(test_df)} samples")
    log(f"  Columns: {list(train_df.columns)}")
    
    print("\nLabel distribution (train):")
    print(train_df[LABEL_COL].value_counts().sort_index())
    
    return train_df, test_df


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

def get_experiments(use_gpu=True, max_epochs=MAX_EPOCHS):
    """
    Define experiment configurations.
    
    Each experiment tests a different combination of:
      - Loss function (CORAL vs CrossEntropy)
      - Image backbone
      - Text/tabular encoder
      - Fusion mechanism (MLP vs Transformer)
    """
    
    # GPU configuration for macOS MPS (Metal Performance Shaders)
    # AutoGluon will automatically detect and use MPS on M1/M2/M3 Macs
    if use_gpu:
        num_gpus = 1
        batch_size = 32
        num_workers = 4
        log("GPU mode enabled - will use MPS on macOS")
    else:
        num_gpus = 0
        batch_size = 8
        num_workers = 0
        log("CPU mode - for faster training, remove --cpu flag to use MPS")
    
    # Base configuration
    base_config = {
        "optim.max_epochs": max_epochs,
        "optim.lr": 1e-4,  # Learning rate
        "env.num_gpus": num_gpus,
        "env.per_gpu_batch_size": batch_size,
        "env.num_workers": num_workers,
        "env.num_workers_inference": num_workers,
        "optim.val_check_interval": 0.5,  # Validate twice per epoch
    }
    
    experiments = {}
    
    # 1. CORAL + ResNet18 (small baseline)
    experiments["coral_resnet18"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "resnet18",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    # 2. CrossEntropy + ResNet18 (baseline comparison)
    experiments["ce_resnet18"] = {
        **base_config,
        "model.timm_image.checkpoint_name": "resnet18",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    # 3. CORAL + ResNet50 (stronger CNN)
    experiments["coral_resnet50"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "resnet50",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    # 4. CORAL + EfficientNet-B0 (efficient architecture)
    experiments["coral_efficientnet_b0"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "efficientnet_b0",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    # 5. CORAL + ConvNeXt-Tiny (modern ConvNet)
    experiments["coral_convnext_tiny"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "convnext_tiny",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    # 6. CORAL + Swin-Base (Vision Transformer)
    experiments["coral_swin_base"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    # 7. CORAL + ResNet18 + ELECTRA-Small (better text encoder)
    experiments["coral_resnet18_electra_small"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "resnet18",
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
    }
    
    # 8. CORAL + ResNet18 + DeBERTa-v3-small (strong text encoder)
    experiments["coral_resnet18_deberta_small"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "resnet18",
        "model.hf_text.checkpoint_name": "microsoft/deberta-v3-small",
    }
    
    # 9. CORAL + ResNet50 + Transformer Fusion
    experiments["coral_resnet50_transformer_fusion"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "resnet50",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
        "model.names": ["timm_image", "hf_text", "fusion_transformer"],
        "model.fusion_transformer.hidden_size": 128,
        "model.fusion_transformer.num_blocks": 2,
    }
    
    # 10. CORAL + ResNet18 + FT-Transformer (tabular-specific)
    experiments["coral_resnet18_ft_transformer"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "resnet18",
        "model.names": ["timm_image", "ft_transformer", "fusion_mlp"],
        "model.ft_transformer.num_blocks": 2,
    }
    
    # 11. CORAL + MobileNetV3 (lightweight, fast)
    experiments["coral_mobilenetv3"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "mobilenetv3_large_100",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    # 12. CORAL + ViT-Base CLIP (CLIP-pretrained vision)
    experiments["coral_vit_clip"] = {
        **base_config,
        "optim.loss_func": "coral",
        "model.timm_image.checkpoint_name": "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    }
    
    return experiments


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

def run_experiment(exp_name, config, train_df, test_df, save_dir="./experiment_models"):
    """Run a single experiment and return results."""
    
    log(f"Starting experiment: {exp_name}", "EXPERIMENT")
    print("\n" + "="*70)
    print(f"  EXPERIMENT: {exp_name}")
    print("="*70)
    
    # Print config summary
    loss = config.get("optim.loss_func", "cross_entropy")
    img = config.get("model.timm_image.checkpoint_name", "default")
    txt = config.get("model.hf_text.checkpoint_name", "N/A")
    fusion = "transformer" if "fusion_transformer" in str(config.get("model.names", "")) else "mlp"
    epochs = config.get("optim.max_epochs", MAX_EPOCHS)
    batch = config.get("env.per_gpu_batch_size", 8)
    gpus = config.get("env.num_gpus", 0)
    
    print(f"\nConfiguration:")
    print(f"  Loss:           {loss}")
    print(f"  Image Backbone: {img}")
    print(f"  Text Encoder:   {txt}")
    print(f"  Fusion:         {fusion}")
    print(f"  Max Epochs:     {epochs}")
    print(f"  Batch Size:     {batch}")
    print(f"  GPUs:           {gpus} {'(MPS on macOS)' if gpus > 0 else '(CPU)'}")
    print(f"  Train Samples:  {len(train_df)}")
    print(f"  Test Samples:   {len(test_df)}")
    print()
    
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Remove existing experiment directory to avoid conflicts
    if os.path.exists(save_path):
        import shutil
        log(f"Removing existing model directory: {save_path}", "INFO")
        shutil.rmtree(save_path)
    
    try:
        start_time = time.time()
        
        # Initialize predictor
        # Note: Using verbosity=2 instead of 3 to avoid NVML errors on macOS
        log("Initializing MultiModalPredictor...")
        predictor = MultiModalPredictor(
            label=LABEL_COL,
            problem_type="multiclass",
            eval_metric="accuracy",
            path=save_path,
            verbosity=2,  # Verbosity 2 for progress without GPU checks
        )
        
        # Train with progress tracking
        log("Starting training...")
        predictor.fit(
            train_data=train_df,
            hyperparameters=config,
            time_limit=TIME_LIMIT,
        )
        
        train_time = time.time() - start_time
        log(f"Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")
        
        # Predict on test set
        log("Running predictions on test set...")
        pred_start = time.time()
        preds = predictor.predict(test_df.drop(columns=[LABEL_COL]))
        pred_time = time.time() - pred_start
        log(f"Predictions completed in {pred_time:.1f}s")
        
        # Compute metrics
        log("Computing metrics...")
        metrics = compute_ordinal_metrics(test_df[LABEL_COL].values, preds.values)
        metrics.update({
            "experiment": exp_name,
            "loss": loss,
            "image_backbone": img,
            "text_encoder": txt,
            "fusion": fusion,
            "time_sec": round(train_time, 1),
            "pred_time_sec": round(pred_time, 1),
        })
        
        print_metrics(metrics, name=exp_name)
        
        # Detailed analysis
        classes = sorted(test_df[LABEL_COL].unique())
        print_confusion_matrix(test_df[LABEL_COL].values, preds.values, classes)
        
        # Probability analysis
        try:
            log("Computing prediction probabilities...")
            probs = predictor.predict_proba(test_df.drop(columns=[LABEL_COL]))
            mean_confidence = probs.max(axis=1).mean()
            print(f"\nProbability Analysis:")
            print(f"  Probabilities sum to 1: {np.allclose(probs.sum(axis=1), 1.0)}")
            print(f"  Mean confidence (max prob): {mean_confidence:.4f}")
            print(f"\nSample predictions (first 5):")
            sample_df = pd.DataFrame({
                'True': test_df[LABEL_COL].values[:5],
                'Pred': preds.values[:5],
            })
            for i, col in enumerate(probs.columns[:5], 1):
                sample_df[f'P(class={i})'] = probs[col].values[:5]
            print(sample_df.to_string(index=False, float_format="%.3f"))
        except Exception as e:            log(f"Could not compute probabilities: {e}", "WARN")
        
        log(f"✅ Experiment {exp_name} completed successfully", "SUCCESS")
        return metrics, predictor
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = str(e)
        
        log(f"❌ Experiment {exp_name} FAILED after {error_time:.1f}s", "ERROR")
        log(f"Error: {error_msg}", "ERROR")
        
        # Note: NVML errors on macOS are harmless warnings about missing NVIDIA drivers
        if "NVML" in error_msg:
            log("Note: NVML warnings are expected on macOS (no NVIDIA GPUs). Try running without --cpu to use MPS GPU", "INFO")
        
        metrics = {
            "experiment": exp_name,
            "accuracy": None,
            "within_one_accuracy": None,
            "mae": None,
            "quadratic_kappa": None,
            "loss": loss,
            "image_backbone": img,
            "text_encoder": txt,
            "fusion": fusion,
            "time_sec": round(error_time, 1),
            "error": error_msg,
        }
        return metrics, None


def run_all_experiments(experiments_to_run, train_df, test_df, experiments_config):
    """Run multiple experiments and collect results."""
    
    log(f"Running {len(experiments_to_run)} experiments")
    log(f"Experiments: {', '.join(experiments_to_run)}")
    
    results = []
    predictors = {}
    
    for i, exp_name in enumerate(experiments_to_run, 1):
        print("\n\n" + "#"*70)
        print(f"# EXPERIMENT {i}/{len(experiments_to_run)}: {exp_name}")
        print("#"*70 + "\n")
        
        if exp_name not in experiments_config:
            log(f"Unknown experiment: {exp_name}", "ERROR")
            continue
        
        config = experiments_config[exp_name]
        metrics, predictor = run_experiment(exp_name, config, train_df, test_df)
        
        results.append(metrics)
        if predictor is not None:
            predictors[exp_name] = predictor
    
    return results, predictors


# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def analyze_results(results_df):
    """Analyze and display experiment results."""
    
    print("\n\n" + "="*100)
    print(" "*40 + "FINAL RESULTS")
    print("="*100 + "\n")
    
    # Sort by within-one accuracy
    display_cols = [
        "experiment", "loss", "image_backbone", "fusion",
        "accuracy", "within_one_accuracy", "mae", "quadratic_kappa", "time_sec"
    ]
    available_cols = [c for c in display_cols if c in results_df.columns]
    results_sorted = results_df[available_cols].sort_values(
        "within_one_accuracy", ascending=False, na_position='last'
    )
    
    print("ALL EXPERIMENTS (sorted by Within-One Accuracy)")
    print("-"*100)
    print(results_sorted.to_string(index=False, float_format="%.4f"))
    print("-"*100)
    
    # Highlight best results
    valid_results = results_df[results_df["within_one_accuracy"].notna()]
    if len(valid_results) > 0:
        best_w1 = valid_results.loc[valid_results["within_one_accuracy"].idxmax()]
        best_acc = valid_results.loc[valid_results["accuracy"].idxmax()]
        best_mae = valid_results.loc[valid_results["mae"].idxmin()]
        best_qwk = valid_results.loc[valid_results["quadratic_kappa"].idxmax()]
        
        print("\n" + "="*100)
        print("BEST RESULTS")
        print("="*100)
        print(f"🏆 Best Within-One Accuracy: {best_w1['experiment']:40s} {best_w1['within_one_accuracy']:.4f} ({best_w1['within_one_accuracy']*100:.2f}%)")
        print(f"🏆 Best Exact Accuracy:      {best_acc['experiment']:40s} {best_acc['accuracy']:.4f} ({best_acc['accuracy']*100:.2f}%)")
        print(f"🏆 Best MAE:                 {best_mae['experiment']:40s} {best_mae['mae']:.4f}")
        print(f"🏆 Best Quadratic Kappa:     {best_qwk['experiment']:40s} {best_qwk['quadratic_kappa']:.4f}")
        print("="*100)
    
    # CORAL vs CrossEntropy comparison
    coral_results = results_df[results_df["loss"] == "coral"]
    ce_results = results_df[results_df["loss"] == "cross_entropy"]
    
    if len(coral_results) > 0 and len(ce_results) > 0:
        print("\n" + "="*100)
        print("CORAL vs CROSS-ENTROPY LOSS (averaged across experiments)")
        print("="*100)
        for metric in ["accuracy", "within_one_accuracy", "mae", "quadratic_kappa"]:
            coral_vals = coral_results[metric].dropna()
            ce_vals = ce_results[metric].dropna()
            if len(coral_vals) > 0 and len(ce_vals) > 0:
                coral_avg = coral_vals.mean()
                ce_avg = ce_vals.mean()
                better = "CORAL" if (coral_avg > ce_avg if metric != "mae" else coral_avg < ce_avg) else "CE"
                diff = abs(coral_avg - ce_avg)
                print(f"  {metric:25s}: CORAL={coral_avg:.4f}  CE={ce_avg:.4f}  Δ={diff:.4f}  → {better} wins")
        print("="*100)


def save_results(results_df, filename="experiment_results.csv"):
    """Save results to CSV file."""
    results_df.to_csv(filename, index=False)
    log(f"Results saved to {filename}")
    print(f"\nResults file: {os.path.abspath(filename)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run CORAL loss experiments for walkability prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                              # Run default experiments with GPU
  python run_experiments.py --cpu                        # Use CPU only
  python run_experiments.py --full-data                  # Use full dataset
  python run_experiments.py --experiments coral_resnet18 ce_resnet18  # Run specific experiments
  python run_experiments.py --epochs 20 --train-samples 5000  # Custom settings
        """
    )
    
    parser.add_argument(
        '--cpu', 
        action='store_true',
        help='Force CPU training (disable MPS GPU)'
    )
    parser.add_argument(
        '--full-data',
        action='store_true',
        help='Use full dataset instead of sampling'
    )
    parser.add_argument(
        '--train-samples',
        type=int,
        default=TRAIN_SAMPLE,
        help=f'Number of training samples (default: {TRAIN_SAMPLE}, use -1 for all)'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=TEST_SAMPLE,
        help=f'Number of test samples (default: {TEST_SAMPLE}, use -1 for all)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=MAX_EPOCHS,
        help=f'Maximum training epochs (default: {MAX_EPOCHS})'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=['coral_resnet18'],
        help='Which experiments to run (default: coral_resnet18)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiment_results.csv',
        help='Output CSV filename (default: experiment_results.csv)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print(" "*15 + "CORAL LOSS EXPERIMENTS")
    print(" "*10 + "Walkability Score Prediction")
    print("="*70 + "\n")
    
    log("Starting experiment suite")
    log(f"Configuration: {'CPU' if args.cpu else 'GPU (MPS)'} mode, {args.epochs} epochs")
    
    # Handle full data flag
    if args.full_data:
        train_samples = None
        test_samples = None
    else:
        train_samples = None if args.train_samples == -1 else args.train_samples
        test_samples = None if args.test_samples == -1 else args.test_samples
    
    # Load data
    train_df, test_df = load_data(train_samples, test_samples)
    
    # Get experiment configurations
    use_gpu = not args.cpu
    experiments_config = get_experiments(use_gpu=use_gpu, max_epochs=args.epochs)
    
    log(f"Available experiments: {len(experiments_config)}")
    for name in experiments_config:
        print(f"  - {name}")
    
    # Validate requested experiments
    invalid = [e for e in args.experiments if e not in experiments_config]
    if invalid:
        log(f"Unknown experiments: {invalid}", "ERROR")
        log(f"Available: {list(experiments_config.keys())}", "ERROR")
        sys.exit(1)
    
    # Run experiments
    results, predictors = run_all_experiments(
        args.experiments, 
        train_df, 
        test_df, 
        experiments_config
    )
    
    # Analyze results
    results_df = pd.DataFrame(results)
    analyze_results(results_df)
    
    # Save results
    save_results(results_df, args.output)
    
    log("All experiments completed", "SUCCESS")
    print("\n" + "="*70)
    print(" "*20 + "EXPERIMENTS FINISHED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
