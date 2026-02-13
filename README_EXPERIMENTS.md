# CORAL Loss Experiments for Walkability Prediction

This directory contains scripts for running systematic experiments comparing different deep learning architectures for ordinal regression using CORAL loss.

## Quick Start

### Run on MacBook with MPS (Apple Silicon GPU)

```bash
# Quick test with default settings (GPU enabled, 500 train/200 test samples)
python run_experiments.py

# Run specific experiments
python run_experiments.py --experiments coral_resnet18 ce_resnet18 coral_resnet50

# Full dataset training (recommended for real experiments)
python run_experiments.py --full-data --epochs 20

# Custom sample sizes
python run_experiments.py --train-samples 5000 --test-samples 1000 --epochs 15
```

### Run on CPU (for testing)

```bash
python run_experiments.py --cpu
```

## MPS (Metal Performance Shaders) on MacBook

**Apple Silicon (M1/M2/M3) Macs have built-in GPU acceleration via MPS!**

### Benefits of using MPS:
- ✅ **3-5x faster** training compared to CPU
- ✅ **Automatically detected** by PyTorch and AutoGluon
- ✅ **Lower power consumption** than external GPUs
- ✅ **Unified memory** - shares RAM with CPU efficiently

### How it works:
1. **Default behavior**: The script automatically uses MPS when available
2. **No CUDA required**: MPS is PyTorch's native macOS GPU backend
3. **Automatic fallback**: If MPS isn't available, uses CPU automatically

### To verify MPS is working:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"
```

### Optimal settings for MPS:
```bash
# Recommended configuration for M1/M2/M3 MacBooks
python run_experiments.py \
    --full-data \
    --epochs 20 \
    --experiments coral_resnet18 coral_resnet50 coral_efficientnet_b0
```

### Batch size considerations:
- **M1/M2 (8GB RAM)**: batch_size=32 (default)
- **M1/M2 Pro/Max (16-32GB)**: Can increase to 64-128
- **M3 Max (36GB+)**: Can increase to 128-256

To modify batch size, edit `run_experiments.py` line 304:
```python
batch_size = 32  # or 64, 128 for more RAM
```

## Available Experiments

The script includes 12 pre-configured experiments:

### Loss Functions:
- **CORAL**: Ordinal regression (cumulative link model)
- **CrossEntropy**: Standard multiclass classification

### Image Backbones:
| Experiment | Backbone | Size | Speed | Notes |
|---|---|---|---|---|
| `coral_resnet18` | ResNet-18 | 11M | Fast | Baseline |
| `coral_resnet50` | ResNet-50 | 25M | Medium | Deeper CNN |
| `coral_efficientnet_b0` | EfficientNet-B0 | 5M | Fast | Efficient scaling |
| `coral_convnext_tiny` | ConvNeXt-Tiny | 28M | Medium | Modern ConvNet |
| `coral_swin_base` | Swin-Base | 88M | Slow | Vision Transformer |
| `coral_mobilenetv3` | MobileNetV3 | 5M | Fast | Mobile-optimized |
| `coral_vit_clip` | ViT-Base CLIP | 87M | Slow | CLIP pretrained |

### Text/Tabular Encoders:
| Experiment | Encoder | Size | Notes |
|---|---|---|---|
| `coral_resnet18_electra_small` | ELECTRA-Small | 14M | Better tabular |
| `coral_resnet18_deberta_small` | DeBERTa-v3-Small | 44M | Strong text |

### Architectural Variations:
| Experiment | Feature | Notes |
|---|---|---|
| `coral_resnet50_transformer_fusion` | Transformer Fusion | Cross-attention between modalities |
| `coral_resnet18_ft_transformer` | FT-Transformer | Transformer for tabular data |

### Baseline Comparison:
| Experiment | Loss | Purpose |
|---|---|---|
| `ce_resnet18` | CrossEntropy | Compare CORAL vs standard classification |

## Command-Line Options

```
usage: run_experiments.py [-h] [--cpu] [--full-data] 
                          [--train-samples TRAIN_SAMPLES]
                          [--test-samples TEST_SAMPLES] 
                          [--epochs EPOCHS]
                          [--experiments EXPERIMENTS [EXPERIMENTS ...]]
                          [--output OUTPUT]

Options:
  --cpu                     Force CPU training (disable MPS)
  --full-data              Use full dataset (~24K train / 6K test)
  --train-samples N        Number of training samples (default: 500, -1 for all)
  --test-samples N         Number of test samples (default: 200, -1 for all)
  --epochs N               Maximum epochs (default: 10)
  --experiments EXP [...]  Which experiments to run (default: coral_resnet18)
  --output FILE            Output CSV filename (default: experiment_results.csv)
```

## Example Workflows

### 1. Quick Test (2-3 minutes on MPS)
```bash
python run_experiments.py --experiments coral_resnet18
```

### 2. Compare Loss Functions (5 minutes)
```bash
python run_experiments.py \
    --experiments coral_resnet18 ce_resnet18 \
    --train-samples 2000 \
    --test-samples 500 \
    --epochs 10
```

### 3. Compare Architectures (15-20 minutes)
```bash
python run_experiments.py \
    --experiments coral_resnet18 coral_resnet50 coral_efficientnet_b0 coral_mobilenetv3 \
    --train-samples 5000 \
    --test-samples 1000 \
    --epochs 15
```

### 4. Full Experiment Suite (1-2 hours on MPS)
```bash
python run_experiments.py \
    --full-data \
    --epochs 20 \
    --experiments coral_resnet18 ce_resnet18 coral_resnet50 \
                  coral_efficientnet_b0 coral_convnext_tiny \
                  coral_resnet18_electra_small \
                  coral_resnet50_transformer_fusion \
    --output full_experiment_results.csv
```

### 5. CPU Testing (slower, but good for debugging)
```bash
python run_experiments.py \
    --cpu \
    --train-samples 200 \
    --test-samples 100 \
    --epochs 5 \
    --experiments coral_resnet18
```

## Output

### Console Output
The script provides detailed progress tracking:
- **Timestamped logs** for each major step
- **Configuration summary** for each experiment
- **Training progress** (via AutoGluon verbosity=3)
- **Validation metrics** during training
- **Final metrics** (accuracy, within-one accuracy, MAE, QWK)
- **Confusion matrix** and error distribution
- **Per-class performance** analysis

### CSV Results
Results are saved to `experiment_results.csv` (or specified filename):
```csv
experiment,loss,image_backbone,text_encoder,fusion,accuracy,within_one_accuracy,mae,quadratic_kappa,time_sec
coral_resnet18,coral,resnet18,prajjwal1/bert-tiny,mlp,0.4500,0.8500,0.6200,0.7800,145.3
...
```

### Model Checkpoints
Trained models are saved to `./experiment_models/{experiment_name}/`

## Understanding the Metrics

### For Ordinal Regression (Rating 1-5):

- **Exact Accuracy**: Percentage of perfect predictions
  - Good: >40%, Excellent: >50%
  
- **Within-One Accuracy**: Percentage of predictions within ±1 of true rating
  - Good: >80%, Excellent: >90%
  - **Most important metric for ordinal data!**
  
- **MAE (Mean Absolute Error)**: Average distance from true rating
  - Good: <0.7, Excellent: <0.5
  
- **Quadratic Kappa**: Agreement metric that penalizes larger errors more
  - Good: >0.6, Excellent: >0.8

## Troubleshooting

### MPS not working?
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, update PyTorch
pip install --upgrade torch torchvision torchaudio

# If still False, you may need PyTorch 2.0+ for MPS support
```

### Out of memory on MPS?
- Reduce batch size in `run_experiments.py` (line 304)
- Use smaller models (resnet18, mobilenetv3)
- Reduce train/test samples

### Slow progress?
- Make sure you're using MPS (not --cpu flag)
- Check Activity Monitor → GPU History to verify GPU usage
- Increase num_workers (line 305) for faster data loading

### Import errors?
```bash
# Reinstall AutoGluon Multimodal
cd /Users/ahmademami/Library/CloudStorage/OneDrive-UNSW/TERM\ 1/Github/aglone/autogluon
pip install -e multimodal/
```

## Advanced: Running Custom Experiments

To add new experiments, edit `run_experiments.py` around line 330:

```python
# Add your custom experiment
experiments["my_custom_experiment"] = {
    **base_config,
    "optim.loss_func": "coral",
    "model.timm_image.checkpoint_name": "resnet34",  # Any timm model
    "model.hf_text.checkpoint_name": "bert-base-uncased",  # Any HF model
    "optim.learning_rate": 5e-5,  # Custom learning rate
}
```

Then run:
```bash
python run_experiments.py --experiments my_custom_experiment
```

## Performance Benchmarks

Approximate training times on M2 MacBook Pro (16GB, MPS):

| Configuration | Samples | Epochs | Time | Accuracy |
|---|---|---|---|---|
| ResNet18 (quick test) | 500 / 200 | 10 | ~3 min | ~45% |
| ResNet18 (medium) | 5000 / 1000 | 15 | ~20 min | ~52% |
| ResNet18 (full) | 24K / 6K | 20 | ~90 min | ~58% |
| ResNet50 (full) | 24K / 6K | 20 | ~150 min | ~61% |
| EfficientNet-B0 (full) | 24K / 6K | 20 | ~120 min | ~60% |

CPU times are typically 3-5x slower.

## Next Steps

1. **Start with quick test**: `python run_experiments.py`
2. **Compare a few models**: `python run_experiments.py --experiments coral_resnet18 coral_resnet50 coral_efficientnet_b0 --train-samples 5000`
3. **Run full experiment**: `python run_experiments.py --full-data --epochs 20 --experiments <best_models>`
4. **Analyze results**: Check CSV output and confusion matrices

## Citation

If using CORAL loss in research, cite:
```
Cao, W., Mirjalili, V., & Raschka, S. (2020). 
Rank consistent ordinal regression for neural networks with application to age estimation. 
Pattern Recognition Letters, 140, 325-331.
```

For AutoGluon:
```
@article{agtabular,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}
```
