# Quick Start Guide - Running Experiments on MacBook MPS

## 🚀 Fastest Way to Start

### Option 1: Using the Quick Run Script (Recommended)

```bash
# Activate conda environment
conda activate ag_env

# Change to project directory
cd "/Users/ahmademami/Library/CloudStorage/OneDrive-UNSW/TERM 1/Github/aglone/autogluon"

# Run quick test
./quick_run.sh test        # 2 minutes - verify everything works
./quick_run.sh quick       # 5 minutes - baseline experiment
./quick_run.sh medium      # 20 minutes - compare 3 architectures
./quick_run.sh full        # 2 hours - full dataset training
```

### Option 2: Using Python Script Directly

```bash
# Activate conda environment
conda activate ag_env

# Run with default settings (uses MPS GPU automatically)
python run_experiments.py

# Run specific experiments
python run_experiments.py --experiments coral_resnet18 coral_resnet50

# Full dataset
python run_experiments.py --full-data --epochs 20
```

## ✅ MPS GPU Verification

Your MacBook has MPS (Metal Performance Shaders) GPU support!
```
✅ PyTorch: 2.7.1
✅ MPS available: True
✅ MPS built: True
```

This means your training will be **3-5x faster** than CPU!

## 📊 What Gets Tracked During Training

The script provides **detailed progress tracking**:

### 1. Before Training
- ✅ Data summary (samples, columns, label distribution)
- ✅ Experiment configuration (model, loss, batch size, GPU)
- ✅ Missing image file warnings

### 2. During Training
- ✅ **Epoch progress** with loss values
- ✅ **Validation metrics** (accuracy, loss) every half epoch
- ✅ **Time per epoch** and estimated completion
- ✅ **Best model tracking** (saves best validation accuracy)
- ✅ **AutoGluon verbosity=3** for maximum detail

Example output during training:
```
[2026-02-13 15:30:42] [INFO] Starting training...
Epoch 1/10:  [=============>            ] 25% | Loss: 1.234 | Time: 00:05
Epoch 1/10:  [==========================] 50% | Val Acc: 0.432 | Val Loss: 1.123
Epoch 1/10:  [==================================] 100% | Loss: 1.098
Best model: epoch=1, val_accuracy=0.432
...
```

### 3. After Training
- ✅ **Final metrics** (accuracy, within-one accuracy, MAE, QWK)
- ✅ **Confusion matrix**
- ✅ **Error distribution** with visual bars
- ✅ **Per-class accuracy** breakdown
- ✅ **Probability analysis** (confidence scores)
- ✅ **Training time** summary

### 4. Final Summary
- ✅ **Results table** sorted by performance
- ✅ **Best model highlights** for each metric
- ✅ **CORAL vs CrossEntropy** comparison
- ✅ **CSV export** for further analysis

## 🔍 Real-Time Progress Monitoring

While training, you can monitor GPU usage:

### Terminal 1: Run experiment
```bash
conda activate ag_env
./quick_run.sh medium
```

### Terminal 2: Monitor GPU usage
```bash
# Watch GPU memory usage (updates every 2 seconds)
watch -n 2 "ps aux | grep python | grep -v grep | head -5"

# Or use Activity Monitor:
# Open Activity Monitor → Window → GPU History
```

You should see the **GPU History** graph jumping up when training, confirming MPS is being used!

## 📝 Example Runs

### Quick Test (2 min)
```bash
conda activate ag_env
./quick_run.sh test
```
**Output:**
- 200 training samples, 100 test samples
- 5 epochs, ResNet18 + CORAL
- Results in `test_results.csv`
- **Purpose:** Verify everything works

### Compare CORAL vs CrossEntropy (15 min)
```bash
conda activate ag_env
./quick_run.sh compare
```
**Output:**
- 5000 training, 1000 test samples
- Both CORAL and CE loss on same model
- Shows which loss is better for ordinal regression
- Results in `compare_results.csv`

### Architecture Comparison (20 min)
```bash
conda activate ag_env
./quick_run.sh medium
```
**Output:**
- 5000 training, 1000 test samples
- ResNet18, ResNet50, EfficientNet-B0
- 15 epochs each
- Shows which architecture works best
- Results in `medium_results.csv`

### Full Training (2 hours)
```bash
conda activate ag_env
./quick_run.sh full
```
**Output:**
- Full dataset (~24K training, 6K test)
- 20 epochs
- Best possible performance
- Results in `full_results.csv`

## 🎯 Custom Experiments

For custom configurations:

```bash
conda activate ag_env

# Custom sample size and epochs
python run_experiments.py \
  --train-samples 10000 \
  --test-samples 2000 \
  --epochs 25 \
  --experiments coral_resnet50 coral_efficientnet_b0

# Multiple experiments with custom output
python run_experiments.py \
  --experiments coral_resnet18 coral_resnet50 coral_convnext_tiny \
  --train-samples 8000 \
  --epochs 20 \
  --output my_experiments.csv

# CPU mode (for debugging)
python run_experiments.py --cpu --train-samples 200 --epochs 5
```

## 📈 Understanding the Output

### Console Output Structure

```
[2026-02-13 15:30:00] [INFO] Loading data...
[2026-02-13 15:30:05] [INFO] Using full training set: 24000 samples
[2026-02-13 15:30:05] [INFO] GPU mode enabled - will use MPS on macOS

======================================================================
  EXPERIMENT: coral_resnet18
======================================================================

Configuration:
  Loss:           coral
  Image Backbone: resnet18
  Text Encoder:   prajjwal1/bert-tiny
  Fusion:         mlp
  Max Epochs:     10
  Batch Size:     32
  GPUs:           1 (MPS on macOS)
  Train Samples:  24000
  Test Samples:   6000

[2026-02-13 15:30:10] [INFO] Initializing MultiModalPredictor...
[2026-02-13 15:30:15] [INFO] Starting training...

... (detailed training progress) ...

======================================================================
  RESULTS: coral_resnet18
======================================================================
  Exact Accuracy:       0.5234 (52.34%)
  Within-One Accuracy:  0.8567 (85.67%)
  MAE:                  0.5821
  Quadratic Kappa:      0.7234
  Training Time:        2456.3s (40.9 min)
======================================================================

Confusion Matrix:
           Pred 1  Pred 2  Pred 3  Pred 4  Pred 5
True 1        432      89      12       3       0
True 2        125     789     234      45       7
True 3         23     345     891     298      43
True 4          2      34     267     654     143
True 5          0       5      38     189     368

Error Distribution (Predicted - True):
  -2:   15 (1.2%) ██
  -1:  843 (14.1%) ██████████████
   0: 3141 (52.3%) ████████████████████████████████████████
  +1: 1789 (29.8%) ██████████████████████████
  +2:  201 (3.4%) ███
  +3:   11 (0.2%) █

...
```

### CSV Output Format

`experiment_results.csv`:
```csv
experiment,loss,image_backbone,text_encoder,fusion,accuracy,within_one_accuracy,mae,quadratic_kappa,time_sec
coral_resnet18,coral,resnet18,prajjwal1/bert-tiny,mlp,0.5234,0.8567,0.5821,0.7234,2456.3
coral_resnet50,coral,resnet50,prajjwal1/bert-tiny,mlp,0.5567,0.8798,0.5234,0.7567,4123.5
...
```

Open in Excel, Google Sheets, or:
```bash
# View in terminal
column -t -s, experiment_results.csv | less -S

# Or with pandas
python -c "import pandas as pd; df = pd.read_csv('experiment_results.csv'); print(df.to_string())"
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
**Solution:** Activate the conda environment first
```bash
conda activate ag_env
```

### "MPS not available"
**Solution:** Check PyTorch version and update if needed
```bash
conda activate ag_env
python -c "import torch; print(torch.__version__)"  # Should be 2.0+
# If older, update: pip install --upgrade torch torchvision
```

### Training is slow
**Solution:** Verify MPS is being used
```bash
# Should show: num_gpus=1 (MPS on macOS)
# If shows num_gpus=0, remove --cpu flag
```

### Out of memory
**Solutions:**
1. Reduce batch size in `run_experiments.py` (line 304)
2. Use smaller models (resnet18, mobilenetv3)
3. Reduce sample sizes: `--train-samples 5000`

### Wrong predictions
**Solutions:**
1. Train longer: `--epochs 25`
2. Use more data: `--full-data`
3. Try different architectures: `--experiments coral_resnet50 coral_efficientnet_b0`

## 🎓 Next Steps

1. **Start with quick test** to verify setup:
   ```bash
   ./quick_run.sh test
   ```

2. **Compare loss functions** to understand CORAL benefits:
   ```bash
   ./quick_run.sh compare
   ```

3. **Try different architectures** to find best model:
   ```bash
   ./quick_run.sh medium
   ```

4. **Run full training** with best configuration:
   ```bash
   python run_experiments.py --full-data --epochs 20 --experiments coral_resnet50
   ```

5. **Analyze results** in the CSV file and confusion matrices

## 📚 More Information

- Full documentation: `README_EXPERIMENTS.md`
- Available models and configurations: See `run_experiments.py` lines 230-430
- AutoGluon Multimodal docs: https://auto.gluon.ai/stable/tutorials/multimodal/index.html
- CORAL loss paper: https://arxiv.org/abs/1901.07884

## 💡 Pro Tips

1. **Use `screen` or `tmux`** for long-running experiments:
   ```bash
   screen -S experiments
   conda activate ag_env
   ./quick_run.sh full
   # Press Ctrl+A, then D to detach
   # Later: screen -r experiments to reattach
   ```

2. **Run multiple experiments in parallel** (if you have enough RAM):
   ```bash
   # Terminal 1
   python run_experiments.py --experiments coral_resnet18 --output exp1.csv &
   
   # Terminal 2
   python run_experiments.py --experiments coral_resnet50 --output exp2.csv &
   ```

3. **Save complete terminal output** for analysis:
   ```bash
   ./quick_run.sh full 2>&1 | tee experiment_log.txt
   ```

4. **Profile GPU usage** during training:
   ```bash
   # Install powermetrics (requires sudo)
   sudo powermetrics --samplers gpu_power -i 1000 -n 0 > gpu_usage.log &
   # Stop with: sudo pkill powermetrics
   ```

---

**Ready to start? Run this:**
```bash
conda activate ag_env
cd "/Users/ahmademami/Library/CloudStorage/OneDrive-UNSW/TERM 1/Github/aglone/autogluon"
./quick_run.sh test
```
