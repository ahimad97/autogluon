#!/bin/bash

# Quick experiment launcher for CORAL loss experiments
# Usage: ./quick_run.sh [test|quick|medium|full|compare|all]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  CORAL Loss Experiments Quick Launcher    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Activate conda environment
echo -e "${YELLOW}Activating ag_env conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate ag_env

# Check if MPS is available
echo -e "${YELLOW}Checking GPU availability...${NC}"
python -c "import torch; mps=torch.backends.mps.is_available(); print(f'MPS (Apple GPU): {\"✅ Available\" if mps else \"❌ Not available\"}'); exit(0 if mps else 1)" || echo -e "${YELLOW}Warning: MPS not available, will use CPU (slower)${NC}"
echo ""

MODE="${1:-quick}"

case "$MODE" in
  
  test)
    echo -e "${GREEN}Running TEST mode (tiny dataset, 5 epochs, ~2 min)${NC}"
    echo "Perfect for: Quick testing, debugging"
    python run_experiments.py \
      --train-samples 200 \
      --test-samples 100 \
      --epochs 5 \
      --experiments coral_resnet18 \
      --output test_results.csv
    ;;
  
  quick)
    echo -e "${GREEN}Running QUICK mode (500 train, 10 epochs, ~5 min on MPS)${NC}"
    echo "Perfect for: Initial exploration, comparing 2-3 models"
    python run_experiments.py \
      --train-samples 500 \
      --test-samples 200 \
      --epochs 10 \
      --experiments coral_resnet18 \
      --output quick_results.csv
    ;;
  
  medium)
    echo -e "${GREEN}Running MEDIUM mode (5K train, 15 epochs, ~20 min on MPS)${NC}"
    echo "Perfect for: Architecture comparison, hyperparameter testing"
    python run_experiments.py \
      --train-samples 5000 \
      --test-samples 1000 \
      --epochs 15 \
      --experiments coral_resnet18 coral_resnet50 coral_efficientnet_b0 \
      --output medium_results.csv
    ;;
  
  full)
    echo -e "${GREEN}Running FULL mode (all data, 20 epochs, ~2 hours on MPS)${NC}"
    echo "Perfect for: Final training, best performance"
    python run_experiments.py \
      --full-data \
      --epochs 20 \
      --experiments coral_resnet18 coral_resnet50 coral_efficientnet_b0 \
      --output full_results.csv
    ;;
  
  compare)
    echo -e "${GREEN}Running COMPARE mode (CORAL vs CrossEntropy, 5K train, ~15 min)${NC}"
    echo "Perfect for: Comparing loss functions"
    python run_experiments.py \
      --train-samples 5000 \
      --test-samples 1000 \
      --epochs 15 \
      --experiments coral_resnet18 ce_resnet18 \
      --output compare_results.csv
    ;;
  
  all)
    echo -e "${GREEN}Running ALL mode (all 12 experiments, full data, ~6 hours on MPS)${NC}"
    echo "Perfect for: Comprehensive comparison"
    read -p "This will take several hours. Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      python run_experiments.py \
        --full-data \
        --epochs 20 \
        --experiments coral_resnet18 ce_resnet18 coral_resnet50 \
                      coral_efficientnet_b0 coral_convnext_tiny \
                      coral_swin_base coral_mobilenetv3 coral_vit_clip \
                      coral_resnet18_electra_small coral_resnet18_deberta_small \
                      coral_resnet50_transformer_fusion coral_resnet18_ft_transformer \
        --output all_results.csv
    else
      echo "Cancelled."
      exit 0
    fi
    ;;
  
  cpu)
    echo -e "${GREEN}Running CPU TEST mode (CPU only, 200 train, 5 epochs)${NC}"
    echo "Perfect for: Debugging without GPU"
    python run_experiments.py \
      --cpu \
      --train-samples 200 \
      --test-samples 100 \
      --epochs 5 \
      --experiments coral_resnet18 \
      --output cpu_test_results.csv
    ;;
  
  *)
    echo -e "${YELLOW}Usage: $0 [mode]${NC}"
    echo ""
    echo "Available modes:"
    echo ""
    echo -e "${GREEN}  test${NC}     - Tiny test (200 samples, 5 epochs, ~2 min)"
    echo -e "${GREEN}  quick${NC}    - Quick run (500 samples, 10 epochs, ~5 min) [DEFAULT]"
    echo -e "${GREEN}  medium${NC}   - Medium run (5K samples, 15 epochs, ~20 min)"
    echo -e "${GREEN}  full${NC}     - Full dataset (24K samples, 20 epochs, ~2 hours)"
    echo -e "${GREEN}  compare${NC}  - Compare CORAL vs CrossEntropy (~15 min)"
    echo -e "${GREEN}  all${NC}      - All 12 experiments, full data (~6 hours)"
    echo -e "${GREEN}  cpu${NC}      - CPU test mode (no GPU)"
    echo ""
    echo "Examples:"
    echo "  ./quick_run.sh test       # Fast test"
    echo "  ./quick_run.sh            # Default quick mode"
    echo "  ./quick_run.sh medium     # Compare architectures"
    echo "  ./quick_run.sh full       # Final training"
    echo ""
    exit 1
    ;;
esac

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Experiment Completed Successfully! 🎉    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved. To view:"
echo "  cat ${MODE}_results.csv"
echo ""
