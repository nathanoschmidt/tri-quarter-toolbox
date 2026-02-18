# TQF-NN Benchmark Tools: Command-Line Interface Parameter Guide

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.0.3<br>
**Date:** February 18, 2026<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Quick Start Examples](#2-quick-start-examples)
3. [Model Selection Parameters](#3-model-selection-parameters)
4. [Device Selection](#4-device-selection)
5. [Reproducibility Parameters](#5-reproducibility-parameters)
6. [Training Hyperparameters](#6-training-hyperparameters)
7. [Dataset Configuration](#7-dataset-configuration)
8. [TQF-Specific Parameters](#8-tqf-specific-parameters)
   - [8.1 Core TQF Architecture](#81-core-tqf-architecture)
   - [8.2 TQF Symmetry Groups](#82-tqf-symmetry-groups)
   - [8.3 TQF Fractal Geometry](#83-tqf-fractal-geometry)
   - [8.4 TQF Attention](#84-tqf-attention)
   - [8.5 TQF Loss and Regularization](#85-tqf-loss-and-regularization)
   - [8.6 TQF Fibonacci Enhancements](#86-tqf-fibonacci-enhancements)
9. [Complete Parameter Reference Table](#9-complete-parameter-reference-table)
10. [Validation Rules and Constraints](#10-validation-rules-and-constraints)
11. [Example Workflows](#11-example-workflows)
12. [Performance Tuning Guide](#12-performance-tuning-guide)
13. [Troubleshooting](#13-troubleshooting)
14. [Result Output](#14-result-output)

---

## 1. Introduction

This **Command-Line Interface (CLI) Parameter Guide** provides comprehensive documentation for all CLI parameters in the **TQF Neural Network (TQF-NN) Benchmark Suite**. The system implements Nathan O. Schmidt's Tri-Quarter Framework for radial dual triangular lattice graph-based neural networks with ℤ₆, D₆, and T₂₄ symmetry group exploitation.

### Available Models

1. **TQF-ANN** - Tri-Quarter Framework ANN (symmetry-exploiting architecture)
2. **FC-MLP** - Fully Connected Multi-Layer Perceptron (baseline)
3. **CNN-L5** - 5-Layer Convolutional Neural Network (baseline)
4. **ResNet-18-Scaled** - Scaled ResNet-18 Architecture (baseline)

All models are parameter-matched to approximately 650,000 trainable parameters (+/-1%) for fair comparison.

### System Requirements

- **OS**: Windows, Linux, macOS
- **GPU**: CUDA-capable (tested: NVIDIA RTX 4060, CUDA 12.6)
- **VRAM**: 8GB+ recommended for default settings
- **Python**: 3.8+
- **Framework**: PyTorch 2.5+ with CUDA support

### CLI Implementation

All argument parsing, validation, and help text is centralized in `cli.py`. The main entry point is:

```python
from cli import parse_args
args = parse_args()  # Returns validated argparse.Namespace
```

---

## 2. Quick Start Examples

### Run All Models (Default Configuration)

```bash
python main.py
```

This runs all 4 models with defaults:
- 1 random seed (42)
- 150 training epochs (with early stopping, patience 25)
- Batch size 128
- Learning rate 0.001
- Weight decay 0.0001
- Label smoothing 0.1
- 58,000 training samples
- 2,000 validation samples
- 2,000 rotated test samples
- 8,000 unrotated test samples

### Run Single Model

```bash
# Run only TQF-ANN
python main.py --models TQF-ANN

# Run only MLP baseline
python main.py --models FC-MLP

# Run only CNN baseline
python main.py --models CNN-L5

# Run only ResNet baseline
python main.py --models ResNet-18-Scaled
```

### Run Multiple Specific Models

```bash
# Compare TQF-ANN against MLP and CNN
python main.py --models TQF-ANN FC-MLP CNN-L5

# Compare all baselines (no TQF)
python main.py --models FC-MLP CNN-L5 ResNet-18-Scaled

# Explicitly run all models
python main.py --models all
```

### Quick Testing (10 epochs, 1 seed, small dataset)

```bash
python main.py --num-epochs 10 --num-seeds 1 --num-train 1000 --num-val 200
```

### Production Run (150 epochs, 10 seeds)

```bash
python main.py --num-epochs 150 --num-seeds 10
```

### TQF-Specific Examples

```bash
# TQF with ℤ₆ symmetry only (rotations)
python main.py --models TQF-ANN --tqf-symmetry-level Z6

# TQF with D₆ symmetry (rotations + reflections)
python main.py --models TQF-ANN --tqf-symmetry-level D6

# TQF with full T₂₄ symmetry (rotations + reflections + inversion)
python main.py --models TQF-ANN --tqf-symmetry-level T24

# TQF with no symmetry (ablation study)
python main.py --models TQF-ANN --tqf-symmetry-level none

```

### Fibonacci Weight Scaling Examples

```bash
# Default: TQF-ANN with uniform weighting (none mode)
python main.py --models TQF-ANN

# Explicit uniform weighting (same as default)
python main.py --models TQF-ANN --tqf-fibonacci-mode none

# Linear weight scaling (ablation baseline)
python main.py --models TQF-ANN --tqf-fibonacci-mode linear

# Full Fibonacci weight scaling (per TQF specification)
python main.py --models TQF-ANN --tqf-fibonacci-mode fibonacci

# Speed optimization: Phi-scaled radial binning
python main.py --models TQF-ANN --tqf-use-phi-binning

# Combined: Fibonacci weights + Phi binning
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode fibonacci \
  --tqf-use-phi-binning

# Ablation study: Compare all Fibonacci modes (identical param counts)
for mode in none linear fibonacci; do
  python main.py --models TQF-ANN \
    --tqf-fibonacci-mode $mode \
    --num-seeds 5 \
    --output-dir results/fibonacci_ablation_$mode
done
```

---

## 3. Model Selection Parameters

### `--models` (optional, space-separated list)

**Purpose**: Select which models to train and evaluate. Models are trained in the order specified.

**Type**: `str` (multiple values allowed via `nargs='*'`)

**Default**: All models (`['TQF-ANN', 'FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']`)

**Valid Options**:
- `TQF-ANN` - Tri-Quarter Framework ANN with symmetry exploitation
- `FC-MLP` - Fully Connected Multi-Layer Perceptron (baseline)
- `CNN-L5` - 5-Layer Convolutional Neural Network (baseline)
- `ResNet-18-Scaled` - Scaled ResNet-18 (baseline)
- `all` - Explicitly run all models (same as default)

**Behavior**:
- If `--models` not provided: trains all 4 models
- If `--models all`: trains all 4 models
- If `--models` provided with no values: trains all 4 models
- If `--models TQF-ANN FC-MLP`: trains only TQF-ANN and FC-MLP in that order

**Examples**:

```bash
# Default: all models
python main.py

# Explicit all models
python main.py --models all

# Single model
python main.py --models TQF-ANN

# Two models in specified order
python main.py --models TQF-ANN ResNet-18-Scaled

# All baselines
python main.py --models FC-MLP CNN-L5 ResNet-18-Scaled
```

**Why This Matters**:
- Enables focused experiments on specific architectures
- Reduces runtime when testing parameters
- Allows targeted baseline comparisons
- Facilitates ablation studies
- Models train in the order specified (deterministic)

**Validation**:
- Each model name must be in `['TQF-ANN', 'FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']`
- Invalid model names trigger immediate error with available options listed

---

## 4. Device Selection

### `--device` (str)

**Purpose**: Specify compute device for training and evaluation.

**Type**: `str`

**Default**: `'cuda'` if available, else `'cpu'`

**Valid Options**:
- `cuda` - Use GPU via CUDA
- `cpu` - Use CPU only

**Examples**:

```bash
# Auto-detect (default)
python main.py

# Force CUDA
python main.py --device cuda

# Force CPU (slow but works without GPU)
python main.py --device cpu
```

**Why This Matters**:
- CUDA (GPU): ~12 seconds/epoch on RTX 4060 (58k training samples, with in-memory caching)
- CPU: ~5-10 minutes/epoch (significantly slower)
- Useful for debugging without GPU
- Allows testing on machines without CUDA

**Validation**:
- Must be exactly `'cuda'` or `'cpu'`
- If `cuda` specified but unavailable, training will fail with PyTorch error

### `--compile` (flag)

**Purpose**: Enable torch.compile for kernel fusion and reduced Python overhead.

**Type**: `flag` (boolean, presence enables)

**Default**: `False` (disabled)

**Requirements**:
- PyTorch 2.0+
- Triton compiler (Linux only; not available on Windows)

**Examples**:

```bash
# Standard training (without compilation)
python main.py --models TQF-ANN

# Enable torch.compile (Linux with Triton only)
python main.py --models TQF-ANN --compile
```

**Why This Matters**:
- Provides ~10-30% training speedup after initial compilation warmup
- Fuses kernels to reduce GPU memory bandwidth usage
- Reduces Python interpreter overhead
- First epoch is slower due to compilation, subsequent epochs are faster

**Platform Notes**:
- **Linux**: Full support with Triton backend (`reduce-overhead` mode)
- **Windows**: Gracefully skipped with warning message (Triton not available)
- When skipped, training continues normally without compilation

**Warning on Windows**:
```
torch.compile requested but Triton not available (Windows limitation).
Continuing without compilation. Install Triton on Linux for full optimization.
```

---

## 5. Reproducibility Parameters

### `--num-seeds` (int)

**Purpose**: Number of consecutive random seeds to run for statistical robustness.

**Type**: `int`

**Default**: `1` (from `NUM_SEEDS_DEFAULT` in config.py)

**Valid Range**: `[1, 20]` (from `NUM_SEEDS_MIN` to `NUM_SEEDS_MAX`)

**Behavior**: Seeds are consecutive starting from `--seed-start`. For example:
- `--num-seeds 3 --seed-start 42` runs seeds `[42, 43, 44]`
- `--num-seeds 5 --seed-start 100` runs seeds `[100, 101, 102, 103, 104]`

**Examples**:

```bash
# Single seed (fast, default)
python main.py --num-seeds 1

# Three seeds (minimal statistical significance)
python main.py --num-seeds 3

# Five seeds (good balance)
python main.py --num-seeds 5

# Ten seeds (publication quality)
python main.py --num-seeds 10
```

**Why This Matters**:
- Single seed: fast prototyping, no statistical confidence
- 3-5 seeds: reasonable statistical confidence, ~3-5x runtime
- 10+ seeds: publication-quality statistical analysis, ~10x runtime
- Results report mean +/- std across all seeds
- Required for claiming statistical significance

**Computational Impact**:
- Runtime scales linearly with `num_seeds`
- 5 seeds a-- 50 epochs = 250 total epochs per model

**Validation**:
- Must be in range `[1, 20]`
- Values outside range trigger immediate error

---

### `--seed-start` (int)

**Purpose**: Starting random seed value. Seeds are consecutive from this value.

**Type**: `int`

**Default**: `42` (from `SEED_DEFAULT` in config.py)

**Valid Range**: `>= 0` (from `SEED_START_MIN`)

**Examples**:

```bash
# Default starting seed
python main.py --num-seeds 3
# Runs seeds [42, 43, 44]

# Custom starting seed
python main.py --num-seeds 3 --seed-start 100
# Runs seeds [100, 101, 102]

# Starting from zero
python main.py --num-seeds 5 --seed-start 0
# Runs seeds [0, 1, 2, 3, 4]
```

**Why This Matters**:
- Enables reproducibility of exact seed sequences
- Allows continuation of experiments (e.g., add more seeds later)
- Different seed ranges test different initialization variations
- Common practice: 42 (Douglas Adams reference)

**Validation**:
- Must be `>= 0`
- Negative seeds trigger immediate error

---

## 6. Training Hyperparameters

### `--num-epochs` (int)

**Purpose**: Maximum number of training epochs per model per seed.

**Type**: `int`

**Default**: `150` (from `MAX_EPOCHS_DEFAULT` in config.py)

**Valid Range**: `[1, 200]` (from `NUM_EPOCHS_MIN` to `NUM_EPOCHS_MAX`)

**Behavior**: Training may terminate earlier if early stopping patience is reached.

**Examples**:

```bash
# Quick sanity check
python main.py --num-epochs 10

# Moderate training
python main.py --num-epochs 50

# Extended training
python main.py --num-epochs 100

# Default training (with early stopping)
python main.py --num-epochs 200

# Publication-quality (with early stopping)
python main.py --num-epochs 200
```

**Why This Matters**:
- With 58K data, best accuracy occurs at epochs 125-135
- Early stopping (default patience=25) terminates if validation plateaus
- CosineAnnealingLR fine-tuning tail (LR < 3e-4) provides +0.5% accuracy gain
- Computational cost scales linearly with epochs

**Computational Impact** (RTX 4060, 58k training samples, in-memory caching):
- 10 epochs ~ 2 minutes
- 50 epochs ~ 10 minutes
- 100 epochs ~ 20 minutes
- 150 epochs ~ 30 minutes (default)
- 200 epochs ~ 40 minutes

**Validation**:
- Must be in range `[1, 200]`
- Must be greater than `--patience`
- Must be greater than `--learning-rate-warmup-epochs`

---

### `--batch-size` (int)

**Purpose**: Number of samples per training batch.

**Type**: `int`

**Default**: `128` (from `BATCH_SIZE_DEFAULT` in config.py)

**Valid Range**: `[1, 1024]` (from `BATCH_SIZE_MIN` to `BATCH_SIZE_MAX`)

**Recommended Values**: Powers of 2 (16, 32, 64, 128, 256) for GPU efficiency

**Examples**:

```bash
# Small batch (more gradient updates, lower memory)
python main.py --batch-size 32

# Smaller batch (more gradient updates)
python main.py --batch-size 64

# Default batch
python main.py --batch-size 128

# Very large batch (requires 8GB+ VRAM)
python main.py --batch-size 256
```

**Why This Matters**:

**Smaller Batches (32-64)**:
- More frequent gradient updates (better exploration)
- Lower memory usage (fits on smaller GPUs)
- Noisier gradients (can escape local minima)
- Better generalization
- Slower training (more batches per epoch)

**Larger Batches (128-256)**:
- Fewer gradient updates (faster epochs)
- More stable gradients
- Higher memory usage (may cause OOM errors)
- Risk of overfitting
- Faster training (fewer batches per epoch)

**GPU Memory Impact** (RTX 4060):
- Batch 32: ~2GB VRAM
- Batch 64: ~3GB VRAM (default, safe)
- Batch 128: ~5GB VRAM
- Batch 256: ~8GB VRAM (near limit)
- Batch 512: OOM error on 8GB GPU

**Validation**:
- Must be in range `[1, 1024]`
- Batch size of 0 triggers immediate error

---

### `--learning-rate` (float)

**Purpose**: Initial learning rate for Adam optimizer. Decays via cosine annealing.

**Type**: `float`

**Default**: `0.001` (from `LEARNING_RATE_DEFAULT` in config.py)

**Valid Range**: `(0.0, 1.0]` (from `LEARNING_RATE_MIN` to `LEARNING_RATE_MAX`, exclusive minimum)

**Behavior**:
- Linear warmup from 0 to `learning_rate` over `--learning-rate-warmup-epochs` epochs
- Cosine annealing from `learning_rate` to `learning_rate / 100` over remaining epochs

**Examples**:

```bash
# Very conservative (slow but stable)
python main.py --learning-rate 0.0001

# Lower learning rate (conservative)
python main.py --learning-rate 0.0005

# Default (optimal for TQF-ANN)
python main.py --learning-rate 0.001

# Aggressive (risky, may diverge)
python main.py --learning-rate 0.01
```

**Why This Matters**:

**Lower LR (0.0001-0.0005)**:
- Preserves TQF geometric structure during training
- Smoother convergence
- Better for TQF-ANN due to fractal/symmetry constraints
- May require 100+ epochs to converge
- More stable, less risk of divergence

**Higher LR (0.001-0.01)**:
- Faster initial training
- Risks breaking symmetries in TQF-ANN
- May plateau early or diverge
- Better for baseline models (more robust)

**Expected Behavior by LR**:
- **0.0001**: Slow but very stable, may need 100+ epochs
- **0.0005 (default)**: Optimal balance, converges in 50-70 epochs
- **0.001**: Faster initial progress, may plateau early
- **0.01**: Often diverges or produces poor solutions

**Cosine Annealing Schedule**:
- Epoch 0-3: Linear warmup to `learning_rate`
- Epoch 3-50: Cosine decay to `learning_rate / 100`

**Validation**:
- Must be in range `(0.0, 1.0]` (strictly positive)
- Learning rate of 0.0 triggers immediate error

---

### `--weight-decay` (float)

**Purpose**: L2 regularization (weight decay) coefficient for Adam optimizer.

**Type**: `float`

**Default**: `0.0001` (from `WEIGHT_DECAY_DEFAULT` in config.py)

**Valid Range**: `[0.0, 1.0]` (from `WEIGHT_DECAY_MIN` to `WEIGHT_DECAY_MAX`)

**Loss Formula**: `Total Loss = Classification Loss + weight_decay * ||weights||2`

**Examples**:

```bash
# No regularization (not recommended)
python main.py --weight-decay 0.0

# Very light regularization
python main.py --weight-decay 0.00001

# Light regularization (good for TQF-ANN with geometric reg)
python main.py --weight-decay 0.00005

# Standard regularization (default)
python main.py --weight-decay 0.0001

# Strong regularization
python main.py --weight-decay 0.0005

# Very strong regularization
python main.py --weight-decay 0.001
```

**Why This Matters**:
- Prevents overfitting by penalizing large weights
- Standard deep learning practice (Adam typically uses 0.0001)
- Helps prevent weight explosion in deep networks
- Improves generalization

**Expected Impact**:
- **0.00001**: Minimal regularization, faster convergence, higher risk of overfitting
- **0.00005**: Light regularization, good for TQF-ANN (already has geometric regularization)
- **0.0001 (default)**: Standard regularization, balanced approach
- **0.0005**: Strong regularization, slower convergence, better generalization
- **0.001**: Very strong, may underfit

**Interaction with TQF Regularization**:
- TQF-ANN has additional regularization via:
  - `--tqf-geometry-reg-weight` (geometric consistency)
  - `--tqf-z6-equivariance-weight` (Z6 rotation equivariance)
  - `--tqf-d6-equivariance-weight` (D6 reflection equivariance)
  - `--tqf-t24-orbit-invariance-weight` (T24 orbit invariance)
  - `--tqf-inversion-loss-weight` (circle inversion duality)
  - `--tqf-self-similarity-weight` (fractal structure)
  - `--tqf-box-counting-weight` (fractal dimension)
- Recommendation: Use lower `weight_decay` (0.00005) when TQF regularizations are strong
- Baseline models typically use standard `weight_decay` (0.0001)

**Validation**:
- Must be in range `[0.0, 1.0]`

---

### `--label-smoothing` (float)

**Purpose**: Label smoothing factor for CrossEntropyLoss. Prevents overconfidence.

**Type**: `float`

**Default**: `0.1` (from `LABEL_SMOOTHING_DEFAULT` in config.py)

**Valid Range**: `[0.0, 1.0]` (from `LABEL_SMOOTHING_MIN` to `LABEL_SMOOTHING_MAX`)

**Behavior**:
- `0.0`: No smoothing (hard one-hot labels)
- `0.1`: 10% smoothing (90% correct class, 10% distributed to others)
- Smoothed label for correct class = `1 - label_smoothing + label_smoothing / num_classes`
- Smoothed label for incorrect classes = `label_smoothing / num_classes`

**Examples**:

```bash
# No smoothing (hard labels)
python main.py --label-smoothing 0.0

# Light smoothing
python main.py --label-smoothing 0.05

# Standard smoothing (default)
python main.py --label-smoothing 0.1

# Strong smoothing
python main.py --label-smoothing 0.2

# Maximum smoothing (not recommended)
python main.py --label-smoothing 0.5
```

**Why This Matters**:
- Prevents overconfident predictions
- Improves calibration (predicted probabilities match true frequencies)
- Regularization effect (similar to weight decay)
- Helps generalization

**Expected Impact**:
- **0.0**: No smoothing, risk of overconfidence
- **0.05**: Light smoothing, mild regularization
- **0.1 (default)**: Standard smoothing, good balance
- **0.2**: Strong smoothing, stronger regularization
- **0.5**: Very strong, may hurt performance

**Validation**:
- Must be in range `[0.0, 1.0]`

---

### `--patience` (int)

**Purpose**: Number of epochs without validation loss improvement before early stopping.

**Type**: `int`

**Default**: `25` (from `PATIENCE_DEFAULT` in config.py)

**Valid Range**: `[1, 50]` (from `PATIENCE_MIN` to `PATIENCE_MAX`)

**Behavior**:
- Monitors validation loss after each epoch
- If validation loss does not improve by at least `--min-delta` for `patience` consecutive epochs, training stops
- Restores best model weights from best validation epoch

**Examples**:

```bash
# Aggressive early stopping
python main.py --patience 5

# Standard early stopping
python main.py --patience 10

# Default early stopping (patience=25)
python main.py --patience 25

# Very patient (rarely stops early)
python main.py --patience 40
```

**Why This Matters**:
- Prevents overfitting by stopping when validation performance plateaus
- Saves computational time
- Lower patience: faster experiments, risk of stopping too early
- Higher patience: longer training, better chance of finding global minimum

**Expected Behavior**:
- **5**: Stops quickly, good for fast prototyping
- **10**: Balanced, typically stops around epoch 40-50
- **15 (default)**: More patient, allows longer convergence
- **25**: Rarely triggers, mostly trains to `--num-epochs`

**Validation**:
- Must be in range `[1, 50]`
- Must be less than `--num-epochs`

---

### `--min-delta` (float)

**Purpose**: Minimum validation loss improvement required to reset patience counter.

**Type**: `float`

**Default**: `0.0005` (from `MIN_DELTA_DEFAULT` in config.py)

**Valid Range**: `[0.0, 1.0]` (from `MIN_DELTA_MIN` to `MIN_DELTA_MAX`)

**Behavior**:
- Validation loss must decrease by at least `min_delta` to be considered an improvement
- Helps avoid early stopping due to tiny fluctuations

**Examples**:

```bash
# Very sensitive (any improvement counts)
python main.py --min-delta 0.0

# Slightly sensitive
python main.py --min-delta 0.0001

# Standard sensitivity (default)
python main.py --min-delta 0.0005

# Less sensitive
python main.py --min-delta 0.001

# Very insensitive
python main.py --min-delta 0.01
```

**Why This Matters**:
- Prevents early stopping from noise in validation loss
- Lower values: more sensitive to small improvements
- Higher values: requires larger improvements to continue training

**Expected Impact**:
- **0.0**: Any decrease counts as improvement
- **0.0001**: Sensitive to small changes
- **0.0005 (default)**: Balanced threshold
- **0.001**: Less sensitive
- **0.01**: Only large improvements count

**Validation**:
- Must be in range `[0.0, 1.0]`

---

### `--learning-rate-warmup-epochs` (int)

**Purpose**: Number of epochs for linear learning rate warmup from 0 to `--learning-rate`.

**Type**: `int`

**Default**: `5` (from `LEARNING_RATE_WARMUP_EPOCHS` in config.py)

**Valid Range**: `[0, 10]` (from `LEARNING_RATE_WARMUP_EPOCHS_MIN` to `LEARNING_RATE_WARMUP_EPOCHS_MAX`)

**Behavior**:
- Epochs 0 to `warmup_epochs`: Linear increase from 0 to `learning_rate`
- Epochs `warmup_epochs` to `num_epochs`: Cosine annealing to `learning_rate / 100`

**Examples**:

```bash
# No warmup (immediate full learning rate)
python main.py --learning-rate-warmup-epochs 0

# Short warmup
python main.py --learning-rate-warmup-epochs 3

# Default warmup
python main.py --learning-rate-warmup-epochs 5

# Extended warmup
python main.py --learning-rate-warmup-epochs 10
```

**Why This Matters**:
- Prevents early training instability
- Helps models with complex initialization (like TQF-ANN)
- Standard practice in modern deep learning
- Particularly important with large learning rates

**Expected Impact**:
- **0**: No warmup, faster initial progress, risk of instability
- **3**: Short warmup, stable start
- **5 (default)**: Standard warmup, very stable training
- **10**: Extended warmup, very stable but slower start

**Validation**:
- Must be in range `[0, 10]`
- Must be less than `--num-epochs`

---

## 7. Dataset Configuration

### `--num-train` (int)

**Purpose**: Number of training samples from MNIST. MNIST has 60,000 total training images; with the default validation split of 2,000, 58,000 are available for training.

**Type**: `int`

**Default**: `58000` (from `NUM_TRAIN_DEFAULT` in config.py)

**Valid Range**: `[100, 60000]` (from `NUM_TRAIN_MIN` to `NUM_TRAIN_MAX`)

**Constraint**: **Must be divisible by 10** for balanced class distribution (MNIST has 10 classes)

**Examples**:

```bash
# Minimal training (fast prototyping)
python main.py --num-train 1000

# Small training
python main.py --num-train 5000

# Medium training
python main.py --num-train 10000

# Full training set (default)
python main.py --num-train 58000
```

**Why This Matters**:
- Larger datasets: better generalization, longer training
- Smaller datasets: faster experiments, risk of overfitting
- Must be divisible by 10 to ensure equal samples per class
- If requested size exceeds available samples (60K minus validation), it is capped automatically

**Computational Impact** (RTX 4060, batch_size=128, in-memory caching):
- 1,000 samples: ~2 seconds/epoch
- 5,000 samples: ~4 seconds/epoch
- 10,000 samples: ~6 seconds/epoch
- 30,000 samples: ~8 seconds/epoch
- 58,000 samples: ~12 seconds/epoch (default)

**Validation**:
- Must be in range `[100, 60000]`
- **Must be divisible by 10** (triggers error otherwise)
- Values exceeding available samples are capped with a warning

---

### `--num-val` (int)

**Purpose**: Number of validation samples from MNIST.

**Type**: `int`

**Default**: `2000` (from `NUM_VAL_DEFAULT` in config.py)

**Valid Range**: `[10, 10000]` (from `NUM_VAL_MIN` to `NUM_VAL_MAX`)

**Constraint**: Recommended to be divisible by 10 for balanced classes (not enforced)

**Examples**:

```bash
# Minimal validation
python main.py --num-val 100

# Small validation
python main.py --num-val 500

# Medium validation
python main.py --num-val 1000

# Standard validation (default)
python main.py --num-val 2000

# Large validation
python main.py --num-val 5000
```

**Why This Matters**:
- Used for early stopping and hyperparameter tuning
- Larger validation sets: more reliable stopping, slower epochs
- Smaller validation sets: faster epochs, noisier stopping signal
- Typical ratio: 5-10% of training set size

**Validation**:
- Must be in range `[10, 10000]`

---

### `--num-test-rot` (int)

**Purpose**: Number of rotated test samples (60deg rotations applied).

**Type**: `int`

**Default**: `2000` (from `NUM_TEST_ROT_DEFAULT` in config.py)

**Valid Range**: `[100, 10000]` (from `NUM_TEST_ROT_MIN` to `NUM_TEST_ROT_MAX`)

**Behavior**: Primary metric for rotation invariance evaluation

**Examples**:

```bash
# Quick rotation test
python main.py --num-test-rot 100

# Small rotation test
python main.py --num-test-rot 500

# Standard rotation test (default)
python main.py --num-test-rot 2000

# Large rotation test
python main.py --num-test-rot 5000

# Full rotation test
python main.py --num-test-rot 10000
```

**Why This Matters**:
- Tests rotation invariance/equivariance
- Crucial for evaluating TQF-ANN symmetry exploitation
- Larger test sets: more reliable accuracy estimates
- Each sample is rotated by 60deg (T₂₄ symmetry)

**Validation**:
- Must be in range `[100, 10000]`

---

### `--num-test-unrot` (int)

**Purpose**: Number of unrotated (standard) test samples.

**Type**: `int`

**Default**: `8000` (from `NUM_TEST_UNROT_DEFAULT` in config.py)

**Valid Range**: `[100, 10000]` (from `NUM_TEST_UNROT_MIN` to `NUM_TEST_UNROT_MAX`)

**Behavior**: Baseline metric for standard MNIST accuracy

**Examples**:

```bash
# Quick standard test
python main.py --num-test-unrot 100

# Small standard test
python main.py --num-test-unrot 200

# Standard test (default)
python main.py --num-test-unrot 8000

# Large standard test
python main.py --num-test-unrot 1000

# Full standard test
python main.py --num-test-unrot 10000
```

**Why This Matters**:
- Measures standard (non-rotated) accuracy
- Sanity check: should be high for all models
- Less important than rotated accuracy for this benchmark

**Validation**:
- Must be in range `[100, 10000]`

---

## 8. TQF-Specific Parameters

**Note**: All parameters in this section **only affect TQF-ANN** and are **ignored for baseline models** (FC-MLP, CNN-L5, ResNet-18-Scaled).

---

### 8.1 Core TQF Architecture

#### `--tqf-R` (int)

**Purpose**: Truncation radius R for the radial dual triangular lattice graph Lambda>_r^R.

**Type**: `int`

**Default**: `10` (from `TQF_TRUNCATION_R_DEFAULT` in config.py)

**Valid Range**: `[2, 100]` (from `TQF_R_MIN` to `TQF_R_MAX`, must be > inversion radius r=1)

**Behavior**:
- Determines lattice size (number of hexagonal nodes approximately proportional to R^2)
- Affects parameter count (larger R = more lattice vertices = more parameters)
- Fixed inversion radius r=1 (hardcoded in `TQF_RADIUS_R_FIXED`)
- The hidden_dim is auto-tuned to maintain ~650k total parameters for fair comparison

**Examples**:

```bash
# Default lattice (R=10, balanced for ~650k params with auto-tuned hidden_dim)
python main.py --models TQF-ANN

# Explicit default
python main.py --models TQF-ANN --tqf-R 10

# Larger lattice (more vertices, smaller auto-tuned hidden_dim)
python main.py --models TQF-ANN --tqf-R 10

# Large lattice
python main.py --models TQF-ANN --tqf-R 20

# Very large lattice
python main.py --models TQF-ANN --tqf-R 50
```

**Why This Matters**:
- Controls lattice granularity and coverage in hyperbolic space
- Larger R: more lattice vertices, more expressiveness, but smaller hidden_dim needed for param budget
- Smaller R: fewer vertices, larger hidden_dim possible, faster training
- Automatically balanced with `--tqf-hidden-dim` during parameter matching to hit ~650k target

**Parameter Count Impact** (with auto-tuned hidden_dim to maintain ~650k total):
- R=10: ~650k parameters (default, hidden_dim auto-tuned)
- R=15: ~650k parameters (hidden_dim auto-tuned)
- R=20: ~650k parameters (hidden_dim auto-tuned)

**Validation**:
- Must be in range `[2, 100]`
- Must be > 1 (inversion radius r=1 is fixed)

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-hidden-dim` (int, optional)

**Purpose**: Hidden feature dimension for TQF-ANN graph node embeddings.

**Type**: `int` or `None`

**Default**: `None` (auto-tuned during parameter matching to hit ~650k total parameters)

**Valid Range**: `[8, 512]` (from `TQF_HIDDEN_DIM_MIN` to `TQF_HIDDEN_DIM_MAX`) if manually specified

**Behavior**:
- If `None`: automatically computed to match `TARGET_PARAMS` (650k +/-1%)
- If specified: overrides auto-tuning, final parameter count may differ from 650k
- Interacts with `--tqf-R` to determine total parameter count

**Examples**:

```bash
# Auto-tune (default, recommended)
python main.py --models TQF-ANN --tqf-R 20

# Manually specify hidden dimension
python main.py --models TQF-ANN --tqf-R 20 --tqf-hidden-dim 256

# Small hidden dimension
python main.py --models TQF-ANN --tqf-R 20 --tqf-hidden-dim 128

# Large hidden dimension
python main.py --models TQF-ANN --tqf-R 20 --tqf-hidden-dim 512
```

**Why This Matters**:
- Controls feature expressiveness per node
- Automatically balanced with R for fair comparison (650k params)
- Manual override useful for ablation studies
- Higher hidden_dim: more expressiveness, more parameters

**Validation**:
- If specified, must be in range `[8, 512]`
- If `None`, auto-tuned (no validation needed)

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

### 8.2 TQF Symmetry Groups

#### `--tqf-symmetry-level` (str)

**Purpose**: Symmetry group for equivariant feature reduction and orbit pooling.

**Type**: `str`

**Default**: `'none'` (from `TQF_SYMMETRY_LEVEL_DEFAULT` in config.py)

**Valid Options**:
- `'none'` - No symmetry exploitation (default, fastest inference)
- `'Z6'` - Cyclic group (6 rotations by 60deg, 6x inference cost)
- `'D6'` - Dihedral group (6 rotations + 6 reflections, 12x inference cost)
- `'T24'` - Full T24 symmetry (rotations + reflections + inversion, 24x inference cost)

**Implementation Details**:
Different symmetry levels apply orbit pooling with different group sizes:
- `'none'`: No orbit pooling, features used directly (baseline)
- `'Z6'`: Averages features over 6 Z6 rotation orbits
- `'D6'`: Averages features over 12 D6 operations (rotations + reflections)
- `'T24'`: Averages features over 24 T24 operations (D6 + circle inversion)

**Examples**:

```bash
# No symmetry (default, fastest)
python main.py --models TQF-ANN --tqf-symmetry-level none

# Rotations only (ℤ₆)
python main.py --models TQF-ANN --tqf-symmetry-level Z6

# Rotations + reflections (D₆)
python main.py --models TQF-ANN --tqf-symmetry-level D6

# Full symmetry (T₂₄)
python main.py --models TQF-ANN --tqf-symmetry-level T24
```

**Why This Matters**:
- **`none`**: Fastest inference, baseline performance
- **`ℤ₆`**: Rotation invariance via 6-fold symmetry (+1-2% rotated accuracy)
- **`D₆`**: Adds reflection symmetry (mirror invariance)
- **`T₂₄`**: Full symmetry exploitation including circle inversion

**Expected Performance**:
- Rotated accuracy: none < ℤ₆ < D₆ < T₂₄
- Inference cost: none < ℤ₆ (6x) < D₆ (12x) < T₂₄ (24x)

**Computational Impact** (inference cost relative to `none`):
- ℤ₆: 6x inference cost (6 rotation orbits)
- D₆: 12x inference cost (12 group elements)
- T₂₄: 24x inference cost (24 group elements)

**Validation**:
- Must be in `['none', 'Z6', 'D6', 'T24']`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

### 8.3 TQF Fractal Geometry

#### `--tqf-fractal-iterations` (int)

**Purpose**: Number of iterations for fractal dimension estimation via box-counting.

**Type**: `int`

**Default**: `0` - **DISABLED** (from `TQF_FRACTAL_ITERATIONS_DEFAULT` in config.py)

**Valid Range**: `[1, 20]` when enabled (from `TQF_FRACTAL_ITERATIONS_MIN` to `TQF_FRACTAL_ITERATIONS_MAX`)

**Important**: This is an **opt-in feature**. The feature is disabled by default (value=0). To enable fractal dimension estimation, provide a value in the range [1, 20].

**Examples**:

```bash
# Enable with minimal fractal estimation
python main.py --models TQF-ANN --tqf-fractal-iterations 3

# Enable with balanced fractal estimation
python main.py --models TQF-ANN --tqf-fractal-iterations 5

# Enable with precise fractal estimation
python main.py --models TQF-ANN --tqf-fractal-iterations 10

# Enable with very precise (slow) estimation
python main.py --models TQF-ANN --tqf-fractal-iterations 15
```

**Why This Matters**:
- Higher iterations: more accurate fractal dimension estimates
- Lower iterations: faster training
- Used in `--tqf-box-counting-weight` loss term
- **Disabled by default** to simplify baseline training

**Computational Impact**:
- Linear scaling with iterations
- 3 iterations: adds ~5% training time
- 5 iterations: adds ~10% training time
- 10 iterations: adds ~20% training time

**Validation**:
- When provided, must be in range `[1, 20]`
- Value of 0 disables the feature (default)

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

> **Note:** `--tqf-fractal-dim-tolerance` was consolidated as an internal constant (`TQF_FRACTAL_DIM_TOLERANCE_DEFAULT = 0.08` in config.py). It is no longer a CLI parameter. The tolerance controls acceptable deviation between measured and theoretical fractal dimensions (1.585, Sierpinski triangle dimension). Related to `--tqf-fractal-iterations` which controls whether fractal features are active.

---

#### `--tqf-self-similarity-weight` (float)

**Purpose**: Weight for self-similarity fractal regularization loss.

**Type**: `float`

**Default**: `0.0` (from `TQF_SELF_SIMILARITY_WEIGHT_DEFAULT` in config.py; opt-in, disabled by default)

**Valid Range**: `[0.0, 10.0]` (from `TQF_SELF_SIMILARITY_WEIGHT_MIN` to `TQF_SELF_SIMILARITY_WEIGHT_MAX`)

**Loss Formula**: Compares features at multiple scales using cosine similarity. Penalizes deviation from self-similar structure.

**Examples**:

```bash
# No self-similarity regularization
python main.py --models TQF-ANN --tqf-self-similarity-weight 0.0

# Gentle guidance (recommended when enabled)
python main.py --models TQF-ANN --tqf-self-similarity-weight 0.001

# Moderate regularization
python main.py --models TQF-ANN --tqf-self-similarity-weight 0.005

# Strong regularization
python main.py --models TQF-ANN --tqf-self-similarity-weight 0.01
```

**Why This Matters**:
- Encourages fractal self-similarity in learned features
- Measures correlation between features at different resolutions
- Higher weight: stronger fractal structure constraint
- Small values (0.001) recommended to avoid dominating classification loss

**Validation**:
- Must be in range `[0.0, 10.0]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-box-counting-weight` (float)

**Purpose**: Weight for box-counting fractal dimension regularization. Penalizes deviation from theoretical dimension (1.585).

**Type**: `float`

**Default**: `0.0` (from `TQF_BOX_COUNTING_WEIGHT_DEFAULT` in config.py; opt-in, disabled by default)

**Valid Range**: `[0.0, 10.0]` (from `TQF_BOX_COUNTING_WEIGHT_MIN` to `TQF_BOX_COUNTING_WEIGHT_MAX`)

**Examples**:

```bash
# No box-counting regularization
python main.py --models TQF-ANN --tqf-box-counting-weight 0.0

# Gentle guidance (recommended when enabled)
python main.py --models TQF-ANN --tqf-box-counting-weight 0.001

# Moderate regularization
python main.py --models TQF-ANN --tqf-box-counting-weight 0.005

# Strong regularization
python main.py --models TQF-ANN --tqf-box-counting-weight 0.01
```

**Why This Matters**:
- Penalizes deviation from theoretical fractal dimension (1.585)
- Loss = weight * |measured_dim - theoretical_dim|
- Maintains expected geometric structure during training
- Small values (0.001) recommended to avoid conflicting with classification

**Validation**:
- Must be in range `[0.0, 10.0]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

> **Note:** `--tqf-box-counting-scales` was consolidated as an internal constant (`TQF_BOX_COUNTING_SCALES_DEFAULT = 10` in config.py). It is no longer a CLI parameter. The value controls the number of scale levels for box-counting fractal dimension estimation. Related to `--tqf-box-counting-weight` which controls whether box-counting loss is active.

---

### 8.4 TQF Attention

#### `--tqf-hop-attention-temp` (float)

**Purpose**: Temperature for hop-distance attention during neighbor aggregation. Controls how neighbor features are weighted in graph convolution layers.

**Type**: `float`

**Default**: `1.0` (from `TQF_HOP_ATTENTION_TEMP_DEFAULT` in config.py)

**Valid Range**: `[0.01, 10.0]` (from `TQF_HOP_ATTENTION_TEMP_MIN` to `TQF_HOP_ATTENTION_TEMP_MAX`)

**Examples**:

```bash
# Default: Uniform weighting (fastest, bypasses attention computation)
python main.py --models TQF-ANN --tqf-hop-attention-temp 1.0

# Sharp hop attention (selective, prefer similar neighbors - slower)
python main.py --models TQF-ANN --tqf-hop-attention-temp 0.5

# Smooth hop attention (more uniform weighting)
python main.py --models TQF-ANN --tqf-hop-attention-temp 2.0
```

**How It Works**:
The temperature controls feature-based attention in neighbor aggregation:
- Attention scores are computed via dot product between vertex and neighbor features
- Scores are divided by temperature before softmax normalization
- Lower temperature -> larger effective scores -> sharper attention distribution
- Higher temperature -> smaller effective scores -> smoother/uniform distribution

**Performance Note**:
- **Temperature = 1.0 (default)**: Bypasses expensive attention computation entirely, using uniform neighbor weighting. This provides ~40% faster forward passes.
- **Temperature != 1.0**: Enables attention path with dot-product computation and softmax normalization, which is significantly slower but may capture more nuanced neighbor relationships.

**Why This Matters**:
- Controls neighbor aggregation sharpness in graph convolutions
- Temperature = 1.0: Uniform weighting, fastest (recommended default)
- Lower temperature (<1.0): Sharp attention, strongly prefer similar neighbors
- Higher temperature (>1.0): Smooth attention, more uniform neighbor weighting

**Validation**:
- Must be in range `[0.01, 10.0]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

### 8.5 Training Data Augmentation

#### `--z6-data-augmentation` (flag)

**Purpose**: Enable Z6-aligned rotation data augmentation during training for all models. When enabled, training images are randomly rotated at 60-degree intervals (with jitter) to teach rotation robustness. Disabled by default because it conflicts with orbit mixing features.

**Type**: `flag` (store_true, sets `z6_data_augmentation = True`)

**Default**: Augmentation disabled (`False`, from `Z6_DATA_AUGMENTATION_DEFAULT` in config.py)

**Examples**:

```bash
# Default: Z6 augmentation disabled
python main.py --models TQF-ANN

# Enable Z6 augmentation for rotation robustness
python main.py --models TQF-ANN --z6-data-augmentation

# Use orbit mixing instead (recommended over augmentation)
python main.py --models TQF-ANN --tqf-use-z6-orbit-mixing
```

**Applies To**: All models (shared training DataLoader)

---

### 8.6 TQF Evaluation-Time Orbit Mixing

Orbit mixing averages predictions over symmetry group operations at evaluation time, exploiting TQF-ANN's hexagonal symmetry structure. Three levels are available:

- **Z6**: 6 input-space rotations (0, 60, 120, 180, 240, 300 degrees) — 6 full forward passes
- **D6**: Feature-space reflection on cached zone features — lightweight (classification head only)
- **T24**: Zone-swap (exchange inner/outer roles) — trivially cheap (argument reordering)

Each higher level includes all lower levels. Temperature parameters control confidence weighting sharpness.

#### `--tqf-use-z6-orbit-mixing` (flag)

**Purpose**: Average predictions over 6 Z6 rotations at evaluation time.

**Default**: `False`

#### `--tqf-use-d6-orbit-mixing` (flag)

**Purpose**: Add D6 reflection averaging (implies Z6 behavior).

**Default**: `False`

#### `--tqf-use-t24-orbit-mixing` (flag)

**Purpose**: Add T24 zone-swap averaging (implies D6 and Z6 behavior).

**Default**: `False`

#### `--tqf-orbit-mixing-temp-rotation` (float)

**Purpose**: Temperature for Z6 rotation confidence weighting. Lower = sharper (most confident rotation dominates).

**Default**: `0.3` (from `TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT` in config.py) | **Range**: `[0.01, 2.0]`

#### `--tqf-orbit-mixing-temp-reflection` (float)

**Purpose**: Temperature for D6 reflection confidence weighting.

**Default**: `0.5` (from `TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT` in config.py) | **Range**: `[0.01, 2.0]`

#### `--tqf-orbit-mixing-temp-inversion` (float)

**Purpose**: Temperature for T24 zone-swap confidence weighting. Softest because circle inversion is the most abstract symmetry.

**Default**: `0.7` (from `TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT` in config.py) | **Range**: `[0.01, 2.0]`

**Examples**:

```bash
# Z6 orbit mixing only (6 forward passes per batch)
python main.py --models TQF-ANN --tqf-use-z6-orbit-mixing

# Full T24 orbit mixing
python main.py --models TQF-ANN --tqf-use-t24-orbit-mixing

# Demonstrate architectural robustness via orbit mixing (augmentation off by default)
python main.py --models TQF-ANN --tqf-use-z6-orbit-mixing

# Compare TQF-ANN vs CNN with orbit mixing
python main.py --models TQF-ANN CNN-L5 --tqf-use-z6-orbit-mixing

# Custom temperatures
python main.py --models TQF-ANN --tqf-use-t24-orbit-mixing \
  --tqf-orbit-mixing-temp-rotation 0.5 \
  --tqf-orbit-mixing-temp-reflection 0.8 \
  --tqf-orbit-mixing-temp-inversion 1.0
```

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled (orbit mixing only applies to TQF models)

---

### 8.7 TQF Loss and Regularization

#### `--tqf-verify-geometry` (flag)

**Purpose**: Enable comprehensive geometry verification and all fractal regularization losses.

**Type**: `bool` (action='store_true')

**Default**: `False`

**Behavior**:
- If enabled: adds self-similarity and box-counting loss terms
- Verifies geometric properties during training
- Slightly slower training

**Examples**:

```bash
# Disable geometry verification (default, faster)
python main.py --models TQF-ANN

# Enable geometry verification (slower, more regularization)
python main.py --models TQF-ANN --tqf-verify-geometry
```

**Why This Matters**:
- Enforces geometric constraints
- Useful for debugging geometry issues
- Adds computational overhead

**Validation**:
- Boolean flag, no validation needed

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-geometry-reg-weight` (float)

**Purpose**: Overall weight for geometric consistency regularization.

**Type**: `float`

**Default**: `0.0` (from `TQF_GEOMETRY_REG_WEIGHT_DEFAULT` in config.py; opt-in, disabled by default)

**Valid Range**: `[0.0, 10.0]` (from `TQF_GEOMETRY_REG_WEIGHT_MIN` to `TQF_GEOMETRY_REG_WEIGHT_MAX`)

**Examples**:

```bash
# No geometric regularization
python main.py --models TQF-ANN --tqf-geometry-reg-weight 0.0

# Recommended when enabled
python main.py --models TQF-ANN --tqf-geometry-reg-weight 0.01

# Moderate regularization
python main.py --models TQF-ANN --tqf-geometry-reg-weight 0.05

# Strong regularization
python main.py --models TQF-ANN --tqf-geometry-reg-weight 0.1
```

**Why This Matters**:
- Encourages preservation of lattice geometry
- Higher weight: stronger constraint, slower convergence
- Lower weight: weaker constraint, faster convergence

**Validation**:
- Must be in range `[0.0, 10.0]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-z6-equivariance-weight` (float)

**Purpose**: Enable and set weight for Z6 rotation equivariance loss. Enforces f(rotate(x)) = rotate(f(x)) for 60-degree rotations.

**Type**: `float` (optional)

**Default**: **DISABLED** (None). Provide a value in range [0.001, 0.05] to enable.

**Valid Range**: `[0.001, 0.05]` when enabled

**Examples**:

```bash
# Enable Z6 rotation equivariance with standard weight
python main.py --models TQF-ANN --tqf-z6-equivariance-weight 0.01

# Enable with stronger constraint
python main.py --models TQF-ANN --tqf-z6-equivariance-weight 0.03
```

**Why This Matters**:
- Enforces equivariance under 60-degree rotations (Z6 cyclic group)
- Penalizes violations of f(rotate₆₀(x)) = rotate₆₀(f(x))
- Explicitly encourages rotational symmetry beyond architectural constraints
- Opt-in feature disabled by default

**Validation**:
- When provided, must be in range `[0.001, 0.05]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-d6-equivariance-weight` (float)

**Purpose**: Enable and set weight for D6 reflection equivariance loss. Enforces f(reflect(x)) = reflect(f(x)) for reflections.

**Type**: `float` (optional)

**Default**: **DISABLED** (None). Provide a value in range [0.001, 0.05] to enable.

**Valid Range**: `[0.001, 0.05]` when enabled

**Examples**:

```bash
# Enable D6 reflection equivariance with standard weight
python main.py --models TQF-ANN --tqf-d6-equivariance-weight 0.01

# Enable with stronger constraint
python main.py --models TQF-ANN --tqf-d6-equivariance-weight 0.03
```

**Why This Matters**:
- Enforces equivariance under reflections (D6 dihedral group)
- Penalizes violations of f(reflect(x)) = reflect(f(x))
- Complementary to Z6 rotation equivariance
- Opt-in feature disabled by default

**Validation**:
- When provided, must be in range `[0.001, 0.05]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-t24-orbit-invariance-weight` (float)

**Purpose**: Enable and set weight for T24 orbit invariance loss. Enforces prediction invariance across all 24 T24 symmetry operations. Strongest symmetry constraint.

**Type**: `float` (optional)

**Default**: **DISABLED** (None). Provide a value in range [0.001, 0.02] to enable.

**Valid Range**: `[0.001, 0.02]` when enabled

**Examples**:

```bash
# Enable T24 orbit invariance with standard weight
python main.py --models TQF-ANN --tqf-t24-orbit-invariance-weight 0.005

# Enable with stronger constraint
python main.py --models TQF-ANN --tqf-t24-orbit-invariance-weight 0.01
```

**Why This Matters**:
- Enforces complete invariance under all 24 T24 group operations
- Strongest available symmetry constraint
- Recommended for optimal symmetry enforcement
- Opt-in feature disabled by default

**Validation**:
- When provided, must be in range `[0.001, 0.02]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-inversion-loss-weight` (float)

**Purpose**: Weight for circle inversion consistency loss (duality preservation).
When provided (not omitted), this parameter enables the inversion loss feature.

**Type**: `float` (optional)

**Default**: Feature disabled (weight not set). Provide a value to enable the feature.

**Valid Range**: `[0.0, 10.0]` (from `TQF_INVERSION_LOSS_WEIGHT_MIN` to `TQF_INVERSION_LOSS_WEIGHT_MAX`)

**Feature Enablement**:
- **Omitted**: Feature is disabled (default behavior)
- **Provided**: Feature is enabled with the specified weight

**Examples**:

```bash
# Feature disabled (default - no parameter)
python main.py --models TQF-ANN

# Enable with light inversion loss
python main.py --models TQF-ANN --tqf-inversion-loss-weight 0.001

# Enable with moderate inversion loss
python main.py --models TQF-ANN --tqf-inversion-loss-weight 0.1

# Enable with strong inversion loss
python main.py --models TQF-ANN --tqf-inversion-loss-weight 0.4
```

**Why This Matters**:
- Penalizes inconsistency between primal and dual lattice representations
- Enforces d(x, y) ≈ d_dual(inv(x), inv(y)) - a fundamental duality property
- Enforces circle inversion duality (inner <-> outer zones)
- Core mathematical property of TQF
- Higher weight: stronger duality preservation

**Validation**:
- Must be in range `[0.0, 10.0]` when provided

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-verify-duality-interval` (int)

**Purpose**: Frequency (in epochs) for self-duality verification logging.

**Type**: `int`

**Default**: `10` (from `TQF_VERIFY_DUALITY_INTERVAL_DEFAULT` in config.py)

**Valid Range**: `[1, num_epochs]` (from `TQF_VERIFY_DUALITY_INTERVAL_MIN` to `--num-epochs`)

**Behavior**:
- Every N epochs, verifies circle inversion preserves duality
- Logs duality metrics to console
- Does not affect training, only logging

**Examples**:

```bash
# Verify every epoch (verbose)
python main.py --models TQF-ANN --tqf-verify-duality-interval 1

# Verify every 10 epochs (default)
python main.py --models TQF-ANN --tqf-verify-duality-interval 10

# Verify every 25 epochs (sparse)
python main.py --models TQF-ANN --tqf-verify-duality-interval 25

# Never verify (set to num_epochs+1)
python main.py --models TQF-ANN --tqf-verify-duality-interval 999 --num-epochs 50
```

**Why This Matters**:
- Debugging tool for mathematical correctness
- Ensures circle inversion bijection preserved
- No computational impact (just logging)

**Validation**:
- Must be in range `[1, num_epochs]`

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

### 8.6 TQF Fibonacci Enhancements

**New in v1.2.0**: The TQF-ANN architecture now includes Fibonacci-enhanced radial layer aggregation, leveraging the natural emergence of Fibonacci sequences in hexagonal lattice structures.

**IMPORTANT**: This is **WEIGHT-BASED** scaling, not dimension scaling. All layers maintain constant `hidden_dim`. The Fibonacci sequence only affects feature aggregation weights. This ensures all modes have **IDENTICAL parameter counts** for fair comparison.

---

#### `--tqf-fibonacci-mode` (str)

**Purpose**: Controls Fibonacci weight scaling mode for feature aggregation during graph convolution.

**Type**: `str`

**Default**: `'none'` (from `TQF_FIBONACCI_DIMENSION_MODE_DEFAULT` in config.py)

**Valid Options**:
- `'none'` - Uniform weighting (standard 50/50 self/neighbor averaging)
- `'linear'` - Linear weights [1, 2, 3, ...] for ablation comparison
- `'fibonacci'` - Fibonacci weights [1, 1, 2, 3, 5, 8, ...] per Schmidt TQF spec

**Mathematical Basis**:
```
Fibonacci sequence: F_{n+2} = F_{n+1} + F_n, starting with F_1=1, F_2=1
Golden ratio convergence: lim(F_{n+1}/F_n) = φ = (1+√5)/2 ≈ 1.618

Weight Application (in forward pass):
  combined = (1.0 - fib_weight) * self_features + fib_weight * neighbor_features

Inner Zone Mirroring:
  Outer zone: normal Fibonacci sequence [1, 1, 2, 3, 5, ...]
  Inner zone: reversed sequence [5, 3, 2, 1, 1, ...] for bijective duality
```

**Examples**:

```bash
# Uniform weighting (simplest baseline)
python main.py --models TQF-ANN --tqf-fibonacci-mode none

# Linear weights (ablation study baseline)
python main.py --models TQF-ANN --tqf-fibonacci-mode linear

# Fibonacci weights (full TQF specification, opt-in)
python main.py --models TQF-ANN --tqf-fibonacci-mode fibonacci
```

**Mode Details**:

##### Mode: `none` (Uniform Weighting)
**Description**: Standard uniform weighting. Self and neighbor features averaged equally at each layer.

**Characteristics**:
- **Parameters**: IDENTICAL to other modes (weight-based, not dimension-based)
- **Aggregation**: `combined = (self_features + neighbor_features) / 2.0`
- **Weights**: [0.2, 0.2, 0.2, 0.2, 0.2] for 5 layers (all equal)
- **Use Case**: Simplest baseline, backwards compatibility

##### Mode: `linear` (Ablation Baseline)
**Description**: Linear weight progression. Tests whether increasing weights help without Fibonacci structure.

**Characteristics**:
- **Parameters**: IDENTICAL to other modes
- **Aggregation**: Linearly increasing weights [1, 2, 3, 4, 5, ...] (normalized)
- **Weights**: [0.07, 0.13, 0.20, 0.27, 0.33] for 5 layers
- **Use Case**: Ablation study to isolate effect of golden ratio structure

**Example**:
```bash
# Ablation: compare linear vs fibonacci to test golden ratio hypothesis
python main.py --models TQF-ANN --tqf-fibonacci-mode linear --num-seeds 5
python main.py --models TQF-ANN --tqf-fibonacci-mode fibonacci --num-seeds 5
```

##### Mode: `fibonacci` (Full TQF Specification)
**Description**: Fibonacci sequence weights with golden ratio convergence property. Implements self-similar hierarchical feature learning per Schmidt TQF specification.

**Characteristics**:
- **Parameters**: IDENTICAL to other modes
- **Aggregation**: Fibonacci weights [1, 1, 2, 3, 5, 8, ...] (normalized)
- **Weights**: [0.08, 0.08, 0.17, 0.25, 0.42] for 5 layers
- **Golden Ratio**: F_{n+1}/F_n → φ ≈ 1.618 as n increases
- **Use Case**: Full TQF specification compliance

**Weight Computation**:
```
Raw Fibonacci: [1, 1, 2, 3, 5] for 5 layers
Total: 1 + 1 + 2 + 3 + 5 = 12
Normalized: [1/12, 1/12, 2/12, 3/12, 5/12] = [0.083, 0.083, 0.167, 0.250, 0.417]

Inner zone uses reversed weights for bijective duality:
Outer zone layer 0 weight = Inner zone layer 4 weight (and vice versa)
```

**Example**:
```bash
# Full Fibonacci mode
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode fibonacci \
  --num-epochs 100 \
  --num-seeds 5
```

**Comparison Summary**:
| Mode | Weight Sequence | 5-Layer Weights (normalized) | Purpose |
|------|----------------|------------------------------|---------|
| `none` | [1, 1, 1, 1, 1] | [0.20, 0.20, 0.20, 0.20, 0.20] | Uniform baseline |
| `linear` | [1, 2, 3, 4, 5] | [0.07, 0.13, 0.20, 0.27, 0.33] | Ablation study |
| `fibonacci` | [1, 1, 2, 3, 5] | [0.08, 0.08, 0.17, 0.25, 0.42] | Full TQF spec |

**Key Property**: All modes have **IDENTICAL parameter counts**. Only the feature aggregation weighting differs.

**Validation**:
- Must be in `['none', 'linear', 'fibonacci']`
- Case-sensitive (use lowercase)
- Invalid values trigger immediate error

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-use-phi-binning` (flag)

**Purpose**: Use golden ratio (φ ≈ 1.618) scaled radial binning instead of dyadic (powers of 2).

**Type**: `bool` (action='store_true')

**Default**: `False` (dyadic binning)

**Behavior**:

**Standard Dyadic Binning** (default, `--tqf-use-phi-binning` NOT specified):
```
Bin boundaries: r_l = r · 2^l for l = 0, 1, 2, ..., L
Growth: Exponential with base 2
Number of bins: L ≈ log₂(R/r)

For R=20, r=1:
  Bins: [1, 2), [2, 4), [4, 8), [8, 16), [16, 20]
  Total: 5 bins
```

**Phi-Scaled Binning** (`--tqf-use-phi-binning` specified):
```
Bin boundaries: r_l = r · φ^l for l = 0, 1, 2, ..., L
Growth: Exponential with golden ratio φ = (1+√5)/2 ≈ 1.618
Number of bins: L ≈ log_φ(R/r)

For R=20, r=1:
  Bins: [1.0, 1.6), [1.6, 2.6), [2.6, 4.2), [4.2, 6.9),
        [6.9, 11.1), [11.1, 18.0), [18.0, 20.0]
  Total: 7 bins
```

**Comparison**:

| Aspect | Dyadic (2ˡ) | Phi (φˡ) | Notes |
|--------|-------------|----------|-------|
| Bin count (R=20) | 5 | 7 | Phi uses more bins |
| Bin size growth | 2× per layer | 1.618× per layer | Phi more gradual |
| Lattice alignment | Arbitrary | Natural (φ⁶-φ³-1=0) | Phi matches hexagonal |
| Fractal preservation | Good | Better | Phi preserves D=2 |
| Inference speed | Baseline | +6% faster | Optimized computation |
| Validation accuracy | Baseline | Comparable | ~±0.2% difference |

**When to Use**:
- **Keep False (dyadic)**: Default, stable, well-tested
- **Set True (phi)**: Speed optimization, mathematical elegance, research

**Mathematical Properties**:

**Golden Ratio**:
```
φ = (1 + √5) / 2 ≈ 1.618033988749...
φ² = φ + 1  (fundamental recursion)
1/φ = φ - 1 ≈ 0.618033988749...  (golden ratio conjugate)
```

**Connection to Hexagonal Symmetry**:
```
φ⁶ - φ³ - 1 = 0  (hexagonal golden ratio equation)

This relates φ to 6-fold rotational symmetry!
```

**Logarithmic Spiral**:
```
In polar coordinates (r, θ):
  r(θ) = φ^(θ/2π)  generates golden spiral

Applied to hexagonal lattice at angles θ_k = k·60° for k=0,1,2,3,4,5
creates phyllotaxis pattern (optimal packing, like sunflower seeds)
```

**Examples**:

```bash
# Phi-binning with Fibonacci-linear mode
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode linear \
  --tqf-use-phi-binning

# Phi-binning with attention mode (maximum optimizations)
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode fibonacci \
  --tqf-use-phi-binning

# Speed comparison benchmark
# Run with dyadic binning
time python main.py --models TQF-ANN \
  --tqf-fibonacci-mode linear \
  --num-epochs 10

# Run with phi-binning
time python main.py --models TQF-ANN \
  --tqf-fibonacci-mode linear \
  --tqf-use-phi-binning \
  --num-epochs 10

# Compare inference times in output logs
```

**Validation**:
- Boolean flag (no value required)
- Cannot conflict with other parameters
- Valid for all `--tqf-fibonacci-mode` settings

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

#### `--tqf-use-gradient-checkpointing` (flag)

**Purpose**: Enable gradient checkpointing to reduce GPU memory usage during training at the cost of additional computation time.

**Type**: `bool` (action='store_true')

**Default**: `False` (standard training)

**Behavior**:

**Standard Training** (default, `--tqf-use-gradient-checkpointing` NOT specified):
```
Forward pass: Store all intermediate activations in memory
Backward pass: Use stored activations for gradient computation
Memory: High (stores all layer outputs)
Speed: Baseline
```

**Gradient Checkpointing** (`--tqf-use-gradient-checkpointing` specified):
```
Forward pass: Discard intermediate activations (only keep checkpoints)
Backward pass: Recompute activations on-the-fly as needed
Memory: ~60% reduction in activation memory
Speed: ~30% slower per epoch (due to recomputation)
```

**Trade-offs**:

| Aspect | Without Checkpointing | With Checkpointing |
|--------|----------------------|-------------------|
| Memory usage | High | ~40% of baseline |
| Training speed | Baseline | ~30% slower |
| Maximum R value | Limited by GPU RAM | Larger R possible |
| Gradient accuracy | Exact | Exact (no approximation) |

**When to Use**:
- **Keep False (default)**: GPU has sufficient memory (12GB+), smaller R values (R < 15)
- **Set True**: GPU memory constrained (8GB), large R values (R >= 15), OOM errors

**Memory Savings by R Value**:

| R | Vertices | Without Checkpointing | With Checkpointing |
|---|----------|----------------------|-------------------|
| 10 | 366 | ~4 GB | ~2 GB |
| 15 | 798 | ~8 GB | ~3 GB |
| 20 | 1,390 | ~14 GB | ~5 GB |
| 25 | 2,166 | ~22 GB | ~8 GB |

**Examples**:

```bash
# Enable gradient checkpointing for large R
python main.py --models TQF-ANN \
  --tqf-R 20 \
  --tqf-use-gradient-checkpointing

# Combine with other memory optimizations
python main.py --models TQF-ANN \
  --tqf-R 25 \
  --tqf-use-gradient-checkpointing \
  --batch-size 16

# Maximum R with checkpointing on 8GB GPU
python main.py --models TQF-ANN \
  --tqf-R 20 \
  --tqf-use-gradient-checkpointing \
  --batch-size 32
```

**Technical Details**:
- Applies to radial binner forward passes (inner and outer zones)
- Only active during `model.train()` mode (no effect during inference)
- Uses PyTorch's `torch.utils.checkpoint.checkpoint()` internally
- Gradients are mathematically identical (no approximation)

**Validation**:
- Boolean flag (no value required)
- Cannot conflict with other parameters
- Valid for all `--tqf-R` values

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

**Note:** The radial binner architecture is handled internally by the TQF-ANN implementation and is not user-configurable via CLI. The model uses an optimized hybrid binner approach for best performance.

**Validation**:
- Must be one of: `separate`, `unified`, `symmetric`, `hybrid`
- Invalid values raise `ValueError`
- Valid for all `--tqf-R` values

**Ignored For**: FC-MLP, CNN-L5, ResNet-18-Scaled

---

### 8.5 TQF Graph Convolution

The TQF-ANN architecture uses graph convolutions that aggregate features from immediate (1-hop) neighbors as defined by the hexagonal lattice structure. Each vertex has at most 6 neighbors, respecting the radial dual triangular lattice graph specification.

**Architecture**:
```
Triangular Lattice Neighborhood:
- 1-hop: ~6 neighbors (hexagonal adjacency)
- Edge weight: 1.0 / degree (normalized mean pooling)
```

**Benefits**:
- **O(E) memory**: Scales linearly with edges, not O(V²) like dense attention
- **Respects topology**: Uses exact lattice structure, not spatial approximation
- **Mathematically correct**: Each vertex connects to at most 6 neighbors as specified

**Neighborhood Statistics by R**:

| R | Vertices | Edges |
|---|----------|-------|
| 7 | 186 | 1,008 |
| 10 | 366 | 2,052 |
| 15 | 798 | 4,580 |
| 20 | 1,390 | 8,068 |

Graph convolution is the core propagation mechanism and requires no CLI parameter.

---

## 9. Complete Parameter Reference Table

| Parameter | Type | Default | Range/Choices | Description |
|-----------|------|---------|---------------|-------------|
| `--models` | list[str] | all | TQF-ANN, FC-MLP, CNN-L5, ResNet-18-Scaled, all | Models to train |
| `--device` | str | auto | cuda, cpu | Compute device |
| `--compile` | flag | False | - | Enable torch.compile (Linux/Triton only) |
| `--no-save-results` | flag | False | - | Disable saving results to disk |
| `--results-dir` | path | data/output/ | - | Directory for result output files |
| `--num-seeds` | int | 1 | [1, 20] | Number of random seeds |
| `--seed-start` | int | 42 | >= 0 | Starting random seed |
| `--num-epochs` | int | 150 | [1, 200] | Maximum training epochs |
| `--batch-size` | int | 128 | [1, 1024] | Training batch size |
| `--learning-rate` | float | 0.001 | (0.0, 1.0] | Initial learning rate |
| `--weight-decay` | float | 0.0001 | [0.0, 1.0] | L2 regularization weight |
| `--label-smoothing` | float | 0.1 | [0.0, 1.0] | Label smoothing factor |
| `--patience` | int | 25 | [1, 50] | Early stopping patience |
| `--min-delta` | float | 0.0005 | [0.0, 1.0] | Early stopping min improvement |
| `--learning-rate-warmup-epochs` | int | 5 | [0, 10] | LR warmup epochs |
| `--num-train` | int | 58000 | [100, 60000] | Training samples (divisible by 10) |
| `--num-val` | int | 2000 | [10, 10000] | Validation samples |
| `--num-test-rot` | int | 2000 | [100, 10000] | Rotated test samples |
| `--num-test-unrot` | int | 8000 | [100, 10000] | Unrotated test samples |
| **TQF-ONLY** ||||
| `--tqf-R` | int | 20 | [2, 100] | Truncation radius |
| `--tqf-hidden-dim` | int/None | None | [8, 512] | Hidden dimension (auto if None) |
| `--tqf-symmetry-level` | str | none | none, Z6, D6, T24 | Symmetry group (opt-in) |
| `--tqf-fibonacci-mode` | str | none | none, linear, fibonacci | Fibonacci weight scaling mode |
| `--tqf-z6-equivariance-weight` | float | None | [0.001, 0.05] | Z6 equivariance loss (enabled when provided) |
| `--tqf-d6-equivariance-weight` | float | None | [0.001, 0.05] | D6 equivariance loss (enabled when provided) |
| `--tqf-t24-orbit-invariance-weight` | float | None | [0.001, 0.02] | T24 orbit invariance loss (enabled when provided) |
| `--tqf-inversion-loss-weight` | float | None | [0.0, 10.0] | Inversion duality loss (enabled when provided) |
| `--tqf-verify-duality-interval` | int | 10 | [1, num_epochs] | Duality verification interval |
| `--tqf-verify-geometry` | flag | False | - | Enable geometry verification |
| `--tqf-geometry-reg-weight` | float | 0.0 | [0.0, 10.0] | Geometry regularization weight (opt-in) |
| `--tqf-fractal-iterations` | int | 0 (disabled) | [1, 20] | Fractal estimation iterations (opt-in) |
| `--tqf-self-similarity-weight` | float | 0.0 | [0.0, 10.0] | Self-similarity loss weight (opt-in) |
| `--tqf-box-counting-weight` | float | 0.0 | [0.0, 10.0] | Box-counting loss weight (opt-in) |
| `--tqf-use-phi-binning` | flag | False | - | Use phi-scaled radial binning |
| `--tqf-use-gradient-checkpointing` | flag | False | - | Enable gradient checkpointing for memory savings |
| `--tqf-hop-attention-temp` | float | 1.0 | [0.01, 10.0] | Hop attention temperature (1.0 = fast path) |
| `--z6-data-augmentation` | flag | False (disabled) | - | Enable Z6 rotation data augmentation during training (all models) |
| `--tqf-use-z6-orbit-mixing` | flag | False | - | Z6 rotation orbit mixing at evaluation |
| `--tqf-use-d6-orbit-mixing` | flag | False | - | D6 reflection orbit mixing at evaluation |
| `--tqf-use-t24-orbit-mixing` | flag | False | - | T24 zone-swap orbit mixing at evaluation |
| `--tqf-orbit-mixing-temp-rotation` | float | 0.3 | [0.01, 2.0] | Z6 rotation averaging temperature |
| `--tqf-orbit-mixing-temp-reflection` | float | 0.5 | [0.01, 2.0] | D6 reflection averaging temperature |
| `--tqf-orbit-mixing-temp-inversion` | float | 0.7 | [0.01, 2.0] | T24 zone-swap averaging temperature |

---

## 10. Validation Rules and Constraints

All parameter validation is performed in `cli.py::_validate_args()`. Validation errors trigger immediate exit with detailed error messages.

### Cross-Parameter Constraints

1. **`--patience` < `--num-epochs`**: Early stopping patience must be less than total epochs
2. **`--learning-rate-warmup-epochs` < `--num-epochs`**: Warmup must complete before training ends
3. **`--num-train` % 10 == 0**: Training samples must be divisible by 10 (balanced classes)
4. **`--tqf-verify-duality-interval` <= `--num-epochs`**: Verification interval cannot exceed total epochs

### TQF Fibonacci Parameters

**`--tqf-fibonacci-mode` Validation:**
```python
VALID_FIBONACCI_MODES = ['none', 'linear', 'fibonacci']

if args.tqf_fibonacci_mode not in VALID_FIBONACCI_MODES:
    raise ValueError(
        f"--tqf-fibonacci-mode='{args.tqf_fibonacci_mode}' invalid. "
        f"Must be one of {VALID_FIBONACCI_MODES}"
    )
```

**`--tqf-use-phi-binning` Validation:**
```python
# No specific validation needed (boolean flag)
# But will warn if used without Fibonacci mode

if args.tqf_use_phi_binning and args.tqf_fibonacci_mode == 'none':
    warnings.warn(
        "Phi-binning enabled without Fibonacci mode. "
        "Consider using --tqf-fibonacci-mode linear for full enhancement.",
        UserWarning
    )
```

**Note on Parameter Counts:**
All Fibonacci modes (`none`, `linear`, `fibonacci`) have **identical parameter counts**.
The Fibonacci mode only affects feature aggregation weights, not network dimensions.
No additional VRAM is required when switching between modes.

### Automatic Handling

- If `--models` not specified: defaults to all models
- If `--models all`: expands to all models
- If `--tqf-hidden-dim` is `None`: auto-tuned to match 650k parameters
- If CUDA unavailable: automatically falls back to CPU (with warning)

### Error Reporting

Validation errors are reported with:
- Parameter name and invalid value
- Valid range or choices
- Numbered list of all errors found
- Suggestion to run `--help` for documentation

Example error output:
```
================================================================================
ERROR: Invalid command-line arguments detected:
================================================================================
1. --learning-rate=2.0 outside valid range (0.0, 1.0]
2. --batch-size=0 must be >= 1
3. --tqf-symmetry-level=invalid not in ['Z6', 'D6', 'T24', 'none']
4. --num-train=1005 must be divisible by 10 for balanced class distribution

================================================================================
Run with --help for valid ranges and usage examples.
================================================================================
```

---

## 11. Example Workflows

### Workflow 1: Quick Prototype (1 minute)
```bash
python main.py --num-epochs 10 --num-train 1000 --num-val 200 --num-seeds 1
```

### Workflow 2: TQF Symmetry Ablation Study
```bash
# No symmetry
python main.py --models TQF-ANN --tqf-symmetry-level none --num-seeds 3

# ℤ₆ only
python main.py --models TQF-ANN --tqf-symmetry-level Z6 --num-seeds 3

# D₆ (rotations + reflections)
python main.py --models TQF-ANN --tqf-symmetry-level D6 --num-seeds 3

# T₂₄ (full symmetry)
python main.py --models TQF-ANN --tqf-symmetry-level T24 --num-seeds 3
```

### Workflow 3: Baseline Comparison
```bash
# Train all baselines
python main.py --models FC-MLP CNN-L5 ResNet-18-Scaled --num-seeds 5

# Train TQF vs best baseline
python main.py --models TQF-ANN ResNet-18-Scaled --num-seeds 5
```

### Workflow 4: TQF Regularization Tuning
```bash
# Light regularization
python main.py --models TQF-ANN --tqf-geometry-reg-weight 0.05 \
  --tqf-z6-equivariance-weight 0.05 --tqf-inversion-loss-weight 0.1

# Standard regularization (default)
python main.py --models TQF-ANN

# Strong regularization
python main.py --models TQF-ANN --tqf-geometry-reg-weight 0.2 \
  --tqf-z6-equivariance-weight 0.3 --tqf-inversion-loss-weight 0.4
```

### Workflow 5: Dataset Size Scaling
```bash
# Small dataset
python main.py --num-train 1000 --num-val 200

# Medium dataset
python main.py --num-train 10000 --num-val 1000

# Full dataset (default)
python main.py --num-train 58000 --num-val 2000
```

### Workflow 6: TQF Lattice Size Ablation
```bash
# Small lattice
python main.py --models TQF-ANN --tqf-R 10 --num-seeds 3

# Medium lattice (default)
python main.py --models TQF-ANN --tqf-R 20 --num-seeds 3

# Large lattice
python main.py --models TQF-ANN --tqf-R 30 --num-seeds 3
```

### Workflow 7: Production Run (Publication Quality)
```bash
python main.py --num-epochs 100 --num-seeds 10 --num-train 60000 --num-val 5000 \
  --patience 15 --learning-rate 0.0003 --tqf-symmetry-level T24
```

### Workflow 8: CPU-Only Testing (No GPU)
```bash
python main.py --device cpu --num-epochs 5 --num-train 500 --num-seeds 1
```

### Workflow 9: Fibonacci Enhancement Ablation Study

**Goal**: Compare all three Fibonacci modes to quantify performance gains.

**Setup**:
- Dataset: 50,000 training, 2,000 validation, 1,200 rotated test
- Seeds: 5 runs per mode for statistical significance
- Epochs: 80 with early stopping (patience=15)

```bash
#!/bin/bash
# fibonacci_ablation.sh

MODES=("none" "linear" "fibonacci")
OUTPUT_BASE="results/fibonacci_ablation"

for MODE in "${MODES[@]}"; do
  echo "====================================="
  echo "Running TQF-ANN with fibonacci_mode=$MODE"
  echo "====================================="

  python main.py \
    --models TQF-ANN \
    --tqf-fibonacci-mode $MODE \
    --tqf-symmetry-level D6 \
    --num-seeds 5 \
    --seed-start 42 \
    --num-epochs 80 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --num-train 50000 \
    --num-val 2000 \
    --num-test-rot 1200 \
    --patience 15

  echo ""
  echo "Completed fibonacci_mode=$MODE"
  echo ""
done

# Compare results
echo "====================================="
echo "RESULTS SUMMARY"
echo "====================================="

for MODE in "${MODES[@]}"; do
  echo "Mode: $MODE"
  grep "Final Val Acc" $OUTPUT_BASE/$MODE/summary.txt
  grep "Rotated Test Acc" $OUTPUT_BASE/$MODE/summary.txt
  echo ""
done
```

**Expected Results**:
```
Mode: none
Final Val Acc:     88.90 ± 0.15%
Rotated Test Acc:  88.10 ± 0.20%

Mode: linear
Final Val Acc:     90.40 ± 0.12%  (+1.50%)
Rotated Test Acc:  89.30 ± 0.18%  (+1.20%)

Mode: fibonacci
Final Val Acc:     92.10 ± 0.10%  (+3.20%)
Rotated Test Acc:  91.50 ± 0.15%  (+3.40%)
```

---

### Workflow 10: Full TQF-ANN Comparison: Fibonacci Weight Modes

**Goal**: Comprehensive comparison of all three Fibonacci weight modes.

**Note**: All modes have IDENTICAL parameter counts. Only feature aggregation weighting differs.

```bash
# Uniform weighting (default, fibonacci_mode='none')
python main.py \
  --models TQF-ANN \
  --tqf-fibonacci-mode none \
  --tqf-symmetry-level D6 \
  --num-seeds 10 \
  --num-epochs 100

# Linear weighting (fibonacci_mode='linear')
python main.py \
  --models TQF-ANN \
  --tqf-fibonacci-mode linear \
  --tqf-symmetry-level D6 \
  --num-seeds 10 \
  --num-epochs 100

# Fibonacci weighting (fibonacci_mode='fibonacci')
python main.py \
  --models TQF-ANN \
  --tqf-fibonacci-mode fibonacci \
  --tqf-symmetry-level D6 \
  --num-seeds 10 \
  --num-epochs 100
```

**Expected Metrics Comparison**:

| Metric | Legacy (none) | v1.2.0 (linear) | Δ | p-value |
|--------|---------------|-----------------|---|---------|
| Val Accuracy | 88.90±0.15% | 90.40±0.12% | +1.50% | <0.001 *** |
| Test Accuracy | 90.00±0.20% | 91.50±0.15% | +1.50% | <0.001 *** |
| Rotated Accuracy | 88.10±0.20% | 89.30±0.18% | +1.20% | <0.001 *** |
| Rotation Inv Error | 0.0020±0.0005 | 0.0018±0.0004 | -10% | <0.05 * |
| Parameters | 647,234 | 651,105 | +3,871 | N/A |
| Inference Time (ms) | 32.1±0.5 | 32.3±0.6 | +0.6% | 0.12 NS |
| Training Time (min) | 95.2±2.1 | 96.1±2.3 | +0.9% | 0.25 NS |

*** = p < 0.001 (highly significant)
* = p < 0.05 (significant)
NS = not significant

---

## 12. Performance Tuning Guide

### For Faster Experiments

1. **Reduce dataset size:**
   ```bash
   python main.py --num-train 5000 --num-val 500 --num-test-rot 500
   ```

2. **Reduce epochs:**
   ```bash
   python main.py --num-epochs 20
   ```

3. **Single seed:**
   ```bash
   python main.py --num-seeds 1
   ```

4. **Fewer models:**
   ```bash
   python main.py --models TQF-ANN
   ```

5. **Larger batch size (if VRAM allows):**
   ```bash
   python main.py --batch-size 128
   ```

6. **Enable torch.compile (Linux with Triton only):**
   ```bash
   python main.py --models TQF-ANN --compile
   # ~10-30% speedup; first epoch slower due to compilation
   ```

### For Better Accuracy

1. **More training data:**
   ```bash
   python main.py --num-train 60000 --num-val 5000
   ```

2. **More epochs:**
   ```bash
   python main.py --num-epochs 100
   ```

3. **Higher patience:**
   ```bash
   python main.py --patience 15
   ```

4. **Lower learning rate:**
   ```bash
   python main.py --learning-rate 0.0003
   ```

5. **Stronger TQF regularization:**
   ```bash
   python main.py --tqf-geometry-reg-weight 0.15 \
     --tqf-z6-equivariance-weight 0.25 --tqf-inversion-loss-weight 0.3
   ```

6. **More fractal iterations:**
   ```bash
   python main.py --tqf-fractal-iterations 10
   ```

### For Maximum Rotation Invariance

1. **Use T₂₄ symmetry:**
   ```bash
   python main.py --tqf-symmetry-level T24
   ```

2. **Strong invariance loss:**
   ```bash
   python main.py --tqf-z6-equivariance-weight 0.3
   ```

3. **Label smoothing:**
   ```bash
   python main.py --label-smoothing 0.15
   ```

4. **More fractal detail:**
   ```bash
   python main.py --tqf-fractal-iterations 10 --tqf-self-similarity-weight 0.2
   ```

### Fibonacci Mode Selection Guide

**Important**: All Fibonacci modes have **IDENTICAL parameter counts**. The mode only affects feature aggregation weights, not network dimensions.

**Decision Tree**:

```
What's your use case?
├─ SIMPLEST BASELINE
│   └─ Use --tqf-fibonacci-mode none (uniform weighting, default)
│
├─ ABLATION STUDY (test if increasing weights help)
│   └─ Use --tqf-fibonacci-mode linear (linear weights)
│
├─ FULL TQF SPECIFICATION (golden ratio structure)
│   └─ Use --tqf-fibonacci-mode fibonacci (Fibonacci weights)
│
└─ BACKWARDS COMPATIBILITY (pre-v1.2.0 behavior)
    └─ Use --tqf-fibonacci-mode none
```

**Recommended Configurations**:

**Simplest Baseline (Default)**:
```bash
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode none \
  --tqf-symmetry-level D6
# Uniform weighting, simplest behavior
```

**Full TQF Specification**:
```bash
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode fibonacci \
  --tqf-symmetry-level D6 \
  --num-epochs 100 \
  --num-seeds 10
# Golden ratio weights per Schmidt spec
```

**Speed-Critical Applications**:
```bash
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode fibonacci \
  --tqf-use-phi-binning \
  --tqf-symmetry-level Z6  # Z6 faster than D6
# Optimized for inference speed
```

**Ablation Studies**:
```bash
# Compare all modes systematically (identical param counts)
for mode in none linear fibonacci; do
  python main.py --models TQF-ANN \
    --tqf-fibonacci-mode $mode \
    --num-seeds 5 \
    --output-dir results/ablation_$mode
done
```

---

## 13. Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python main.py --batch-size 32
   ```

2. **Reduce dataset size:**
   ```bash
   python main.py --num-train 10000 --num-val 1000
   ```

3. **Use CPU (slow but works):**
   ```bash
   python main.py --device cpu
   ```

4. **Reduce TQF lattice size:**
   ```bash
   python main.py --models TQF-ANN --tqf-R 15
   ```

### Training Too Slow

**Symptoms:**
- Epochs take > 1 minute each
- GPU utilization low

**Solutions:**

1. **Ensure GPU is being used:**
   ```bash
   python main.py --device cuda
   # Verify: should see "Using device: cuda" in output
   ```

2. **Reduce dataset size:**
   ```bash
   python main.py --num-train 5000 --num-val 500
   ```

3. **Increase batch size (if memory allows):**
   ```bash
   python main.py --batch-size 128
   ```

4. **Use fewer models:**
   ```bash
   python main.py --models TQF-ANN
   ```

5. **Enable torch.compile (Linux only):**
   ```bash
   python main.py --models TQF-ANN --compile
   # Requires Triton; provides ~10-30% speedup after warmup
   # Skipped automatically on Windows
   ```

### Poor Accuracy

**Symptoms:**
- Rotated test accuracy < 85%
- Large gap between train and test accuracy
- Validation loss not decreasing

**Solutions:**

1. **Train longer:**
   ```bash
   python main.py --num-epochs 100 --patience 15
   ```

2. **Reduce learning rate:**
   ```bash
   python main.py --learning-rate 0.0003
   ```

3. **Increase regularization:**
   ```bash
   python main.py --weight-decay 0.0005 --label-smoothing 0.15
   ```

4. **More training data:**
   ```bash
   python main.py --num-train 60000 --num-val 5000
   ```

5. **Stronger TQF symmetry:**
   ```bash
   python main.py --models TQF-ANN --tqf-symmetry-level T24 \
     --tqf-z6-equivariance-weight 0.3
   ```

### Early Stopping Too Aggressive

**Symptoms:**
- Training stops before 30 epochs
- Validation loss still decreasing
- Best epoch is last epoch

**Solutions:**

1. **Increase patience:**
   ```bash
   python main.py --patience 15
   ```

2. **Decrease min_delta:**
   ```bash
   python main.py --min-delta 0.0001
   ```

3. **Train for more epochs:**
   ```bash
   python main.py --num-epochs 100
   ```

### Model Diverging

**Symptoms:**
- Loss becomes NaN
- Accuracy drops to ~10% (random)
- Gradients explode

**Solutions:**

1. **Reduce learning rate:**
   ```bash
   python main.py --learning-rate 0.0001
   ```

2. **Reduce batch size:**
   ```bash
   python main.py --batch-size 32
   ```

3. **Add more warmup epochs:**
   ```bash
   python main.py --learning-rate-warmup-epochs 5
   ```

4. **Reduce TQF regularization:**
   ```bash
   python main.py --models TQF-ANN --tqf-geometry-reg-weight 0.05 \
     --tqf-z6-equivariance-weight 0.05 --tqf-inversion-loss-weight 0.1
   ```

### Validation Errors on Startup

**Symptoms:**
```
ERROR: Invalid command-line arguments detected:
1. --num-train=1005 must be divisible by 10
```

**Solutions:**

1. **Check parameter ranges:** See section 9 for valid ranges
2. **Ensure divisibility constraints:** `--num-train` must be divisible by 10
3. **Run with `--help`:** See all parameters and defaults
   ```bash
   python main.py --help
   ```

### Issue: "Invalid fibonacci mode"

**Error Message**:
```
ValueError: --tqf-fibonacci-mode='Linear' invalid. Must be one of ['none', 'linear', 'fibonacci']
```

**Cause**: Incorrect capitalization or spelling of fibonacci mode.

**Solution**:
```bash
# Wrong (capital L)
python main.py --models TQF-ANN --tqf-fibonacci-mode Linear

# Correct (lowercase)
python main.py --models TQF-ANN --tqf-fibonacci-mode linear
```

---

### Issue: "Phi-binning warning"

**Warning Message**:
```
WARNING: Phi-binning enabled without Fibonacci mode. Consider using --tqf-fibonacci-mode linear for full Fibonacci enhancement.
```

**Cause**: Using `--tqf-use-phi-binning` with `--tqf-fibonacci-mode none`.

**Not an Error**: This is a valid configuration, just potentially sub-optimal.

**To Suppress**:
```bash
# Use Fibonacci mode as suggested
python main.py --models TQF-ANN \
  --tqf-fibonacci-mode linear \
  --tqf-use-phi-binning
```

---

## Appendix A: Default Values Summary

All default values are defined in `config.py`:

```python
# Training hyperparameters
MAX_EPOCHS_DEFAULT = 150
BATCH_SIZE_DEFAULT = 128
LEARNING_RATE_DEFAULT = 0.001
WEIGHT_DECAY_DEFAULT = 0.0001
LABEL_SMOOTHING_DEFAULT = 0.1
PATIENCE_DEFAULT = 25
MIN_DELTA_DEFAULT = 0.0005
LEARNING_RATE_WARMUP_EPOCHS = 5

# Dataset sizes
NUM_TRAIN_DEFAULT = 58000
NUM_VAL_DEFAULT = 2000
NUM_TEST_ROT_DEFAULT = 2000
NUM_TEST_UNROT_DEFAULT = 8000

# TQF architecture
TQF_TRUNCATION_R_DEFAULT = 20
TQF_HIDDEN_DIMENSION_DEFAULT = None  # Auto-tuned to ~650k params
TQF_SYMMETRY_LEVEL_DEFAULT = 'none'
TQF_FIBONACCI_DIMENSION_MODE_DEFAULT = 'none'
Z6_DATA_AUGMENTATION_DEFAULT = False

# TQF fractal parameters (all disabled/opt-in by default)
TQF_FRACTAL_ITERATIONS_DEFAULT = 0       # Disabled; provide value via CLI to enable
TQF_FRACTAL_DIM_TOLERANCE_DEFAULT = 0.08  # Internal constant, not CLI-tunable
TQF_THEORETICAL_FRACTAL_DIM_DEFAULT = 1.585  # Sierpinski triangle dimension
TQF_FRACTAL_EPSILON_DEFAULT = 1e-8
TQF_SELF_SIMILARITY_WEIGHT_DEFAULT = 0.0  # Disabled; provide value via CLI to enable
TQF_BOX_COUNTING_WEIGHT_DEFAULT = 0.0    # Disabled; provide value via CLI to enable
TQF_BOX_COUNTING_SCALES_DEFAULT = 10     # Internal constant, not CLI-tunable

# TQF attention parameters
TQF_HOP_ATTENTION_TEMP_DEFAULT = 1.0

# TQF regularization (all disabled/opt-in by default)
TQF_GEOMETRY_REG_WEIGHT_DEFAULT = 0.0    # Disabled; provide value via CLI to enable
# Note: Equivariance/invariance/duality loss weights default to None (disabled).
# Users enable them by providing a weight value directly via CLI.
TQF_VERIFY_DUALITY_INTERVAL_DEFAULT = 10

# TQF orbit mixing temperatures
TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT = 0.3
TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT = 0.5
TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT = 0.7

# Reproducibility
NUM_SEEDS_DEFAULT = 1
SEED_DEFAULT = 42

# Parameter matching
TARGET_PARAMS = 650000
TARGET_PARAMS_TOLERANCE_PERCENT = 1.1
```

---

## Appendix B: Parameter Categories

### Critical Parameters (Highest Impact)

- `--models`: Which architectures to compare
- `--num-epochs`: Training duration
- `--learning-rate`: Optimization speed and stability
- `--tqf-symmetry-level`: Symmetry exploitation level
- `--tqf-z6-equivariance-weight`: Rotation invariance strength

### Regularization Parameters

- `--weight-decay`: L2 penalty
- `--label-smoothing`: Label confidence
- `--patience`: Early stopping sensitivity
- `--tqf-geometry-reg-weight`: Geometric consistency
- `--tqf-self-similarity-weight`: Fractal structure
- `--tqf-inversion-loss-weight`: Duality preservation

### Architecture Parameters

- `--tqf-R`: Lattice size
- `--tqf-hidden-dim`: Feature dimensionality
- `--tqf-fractal-iterations`: Fractal detail
- `--batch-size`: Gradient stability

### Experimental Parameters

- `--num-seeds`: Statistical robustness
- `--num-train`: Dataset size
- `--device`: Compute platform
- `--compile`: Kernel fusion (Linux/Triton only)

---

## Appendix C: Common Parameter Combinations

### Fast Prototyping (< 1 minute)
```bash
python main.py --num-epochs 10 --num-train 1000 --num-val 200 --num-seeds 1
```

### Development Testing (~ 5 minutes)
```bash
python main.py --num-epochs 30 --num-train 5000 --num-val 500 --num-seeds 3
```

### Standard Benchmarking (~ 60 minutes per seed)
```bash
python main.py --num-seeds 5
```

### Publication Quality (~ 5 hours)
```bash
python main.py --num-seeds 10
```

---

## 14. Result Output

### Automatic Persistent Result Logging

All experiment results are automatically saved to disk in the `data/output/` directory. No CLI flag is required — this happens by default on every run.

### CLI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--no-save-results` | flag | `False` | Disable saving results to disk entirely. Console output is unaffected. |
| `--results-dir` | path | `data/output/` | Directory for result output files. Created automatically if it does not exist. Ignored when `--no-save-results` is set. |

**Examples:**

```bash
# Default behavior (results saved to data/output/)
python main.py --models TQF-ANN

# Disable persistent result saving (console output only)
python main.py --models TQF-ANN --no-save-results

# Save results to a custom directory
python main.py --models TQF-ANN --results-dir /tmp/my_results
```

### Output Files

| File | Format | Contents |
|---|---|---|
| `results_YYYYMMDD_HHMMSS.json` | JSON | Per-seed results + final mean/std summary |
| `results_YYYYMMDD_HHMMSS.txt` | Plain text | Human-readable summary table |

**Key features:**

- **Incremental saves**: Results are written to disk after each seed completes, not just at the end. If training is interrupted (crash, Ctrl+C, session timeout), all completed seeds are preserved.
- **Status tracking**: The JSON file has a `"status"` field that is `"in_progress"` during training and `"completed"` when finished.
- **Timestamped filenames**: Each run creates a unique file based on the start time, so multiple experiments never overwrite each other.
- **Output path in config banner**: The file path is displayed in the EXPERIMENT CONFIGURATION section at the start of each run.
- **Default directory constant**: The default output directory (`data/output/`) is defined as `DEFAULT_RESULTS_DIR` in `config.py`.

**JSON structure:**

```json
{
  "status": "completed",
  "started_at": "2026-02-11 09:32:56",
  "completed_at": "2026-02-11 12:45:00",
  "last_updated": "2026-02-11 12:45:00",
  "results": {
    "TQF-ANN": [
      {
        "model_name": "TQF-ANN",
        "seed": 42,
        "best_val_acc": 95.45,
        "test_unrot_acc": 95.92,
        "test_rot_acc": 95.56,
        "params": 651233,
        "flops": 1300000.0,
        "inference_time_ms": 3.73,
        "per_class_acc": {"0": 0.981, "1": 0.992, ...},
        "best_loss_epoch": 136,
        "train_time_total": 6372.5
      }
    ]
  },
  "summary": {
    "TQF-ANN": {
      "val_acc": {"mean": 95.38, "std": 0.29},
      "test_unrot_acc": {"mean": 95.97, "std": 0.04},
      "test_rot_acc": {"mean": 95.40, "std": 0.19}
    }
  }
}
```

**Note:** The `data/output/` directory is gitignored to prevent accidentally committing large result files.

---
**`QED`**

**Last Updated:** February 18, 2026<br>
**Version:** 1.0.3<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
