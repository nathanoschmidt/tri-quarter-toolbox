# TQF-NN Benchmark Tools: Quick Start Guide

**Get it fired up ASAP with minimum hassle (well, at least that's the intent)**

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.1.1<br>
**Date:** February 27, 2026<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [1. Installation](#1-installation)
- [2. Run Your First Experiment](#2-run-your-first-experiment)
- [3. Interpret Results](#3-interpret-results)
- [4. Explore the Mighty CLI](#4-explore-the-mighty-cli)
- [5. Common Workflows](#5-common-workflows)
- [6. Testing](#6-testing)
- [7. Next Steps](#7-next-steps)

---

## 1. Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU drivers + CUDA 12.6 toolkit (for RTX 4060 acceleration)
- Git

### Quick Install

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nathanoschmidt/tri-quarter-toolbox.git
   cd machine_learning/tqf-nn_benchmark
   ```

2. **Create & Activate Virtual Environment:**

   ```bash
   # Windows (PowerShell / Command Prompt)
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA 12.6 Support (critical for RTX 4060):**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

   Verify GPU is detected:

   ```python
   python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

   Expected output example:
   ```
   PyTorch: 2.5.0+cu126 | CUDA: True | Device: NVIDIA GeForce RTX 4060 Laptop GPU
   ```

4. **Install Project Dependencies:**

   ```bash
   # Full development + testing stack (strongly recommended)
   pip install -r requirements-dev.txt

   # Minimal runtime only (skip tests/linting)
   # pip install -r requirements.txt
   ```

5. **Quick Smoke Test:**

   ```bash
   pytest --version
   python src/main.py --help
   ```

### CPU-Only Installation (no GPU / no CUDA)

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements-dev.txt
   ```

### Next Steps

- Run full test suite: `pytest -v` (expected: 758 passed, 6 skipped)
- Run quick non-slow test suite: `pytest -v -m "not slow"` (expected: 679 passed, 6 skipped, 79 deselected)
- View CLI options: `python src/main.py --help`
- Run an experiment--see next section!

**Hassle mode?** See more detailed troubleshooting in [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

---

## 2. Run Your First Experiment

**Get blasting immediately with a single command:**

```bash
(venv) $ python src/main.py --tqf-use-z6-orbit-mixing
```

**What happens:**
- Trains the parameter-matched TQF-ANN (∼650K params) and three strong non-TQF baselines
- Uses ℤ₆ orbit mixing at evaluation time for TQF-ANN
- Automatically downloads & prepares the rotated MNIST datasets
- Prints a clean apples-to-apples comparison table with Val/Test/Rotated Test accuracy ± std

**Quick alternative (TQF-ANN model training only):**

```bash
(venv) $ python src/main.py --models TQF-ANN --tqf-use-z6-orbit-mixing
```

**Adjust training duration and data size (very common for quick debugging or scaling experiments):**

```bash
# Short run: 10 epochs, only 10,000 training samples
(venv) $ python src/main.py --models TQF-ANN --num-seeds 1 --num-epochs 10 --num-train 10000

# Medium run: 40 epochs on full training set (~50k samples)
(venv) $ python src/main.py --models TQF-ANN FC-MLP --num-seeds 2 --num-epochs 40 --num-train 50000
```

**That's it—now you're cookin' with gas and benchmarkin' the TQF-ANN!**
(Full CLI options—including patience, learning rate, symmetry losses, and many more—are documented in [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md)).

---

## 3. Interpret Results

After an experiment finishes, results are available in **two places**:

1. **Console output** — a summary table printed to stdout (see below)
2. **Persistent files** — automatically saved to `data/output/results_YYYYMMDD_HHMMSS.json` (and a companion `.txt` summary)

Results are saved **incrementally after each seed completes**, so even if training is interrupted (crash, session timeout, Ctrl+C), all completed seeds are preserved on disk. The output path is shown in the experiment configuration banner at startup.

### Console Output

You'll see a clean summary table printed to the console as something like (example format):

```
FINAL MODEL COMPARISON (Mean +/- Std)
========================================================================================================================
Model                 Val Acc (%)     Test Acc (%)    Rot Acc (%)      Params (k)        FLOPs (M)      Inf Time (ms)
------------------------------------------------------------------------------------------------------------------------
FC-MLP                98.50+/-0.00    98.47+/-0.00    38.28+/-0.00     648.4+/- 0.0       1.3+/- 0.0     0.42+/-0.00
CNN-L5                99.25+/-0.00    99.39+/-0.00    43.43+/-0.00     654.0+/- 0.0      82.3+/- 0.0     1.92+/-0.00
ResNet-18-Scaled      99.50+/-0.00    99.40+/-0.00    43.33+/-0.00     654.6+/- 0.0     255.3+/- 0.0     1.88+/-0.00
TQF-ANN               97.05+/-0.00    97.75+/-0.00    66.57+/-0.00     651.2+/- 0.0       1.3+/- 0.0     5.00+/-0.00
========================================================================================================================
```

**Key columns explained:**
- **Val Acc / Test Acc**: Standard MNIST accuracy (unrotated)
- **Rot Acc**: Rotation-invariant test accuracy
- **Params (k)**: Trainable parameters (should stay ~650k across models for fair comparison)
- **FLOPs (M)**: Approximate forward-pass compute
- **Inf Time (ms)**: Average inference time per sample on your hardware

**Quick interpretation tips:**
- **TQF-ANN strength** — Look primarily at **Rot Acc** column. A main feature of TQF is rotation invariance thanks to explicit ℤ₆/D₆/𝕋₂₄ geometric structure.
- **Greatness** — If Rot Acc is much higher than baselines while Val/Test Acc is competitive, that's success.
- **Statistical reliability** — With `--num-seeds ≥ 3`, the ± values give you a sense of variance. Bigger spreads → need more seeds or longer training.
- **Next steps** — The most impactful option is `--tqf-use-z6-orbit-mixing` for rotation robustness. Also try different data sizes (`--num-train`, `--num-epochs`) to see what drives accuracy.

### Persistent Result Files

All results are automatically saved to `data/output/` in two formats:

- **JSON** (`results_YYYYMMDD_HHMMSS.json`) — machine-readable, includes per-seed results and final summary with mean/std
- **TXT** (`results_YYYYMMDD_HHMMSS.txt`) — human-readable summary table (same format as the console output)

The JSON file is updated **incrementally after each seed** with status `"in_progress"`, then marked `"completed"` with the full summary when the experiment finishes. This means partial results survive crashes.

**Controlling result output:**
- `--no-save-results` — disable persistent result saving entirely (console output still works)
- `--results-dir /path/to/dir` — save results to a custom directory instead of the default `data/output/`

Happy experimenting and benchmarking! Enjoy the symmetries! 🚀

---

### 4. Explore the Mighty CLI

The command-line interface (CLI) in `src/main.py` gives you flexible control over experiments, symmetry levels, training settings, and debugging features. Run this for the complete help:

```bash
python src/main.py --help
```

Here are some examples of commands:

```bash
# Default run: trains & evaluates all four models (~650k params each) with default settings
python src/main.py
```

```bash
# Quick smoke test run on a small dataset
python src/main.py --num-train 4000 --num-epochs 12 --patience 8
```

```bash
# Default baseline run: trains & evaluates the three baseline models (~650k params each)
python src/main.py --models FC-MLP CNN-L5 ResNet-18-Scaled
```

```bash
# TQF-ANN + Z6 orbit mixing (very common combo)
python src/main.py --models TQF-ANN --tqf-use-z6-orbit-mixing
```

```bash
# Change truncation radius R (controls lattice size; sparsifying lattice because hidden_dim auto-tunes to keep ~650k params)
python src/main.py --models TQF-ANN --tqf-R 30
```

```bash
# Multi-seed for reliable statistics (recommended for serious comparisons)
python src/main.py --models TQF-ANN ResNet-18-Scaled --num-seeds 5 --num-epochs 90 --patience 22
```

Some common flags to remember:

- `--models`              space- or comma-separated list (e.g. `TQF-ANN ResNet-18-Scaled`)
- `--num-seeds`                      number of independent runs
- `--num-train`                      training set size
- `--num-epochs`                     max epochs
- `--tqf-use-z6-orbit-mixing`        ℤ₆ evaluation-time orbit mixing
- `--tqf-use-d6-orbit-mixing`        D₆ evaluation-time orbit mixing
- `--tqf-use-t24-orbit-mixing`       full 𝕋₂₄ evaluation-time orbit mixing
- `--tqf-t24-orbit-invariance-weight`  𝕋₂₄ orbit invariance loss (opt-in)
- `--z6-data-augmentation`            enable ℤ₆ rotation data augmentation (disabled by default)
- `--compile`                        (Linux only)

ℤ₆ orbit mixing quality enhancements (all eval-time, opt-in, require `--tqf-use-z6-orbit-mixing`):

- `--tqf-z6-orbit-mixing-confidence-mode`        `max_logit` (default) or `margin`
- `--tqf-z6-orbit-mixing-aggregation-mode`       `logits` (default), `probs`, or `log_probs`
- `--tqf-z6-orbit-mixing-top-k`                 keep only K most confident variants (default: all 6)
- `--tqf-z6-orbit-mixing-adaptive-temp`          per-sample entropy-based temperature scaling
- `--tqf-z6-orbit-mixing-rotation-mode`          `bilinear` (default) or `bicubic`
- `--tqf-z6-orbit-mixing-rotation-padding-mode`  `zeros` (default) or `border`
- `--tqf-z6-orbit-mixing-rotation-pad`           reflect-pad before rotate (0 = off, try 2–4)

Training-time augmentation/loss additions (opt-in):

- `--non-rotation-data-augmentation`           random crop + brightness/contrast jitter
- `--tqf-z6-orbit-consistency-weight`            ℤ₆ orbit consistency self-distillation loss weight
- `--tqf-z6-orbit-consistency-rotations`         extra rotation passes for consistency loss (default: 2)

For the full list of command parameters (including weights, TQF lattice radius, symmetry levels, etc.), see [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md).

---

## 5. Common Workflows

### Compare TQF Against Best Baseline

```bash
python src/main.py --models TQF-ANN ResNet-18-Scaled
```

**What this does:**
- Trains only TQF-ANN and ResNet-18-Scaled (~650K params each)
- Produces clean head-to-head comparison on rotated MNIST
- Produces rotated MNIST evaluation for fair TQF comparison

### Orbit Mixing Ablation Study

```bash
# No orbit mixing (baseline)
python src/main.py --models TQF-ANN --num-seeds 2

# Z6 orbit mixing (6 input-space rotations)
python src/main.py --models TQF-ANN --tqf-use-z6-orbit-mixing --num-seeds 2

# D6 orbit mixing (rotations + reflections)
python src/main.py --models TQF-ANN --tqf-use-d6-orbit-mixing --num-seeds 2

# T24 orbit mixing (full symmetry)
python src/main.py --models TQF-ANN --tqf-use-t24-orbit-mixing --num-seeds 2
```

**What this does:**
- Compares how much each level of evaluation-time orbit mixing improves rotated MNIST accuracy
- Each higher level includes all lower levels
- Z6: 6 full forward passes (input-space rotations)
- D6: adds feature-space reflection (lightweight, classification head only)
- T24: adds zone-swap (trivially cheap, argument reordering)

### Orbit Invariance Loss Training

> **Note:** This training-time loss feature is experimental. In practice it has not shown accuracy improvements. For rotation robustness, prefer `--tqf-use-z6-orbit-mixing` instead.

Feature is disabled by default and enabled by providing a weight value:

```bash
# Train with 𝕋₂₄ orbit invariance loss
python src/main.py --models TQF-ANN --tqf-t24-orbit-invariance-weight 0.005
```

**What this does:**
- Adds an auxiliary loss term that penalizes violations of T24 orbit invariance during training
- 𝕋₂₄ loss:  Orbit consistency across the full 24-element group

### Rotation Robustness: Data Augmentation vs. Architecture

TQF-ANN achieves rotation robustness from two sources: **Z6 data augmentation** (training-time rotation of images at 60-degree intervals) and **architectural symmetry** (hexagonal lattice geometry). You can isolate and compare these contributions:

```bash
# Standard training: no data augmentation (default), no orbit mixing
python src/main.py --models TQF-ANN

# With Z6 data augmentation for rotation robustness
python src/main.py --models TQF-ANN --z6-data-augmentation

# Architectural robustness via orbit mixing (no augmentation)
python src/main.py --models TQF-ANN --tqf-use-z6-orbit-mixing

# Apples-to-apples comparison: TQF-ANN vs CNN with orbit mixing
# Shows TQF-ANN's geometric advantage over non-symmetric architectures
python src/main.py --models TQF-ANN CNN-L5 --tqf-use-z6-orbit-mixing

# Full T24 orbit mixing (Z6 rotations + D6 reflections + zone-swap)
python src/main.py --models TQF-ANN --tqf-use-t24-orbit-mixing
```

**What this does:**
- `--z6-data-augmentation` enables training-time rotation augmentation for rotation robustness (disabled by default to avoid conflicts with orbit mixing)
- `--tqf-use-z6-orbit-mixing` averages predictions over 6 input-space rotations (0, 60, ..., 300 degrees) at evaluation time, leveraging TQF-ANN's hexagonal symmetry

**Temperature tuning** (optional, for advanced users):
```bash
# Custom confidence weighting temperatures
python src/main.py --models TQF-ANN --tqf-use-t24-orbit-mixing \
  --tqf-z6-orbit-mixing-temp-rotation 0.5 \
  --tqf-d6-orbit-mixing-temp-reflection 0.5 \
  --tqf-t24-orbit-mixing-temp-inversion 0.7
```
Lower temperatures produce sharper weighting (most confident variant dominates); higher temperatures approach uniform averaging.

### Manual Hyperparameter Sweep (Simple Loop Example)

In Windows PowerShell and/or Linux/macOS bash you can implement a "sweep"/loop to grab results for different values of a single variable:

```bash
# Learning rate sweep (Windows PowerShell & Linux/macOS bash)
for lr in 0.0001 0.0003 0.0005 0.001; do
  python src/main.py --models TQF-ANN --learning-rate $lr --num-seeds 2 --num-epochs 60
done
```

Or alternatively in Python (cross-platform, more flexible) you can implement it as well:

```python
import subprocess
from pathlib import Path

MAIN_SCRIPT: Path = Path("src/main.py")

learning_rates: list[float] = [0.0001, 0.0003, 0.0005, 0.001]

for lr in learning_rates:
    print(f"\n=== Running learning rate = {lr} ===")
    cmd = [
        "python", str(MAIN_SCRIPT),
        "--models", "TQF-ANN",
        "--learning-rate", str(lr),
        "--num-seeds", "2",
        "--num-epochs", "60",
        "--patience", "15"
    ]
    subprocess.run(cmd, check=True)
```

**What this does:**
- Systematically tests one hyperparameter while keeping others fixed
- Easily adapt for geometry reg weight, inversion loss weight, truncation radius R, etc.
- Results appear sequentially in console/logs — compare manually or collect in a spreadsheet

These workflows cover most day-to-day usage: direct comparison, symmetry group ablation, loss ablation, hierarchical weighting experiments, and simple hyperparameter tuning.

---

## 6. Testing

Run the automated test suite to verify everything works (requires `requirements-dev.txt`):

```bash
pytest tests/ -v
```

For parallel execution (faster, recommended on multi-core systems):

```bash
pytest tests/ -v -n auto
```

For coverage report (opens in browser):

```bash
pytest tests/ -v --cov=src/ --cov-report=html
```

**Cross-Platform Notes:**
- **Windows:** Use Command Prompt or PowerShell; ensure Python is in PATH.
- **Linux/macOS:** Use terminal; no additional steps needed.

All tests should pass with no failures or errors. Hassles? See the troubleshooting section of [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

---

## 7. Next Steps

You're ready—get your gameface on and get to training!

1. **Run your first experiment (recommended starting point)**
   Quick single-seed comparison of TQF-ANN vs three baselines on rotated MNIST:
   ```bash
   python src/main.py --num-seeds 1 --num-epochs 30
   ```

2. **See all available options and flags**
   ```bash
   python src/main.py --help
   ```
   Try useful ones to experiment with:
   - `--tqf-use-z6-orbit-mixing` (recommended — ℤ₆ orbit mixing for rotation robustness)
   - `--tqf-use-z6-orbit-mixing --tqf-z6-orbit-mixing-rotation-pad 2` (orbit mixing with corner artefact reduction)
   - `--batch-size 128 --learning-rate 0.0005`
   - `--device cpu` (fallback when testing without GPU)

3. **Run the full test suite** (strongly recommended before long runs)
   ```bash
   pytest tests/ -v -n auto
   ```
   Or with coverage (opens report in browser):
   ```bash
   pytest tests/ -v --cov=src/ --cov-report=html
   ```

4. **Monitor GPU usage during training**
   Open a second terminal/PowerShell and run:
   ```bash
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
   ```

5. **Explore advanced/comprehensive/L33T documentation**
   - Full CLI reference → [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md)
   - Architecture overview → [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md)
   - API documentation → [`doc/API_REFERENCE.md`](doc/API_REFERENCE.md)
   - Data set documentation → [`data/DATASET_README.md`](data/DATASET_README.md)
   - Test suite guide & structure → [`tests/TESTS_README.md`](tests/TESTS_README.md)

6. **Share results or get help**
   - See all options with descriptions:
      ```bash
      python src/main.py --help
      ```
   - Interesting accuracy / symmetry behavior? Bug?
     Open a GitHub issue or send me an email (nate.o.schmidt@coldhammer.net)
   - Follow updates / share thoughts: @RealColdHammer on X

---

**`QED`**

**Last Updated:** February 27, 2026<br>
**Version:** 1.1.1<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
