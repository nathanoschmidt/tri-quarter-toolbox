# TQF-NN Benchmark Tools: Quick Start Guide

**Get it fired up ASAP with minimum hassle (well, at least that's the intent)**

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.0.0<br>
**Date:** February 7, 2026<br>

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

- Run full test suite: `pytest -v`
- Run quick non-slow test suite: `pytest -v -m "not slow"`
- View CLI options: `python src/main.py --help`
- Run an experiment--see next section!

**Hassle mode?** See more detailed troubleshooting in [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

---

## 2. Run Your First Experiment

**Get blasting immediately with a single command:**

```bash
(venv) $ python src/main.py --tqf-symmetry-level D6
```

**What happens:**
- Trains the parameter-matched TQF-ANN (∼650K params) and three strong non-TQF baselines
- Uses D₆ symmetry
- Automatically downloads & prepares the rotated MNIST datasets
- Prints a clean apples-to-apples comparison table with Val/Test/Rotated Test accuracy ± std

**Quick alternative (single-seed, TQF-ANN model training only):**

```bash
(venv) $ python src/main.py --models TQF-ANN --num-seeds 1
```

**Adjust training duration and data size (very common for quick debugging or scaling experiments):**

```bash
# Short run: 10 epochs, only 10,000 training samples
(venv) $ python src/main.py --models TQF-ANN --num-seeds 1 --num-epochs 10 --num-train 10000

# Medium run: 40 epochs on full training set (~50k samples)
(venv) $ python src/main.py --models TQF-ANN MLP --num-seeds 2 --num-epochs 40 --num-train 50000
```

**That's it—now you're cookin' with gas and benchmarkin' the TQF-ANN!**
(Full CLI options—including patience, learning rate, symmetry losses, and many more—are documented in [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md)).

---

## 3. Interpret Results

After an experiment finishes, you'll see a clean summary table printed to the console as something like (example format):

```
============================================================================================
Model                Val Acc    Test Acc   Rot Acc    Params (k)   FLOPs (M)   Inf Time (ms)
--------------------------------------------------------------------------------------------
TQF-ANN             95.4±0.3   95.7±0.2   95.9±0.4     648.2±1.1     12.4±0.3      4.8±0.2
MLP                 97.1±0.4   97.5±0.3   89.2±1.1     649.8±0.9     11.9±0.2      3.2±0.1
CNN                 98.2±0.2   98.6±0.2   94.5±0.7     651.3±1.4     18.7±0.5      5.1±0.3
ResNet              98.5±0.3   98.8±0.2   95.8±0.6     647.9±1.0     22.1±0.4      6.9±0.4
============================================================================================
```

**Key columns explained:**
- **Val Acc / Test Acc**: Standard MNIST accuracy (unrotated)
- **Rot Acc**: Rotation-invariant test accuracy
- **Params (k)**: Trainable parameters (should stay ~650k across models for fair comparison)
- **FLOPs (M)**: Approximate forward-pass compute
- **Inf Time (ms)**: Average inference time per sample on your hardware

**Quick interpretation tips:**
- **TQF-ANN strength** — Look primarily at **Rot Acc** column. A main feature of TQF is rotation invariance thanks to explicit Z₆/D₆/𝕋₂₄ geometric structure and orbit mixing.
- **Greatness** — If Rot Acc is much higher than baselines while Val/Test Acc is competitive, that's success.
- **Statistical reliability** — With `--num-seeds ≥ 3`, the ± values give you a sense of variance. Bigger spreads → need more seeds or longer training.
- **Next steps** — Try ablations with different symmetry levels (`--tqf-symmetry-level none/Z6/D6/T24`), loss weights, or data sizes (`--num-train`, `--num-epochs`) to see what drives the rotation gains.

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
python src/main.py --models FC-MLP CNN-L5 ResNet18Scaled
```

```bash
# TQF-ANN + D6 symmetry + D6 equivariance enforcement (very common combo)
python src/main.py --models TQF-ANN --tqf-symmetry-level D6 --tqf-d6-equivariance-weight 0.01
```

```bash
# Change truncation radius R (controls lattice size; hidden_dim auto-tunes to keep ~650k params)
python src/main.py --models TQF-ANN --tqf-R 15 --tqf-symmetry-level D6
```

```bash
# Multi-seed for reliable statistics (recommended for serious comparisons)
python src/main.py --models TQF-ANN ResNet18Scaled --num-seeds 5 --num-epochs 90 --patience 22
```

Most frequently used flags to remember:

- `--models`              space- or comma-separated list (e.g. `TQF-ANN ResNet18Scaled`)
- `--tqf-symmetry-level`             `none` / `Z6` / `D6` / `T24`
- `--num-seeds`                      number of independent runs
- `--num-train`                      training set size
- `--num-epochs`                     max epochs
- `--tqf-z6-equivariance-weight`     Z6 rotation equivariance loss (opt-in)
- `--tqf-d6-equivariance-weight`     D6 reflection equivariance loss (opt-in)
- `--tqf-t24-orbit-invariance-weight`  T24 orbit invariance loss (opt-in)
- `--verify-geometry`
- `--compile`                        (Linux only)

For the full list of command parameters (including weights, TQF lattice radius, Fibonacci modes, etc.), see [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md).

---

## 5. Common Workflows

### Compare TQF Against Best Baseline

```bash
python src/main.py --models TQF-ANN ResNet-18-Scaled
```

**What this does:**
- Trains only TQF-ANN and ResNet-18-Scaled (~650K params each)
- Produces clean head-to-head comparison on rotated MNIST
- Uses Z₆ orbit mixing for fair TQF evaluation

### Symmetry Level Ablation Study

```bash
# Test impact of different symmetry groups
for sym in none Z6 D6 T24; do
  python src/main.py --models TQF-ANN --tqf-symmetry-level $sym --num-seeds 2
done
```

**What this does:**
- none:      No explicit symmetry enforcement (baseline)
- Z₆:       Cyclic group — 60° rotations only
- D₆:       Dihedral group — rotations + reflections
- 𝕋₂₄:      Full inversive hexagonal dihedral group (D₆ ⋊ ℤ₂) — rotations + reflections + circle inversion
- Lets you directly compare how much each layer of symmetry improves rotated MNIST accuracy

### Equivariance/Orbit Loss Training

Features are disabled by default and enabled by providing a weight value:

```bash
# Train with explicit Z₆ rotation equivariance enforcement
python src/main.py --models TQF-ANN --tqf-z6-equivariance-weight 0.01

# Train with D₆ reflection equivariance enforcement
python src/main.py --models TQF-ANN --tqf-d6-equivariance-weight 0.01

# Train with full 𝕋₂₄ orbit invariance
python src/main.py --models TQF-ANN --tqf-t24-orbit-invariance-weight 0.005

# Combine all symmetry losses for maximum enforcement
python src/main.py --models TQF-ANN \
  --tqf-z6-equivariance-weight 0.01 \
  --tqf-d6-equivariance-weight 0.01 \
  --tqf-t24-orbit-invariance-weight 0.005
```

**What this does:**
- Adds auxiliary loss terms that directly penalize symmetry violations during training
- Z₆ loss:   Feature consistency under 60° rotations
- D₆ loss:   Feature consistency under reflections
- 𝕋₂₄ loss:  Orbit consistency across the full 24-element group
- Helps the model internalize the radial dual triangular lattice geometry

### Fibonacci Mode Comparison

```bash
# Compare all three Fibonacci modes (weight-based scaling)
for mode in none linear fibonacci; do
  python src/main.py --models TQF-ANN --tqf-fibonacci-mode $mode --num-seeds 3
done
```

**What this does:**
- none:      Uniform weights — baseline behavior
- linear:    Linear weighting (1,2,3,…) across radial zones
- fibonacci: Full Fibonacci sequence weighting (1,1,2,3,5,…) — strongest hierarchical bias
- All modes keep ~same parameter count (weight scaling, not dimension scaling)

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
- Easily adapt for geometry reg weight, inversion loss weight, truncation radius R, self-similarity weight, hop attention temperature, etc.
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
   - `--tqf-symmetry-level T24` (full inversive hexagonal symmetry)
   - `--tqf-inversion-loss-weight 0.001 --tqf-t24-orbit-invariance-weight 0.005`
   - `--tqf-fibonacci-mode fibonacci`
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
   - Symmetry enforcement details → [`doc/SYMMETRY_ENFORCEMENT.md`](doc/SYMMETRY_ENFORCEMENT.md)
   - Test suite guide & structure → [`doc/TESTS_README.md`](doc/TESTS_README.md)
   - API documentation → [`doc/API_REFERENCE.md`](doc/API_REFERENCE.md)
   - Scientific background → TechRxiv paper (link in [`README.md`](README.md))

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

**Last Updated:** February 7, 2026<br>
**Version:** 1.0.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
