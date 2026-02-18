# Tri-Quarter Framework Neural Network (TQF-NN) Benchmark Tools: README

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

- [1. Overview](#1-overview)
- [2. Key Features](#2-key-features)
- [3. Installation](#3-installation)
- [4. Quick Start](#4-quick-start)
- [5. CLI Usage](#5-cli-usage)
- [6. Models](#6-models)
- [7. Datasets](#7-datasets)
- [8. Development](#8-development)
- [9. License](#9-license)
- [10. Acknowledgement](#10-acknowledgement)
- [11. References](#11-references)
- [12. Future Adventures](#12-future-adventures)

---

## 1. Overview

This hobby project is an original set of machine learning benchmark tools in the ***Tri-Quarter Toolbox*** which implements Nathan O. Schmidt's ***Tri-Quarter Framework (TQF)*** for ***radial dual triangular lattice graph-based neural networks (TQF-NN)*** with ℤ₆, D₆, and 𝕋₂₄ symmetry group exploitation. Fueled by past and current academic science work (2007-2017 and 2025-present), guided by rigorous creativity, developed via the methods of science, mathematics, and engineering (accelerated with some modern AI assistance), and driven by passion & hard work... It is designed for researchers and practitioners exploring geometric deep learning via TQF symmetry groups (ℤ₆, D₆, 𝕋₂₄), symmetry-aware architectures, and first-principles neural network design, with built-in benchmarks for "apples-to-apples" comparisons on rotated MNIST datasets using PyTorch on CUDA-enabled hardware.

The primary goals are:
- **Design and Implement *TQF Artificial Neural Network (TQF-ANN)* Architecture**: Apply TQF radial dual triangular lattice graph (truncated) to neural networks (NN) via first principles by leveraging the ℤ₆, D₆, and 𝕋₂₄ symmetry groups for encodings and architecture where practical/useful.
- **Conduct Rotated MNIST Benchmarks**: Train and evaluate TQF-ANN on rotated MNIST datasets (60° increments aligned with ℤ₆), measuring rotational invariance and inversion consistency.
- **Setup "Apples-to-Apples" Comparisons**: Benchmark TQF-ANN against 3 non-TQF-ANN models (e.g., parameter-matched FC-MLP, CNN-L5, ResNet-18-Scaled) on equivalent hardware (e.g., Intel i7, PyTorch CUDA 12.6, NVIDIA GeForce RTX 4060 Laptop GPU), ensuring ~650K params tolerance for fair evaluation.
- **Prepare for *TQF Spiking Neural Network (TQF-SNN)* Extension**: Lay groundwork for spiking neural network (SNN) architecture without implementation yet, preparing for future phases.
- **Promote Reproducibility and Best Practices**: Use fixed seeds, stratified sampling, and modular code for deterministic results.

---

## 2. Key Features

- 🧬 **First-Principles TQF-NN Architecture:** Radial dual triangular lattice graph (truncated) with explicit Eisenstein integer coordinates, true 6-neighbor hexagonal adjacency, phase-pair directional encoding, and circle inversion bijective duality between inner & outer zones.
- 🔄 **Symmetry Group Enforcement:** Native ℤ₆ (rotational), D₆ (dihedral), and 𝕋₂₄ (inversive hexagonal dihedral) support with optional geometry regularization + equivariance/inversion orbit consistency losses.
- 📊 **Rotated MNIST Benchmark Suite:** 60°-aligned augmentations for true ℤ₆-equivariant evaluation with robust rotational accuracy.
- ⚖️ **Apples-to-Apples Comparisons:** Parameter-matched (~650k ±1%) training & evaluation of TQF-ANN vs 3 non-TQF ANN baselines (FC-MLP, CNN-L5, ResNet-18-Scaled); groundwork laid for future non-TQF-SNN baselines.
- 🧪 **High Test Coverage & Reproducibility:** Over 85–95% coverage on core lattice, symmetry, and training components; fixed seeds, stratified sampling, deterministic algorithms where possible.
- 💾 **Developer-Friendly CLI & Tooling:** Rich command-line interface, automatic parameter tuning/matching, detailed logging, and VS Code + Windows/Linux/macOS + RTX 4060 Laptop GPU friendly (CUDA 12.6 / PyTorch 2.5+) or equivalent box.
- 📜 **MIT Licensed Experimental Science Project:** After-hours hobby R&D focused on geometric deep learning and symmetry-aware neural architectures.

---

## 3. Installation

See detailed instructions in [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

### Quick Version (CUDA 12.6)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
```

### CPU-Only (All Platforms):
```bash
pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

### Verify:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
pytest --version
```

For macOS or other configurations, see [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

---

## 4. Quick Start

Activate your virtual environment and run from the project root directory.

**Windows (PowerShell/Command Prompt):**
```bash
# Activate venv (if not already)
venv\Scripts\activate
# Default benchmark (all 4 models, 150 epochs max, 58k train samples)
python src\main.py
```

**Linux/macOS:**
```bash
# Activate venv (if not already)
source venv/bin/activate
# Default benchmark (all 4 models, 150 epochs max, 58k train samples)
python src/main.py
```

For a quick smoke test prototype run (10 epochs, 1k samples):
```bash
python src/main.py --num-epochs 10 --num-train 1000
```

Note: Particularly for TQF-ANN training, we expect the first epoch training time to be noticeably longer than the subsequent epoch training times due to runtime optimizations like cuDNN auto-tuning and CUDA initialization overhead.

See more examples in [`QUICKSTART.md`](QUICKSTART.md) and full CLI reference in [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md).

---

## 5. CLI Usage

Centralized argument parsing and validation is in `cli.py`.

### Quick Start
Activate your virtual environment (recommended for reproducibility):
- **Windows**: `venv\Scripts\activate`
- **Linux/macOS**: `source venv/bin/activate`

Run all models with defaults (~650K params each):
```bash
python src/main.py
```

Most common flags:
```bash
--models                      List of models to run (comma-separated; default: all)
--num-epochs                  Training epochs (default: 150)
--num-train                   Training set size (default: 58000)
--num-val                     Validation set size (default: 2000)
--num-seeds                   Number of random seeds (default: 1)
--learning-rate               Initial LR (default: 0.001)
--batch-size                  Batch size (default: 128)
--patience                    Early stopping patience (default: 25)
--tqf-symmetry-level          none | Z6 | D6 | T24 (default: none)
--tqf-use-z6-orbit-mixing     TQF Z6 orbit mixing (competes/conflicts with TQF Z6 augmentation)
--z6-data-augmentation        Enable Z6 data augmentation during training (disabled by default)
--tqf-z6-equivariance-weight  Enable and set weight for Z6 equivariance loss (range [0.001, 0.05])
--device                      cpu | cuda (default: cuda if available)
--compile                     Enable PyTorch model compilation (Linux/macOS only)
```

**Note**: Commands work on both Windows (Command Prompt/PowerShell) and Linux/macOS (terminal/bash). Use `python` (or `python3` on some Linux systems) assuming it's in your PATH.

### Orbit Mixing (TQF-ANN)
Orbit mixing is an evaluation-time ensemble technique in TQF-ANN that adaptively averages logits from symmetry-transformed sector features, improving prediction robustness and rotational invariance without additional training overhead. It operates in feature space using group operations (rotations, reflections, inversions) and temperature-scaled softmax weighting based on prediction confidence. This leverages the TQF lattice's inherent symmetries (ℤ₆, D₆, 𝕋₂₄) to produce more consistent outputs across transformations.

Key CLI flags (focus on ℤ₆ for simplicity and efficiency):
- `--tqf-use-z6-orbit-mixing`: Enables ℤ₆ orbit mixing (6 rotations: 0°, 60°, 120°, 180°, 240°, 300°) during evaluation. Recommended as the primary option for balanced performance gains on rotated MNIST. (Default: false)
- `--tqf-use-d6-orbit-mixing`: Enables D₆ orbit mixing (ℤ₆ rotations + 6 reflections). Use for stronger reflection invariance.
- `--tqf-use-t24-orbit-mixing`: Enables full 𝕋₂₄ orbit mixing (D₆ + circle inversions). Ideal for exploiting inner/outer zone duality, but requires inversion support.
- Temperature controls (adjust weighting softness):
  - `--tqf-orbit-mixing-temp-rotation`: Temperature for rotation averaging (default: 0.3).
  - `--tqf-orbit-mixing-temp-reflection`: Temperature for reflection averaging (default: 0.5).
  - `--tqf-orbit-mixing-temp-inversion`: Temperature for inversion averaging (default: 0.7).

**Important Note:** Orbit mixing conflicts with training-time data augmentation via `--z6-data-augmentation`, as the latter introduces explicit rotations that may interfere with orbit-based ensembles. Keep `--z6-data-augmentation` off (default) when enabling any orbit mixing to avoid suboptimal results.

Example: Enable ℤ₆ orbit mixing for evaluation robustness:
```
python main.py --models TQF-ANN --tqf-use-z6-orbit-mixing
```

### Symmetry Enforcement Losses (TQF-ANN)
Explicit symmetry enforcement through equivariance and/or invariance losses during training.
Features are disabled by default and enabled by providing a weight value:
```bash
# Z6 rotation equivariance loss (disabled by default)
--tqf-z6-equivariance-weight 0.01           Enable and set weight for Z6 loss

# D6 reflection equivariance loss (disabled by default)
--tqf-d6-equivariance-weight 0.01           Enable and set weight for D6 loss

# T24 orbit invariance loss (disabled by default)
--tqf-t24-orbit-invariance-weight 0.005     Enable and set weight for T24 loss
```

**Example: Train with Z6 equivariance loss over 50 epochs max**
```bash
python src/main.py --models TQF-ANN --tqf-z6-equivariance-weight 0.01 --num-epochs 50
```

**Example: Train with full symmetry enforcement over 150 epochs max**
```bash
python src/main.py --models TQF-ANN --tqf-z6-equivariance-weight 0.01 --tqf-d6-equivariance-weight 0.01 --tqf-t24-orbit-invariance-weight 0.005 --num-epochs 150
```

### Fibonacci Weight Scaling Modes (TQF-ANN)
Control hierarchical feature learning with Fibonacci weight scaling. All modes have identical parameter counts - only the feature aggregation weights differ:
```bash
# Fibonacci weight mode options
--tqf-fibonacci-mode none           Uniform weighting (default)
--tqf-fibonacci-mode linear         Linear weights [1,2,3,...] (ablation baseline)
--tqf-fibonacci-mode fibonacci      Fibonacci weights [1,1,2,3,5,...] (opt-in)

# Optional: Golden ratio radial binning
--tqf-use-phi-binning               Use φ-scaled bins (faster inference)
```

**Example: Train with uniform weighting (default)**
```bash
python src/main.py --models TQF-ANN --tqf-fibonacci-mode none --num-epochs 50
```

**Example: Full Fibonacci specification**
```bash
python src/main.py --models TQF-ANN --tqf-fibonacci-mode fibonacci --tqf-symmetry-level D6 --num-epochs 100
```

See full documentation in [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md).

---

## 6. Models

| Model            | Type     | Approx. Params | Key Characteristics                          |
|------------------|----------|----------------|----------------------------------------------|
| TQF-ANN          | Custom   | ~650k          | Radial dual triangular lattice + symmetries  |
| FC-MLP           | Baseline | ~650k          | Fully-connected, parameter-matched           |
| CNN-L5           | Baseline | ~650k          | 5-layer convnet, parameter-matched           |
| ResNet-18-Scaled | Baseline | ~650k          | Scaled-down ResNet-18, parameter-matched     |

All models deliberately matched to ~650,000 trainable parameters (±1.1%) for fair comparison. Models are implemented in cross-platform Python code using PyTorch, ensuring full compatibility with Windows, Linux, and macOS operating systems (CPU/GPU support via CUDA or CPU fallback).

---

## 7. Datasets

Work hard, train hard!

For complete dataset documentation, see [`data/DATASET_README.md`](data/DATASET_README.md).

- **Training / Validation**: Stratified subsets of the standard MNIST training set (default: 58,000 train + 2,000 validation = 60,000 total images). Uses class-balanced sampling for reproducibility and fairness.
- **Test (unrotated)**: Stratified subset of MNIST test set (default: 8,000 images).
- **Test (rotated)**: Rotated versions of test samples at 0°, 60°, 120°, 180°, 240°, 300° (ℤ₆-aligned; default: 2,000 base × 6 = 12,000 images total).

Datasets are automatically downloaded from PyTorch (if not present), organized into class-specific folders as PNG images, and rotated (if needed) during setup. This process is cross-platform compatible (Windows/Linux/macOS) using standard libraries like Pillow (PIL) for image handling and os.path for file paths—no platform-specific code required.

---

## 8. Development

- **IDE (Recommended)**: Visual Studio Code (works on Windows, Linux, and macOS)
- **Python**: 3.8+
- **Virtual Environment**: Recommended for isolation (e.g., `python -m venv venv` on Windows/Linux/macOS; activate with `venv\Scripts\activate` on Windows or `source venv/bin/activate` on Linux/macOS)
- **Dev Dependencies**: Install via `pip install -r requirements-dev.txt` (includes black, mypy, flake8, pytest, and coverage tools)
- **Formatting**: black (run `black .` to format all code)
- **Type checking**: mypy (run `mypy .` for static type analysis)
- **Linting**: flake8 (run `flake8 .` to check code style)
- **Testing**: pytest (run `pytest tests/ -v` for verbose output)

Run full test suite with coverage (cross-platform):

```bash
pytest tests/ -v --cov=. --cov-report=html
```

Run non-slow test suite (excludes integration/slow tests):

```bash
pytest tests/ -v -m "not slow"
```

---

## 9. License

```text
MIT License

Copyright (c) 2026 Nathan O. Schmidt, Cold Hammer Research & Development LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 10. Acknowledgement

See our gratitude in [`ACKNOWLEDGEMENT.md`](ACKNOWLEDGEMENT.md).

---

## 11. References

- **Schmidt, Nathan O.** (2025). *The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal Dihedral Symmetry Group 𝕋₂₄*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1339304](https://www.techrxiv.org/users/906377/articles/1339304).

---

## 12. Future Adventures

See some of our future TODO items in [`FUTURE_TODO.md`](FUTURE_TODO.md).

---

**`QED`**

**Last Updated:** February 18, 2026<br>
**Version:** 1.0.3<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
