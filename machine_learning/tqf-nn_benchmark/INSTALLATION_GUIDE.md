# TQF-NN Benchmark Tools: Installation Guide

**Aiming to minimize hassles and headaches**

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

- [1. Prerequisites](#1-prerequisites)
- [2. Platform-Specific Quick Install](#2-platform-specific-quick-install)
- [3. Verification & First Checks](#3-verification--first-checks)
- [4. Troubleshooting](#4-troubleshooting)
- [5. Next Steps](#5-next-steps)

---

## 1. Prerequisites

Before installing, ensure your system meets these requirements:

- **Operating System:** Windows, Linux, or macOS
- **Python:** 3.8+ from [python.org/downloads](https://www.python.org/downloads/)
- **PyTorch:** 2.5+ with CUDA 12.6 support (for NVIDIA GPUs) or MPS (Apple Silicon)
- **Hardware:** NVIDIA GPU with CUDA support recommended (e.g. RTX 4060 Laptop GPU); CPU fallback is functional but slower
- **Disk Space:** At least 2 GB free (datasets + models + dependencies)
- **Internet:** Required for initial dependency & MNIST download

**GPU Setup Note (Windows/Linux):**
Install latest NVIDIA driver + CUDA Toolkit 12.6 from
[developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

---

## 2. Platform-Specific Quick Install

We assume the repository is already cloned:

```bash
git clone https://github.com/nathanoschmidt/tri-quarter-toolbox.git
cd machine_learning/tqf-nn_benchmark
```

Use a virtual environment (strongly recommended).

### Windows (PowerShell or Command Prompt)

```powershell
# 1. Create + activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Upgrade tooling
python -m pip install --upgrade pip setuptools wheel

# 3. Install PyTorch + CUDA 12.6 (RTX 4060 compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install project dependencies
pip install -r requirements.txt

# Optional — full development & testing tools
pip install -r requirements-dev.txt
```

### Linux

```bash
# 1. Create + activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade tooling
python3 -m pip install --upgrade pip setuptools wheel

# 3. Install PyTorch + CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# (CPU-only alternative if needed)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install project dependencies
pip install -r requirements.txt

# Optional — development tools
pip install -r requirements-dev.txt
```

### macOS

```bash
# 1. Create + activate venv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade tooling
python3 -m pip install --upgrade pip setuptools wheel

# 3. Install PyTorch (MPS acceleration automatic on Apple Silicon)
pip install torch torchvision

# (Intel Mac CPU-only fallback)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install project dependencies
pip install -r requirements.txt

# Optional — development tools
pip install -r requirements-dev.txt
```

---

## 3. Verification & First Checks

Run these commands inside the activated virtual environment.

### 3.1 Basic Environment

```bash
python --version
pip --version
pip list | grep -E "torch|numpy|scipy|pillow|pytest"
```

### 3.2 PyTorch + Accelerator Status

```bash
python -c "import torch; \
    print('PyTorch:        ', torch.__version__); \
    print('CUDA available: ', torch.cuda.is_available()); \
    print('CUDA version:   ', torch.version.cuda); \
    print('Device count:   ', torch.cuda.device_count()); \
    print('Current device: ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \
          'MPS' if torch.backends.mps.is_available() else 'CPU')"
```

**Expected output (e.g., your Windows box):**

```
PyTorch:         2.5.0+cu126  (or newer)
CUDA available:  True
CUDA version:    12.6
Device count:    1
Current device:  NVIDIA GeForce RTX 4060 Laptop GPU
```

### 3.3 Quick Smoke Tests

```bash
# Non-slow tests
pytest tests/ -v -m "not slow"

# Full test suite (includes some computationally expensive runtime performance testing)
pytest tests/ -v

# With coverage report (recommended during development)
pytest tests/ -v --cov=. --cov-report=term-missing
```

If at least the non-slow tests pass, the installation is healthy.

---

## 4. Troubleshooting

### CUDA / GPU Not Detected

- Run `nvidia-smi` → should list RTX 4060 + driver ≥ 535.xx
- Check torch wheel: `pip show torch` should contain `+cu126`
- Reinstall if needed:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Wrong / Missing Environment

Always verify `(venv)` prefix in prompt.
Reactivate if needed:

```powershell
# Windows
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

### NumPy 2.x Conflict

```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### CUDA Out-of-Memory During Training

- Lower batch size: `--batch-size 64` or `32`
- Smaller lattice: `--tqf-R 10` or smaller truncation radius
- Clear cache before run:

```python
import torch
torch.cuda.empty_cache()
```

### Still Bogged Down in the Hassle Swamp?

Collect & share when opening issue:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
nvidia-smi
pip list | grep -E 'torch|numpy|scipy|pillow|pytest'
```

See contact info below.

---

## 5. Next Steps

You're ready to launch! Have fun exploring the TQF-NN!

We recommend to check out [`README.md`](README.md) and [`QUICKSTART.md`](QUICKSTART.md).

If you're still wrestling with hassles, or if you're interesting in the heavier documentation, then feel free to check out the advanced/comprehensive/L33T documentation:

   - Full CLI reference → [`doc/CLI_PARAMETER_GUIDE.md`](doc/CLI_PARAMETER_GUIDE.md)
   - Architecture overview → [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md)
   - API documentation → [`doc/API_REFERENCE.md`](doc/API_REFERENCE.md)
   - Data set documentation → [`data/DATASET_README.md`](data/DATASET_README.md)
   - Test suite guide & structure → [`tests/TESTS_README.md`](tests/TESTS_README.md)

---

**`QED`**

**Last Updated:** February 18, 2026<br>
**Version:** 1.0.3<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
