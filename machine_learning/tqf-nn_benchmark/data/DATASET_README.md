# TQF-NN Benchmark Tools: DATASET README

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.1.0<br>
**Date:** February 26, 2026<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Directory Structure](#2-directory-structure)
- [3. Automatic Dataset Generation](#3-automatic-dataset-generation)
- [4. Dataset Contents](#4-dataset-contents)
- [5. Reproducibility](#5-reproducibility)
- [6. Manual Dataset Preparation](#6-manual-dataset-preparation)
- [7. Storage Requirements](#7-storage-requirements)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Overview

This directory contains the MNIST dataset and its processed variants used for training and evaluating TQF-ANN and baseline models. The data is automatically downloaded, organized, and preprocessed by the `src/prepare_datasets.py` script.

**Key Features:**
- **Automatic Download:** MNIST data is fetched from PyTorch's dataset repository on first run
- **Organized Structure:** Images organized into class-specific folders (PNG format)
- **Rotated Test Sets:** Z₆-aligned rotations (0°, 60°, 120°, 180°, 240°, 300°) for evaluating rotational equivariance
- **Stratified Sampling:** Class-balanced splits for fair training/validation/testing
- **Reproducible:** Fixed random seeds ensure deterministic dataset preparation

**Scientific Rationale:**
The 60-degree rotation increments align with the hexagonal (Z₆) symmetry of the TQF architecture, providing a natural testbed for evaluating rotational equivariance. The class-balanced organization ensures fair representation across all digit classes during training and evaluation.

---

## 2. Directory Structure

After running `src/prepare_datasets.py` or any training script, the following directory structure will be automatically created:

```
data/
├── DATASET_README.md           # This file
└── mnist/                      # MNIST data root (auto-generated, git-ignored)
    ├── MNIST/                  # PyTorch downloaded raw data
    │   ├── raw/                # Original MNIST binary files
    │   │   ├── train-images-idx3-ubyte
    │   │   ├── train-labels-idx1-ubyte
    │   │   ├── t10k-images-idx3-ubyte
    │   │   └── t10k-labels-idx1-ubyte
    │   └── processed/          # PyTorch processed tensors
    │       ├── training.pt
    │       └── test.pt
    │
    └── organized/              # Reorganized class-specific PNG images
        ├── train/              # Training set (60,000 images)
        │   ├── class_0/        # Digit 0 images
        │   ├── class_1/        # Digit 1 images
        │   ├── ...
        │   └── class_9/        # Digit 9 images
        │
        ├── test/               # Test set (10,000 images)
        │   ├── class_0/
        │   ├── class_1/
        │   ├── ...
        │   └── class_9/
        │
        └── rotated_test/       # Rotated test set (60,000 images = 10,000 × 6 angles)
            ├── class_0/        # Digit 0 rotated at all 6 angles
            ├── class_1/        # Digit 1 rotated at all 6 angles
            ├── ...
            └── class_9/        # Digit 9 rotated at all 6 angles
```

**Important:** The entire `data/mnist/` directory is automatically generated and listed in `.gitignore`. Do not manually create these subdirectories or commit dataset files to version control.

---

## 3. Automatic Dataset Generation

### When Does Generation Occur?

Dataset preparation happens automatically when:
1. Running any training script (`python src/main.py`)
2. Calling `get_dataloaders()` from `src/prepare_datasets.py`
3. Manually executing `python src/prepare_datasets.py`

### What Gets Generated?

**Step 1: Download and Organization**
- Downloads MNIST from PyTorch (if not already present)
- Converts all 70,000 images (60K train + 10K test) to PNG format
- Organizes into class-specific folders: `organized/train/class_0/` through `class_9/`, and similarly for test

**Step 2: Rotated Test Set Creation**
- Generates 6 rotated versions of each test image (Z₆ symmetry angles)
- Creates 60,000 total images (10,000 original × 6 rotations)
- Uses BICUBIC interpolation for high-quality rotations
- Saves to `organized/rotated_test/class_0/` through `class_9/`

**Step 3: Stratified Sampling**
- Applies class-balanced sampling during DataLoader creation
- Ensures proportional representation across all 10 digit classes
- Uses fixed random seed (42) for reproducible splits

---

## 4. Dataset Contents

### 4.1 Training Set

**Location:** `mnist/organized/train/`
**Size:** 60,000 images (28×28 grayscale PNG)
**Organization:** 10 class folders, ~6,000 images per class
**Usage:** Split into training and validation subsets via stratified sampling

**Example filename:** `train_00042_label_3.png`
- `train` = training split
- `00042` = image index
- `label_3` = digit class (3)

### 4.2 Unrotated Test Set

**Location:** `mnist/organized/test/`
**Size:** 10,000 images (28×28 grayscale PNG)
**Organization:** 10 class folders, ~1,000 images per class
**Usage:** Baseline evaluation on standard MNIST test set

**Example filename:** `test_00123_label_7.png`
- `test` = test split
- `00123` = image index
- `label_7` = digit class (7)

### 4.3 Rotated Test Set

**Location:** `mnist/organized/rotated_test/`
**Size:** 60,000 images (28×28 grayscale PNG)
**Organization:** 10 class folders, ~6,000 images per class (1,000 per angle)
**Rotation Angles:** 0°, 60°, 120°, 180°, 240°, 300° (Z₆ group)
**Usage:** Evaluating rotational robustness and equivariance

**Example filename:** `test_00123_label_7_rot_120.png`
- `test_00123` = original test image index
- `label_7` = digit class (7)
- `rot_120` = rotation angle (120 degrees)

**Scientific Rationale:** The 60-degree increments correspond to the Z₆ cyclic group, which is a subgroup of the T₂₄ inversive hexagonal dihedral group. This alignment allows direct testing of the TQF architecture's symmetry exploitation capabilities.

---

## 5. Reproducibility

All dataset preparation is fully reproducible across runs and machines:

- **Fixed Random Seed:** Global seed = 42 (set in `prepare_datasets.py`)
- **Deterministic Sampling:** NumPy and PyTorch random generators seeded
- **Stratified Splits:** Class-balanced validation sampling ensures consistent proportions
- **Rotation Consistency:** Deterministic rotation angles (no random jitter in test sets)
- **File Ordering:** Sorted filenames ensure consistent iteration order

**Cross-Platform Compatibility:** Dataset preparation works identically on Windows, Linux, and macOS using standard libraries (os.path, PIL, PyTorch).

---

## 6. Manual Dataset Preparation

While automatic preparation is recommended, you can manually trigger dataset generation:

### Option 1: Run Preparation Script Directly

```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Run preparation script
python src/prepare_datasets.py
```

**Expected output:**
```
Base MNIST organized in:
  <path>/data/mnist/organized/train
  <path>/data/mnist/organized/test

Rotated test set created in <path>/data/mnist/organized/rotated_test
  (10000 original x 6 rotations = 60000 images)

================================================================================
DATALOADERS READY AND FULLY REPRODUCIBLE
================================================================================
  Train samples: 58000
  Validation samples: 2000
  Rotated test samples: 2000
  Unrotated test samples: 8000
================================================================================
```

### Option 2: Import and Call from Python

```python
from prepare_datasets import download_and_organize_mnist, create_rotated_test

# Download and organize base MNIST
download_and_organize_mnist()

# Create rotated test set
create_rotated_test()
```

### Option 3: Let Training Scripts Handle It

Simply run any training command—datasets will be prepared automatically if missing:

```bash
python src/main.py
```

---

## 7. Storage Requirements

| Component           | Size (Approx.) | Description                                      |
|---------------------|----------------|--------------------------------------------------|
| Raw MNIST           | ~50 MB         | Original binary files from PyTorch               |
| Organized Train     | ~60 MB         | 60,000 PNG images in class folders               |
| Organized Test      | ~10 MB         | 10,000 PNG images in class folders               |
| Rotated Test        | ~60 MB         | 60,000 rotated PNG images (6 angles × 10K)       |
| **Total**           | **~180 MB**    | Full dataset including all variants              |

**Note:** Actual sizes may vary slightly depending on PNG compression. Plan for at least 200-300 MB of free disk space to accommodate temporary files during generation.

---

## 8. Troubleshooting

### Dataset Not Found / Missing Directories

**Symptom:** Error messages about missing directories or data files.

**Solution:** Run the preparation script manually:
```bash
python src/prepare_datasets.py
```

All necessary directories (`data/mnist/`, subdirectories) will be created automatically. You do NOT need to manually create any folders.

### Download Failures

**Symptom:** Network errors or timeout during MNIST download.

**Causes:**
- Firewall blocking PyTorch dataset URLs
- Proxy configuration issues
- Intermittent network connectivity

**Solutions:**
1. **Retry:** Simply run the script again—download will resume if partial
2. **Manual Download:** Download MNIST from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) and place files in `data/mnist/MNIST/raw/`
3. **Check Proxy:** Configure proxy settings if behind corporate firewall

### Incorrect Image Counts

**Symptom:** Fewer images than expected in organized directories.

**Solution:** Delete `data/mnist/` entirely and regenerate:
```bash
# Windows
rmdir /s data\mnist

# Linux/macOS
rm -rf data/mnist

# Then regenerate
python src/prepare_datasets.py
```

### Disk Space Issues

**Symptom:** "No space left on device" or similar errors.

**Solution:**
- Ensure at least 300 MB free disk space
- Clear temporary files in system temp directory
- Consider using a different drive with more space (modify `DATA_DIR` in `src/prepare_datasets.py`)

### Permission Errors

**Symptom:** "Permission denied" when writing files.

**Solution:**
- Ensure write permissions for the `data/` directory
- Run script as user with appropriate permissions (avoid sudo unless necessary)
- On Windows, check if files are locked by another process

### Rotated Images Look Wrong

**Symptom:** Rotated test images appear distorted or incorrectly rotated.

**Cause:** Possible corruption during generation or incorrect interpolation.

**Solution:** Regenerate rotated test set:
```bash
# Delete rotated test directory
# Windows: rmdir /s data\mnist\organized\rotated_test
# Linux/macOS: rm -rf data/mnist/organized/rotated_test

# Regenerate (only rotated test, keeps other data)
python src/prepare_datasets.py
```

---

**`QED`**

**Last Updated:** February 26, 2026<br>
**Version:** 1.1.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
