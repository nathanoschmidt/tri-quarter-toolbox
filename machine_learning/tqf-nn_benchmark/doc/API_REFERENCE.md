# TQF-NN Benchmark Tools: API Reference

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.1.0<br>
**Last Updated:** February 26, 2026<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dependencies & Requirements](#2-dependencies--requirements)
3. [Command-Line Interface (`cli.py`)](#3-command-line-interface-clipy)
4. [Model Architectures](#4-model-architectures)
5. [Training Engine (`engine.py`)](#5-training-engine-enginepy)
6. [Evaluation Utilities (`evaluation.py`)](#6-evaluation-utilities-evaluationpy)
7. [Dataset Preparation (`prepare_datasets.py`)](#7-dataset-preparation-prepare_datasetspy)
8. [Configuration (`config.py`)](#8-configuration-configpy)
9. [Dual Metrics & Geometry (`dual_metrics.py`)](#9-dual-metrics--geometry-dual_metricspy)
10. [Parameter Matching (`param_matcher.py`)](#10-parameter-matching-param_matcherpy)
11. [Logging & Output Formatting](#11-logging--output-formatting)
12. [Type Definitions & Conventions](#12-type-definitions--conventions)
13. [Complete Usage Examples](#13-complete-usage-examples)
14. [Testing & Validation](#14-testing--validation)

---

## 1. Introduction

This API reference provides comprehensive documentation for all public functions, classes, and modules in the TQF-NN benchmark suite. The codebase implements Nathan O. Schmidt's Tri-Quarter Framework (TQF), applying radial dual triangular lattice graph structures to neural network architectures with ℤ₆, D₆, and T₂₄ symmetry groups.

### Design Philosophy

- **Type Safety**: All new variable declarations use Python type hints
- **ASCII Compatibility**: All code is LaTeX-compatible (ASCII-only)
- **Modularity**: Clear separation between models, training, evaluation, and utilities
- **Reproducibility**: Deterministic seeding and comprehensive logging
- **Fair Comparison**: Automatic parameter matching across architectures

### Module Overview

| Module | Purpose | Lines | Key Components |
|--------|---------|-------|----------------|
| `cli.py` | Command-line interface | 850 | Argument parsing, validation |
| `config.py` | Configuration constants | 1,232 | Hyperparameters, defaults |
| `models_tqf.py` | TQF-ANN architecture | 700 | Layer aggregation, symmetry ops |
| `models_baseline.py` | Baseline models | 525 | MLP, CNN, ResNet |
| `dual_metrics.py` | Geometric operations | 970 | Dual metrics |
| `engine.py` | Training orchestration | 996 | Multi-seed experiments |
| `evaluation.py` | Metrics computation | 879 | 18 evaluation functions |
| `param_matcher.py` | Parameter tuning | 712 | Fair model comparison |
| `prepare_datasets.py` | Data loading | 429 | MNIST rotation handling |
| `logging_utils.py` | Progress tracking | 414 | Console output formatting |
| `output_formatters.py` | Results presentation | 897 | Tables, summaries |
| `main.py` | Experiment entry | 203 | CLI integration |

---

## 2. Dependencies & Requirements

### Runtime Dependencies

**Required for all users** (`requirements.txt`):

```text
torch>=2.5.0
torchvision>=0.20.0
numpy>=1.24.0,<2.0.0
scipy>=1.10.0
Pillow>=10.0.0
ptflops>=0.7.0
```

**Tested Configuration:**
- CUDA 12.6
- NVIDIA GeForce RTX 4060 Laptop GPU
- Intel i7 processor
- Visual Studio Code

### Development Dependencies

**For testing and development** (`requirements-dev.txt`):

```text
# All runtime dependencies plus:
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-xdist>=3.3.0

# Optional (commented out by default):
# black>=23.0.0         # Code formatting
# mypy>=1.5.0           # Type checking
# matplotlib>=3.7.0     # Visualization
# tensorboard>=2.14.0   # Training visualization
```

### Installation

```bash
# Minimal installation (runtime only)
pip install -r requirements.txt

# Full development environment
pip install -r requirements-dev.txt
```

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed setup instructions.

---

## 3. Command-Line Interface (`cli.py`)

The CLI module provides argument parsing, validation, and logging setup for experiments.

### Functions

#### `get_all_model_names() -> List[str]`

**Description:** Returns list of all available model names from the model registry.

**Returns:**
- `List[str]`: Model names, e.g., `['TQF-ANN', 'FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']`

**Example:**
```python
from cli import get_all_model_names

models: List[str] = get_all_model_names()
print(f"Available models: {models}")
# Output: Available models: ['TQF-ANN', 'FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']
```

---

#### `parse_args() -> argparse.Namespace`

**Description:** Parses and validates command-line arguments with comprehensive help text and range checking.

**Returns:**
- `argparse.Namespace`: Validated arguments object

**Key Arguments (Organized by Group):**

**General Settings:**
- `--models`: Models to train (space-separated)
  - Type: `List[str]`
  - Default: All models
  - Choices: `['TQF-ANN', 'FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']` or `'all'`
  - Example: `--models TQF-ANN FC-MLP`

- `--device`: Compute device
  - Type: `str`
  - Default: `'cuda'` if available, else `'cpu'`
  - Choices: `['cuda', 'cpu']`

**Reproducibility:**
- `--num-seeds`: Number of random seeds for statistical robustness
  - Type: `int`
  - Default: 1
  - Range: [1, 20]
  - Why: Multiple seeds provide confidence intervals and statistical significance

- `--seed-start`: Starting random seed
  - Type: `int`
  - Default: 42
  - Range: >= 0
  - Why: Seeds run consecutively from this value

**Training Hyperparameters:**
- `--num-epochs`: Maximum training epochs per seed
  - Type: `int`
  - Default: 150
  - Range: [1, 200]
  - Why: Best accuracy occurs at epochs 125-135 with 58K data

- `--batch-size`: Mini-batch size
  - Type: `int`
  - Default: 128
  - Range: [1, 1024]
  - Why: Powers of 2 recommended for GPU efficiency

- `--learning-rate`: Initial learning rate
  - Type: `float`
  - Default: 0.001
  - Range: (0.0, 1.0]
  - Why: Cosine annealing with warmup applied

- `--learning-rate-warmup-epochs`: LR warmup epochs
  - Type: `int`
  - Default: 5
  - Range: [0, 10]

- `--weight-decay`: L2 regularization
  - Type: `float`
  - Default: 0.0001
  - Range: [0.0, 1.0]

- `--label-smoothing`: Label smoothing factor
  - Type: `float`
  - Default: 0.1
  - Range: [0.0, 1.0]

**Early Stopping:**
- `--patience`: Epochs without improvement before stopping
  - Type: `int`
  - Default: 25
  - Range: [1, 50]

- `--min-delta`: Minimum improvement threshold
  - Type: `float`
  - Default: 0.0005
  - Range: [0.0, 1.0]

**Dataset Configuration:**
- `--num-train`: Training samples (total)
  - Type: `int`
  - Default: 58000 (max usable with default 2K validation split)
  - Range: [100, 60000]
  - Note: Must be divisible by 10

- `--num-val`: Validation samples
  - Type: `int`
  - Default: 2000
  - Range: [10, 10000]

- `--num-test-rot`: Rotated test samples
  - Type: `int`
  - Default: 2000
  - Range: [100, 10000]

- `--num-test-unrot`: Unrotated test samples
  - Type: `int`
  - Default: 8000
  - Range: [100, 10000]

**TQF Architecture (TQF-ANN only):**
- `--tqf-R`: Lattice truncation radius
  - Type: `int`
  - Default: 20 (from `TQF_TRUNCATION_R_DEFAULT`)
  - Range: [2, 100]
  - Why: Number of hexagonal nodes proportional to R^2

- `--tqf-hidden-dim`: TQF hidden dimension
  - Type: `int` or `None`
  - Default: `None` (auto-tuned to match ~650K params)
  - Range: [8, 512] if manually specified

- `--tqf-symmetry-level`: Symmetry group for orbit pooling
  - Type: `str`
  - Default: `'none'` (from `TQF_SYMMETRY_LEVEL_DEFAULT`, orbit pooling disabled)
  - Choices: `['none', 'Z6', 'D6', 'T24']`

**TQF Equivariance Losses (TQF-ANN only):**
Features are disabled by default and enabled by providing a weight value.
- `--tqf-z6-equivariance-weight`: Enable and set weight for Z6 rotation equivariance loss (disabled by default)
- `--tqf-d6-equivariance-weight`: Enable and set weight for D6 reflection equivariance loss (disabled by default)

**TQF Invariance Losses (TQF-ANN only):**
- `--tqf-t24-orbit-invariance-weight`: Enable and set weight for T24 orbit invariance loss (disabled by default)

**TQF Duality Losses (TQF-ANN only):**
- `--tqf-inversion-loss-weight`: Enable and set weight for circle inversion duality loss (disabled by default)
- `--tqf-verify-duality-interval`: Epochs between duality checks (default: 10)

**TQF Geometry Regularization (TQF-ANN only, opt-in):**
- `--tqf-verify-geometry`: Enable geometry verification
- `--tqf-geometry-reg-weight`: Weight (default: 0.0, disabled)

**Example:**
```python
from cli import parse_args

args = parse_args()
print(f"Training {args.models} for {args.num_epochs} epochs")
print(f"TQF symmetry: {args.tqf_symmetry_level}")
```

**Command-Line Usage:**
```bash
# Train all models with default settings
python src/main.py

# Train all models with 10 seeds
python src/main.py --num-seeds 10

# Train only TQF-ANN with T24 symmetry
python src/main.py --models TQF-ANN --tqf-symmetry-level T24

# Custom dataset sizes
python src/main.py --num-train 10000 --num-val 1000 --num-test-rot 2000

# Quick test run
python src/main.py --num-epochs 10 --num-train 1000 --num-val 200
```

---

#### `_validate_args(args: argparse.Namespace) -> None`

**Description:** Internal function that validates argument ranges and consistency. Called automatically by `parse_args()`.

**Args:**
- `args`: Parsed arguments to validate

**Raises:**
- `ValueError`: If any argument is outside valid range or logically inconsistent

**Validation Rules:**
- All numeric arguments must be within specified ranges
- Learning rate must be positive
- Batch size must be at least 1
- TQF parameters must satisfy lattice construction requirements
- Model names must exist in registry

**Example Error Messages:**
```
ValueError: --learning-rate=2.0 outside valid range (0.0, 1.0]
ValueError: --batch-size=0 must be >= 1
ValueError: --tqf-symmetry-level='invalid' not in ['none', 'Z6', 'D6', 'T24']
```

---

#### `setup_logging() -> None`

**Description:** Configures Python logging system with consistent formatting.

**Behavior:**
- Sets log level based on `--log-level` argument
- Formats: `[TIMESTAMP] LEVEL: message`
- Outputs to console (stdout)
- Includes module name and line number in DEBUG mode

**Example:**
```python
from cli import setup_logging

setup_logging()
import logging

logging.info("Experiment starting")
logging.debug("Debug information")
# Output:
# [2026-01-19 10:30:45] INFO: Experiment starting
# [2026-01-19 10:30:45] DEBUG: Debug information (module.py:42)
```

---

## 4. Model Architectures

### TQF-ANN (`models_tqf.py`)

The Tri-Quarter Framework Artificial Neural Network implements Schmidt's radial dual triangular lattice graph architecture with symmetry exploitation.

#### `class TQFANN(nn.Module)`

**Description:** Main TQF-ANN model with symmetry-preserving operations and dual zone structure.

**Architecture:**
1. **Boundary Encoder**: Maps input to unit-norm lattice boundary (6 vertices for r=1)
2. **Radial Bins**: Partitions outer zone by integer norms
3. **Layer Aggregation**: Linear layer combination with uniform weighting
4. **Dual Output**: Inner zone (via circle inversion) provides dual predictions
5. **Symmetry Operations**: ℤ₆, D₆, or T₂₄ group actions for equivariance

**Constructor:**
```python
TQFANN(
    in_features: int = 784,
    hidden_dim: Optional[int] = None,
    num_classes: int = 10,
    R: int = 20,
    r: float = 1.0,
    symmetry_level: str = 'none',
    use_dual_output: bool = True,
    use_gradient_checkpointing: bool = False,
    dropout: float = 0.2
)
```

**Additional Parameters:**

- `use_gradient_checkpointing` (bool): Enable gradient checkpointing for memory optimization
  - **Default**: `False` (standard training with all activations stored)
  - **Purpose**: Trade compute time for memory savings during training
  - **Effect when True**:
    - Discards intermediate activations during forward pass
    - Recomputes activations on-the-fly during backward pass
    - Reduces activation memory by ~60%
    - Increases training time by ~30% per epoch
  - **Use cases**:
    - GPU memory constrained (8GB or less)
    - Large R values (R >= 15)
    - OOM errors during training
  - **Implementation details**:
    - Applies to radial binner forward passes (inner and outer zones)
    - Only active during `model.train()` mode
    - Uses PyTorch's `torch.utils.checkpoint.checkpoint()` internally
    - Gradients are mathematically identical (no approximation)
  - **Recommendation**: Keep `False` unless hitting OOM errors; enable for R >= 15 on 8GB GPUs

**Graph Convolution:**

The TQF-ANN architecture uses graph convolutions that aggregate features from immediate (1-hop) neighbors as defined by the hexagonal lattice structure:

- **1-hop neighbors**: ~6 vertices (hexagonal adjacency), weight = 1.0/degree
- **Memory complexity**: O(E) linear in edges, not O(V²) like dense attention

Each vertex connects to at most 6 neighbors, respecting the radial dual triangular lattice graph specification.

**Updated Parameters:**

- `hidden_dim` (Optional[int]): Feature dimension distributed across lattice
  - **Default**: `None` (auto-tuned to ~650K parameters)
  - **Range**: [8, 512]
  - **Auto-tuning**: Adjusts to maintain ~650K parameter budget

- `R` (int): Truncation radius for lattice construction
  - **Default**: 20 (updated from 18 for better coverage)
  - **Range**: [2, 100], must be > inversion radius (r=1.0)

**Example Usage:**

```python
from models_tqf import TQFANN
import torch

# Default: Baseline mode (no orbit pooling, fastest)
model = TQFANN(
    hidden_dim=128,
    R=20
    # symmetry_level='none' is default (no orbit pooling)
)

# With D6 symmetry enabled (opt-in for rotation invariance)
model_d6 = TQFANN(
    hidden_dim=128,
    R=20,
    symmetry_level='D6'  # Enable D6 orbit pooling
)

# Standard forward pass
x = torch.randn(32, 784)
logits = model(x)
print(f"Output shape: {logits.shape}")  # (32, 10)
print(f"Parameters: {model.count_parameters():,}")
```

---

**Parameters:**

- `in_features` (int): Input feature dimension
  - **Default**: 784 (28x28 MNIST)

- `hidden_dim` (Optional[int]): Feature dimension distributed across lattice
  - **Default**: None (auto-tuned to match ~650K params)
  - **Range**: [8, 512] if manually specified
  - **Why**: Controls total model capacity
  - **Effect**: Larger -> more expressive, higher memory usage

- `num_classes` (int): Number of output classes
  - **Default**: 10 (MNIST digits)

- `R` (int): Truncation radius for lattice construction
  - **Default**: 20 (from `TQF_TRUNCATION_R_DEFAULT`)
  - **Range**: [2, 100], must be > inversion radius (r=1)
  - **Why**: Determines lattice size (vertices proportional to R^2)
  - **Effect**: Larger R -> more vertices, hidden_dim auto-adjusted to maintain param budget

- `r` (float): Inversion radius (fixed)
  - **Default**: 1.0 (from `TQF_RADIUS_R_FIXED`)
  - **Note**: Hardcoded, not user-configurable

- `symmetry_level` (str): Symmetry group for orbit aggregation
  - **Default**: `'none'` (from `TQF_SYMMETRY_LEVEL_DEFAULT`, orbit pooling disabled)
  - **Choices**: `['none', 'Z6', 'D6', 'T24']`
  - **Why**:
    - `'none'`: No symmetry (baseline, fastest)
    - `'Z6'`: 6-fold rotational symmetry (60-degree increments)
    - `'D6'`: Rotations + 6 reflection axes (dihedral group)
    - `'T24'`: Full group with circle inversion (order 24)
  - **Recommendation**: Start with 'none' for baseline; enable D6 for rotation invariance tasks

- `use_dual_output` (bool): Enable dual zone output
  - **Default**: True

- `use_dual_metric` (bool): Enable dual metric computation
  - **Default**: True

- `dropout` (float): Dropout probability
  - **Default**: 0.2

- `verify_geometry` (bool): Enable geometry verification
  - **Default**: False

**Regularization Weights:**

- `geometry_reg_weight` (float): Weight for geometric regularization
  - **Default**: 0.0 (from `TQF_GEOMETRY_REG_WEIGHT_DEFAULT`, disabled; opt-in via CLI)

**Methods:**

##### `forward(x: torch.Tensor, return_inv_loss: bool = False, return_geometry_loss: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]`

**Description:** Forward pass through TQF-ANN with optional auxiliary loss outputs.

**Args:**
- `x` (torch.Tensor): Input batch
  - **Shape**: (batch_size, 784) for flattened MNIST, or (batch_size, 1, 28, 28) for image format
  - **Range**: Typically normalized to [0, 1] or [-1, 1]

- `return_inv_loss` (bool): Return inversion consistency loss
  - **Default**: False
  - **Purpose**: Enable to get circle inversion duality loss

- `return_geometry_loss` (bool): Return geodesic verification loss
  - **Default**: False
  - **Purpose**: Enable to get geometric consistency loss

**Returns:**
- If both flags False:
  - `torch.Tensor`: Logits, shape (batch_size, num_classes)

- If `return_inv_loss=True`:
  - `Tuple[torch.Tensor, torch.Tensor]`: (logits, inv_loss)
    - Logits: (batch_size, num_classes)
    - inv_loss: Scalar inversion consistency loss

- If `return_geometry_loss=True`:
  - `Tuple[torch.Tensor, torch.Tensor]`: (logits, geometry_loss)
    - Logits: (batch_size, num_classes)
    - geometry_loss: Scalar geodesic verification loss

- If both flags True:
  - `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: (logits, inv_loss, geometry_loss)

**Example:**
```python
from models_tqf import TQFANN
import torch

# Initialize model (with D6 symmetry enabled for this example)
model: TQFANN = TQFANN(R=20, symmetry_level='D6')

# Standard forward pass
x: torch.Tensor = torch.randn(32, 784)  # Batch of 32 images
logits: torch.Tensor = model(x)
print(logits.shape)  # torch.Size([32, 10])

# With inversion loss
logits, inv_loss = model(x, return_inv_loss=True)
print(f"Inversion loss: {inv_loss.item():.6f}")

# With geometry loss
logits, geo_loss = model(x, return_geometry_loss=True)
print(f"Geometry loss: {geo_loss.item():.6f}")

# With both losses
logits, inv_loss, geo_loss = model(x, return_inv_loss=True, return_geometry_loss=True)
print(f"Inversion: {inv_loss.item():.6f}, Geometry: {geo_loss.item():.6f}")
```

---

##### `count_parameters() -> int`

**Description:** Counts total trainable parameters in the model.

**Returns:**
- `int`: Number of trainable parameters

**Example:**
```python
model: TQFANN = TQFANN(R=20, hidden_dim=512)
params: int = model.count_parameters()
print(f"Total parameters: {params:,}")
# Output: Total parameters: ~650,000
```

---

##### `verify_self_duality() -> Dict[str, float]`

**Description:** Verifies that the discrete metric satisfies self-duality: d(iota_r(v), iota_r(w)) = d(v, w).

**Returns:**
- `Dict[str, float]`: Verification metrics
  - `'max_error'`: Maximum duality violation
  - `'mean_error'`: Average duality violation
  - `'num_pairs_checked'`: Number of vertex pairs tested
  - `'passes'`: Boolean (1.0 if max_error < tolerance, else 0.0)

**Example:**
```python
model: TQFANN = TQFANN(R=20, verify_geometry=True)
duality_metrics: Dict[str, float] = model.verify_self_duality()

print(f"Max duality error: {duality_metrics['max_error']:.2e}")
print(f"Mean duality error: {duality_metrics['mean_error']:.2e}")
print(f"Passes: {duality_metrics['passes'] == 1.0}")
```

---

#### Helper Classes in `models_tqf.py`

##### `class BoundaryZoneEncoder(nn.Module)`

**Description:** Maps input features to the lattice boundary zone (unit norm ring).

**Purpose:** Provides initial lattice embedding from raw input.

**Key Methods:**
- `forward(x)`: Maps input -> boundary features
  - Input shape: (batch, input_dim)
  - Output shape: (batch, num_boundary_vertices, hidden_dim)

---

##### `class RadialBinner(nn.Module)`

**Description:** Partitions outer zone into radial bins by integer norm values.

**Purpose:** Creates radial structure with dual metric binning strategies.

**Key Methods:**
- `forward(boundary_feats)`: Propagates features from boundary -> bins
- `get_bin_assignments()`: Returns vertex -> bin mapping

**Binning Methods:**
- `'linear'`: Uniform bin widths
- `'dyadic'`: Powers of 2
- `'dual_metric'`: Based on ContinuousDualMetric distances

---

##### `class DualOutputHead(nn.Module)`

**Description:** Dual prediction head operating on both outer and inner zones.

**Purpose:** Combines outer zone features with their inverted (inner zone) counterparts for robust predictions.

**Key Methods:**
- `forward(outer_feats, inner_feats)`: Fuses dual zone information
  - Returns: Combined logits with inversion consistency

---

##### `class SymmetryReducer(nn.Module)`

**Description:** Applies ℤ₆, D₆, or T₂₄ symmetry group operations for orbit aggregation.

**Purpose:** Enforces equivariance under symmetry transformations.

**Key Methods:**
- `forward(feats, symmetry_level)`: Aggregates features under group action
  - `'Z6'`: 6-fold rotations (k x 60-degrees for k = 0, ..., 5)
  - `'D6'`: Rotations + 6 reflections
  - `'T24'`: Full group including inversion

**Symmetry Operations:**
- `rotate_feats(feats, angle)`: Rotates feature arrangement
- `reflect_feats(feats, axis)`: Reflects across specified axis
- `invert_feats(feats)`: Applies circle inversion transformation

---

##### `class LabelSmoothingCrossEntropy(nn.Module)`

**Description:** Cross-entropy loss with label smoothing regularization.

**Purpose:** Prevents overconfident predictions, improves generalization.

**Parameters:**
- `smoothing` (float): Smoothing factor, default 0.1
  - Range: [0.0, 1.0]
  - 0.0 = standard cross-entropy
  - 0.1 = 10% of probability mass redistributed uniformly

---

##### `class StochasticDepth(nn.Module)`

**Description:** Randomly drops layers during training (ResNet regularization).

**Purpose:** Prevents co-adaptation, similar to dropout but for entire layers.

**Parameters:**
- `drop_prob` (float): Probability of dropping layer
  - Default: 0.1
  - Range: [0.0, 1.0]

---

### Baseline Models (`models_baseline.py`)

Provides standard architectures for comparison with TQF-ANN.

#### `class FCMLP(nn.Module)`

**Description:** Fully-connected multi-layer perceptron with configurable depth and width.

**Constructor:**
```python
FCMLP(
    input_dim: int = 784,
    hidden_sizes: Optional[List[int]] = None,  # Auto-tuned to ~650K params if None
    num_classes: int = 10,
    dropout: float = 0.2,
    activation: str = 'relu'
)
```

**Parameters:**
- `input_dim`: Flattened input size (default 784 for 28x28 MNIST)
- `hidden_sizes`: List of hidden layer sizes (optional)
  - Default `None` -> Auto-tuned to ~650K params (matches TQF-ANN)
- `num_classes`: Output classes
- `dropout`: Dropout rate for regularization
- `activation`: Activation function, choices: `['relu', 'gelu', 'silu']`

**Architecture:**
```
Input (784) -> FC(460) -> ReLU -> Dropout
            -> FC(460) -> ReLU -> Dropout
            -> FC(460) -> ReLU -> Dropout
            -> FC(10)
```

---

#### `class CNNL5(nn.Module)`

**Description:** 5-layer convolutional neural network.

**Constructor:**
```python
CNNL5(
    num_classes: int = 10,
    conv_channels: Optional[List[int]] = None,  # Auto-tuned to ~650K params if None
    fc_size: Optional[int] = None,  # Auto-tuned if None
    dropout: float = 0.2
)
```

**Parameters:**
- `num_classes`: Output classes
- `channels`: Channel counts for conv layers
  - Default `[32, 64, 96]` tuned for ~650K params
- `fc_size`: Fully connected layer size after convolutions
- `dropout`: Dropout rate

**Architecture:**
```
Input (1x28x28) -> Conv(32, 3x3) -> ReLU -> MaxPool(2x2)
                -> Conv(64, 3x3) -> ReLU -> MaxPool(2x2)
                -> Conv(96, 3x3) -> ReLU -> MaxPool(2x2)
                -> Flatten -> FC(256) -> ReLU -> Dropout -> FC(10)
```

---

#### `class ResNet18_Scaled(nn.Module)`

**Description:** Scaled-down ResNet-18 variant matched to ~650K parameters.

**Constructor:**
```python
ResNet18_Scaled(
    num_classes: int = 10,
    num_blocks: int = 2,
    base_channels: int = 32
)
```

**Parameters:**
- `num_classes`: Output classes
- `num_blocks`: Residual blocks per stage (default 2)
- `base_channels`: Initial channel count (default 32)
  - Standard ResNet-18 uses 64, scaled down for param matching

**Architecture:**
```
Input (1x28x28) -> Conv(32, 7x7, stride=2) -> BN -> ReLU -> MaxPool
                -> [ResBlock(32)] x 2
                -> [ResBlock(64)] x 2  (stride=2 first block)
                -> [ResBlock(128)] x 2 (stride=2 first block)
                -> [ResBlock(256)] x 2 (stride=2 first block)
                -> AdaptiveAvgPool -> FC(10)
```

---

#### `get_model(model_name: str, **kwargs) -> nn.Module`

**Description:** Factory function to instantiate models by name.

**Args:**
- `model_name` (str): Model identifier
  - Choices: `['TQF-ANN', 'FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']`
- `**kwargs`: Model-specific keyword arguments

**Returns:**
- `nn.Module`: Initialized model

**Example:**
```python
from models_baseline import get_model

# Instantiate MLP with custom config
mlp = get_model('FC-MLP', hidden_dims=[512, 512], dropout=0.3)

# Instantiate TQF-ANN
from models_tqf import TQFANN
tqf = get_model('TQF-ANN', R=20, symmetry_level='T24')

# Parameter counts
print(f"MLP: {mlp.count_parameters():,} params")
print(f"TQF: {tqf.count_parameters():,} params")
```

---

## 5. Training Engine (`engine.py`)

Orchestrates training loops, multi-seed experiments, and statistical comparisons.

### `class TrainingEngine`

**Description:** Encapsulates training logic with epoch-level control, metrics tracking, and integrated optimizer/scheduler setup. Creates Adam optimizer with weight decay (L2 regularization) internally.

**Constructor:**
```python
TrainingEngine(
    model: nn.Module,
    device: torch.device,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    label_smoothing: float = 0.1,
    use_geometry_reg: bool = False,
    geometry_weight: float = 0.0,
    use_amp: bool = True,
    warmup_epochs: int = 5,
    num_epochs: int = 150
)
```

**Parameters:**
- `model` (nn.Module): Neural network to train
- `device` (torch.device): Computation device (CPU or CUDA)
- `learning_rate` (float): Learning rate for Adam optimizer. Default: 0.001
- `weight_decay` (float): L2 regularization coefficient for Adam optimizer. Adds penalty term `weight_decay * ||W||^2` to loss, encouraging smaller weights to prevent overfitting. Default: 0.0001 (1e-4). Use lower values (e.g., 5e-5) when TQF regularizations are active.
- `label_smoothing` (float): Label smoothing factor for CrossEntropyLoss (0=hard labels, 0.1=standard). Default: 0.1
- `use_geometry_reg` (bool): Enable geometric preservation loss (TQF-ANN only). Default: False
- `geometry_weight` (float): Weight for geometric regularization loss. Default: 0.0
- `use_amp` (bool): Enable automatic mixed precision for faster training on RTX GPUs. Default: True
- `warmup_epochs` (int): Number of epochs for linear LR warmup (0 disables). Default: 5
- `num_epochs` (int): Total training epochs for scheduler T_max calculation. Default: 150

**Internal Components (created automatically):**
- Adam optimizer with specified `learning_rate` and `weight_decay`
- CosineAnnealingLR scheduler with optional linear warmup
- CrossEntropyLoss criterion with `label_smoothing`
- GradScaler for automatic mixed precision (if `use_amp=True`)

**Key Methods:**

##### `train_epoch(epoch: int) -> Dict[str, float]`

Trains model for one epoch.

**Returns:**
- `Dict[str, float]`: Training metrics
  - `'loss'`: Average training loss
  - `'accuracy'`: Training accuracy (%)
  - `'geometry_loss'`: Geometry regularization (TQF only)
  - `'inv_loss'`: Inversion duality loss (TQF only)

---

##### `evaluate(data_loader: DataLoader, desc: str = "Eval") -> Dict[str, float]`

Evaluates model on given dataset.

**Args:**
- `data_loader`: DataLoader for evaluation dataset
- `desc`: Description for progress bar (default: "Eval")

**Returns:**
- `Dict[str, float]`: Evaluation metrics
  - `'loss'`: Average loss
  - `'accuracy'`: Accuracy (%)

---

##### `train(train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, ...) -> Dict[str, Any]`

Executes full training loop with early stopping and learning rate scheduling.

**Args:**
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `num_epochs`: Maximum training epochs
- Additional kwargs for TQF-specific losses (see engine.py)

**Returns:**
- `Dict[str, Any]`: Training history
  - `'train_losses'`: List[float]
  - `'train_accs'`: List[float]
  - `'val_losses'`: List[float]
  - `'val_accs'`: List[float]
  - `'best_val_acc'`: float
  - `'best_epoch'`: int
  - `'stopped_early'`: bool

**Example:**
```python
from engine import TrainingEngine
from models_baseline import get_model
import torch

# Setup model and device
model = get_model('TQF-ANN', R=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training engine with weight decay (L2 regularization)
engine = TrainingEngine(
    model=model,
    device=device,
    learning_rate=0.0003,
    weight_decay=0.0001,        # L2 regularization (standard)
    label_smoothing=0.1,
    use_geometry_reg=True,      # Enable TQF geometric loss
    geometry_weight=0.1,
    warmup_epochs=5,
    num_epochs=50
)

# For TQF-ANN with strong geometric regularization, use lower weight_decay:
engine_tqf = TrainingEngine(
    model=model,
    device=device,
    learning_rate=0.0003,
    weight_decay=0.00005,       # Lower weight_decay when TQF regs active
    use_geometry_reg=True,
    geometry_weight=0.1,
    num_epochs=50
)
```

---

### Top-Level Training Functions

#### `set_seed(seed: int) -> None`

**Description:** Sets all random seeds for reproducibility.

**Args:**
- `seed` (int): Random seed value

**Effect:**
- Sets Python `random` seed
- Sets NumPy seed
- Sets PyTorch CPU seed
- Sets PyTorch CUDA seed (all devices)
- Sets cuDNN deterministic mode

**Example:**
```python
from engine import set_seed

set_seed(42)
# All subsequent random operations are deterministic
```

---

#### `rotate_batch_images(images: torch.Tensor, angle_degrees: float) -> torch.Tensor`

**Description:** Rotates a batch of images by specified angle.

**Args:**
- `images` (torch.Tensor): Image batch, shape (B, C, H, W)
- `angle_degrees` (float): Rotation angle in degrees

**Returns:**
- `torch.Tensor`: Rotated images, same shape as input

**Method:**
- Uses `torchvision.transforms.functional.rotate`
- Preserves image dimensions via appropriate padding
- Interpolation: Bilinear

**Example:**
```python
from engine import rotate_batch_images
import torch

images: torch.Tensor = torch.randn(32, 1, 28, 28)
rotated: torch.Tensor = rotate_batch_images(images, 60.0)  # 60-degree rotation
print(rotated.shape)  # torch.Size([32, 1, 28, 28])
```

---

#### `run_single_seed_experiment(...) -> Dict[str, Any]`

**Description:** Runs a complete training + evaluation experiment for a single random seed.

**Args:**
```python
run_single_seed_experiment(
    model_name: str,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_rotated_loader: DataLoader,
    test_unrotated_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_model: bool = False,
    output_dir: str = 'results',
    **model_kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `model_name`: Name from model registry
- `seed`: Random seed for this run
- `train_loader`, `val_loader`, `test_rotated_loader`, `test_unrotated_loader`: Data loaders
- `num_epochs`: Training epochs
- `learning_rate`: Initial learning rate
- `device`: Computation device
- `save_model`: Whether to save checkpoint
- `output_dir`: Results directory
- `**model_kwargs`: Model-specific arguments (e.g., `R=18` for TQF-ANN)

**Returns:**
- `Dict[str, Any]`: Complete results dictionary
  - `'model_name'`: str
  - `'seed'`: int
  - `'train_history'`: Training metrics per epoch
  - `'best_val_acc'`: float
  - `'test_rotated_acc'`: float
  - `'test_unrotated_acc'`: float
  - `'rotation_invariance_score'`: float (0-1, higher is better)
  - `'per_class_accuracies'`: Dict[int, float]
  - `'num_parameters'`: int
  - `'training_time_sec'`: float
  - `'inference_time_ms'`: float

**Example:**
```python
from engine import run_single_seed_experiment

results: Dict[str, Any] = run_single_seed_experiment(
    model_name='TQF-ANN',
    seed=42,
    train_loader=train_loader,
    val_loader=val_loader,
    test_rotated_loader=test_rotated_loader,
    test_unrotated_loader=test_unrotated_loader,
    num_epochs=30,
    learning_rate=0.0003,
    device=torch.device('cuda'),
    R=18,
    symmetry_level='D6'  # Enable D6 orbit pooling (opt-in)
)

print(f"Test accuracy (rotated): {results['test_rotated_acc']:.2f}%")
print(f"Rotation invariance: {results['rotation_invariance_score']:.3f}")
print(f"Parameters: {results['num_parameters']:,}")
```

---

#### `run_multi_seed_experiment(...) -> Dict[str, Any]`

**Description:** Runs experiments across multiple random seeds for statistical robustness.

**Args:**
```python
run_multi_seed_experiment(
    model_name: str,
    seeds: List[int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_rotated_loader: DataLoader,
    test_unrotated_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_models: bool = False,
    output_dir: str = 'results',
    **model_kwargs
) -> Dict[str, Any]
```

**Parameters:**
- Same as `run_single_seed_experiment`, except:
- `seeds` (List[int]): List of random seeds to run

**Returns:**
- `Dict[str, Any]`: Aggregated results
  - `'model_name'`: str
  - `'seed_results'`: List[Dict] - individual seed results
  - `'mean_test_rotated_acc'`: float
  - `'std_test_rotated_acc'`: float
  - `'mean_test_unrotated_acc'`: float
  - `'std_test_unrotated_acc'`: float
  - `'mean_rotation_invariance'`: float
  - `'std_rotation_invariance'`: float
  - `'ci_95_lower'`: float (95% confidence interval lower bound)
  - `'ci_95_upper'`: float (95% confidence interval upper bound)
  - `'num_parameters'`: int (constant across seeds)

**Statistical Methods:**
- Mean and standard deviation across seeds
- 95% confidence intervals using Student's t-distribution
- Handles NaN values gracefully

**Example:**
```python
from engine import run_multi_seed_experiment

results: Dict[str, Any] = run_multi_seed_experiment(
    model_name='TQF-ANN',
    seeds=[42, 43, 44, 45, 46],
    train_loader=train_loader,
    val_loader=val_loader,
    test_rotated_loader=test_rotated_loader,
    test_unrotated_loader=test_unrotated_loader,
    num_epochs=30,
    learning_rate=0.0003,
    device=torch.device('cuda'),
    R=18,
    symmetry_level='D6'  # Enable D6 orbit pooling (opt-in)
)

print(f"Mean test accuracy: {results['mean_test_rotated_acc']:.2f}% "
      f"+/- {results['std_test_rotated_acc']:.2f}%")
print(f"95% CI: [{results['ci_95_lower']:.2f}%, {results['ci_95_upper']:.2f}%]")
```

---

#### `compare_models_statistical(results_list: List[Dict]) -> Dict[str, Any]`

**Description:** Performs pairwise statistical comparisons between models.

**Args:**
- `results_list` (List[Dict]): List of results from `run_multi_seed_experiment`

**Returns:**
- `Dict[str, Any]`: Comparison statistics
  - `'comparisons'`: List[Dict] - pairwise t-tests
    - Each comparison contains:
      - `'model_a'`: str
      - `'model_b'`: str
      - `'mean_diff'`: float (A - B)
      - `'p_value'`: float (two-tailed t-test)
      - `'significant'`: bool (p < 0.05)
      - `'cohens_d'`: float (effect size)
  - `'best_model'`: str (highest mean accuracy)

**Statistical Tests:**
- Independent samples t-test (two-tailed)
- Cohen's d effect size
- Significance level alpha = 0.05

**Example:**
```python
from engine import run_multi_seed_experiment, compare_models_statistical

# Run experiments
tqf_results = run_multi_seed_experiment(model_name='TQF-ANN', seeds=[42, 43, 44], ...)
mlp_results = run_multi_seed_experiment(model_name='FC-MLP', seeds=[42, 43, 44], ...)

# Statistical comparison
comparison = compare_models_statistical([tqf_results, mlp_results])

for comp in comparison['comparisons']:
    print(f"{comp['model_a']} vs {comp['model_b']}:")
    print(f"  Mean difference: {comp['mean_diff']:.2f}%")
    print(f"  p-value: {comp['p_value']:.4f}")
    print(f"  Significant: {comp['significant']}")
    print(f"  Cohen's d: {comp['cohens_d']:.3f}")

print(f"\nBest model: {comparison['best_model']}")
```

---

#### `print_final_comparison_table(results_list: List[Dict]) -> None`

**Description:** Prints formatted comparison table of all models.

**Args:**
- `results_list` (List[Dict]): List of multi-seed experiment results

**Output Format:**
```
================================================================================
FINAL MODEL COMPARISON (Mean +/- Std across N seeds)
================================================================================
Model           Params      Test-Rot    Test-Unrot  Rot-Inv     Inference
--------------------------------------------------------------------------------
TQF-ANN        650,123     95.23+/-0.15  96.45+/-0.12  0.986+/-0.002  2.3 ms
FC-MLP         648,030     93.12+/-0.21  95.67+/-0.18  0.973+/-0.004  1.8 ms
CNN-L5         649,800     94.56+/-0.19  96.01+/-0.14  0.984+/-0.003  2.1 ms
ResNet-18      651,200     94.89+/-0.17  96.23+/-0.13  0.985+/-0.002  2.5 ms
================================================================================
```

**Columns:**
- **Params**: Total trainable parameters
- **Test-Rot**: Rotated test accuracy (mean +/- std)
- **Test-Unrot**: Unrotated test accuracy (mean +/- std)
- **Rot-Inv**: Rotation invariance score (mean +/- std)
- **Inference**: Average inference time per sample

---

## 6. Evaluation Utilities (`evaluation.py`)

Comprehensive metrics computation and statistical analysis.

### Core Evaluation Functions

#### `compute_per_class_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int = 10) -> Dict[int, float]`

**Description:** Computes accuracy for each class separately.

**Args:**
- `model`: Trained model
- `data_loader`: Test data iterator
- `device`: Computation device
- `num_classes`: Number of classes

**Returns:**
- `Dict[int, float]`: Class -> accuracy mapping
  - Keys: 0, 1, ..., num_classes-1
  - Values: Accuracy (0-100%)

**Example:**
```python
from evaluation import compute_per_class_accuracy

per_class: Dict[int, float] = compute_per_class_accuracy(
    model=model,
    data_loader=test_loader,
    device=device,
    num_classes=10
)

for class_id, acc in per_class.items():
    print(f"Class {class_id}: {acc:.2f}%")
```

---

#### `compute_rotation_invariance_error(model: nn.Module, data_loader: DataLoader, device: torch.device, rotation_angles: List[float] = [0, 60, 120, 180, 240, 300]) -> float`

**Description:** Measures rotation invariance by comparing predictions across rotations.

**Args:**
- `model`: Trained model
- `data_loader`: Test data (unrotated)
- `device`: Computation device
- `rotation_angles`: Angles to test (degrees)

**Returns:**
- `float`: Invariance error (0-1, lower is better)
  - 0 = perfect invariance (predictions identical across rotations)
  - 1 = no invariance (completely different predictions)

**Method:**
- For each sample, compute predictions at all rotation angles
- Measure prediction variance across rotations
- Average across dataset

**Example:**
```python
from evaluation import compute_rotation_invariance_error

inv_error: float = compute_rotation_invariance_error(
    model=model,
    data_loader=test_loader,
    device=device,
    rotation_angles=[0, 60, 120, 180, 240, 300]
)

print(f"Rotation invariance error: {inv_error:.4f}")
print(f"Rotation invariance score: {1.0 - inv_error:.4f}")
```

---

#### `adaptive_orbit_mixing(logits_per_variant, temperature, confidence_mode, aggregation_mode, top_k, adaptive_temp, adaptive_temp_alpha) -> torch.Tensor`

**Description:** Combines multiple prediction variants using confidence-weighted averaging. Supports multiple confidence scoring modes, aggregation spaces, top-K filtering, and adaptive per-sample temperature scaling.

**Args:**
- `logits_per_variant`: `List[torch.Tensor]` — logit tensors, each `(batch, num_classes)`
- `temperature`: `float` — softmax temperature for confidence weighting (default: `0.5`, lower = sharper)
- `confidence_mode`: `str` — `'max_logit'` (default) or `'margin'` (top-1 minus top-2 logit)
- `aggregation_mode`: `str` — `'logits'` (default), `'probs'`, or `'log_probs'` (geometric mean)
- `top_k`: `Optional[int]` — keep only the K most confident variants (default: `None` = all)
- `adaptive_temp`: `bool` — enable per-sample entropy-based temperature scaling (default: `False`)
- `adaptive_temp_alpha`: `float` — sensitivity of adaptive temperature (default: `1.0`)

**Returns:**
- `torch.Tensor`: Weighted average in the chosen aggregation space `(batch, num_classes)`

---

#### `compute_orbit_consistency_loss(model, inputs, base_logits, num_rotations) -> Optional[torch.Tensor]`

**Description:** Computes orbit consistency self-distillation loss (training-time only). Creates a stop-gradient ensemble from the base logits plus extra rotated forward passes, then penalises each rotation variant for diverging from the ensemble via KL divergence. Returns `None` for non-TQF models.

**Args:**
- `model`: `nn.Module` — TQF-ANN model instance (must have `dual_output` attribute)
- `inputs`: `torch.Tensor` — batch inputs `(batch, 784)` from the current training step
- `base_logits`: `torch.Tensor` — logits from the current forward pass `(batch, num_classes)`, must still be in the computation graph
- `num_rotations`: `int` — number of extra Z6 rotation passes sampled from {60°, 120°, 180°, 240°, 300°} (default: `2`)

**Returns:**
- `Optional[torch.Tensor]`: Scalar KL-divergence loss averaged over all variants, or `None` for non-TQF models

**Example:**
```python
from evaluation import compute_orbit_consistency_loss

# During training
base_logits = model(inputs, return_inv_loss=False)
ce_loss = criterion(base_logits, labels)

consistency_loss = compute_orbit_consistency_loss(model, inputs, base_logits, num_rotations=2)
if consistency_loss is not None:
    total_loss = ce_loss + 0.01 * consistency_loss
else:
    total_loss = ce_loss
```

---

#### `classify_from_sector_features(model: nn.Module, outer_sector_feats: torch.Tensor, inner_sector_feats: torch.Tensor, swap_weights: bool = False) -> torch.Tensor`

**Description:** Runs the TQF-ANN classification pipeline on pre-computed zone features. Mirrors the classification steps in `TQFANN.forward()` — shared classification head, sector weight einsum, and confidence-weighted ensemble.

**Args:**
- `model`: TQFANN model instance (must have `dual_output` attribute)
- `outer_sector_feats`: Outer zone features `(batch, 6, hidden_dim)`
- `inner_sector_feats`: Inner zone features `(batch, 6, hidden_dim)`
- `swap_weights` (bool): If `True`, swaps inner/outer zone weighting for T24 zone-swap operations. **Default**: `False`

**Returns:**
- `torch.Tensor`: Ensemble logits `(batch, num_classes)`

---

#### `evaluate_with_orbit_mixing(model, loader, device, use_z6, use_d6, use_t24, temp_rotation, temp_reflection, temp_inversion, confidence_mode, aggregation_mode, top_k, adaptive_temp, adaptive_temp_alpha, rotation_mode, rotation_padding_mode, pad_before_rotate, use_amp, verbose) -> Tuple[float, float]`

**Description:** Evaluates TQF-ANN with hierarchical orbit mixing preserving inner/outer mirroring. Three independent levels: Z6 input-space rotation, D6 feature-space reflection, T24 zone-swap. All Z6 quality enhancement parameters from `adaptive_orbit_mixing` and `rotate_batch_images` are threaded through.

**Args:**
- `model`: TQFANN model instance
- `loader`: DataLoader for evaluation
- `device`: Torch device
- `use_z6/use_d6/use_t24`: Enable each symmetry level
- `temp_rotation/temp_reflection/temp_inversion`: Per-level temperatures
- `confidence_mode`: `'max_logit'` or `'margin'` (passed to `adaptive_orbit_mixing`)
- `aggregation_mode`: `'logits'`, `'probs'`, or `'log_probs'`
- `top_k`: Keep only K most confident Z6 variants (`None` = all)
- `adaptive_temp`: Enable per-sample adaptive temperature
- `adaptive_temp_alpha`: Adaptive temperature entropy sensitivity
- `rotation_mode`: Grid-sample interpolation (`'bilinear'` or `'bicubic'`)
- `rotation_padding_mode`: Grid-sample padding (`'zeros'` or `'border'`)
- `pad_before_rotate`: Reflect-pad before rotation, crop after (0 = disabled)
- `use_amp`: Enable automatic mixed precision
- `verbose`: Log per-batch progress

**Returns:**
- `Tuple[float, float]`: `(average_loss, accuracy_percent)`

**Example:**
```python
from evaluation import evaluate_with_orbit_mixing

# Standard Z6 orbit mixing with all defaults
loss, acc = evaluate_with_orbit_mixing(
    model=model,
    loader=test_loader_rot,
    device=device,
    use_z6=True,
)
print(f"Z6 orbit mixing: {acc:.2f}%")

# Z6 with mark3 quality enhancements
loss, acc = evaluate_with_orbit_mixing(
    model=model,
    loader=test_loader_rot,
    device=device,
    use_z6=True,
    confidence_mode='margin',
    aggregation_mode='log_probs',
    top_k=4,
)
print(f"Enhanced Z6 orbit mixing: {acc:.2f}%")
```

---

#### `measure_inference_time(model: nn.Module, data_loader: DataLoader, device: torch.device, num_batches: int = 10) -> float`

**Description:** Measures average inference time per sample.

**Args:**
- `model`: Trained model
- `data_loader`: Data iterator
- `device`: Computation device
- `num_batches`: Number of batches to average over

**Returns:**
- `float`: Inference time in milliseconds per sample

**Method:**
- Warm-up: 5 batches to initialize CUDA
- Timed inference: Average over `num_batches`
- Excludes data loading time (only model forward pass)

**Example:**
```python
from evaluation import measure_inference_time

inf_time_ms: float = measure_inference_time(
    model=model,
    data_loader=test_loader,
    device=device,
    num_batches=20
)

print(f"Inference time: {inf_time_ms:.2f} ms/sample")
```

---

#### `estimate_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> float`

**Description:** Estimates floating-point operations (FLOPs) for a forward pass.

**Args:**
- `model`: PyTorch module
- `input_shape`: Input tensor shape, e.g., `(1, 28, 28)`

**Returns:**
- `float`: Estimated FLOPs

**Method:**
- Analytical estimation based on layer types
- Linear layers: 2 x input_dim x output_dim
- Conv2D: 2 x kernel_ops x output_spatial_size
- Approximate for complex modules

**Example:**
```python
from evaluation import estimate_model_flops

flops: float = estimate_model_flops(model, input_shape=(1, 28, 28))
print(f"Estimated FLOPs: {flops / 1e6:.1f}M")
```

---

#### `compute_statistical_significance(results_a: Dict, results_b: Dict, alpha: float = 0.05) -> Dict[str, Any]`

**Description:** Performs statistical comparison between two models' multi-seed results.

**Args:**
- `results_a`, `results_b`: Multi-seed experiment results
- `alpha`: Significance level (default 0.05)

**Returns:**
- `Dict[str, Any]`:
  - `'t_statistic'`: float
  - `'p_value'`: float
  - `'significant'`: bool (True if p < alpha)
  - `'mean_diff'`: float (A - B)
  - `'cohens_d'`: float (effect size)
  - `'confidence_interval'`: Tuple[float, float] (95% CI of difference)

**Example:**
```python
from evaluation import compute_statistical_significance

sig_test = compute_statistical_significance(tqf_results, mlp_results)

print(f"Mean difference: {sig_test['mean_diff']:.2f}%")
print(f"p-value: {sig_test['p_value']:.4f}")
print(f"Significant: {sig_test['significant']}")
print(f"Effect size (Cohen's d): {sig_test['cohens_d']:.3f}")
```

---

#### `compute_inversion_consistency_metrics(model: TQFANN, data_loader: DataLoader, device: torch.device) -> Dict[str, float]`

**Description:** Measures consistency between outer and inner zone predictions (TQF-ANN only).

**Args:**
- `model`: TQF-ANN model
- `data_loader`: Test data
- `device`: Computation device

**Returns:**
- `Dict[str, float]`:
  - `'mean_consistency'`: Average agreement (0-1)
  - `'std_consistency'`: Standard deviation
  - `'min_consistency'`: Worst-case sample
  - `'max_consistency'`: Best-case sample

**Method:**
- Extract outer and inner zone logits from model
- Compute prediction agreement: 1.0 if same class, 0.0 otherwise
- Average across dataset

---

---

## 7. Dataset Preparation (`prepare_datasets.py`)

Handles MNIST loading, rotation, and data loader creation.

### `class RotatedMNIST(Dataset)`

**Description:** MNIST dataset with on-the-fly rotation augmentation.

**Constructor:**
```python
RotatedMNIST(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True,
    rotation_angles: List[float] = [0, 60, 120, 180, 240, 300]
)
```

**Parameters:**
- `root`: Data directory path
- `train`: True for training set, False for test set
- `transform`: Additional transforms (applied after rotation)
- `download`: Download MNIST if not present
- `rotation_angles`: Rotation angles to apply (degrees)

**Methods:**

##### `__getitem__(index: int) -> Tuple[torch.Tensor, int]`

Returns rotated image and label.

**Returns:**
- `Tuple[torch.Tensor, int]`:
  - Image: (1, 28, 28) tensor
  - Label: Class integer (0-9)

---

##### `__len__() -> int`

Returns dataset size.

---

### Data Loader Creation Functions

#### `prepare_mnist_dataloaders(...) -> Tuple[DataLoader, ...]`

**Description:** Creates all required data loaders for experiments.

**Args:**
```python
prepare_mnist_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    num_train: int = 5000,
    num_val: int = 1000,
    num_test_rotated: int = 1000,
    num_test_unrotated: int = 1000,
    rotation_angles: List[float] = [0, 60, 120, 180, 240, 300],
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]
```

**Parameters:**
- `data_dir`: Directory for MNIST data
- `batch_size`: Mini-batch size
- `num_workers`: Data loading threads
- `num_train`: Training samples per class
- `num_val`: Validation samples per class
- `num_test_rotated`: Rotated test samples per class
- `num_test_unrotated`: Unrotated test samples per class
- `rotation_angles`: Angles for rotation augmentation
- `seed`: Random seed for reproducibility

**Returns:**
- `Tuple[DataLoader, DataLoader, DataLoader, DataLoader]`:
  1. Training loader (with rotations)
  2. Validation loader (with rotations)
  3. Rotated test loader
  4. Unrotated test loader

**Example:**
```python
from prepare_datasets import prepare_mnist_dataloaders

train_loader, val_loader, test_rot_loader, test_unrot_loader = \
    prepare_mnist_dataloaders(
        data_dir='./data',
        batch_size=128,
        num_train=6000,
        num_val=1000,
        num_test_rotated=1500,
        num_test_unrotated=1500,
        rotation_angles=[0, 60, 120, 180, 240, 300],
        seed=42
    )

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

---

## 8. Configuration (`config.py`)

Centralized constants and hyperparameter defaults.

### Key Configuration Categories

#### Training Hyperparameters

```python
# Default training settings
NUM_SEEDS_DEFAULT: int = 1
MAX_EPOCHS_DEFAULT: int = 150
BATCH_SIZE_DEFAULT: int = 128
LEARNING_RATE_DEFAULT: float = 0.001
NUM_WORKERS_DEFAULT: int = 0
PIN_MEMORY_DEFAULT: bool = True

# Optimizer settings
WEIGHT_DECAY_DEFAULT: float = 0.0001  # 1e-4
LABEL_SMOOTHING_DEFAULT: float = 0.1
DROPOUT_DEFAULT: float = 0.2

# Learning rate schedule
LEARNING_RATE_WARMUP_EPOCHS: int = 5
SCHEDULER_T_MAX_DEFAULT: int = 145  # MAX_EPOCHS - WARMUP

# Early stopping
PATIENCE_DEFAULT: int = 25
MIN_DELTA_DEFAULT: float = 0.0005
```

---

#### TQF-Specific Parameters

```python
# Lattice structure
TQF_TRUNCATION_R_DEFAULT: int = 20
TQF_RADIUS_R_FIXED: float = 1.0  # Fixed inversion radius

# Model capacity
TQF_HIDDEN_DIMENSION_DEFAULT: int = 512

# Symmetry
TQF_SYMMETRY_LEVEL_DEFAULT: str = 'none'
TQF_SYMMETRY_CHOICES: List[str] = ['none', 'Z6', 'D6', 'T24']

# Z6 augmentation (applies to all models)
Z6_DATA_AUGMENTATION_DEFAULT: bool = False

# Regularization weights (all opt-in, disabled by default)
TQF_GEOMETRY_REG_WEIGHT_DEFAULT: float = 0.0

# Verification
TQF_VERIFY_DUALITY_INTERVAL_DEFAULT: int = 10
```

---

#### Dataset Configuration

```python
# MNIST dataset sizes
NUM_TRAIN_DEFAULT: int = 58000
NUM_VAL_DEFAULT: int = 2000
NUM_TEST_ROT_DEFAULT: int = 2000
NUM_TEST_UNROT_DEFAULT: int = 8000

# Rotation angles (degrees)
ROTATION_ANGLES_DEFAULT: List[float] = [0, 60, 120, 180, 240, 300]

# Data augmentation
USE_DATA_AUGMENTATION: bool = True
AUGMENTATION_PROB: float = 0.5
```

---

#### Parameter Matching

```python
# Target parameter count for fair comparison
TARGET_PARAMS: int = 650000
TARGET_PARAMS_TOLERANCE_PERCENT: float = 1.1

# Baseline model configurations (auto-tuned to match TQF)
MLP_HIDDEN_DIMS_DEFAULT: List[int] = [460, 460, 460]
CNN_CHANNELS_DEFAULT: List[int] = [32, 64, 96]
CNN_FC_SIZE_DEFAULT: int = 256
RESNET_NUM_BLOCKS_DEFAULT: int = 2
RESNET_BASE_CHANNELS_DEFAULT: int = 32
```

---

#### Validation Ranges

```python
# CLI argument ranges
NUM_SEEDS_MIN: int = 1
NUM_SEEDS_MAX: int = 20
NUM_EPOCHS_MIN: int = 1
NUM_EPOCHS_MAX: int = 200
BATCH_SIZE_MIN: int = 1
BATCH_SIZE_MAX: int = 1024
LEARNING_RATE_MIN: float = 0.0
LEARNING_RATE_MAX: float = 1.0

TQF_HIDDEN_DIM_MIN: int = 8
TQF_HIDDEN_DIM_MAX: int = 512
TQF_TRUNCATION_R_MIN: int = 2
TQF_TRUNCATION_R_MAX: int = 100
```

**Usage:**
```python
from config import *

print(f"Default learning rate: {LEARNING_RATE_DEFAULT}")
print(f"TQF truncation radius: {TQF_TRUNCATION_R_DEFAULT}")
print(f"Target params: {TARGET_PARAMS:,}")
```

---

## 9. Dual Metrics & Geometry (`dual_metrics.py`)

Implements discrete and continuous dual metrics for lattice operations.

### `class DiscreteDualMetric`

**Description:** Graph-based dual metric for discrete lattice operations.

**Constructor:**
```python
DiscreteDualMetric(
    adjacency: Dict[int, List[int]],
    validate: bool = True
)
```

**Parameters:**
- `adjacency`: Graph adjacency list (vertex_id -> neighbor_ids)
- `validate`: Check adjacency structure validity

**Methods:**

##### `compute_hop_distance(source: int, target: int) -> int`

**Description:** Computes shortest path length (hop count) between vertices.

**Args:**
- `source`, `target`: Vertex IDs

**Returns:**
- `int`: Hop count (large finite value if unreachable)

**Method:** Breadth-first search

**Example:**
```python
from dual_metrics import DiscreteDualMetric

adjacency: Dict[int, List[int]] = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

metric: DiscreteDualMetric = DiscreteDualMetric(adjacency)
dist: int = metric.compute_hop_distance(0, 3)
print(f"Distance: {dist} hops")  # Distance: 2 hops
```

---

##### `verify_self_duality(inversion_map: Dict[int, int], tolerance: float = 1e-6) -> Dict[str, float]`

**Description:** Verifies self-duality: d(iota(v), iota(w)) = d(v, w) for all v, w.

**Args:**
- `inversion_map`: Vertex -> inverted vertex mapping
- `tolerance`: Numerical tolerance for equality

**Returns:**
- `Dict[str, float]`:
  - `'max_error'`: Maximum duality violation
  - `'mean_error'`: Average violation
  - `'num_pairs_checked'`: Number of vertex pairs tested
  - `'passes'`: 1.0 if passes, 0.0 otherwise

---

### `class ContinuousDualMetric`

**Description:** Continuous dual metric for radial binning in Euclidean space.

**Constructor:**
```python
ContinuousDualMetric(
    R: float,
    r: float = 1.0,
    binning_method: str = 'dyadic'
)
```

**Parameters:**
- `R`: Truncation radius
- `r`: Inversion radius
- `binning_method`: Binning strategy
  - `'linear'`: Uniform bin widths
  - `'dyadic'`: Powers of 2

**Methods:**

##### `compute_dual_norm(z: complex) -> float`

**Description:** Computes dual norm: |z|_dual = max(|z|, r^2/|z|).

**Args:**
- `z` (complex): Point in complex plane

**Returns:**
- `float`: Dual norm value

**Example:**
```python
from dual_metrics import ContinuousDualMetric

metric: ContinuousDualMetric = ContinuousDualMetric(R=3, r=1.0)
z: complex = 2 + 3j
dual_norm: float = metric.compute_dual_norm(z)
print(f"Dual norm of {z}: {dual_norm:.4f}")
```

---

##### `assign_to_bin(z: complex) -> int`

**Description:** Assigns a point to a radial bin based on dual norm.

**Args:**
- `z` (complex): Point to assign

**Returns:**
- `int`: Bin index (0 to num_bins-1)

---

##### `get_bin_centers() -> List[float]`

**Description:** Returns radial centers of all bins.

**Returns:**
- `List[float]`: Bin center radii

---

### Utility Functions

#### `build_triangular_lattice_zones(R: float, r: float = 1.0) -> Tuple[Dict, Dict, Dict]`

**Description:** Constructs outer and inner zones of radial dual triangular lattice.

**Args:**
- `R`: Truncation radius
- `r`: Inversion radius

**Returns:**
- `Tuple[Dict, Dict, Dict]`:
  1. Outer zone adjacency
  2. Inner zone adjacency
  3. Inversion map (outer -> inner)

**Example:**
```python
from dual_metrics import build_triangular_lattice_zones

outer_adj, inner_adj, inv_map = build_triangular_lattice_zones(R=10, r=1.0)

print(f"Outer zone vertices: {len(outer_adj)}")
print(f"Inner zone vertices: {len(inner_adj)}")
print(f"Inversion pairs: {len(inv_map)}")
```

---

## 10. Parameter Matching (`param_matcher.py`)

Ensures fair comparison by matching model parameter counts.

### Target Parameter Configuration

```python
TARGET_PARAMS: int = 650000  # ~650K parameters
TARGET_PARAMS_TOLERANCE_PERCENT: float = 1.1  # +/-1.1% acceptable
```

### TQF Parameter Estimation

#### `estimate_tqf_params(R: int, d_hidden: int, num_classes: int = 10) -> int`

**Description:** Estimates TQF-ANN parameter count.

**Args:**
- `R`: Truncation radius
- `d_hidden`: Hidden dimension
- `num_classes`: Output classes

**Returns:**
- `int`: Estimated parameter count

**Formula:**
- Boundary encoder: 784 x 6 x d_hidden
- Radial bins: R x d_hidden^2
- Output heads: d_hidden x num_classes x 2 (dual zones)

---

#### `tune_d_for_params(R: int, target_params: int = TARGET_PARAMS, tolerance_percent: float = TARGET_PARAMS_TOLERANCE_PERCENT) -> int`

**Description:** Auto-tunes hidden dimension to match target parameter count.

**Args:**
- `R`: Truncation radius (fixed)
- `target_params`: Target parameter count
- `tolerance_percent`: Acceptable deviation (%)

**Returns:**
- `int`: Optimal hidden dimension

**Method:** Binary search with parameter estimation

**Example:**
```python
from param_matcher import tune_d_for_params, estimate_tqf_params

R: int = 18

d_opt: int = tune_d_for_params(R, target_params=650000)
params: int = estimate_tqf_params(R, d_opt)

print(f"Optimal d_hidden: {d_opt}")
print(f"Actual parameters: {params:,}")
print(f"Target: 650,000")
print(f"Deviation: {abs(params - 650000) / 650000 * 100:.2f}%")
```

---

### Baseline Parameter Tuning

#### `tune_mlp_for_params(target_params: int = TARGET_PARAMS, tolerance_percent: float = TARGET_PARAMS_TOLERANCE_PERCENT) -> List[int]`

**Description:** Finds MLP architecture matching target parameters.

**Returns:**
- `List[int]`: Hidden layer sizes, e.g., `[460, 460, 460]`

---

#### `tune_cnn_for_params(target_params: int = TARGET_PARAMS, tolerance_percent: float = TARGET_PARAMS_TOLERANCE_PERCENT) -> Tuple[List[int], int]`

**Description:** Finds CNN architecture matching target parameters.

**Returns:**
- `Tuple[List[int], int]`: (channel_list, fc_size)
  - Example: `([32, 64, 96], 256)`

---

#### `tune_resnet_for_params(target_params: int = TARGET_PARAMS, tolerance_percent: float = TARGET_PARAMS_TOLERANCE_PERCENT) -> Tuple[int, int]`

**Description:** Finds ResNet architecture matching target parameters.

**Returns:**
- `Tuple[int, int]`: (num_blocks, base_channels)
  - Example: `(2, 32)`

---

---

## 11. Logging & Output Formatting

### Logging Utilities (`logging_utils.py`)

#### `class ProgressLogger`

**Description:** Lightweight progress logger for epoch-level tracking.

**Constructor:**
```python
ProgressLogger(total_epochs: int)
```

**Methods:**

##### `update(epoch: int, train_loss: float, val_loss: float, val_acc: float, extras: Optional[Dict[str, float]] = None) -> None`

Logs epoch progress with metrics.

##### `finish() -> None`

Marks completion of training.

**Example:**
```python
from logging_utils import ProgressLogger

logger: ProgressLogger = ProgressLogger(total_epochs=50)

for epoch in range(50):
    # ... training ...
    logger.update(
        epoch=epoch,
        train_loss=1.2,
        val_loss=1.0,
        val_acc=92.5,
        extras={'lr': 0.0003}
    )

logger.finish()
```

---

### Output Formatting (`output_formatters.py`)

Provides formatted output for results presentation.

#### Typography Constants

```python
# Separator characters
MAJOR_SEP_CHAR: str = "="
MINOR_SEP_CHAR: str = "-"

# Standard widths
WIDTH_NARROW: int = 80
WIDTH_STANDARD: int = 92
WIDTH_WIDE: int = 120
WIDTH_EXTRA: int = 175
```

#### Key Functions

##### `print_section_header(title: str, char: str = '=', width: int = 92) -> None`

Prints titled section header.

##### `format_accuracy(value: float, width: int = 6) -> str`

Formats accuracy as percentage: `"95.23%"`

##### `format_params(params: int) -> str`

Formats parameter count: `"650k"` or `"1.2M"`

##### `format_time_seconds(seconds: float) -> str`

Formats duration: `"1.5 min"`, `"2.3 hr"`

##### `print_final_results_table(results: List[Dict]) -> None`

Prints comprehensive results table.

---

## 12. Type Definitions & Conventions

### Common Type Aliases

```python
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Model types
ModelType = nn.Module
OptimizerType = torch.optim.Optimizer
SchedulerType = torch.optim.lr_scheduler._LRScheduler

# Data types
TensorType = torch.Tensor
DeviceType = torch.device
DataLoaderType = DataLoader

# Result types
MetricsDict = Dict[str, float]
AuxOutputDict = Dict[str, torch.Tensor]
SeedResultsDict = Dict[str, Any]
```

### Type Hint Conventions

All new variable declarations use type hints:

```python
# Scalars
epochs: int = 50
learning_rate: float = 0.0003
model_name: str = 'TQF-ANN'
is_training: bool = True

# Collections
hidden_dims: List[int] = [256, 512, 256]
metrics: Dict[str, float] = {'accuracy': 95.2, 'loss': 0.12}
results: Tuple[float, float] = (0.95, 0.02)

# Optional values
checkpoint_path: Optional[str] = None
scheduler: Optional[SchedulerType] = None

# Callables
loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss()
```

---

## 13. Complete Usage Examples

### Example 1: Single Model Training

```python
from cli import parse_args, setup_logging
from prepare_datasets import prepare_mnist_dataloaders
from engine import run_single_seed_experiment
import torch

# Setup
setup_logging()
args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
train_loader, val_loader, test_rot_loader, test_unrot_loader = \
    prepare_mnist_dataloaders(
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test_rotated=args.num_test_rotated,
        num_test_unrotated=args.num_test_unrotated
    )

# Train
results = run_single_seed_experiment(
    model_name='TQF-ANN',
    seed=42,
    train_loader=train_loader,
    val_loader=val_loader,
    test_rotated_loader=test_rot_loader,
    test_unrotated_loader=test_unrot_loader,
    num_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    device=device,
    R=args.tqf_truncation_R,
    symmetry_level=args.tqf_symmetry_level
)

print(f"Test accuracy (rotated): {results['test_rotated_acc']:.2f}%")
print(f"Rotation invariance: {results['rotation_invariance_score']:.3f}")
```

---

### Example 2: Multi-Model Comparison

```python
from engine import run_multi_seed_experiment, compare_models_statistical, print_final_comparison_table

# Models to compare
model_names = ['TQF-ANN', 'FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']
seeds = [42, 43, 44, 45, 46]

# Run experiments
all_results = []
for model_name in model_names:
    results = run_multi_seed_experiment(
        model_name=model_name,
        seeds=seeds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_rotated_loader=test_rot_loader,
        test_unrotated_loader=test_unrot_loader,
        num_epochs=30,
        learning_rate=0.0003,
        device=device
    )
    all_results.append(results)

# Statistical comparison
comparison = compare_models_statistical(all_results)
print_final_comparison_table(all_results)

# Pairwise significance tests
for comp in comparison['comparisons']:
    print(f"{comp['model_a']} vs {comp['model_b']}: "
          f"p={comp['p_value']:.4f}, "
          f"Cohen's d={comp['cohens_d']:.3f}")
```

---

### Example 3: Custom TQF Configuration

```python
from models_tqf import TQFANN
from param_matcher import tune_d_for_params, estimate_tqf_params

# Custom configuration
R: int = 25
symmetry_level: str = 'T24'

# Auto-tune hidden dimension
d_hidden: int = tune_d_for_params(R, target_params=650000)
print(f"Auto-tuned hidden_dim: {d_hidden}")

# Initialize model
model: TQFANN = TQFANN(
    hidden_dim=d_hidden,
    R=R,
    symmetry_level=symmetry_level,
    geometry_reg_weight=0.15,
    inv_loss_weight=0.25,
    verify_duality_interval=10,
    verify_geometry=True
)

# Verify parameter count
actual_params: int = model.count_parameters()
print(f"Actual parameters: {actual_params:,}")
print(f"Target: 650,000")
print(f"Deviation: {abs(actual_params - 650000) / 650000 * 100:.2f}%")

# Verify duality
duality_metrics = model.verify_self_duality()
print(f"Duality check passed: {duality_metrics['passes'] == 1.0}")
```

---

### Example 4: Evaluation Only (Pre-Trained Model)

```python
from evaluation import (
    compute_per_class_accuracy,
    compute_rotation_invariance_error,
    measure_inference_time,
    estimate_model_flops
)
import torch

# Load pre-trained model
model: nn.Module = torch.load('checkpoints/tqf_ann_best.pth')
model.eval()
device = torch.device('cuda')
model.to(device)

# Per-class accuracy
per_class: Dict[int, float] = compute_per_class_accuracy(
    model, test_loader, device
)
print("Per-class accuracies:")
for class_id, acc in per_class.items():
    print(f"  Class {class_id}: {acc:.2f}%")

# Rotation invariance
inv_error: float = compute_rotation_invariance_error(
    model, test_loader, device
)
print(f"Rotation invariance score: {1.0 - inv_error:.4f}")

# Inference time
inf_time: float = measure_inference_time(model, test_loader, device)
print(f"Inference time: {inf_time:.2f} ms/sample")

# FLOPs
flops: float = estimate_model_flops(model, input_shape=(1, 28, 28))
print(f"Estimated FLOPs: {flops / 1e6:.1f}M")
```

---

## 14. Testing & Validation

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module tests
pytest tests/test_tqf_ann.py -v
pytest tests/test_dual_metrics.py -v
pytest tests/test_param_matcher.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Organization

- `test_cli.py`: CLI argument parsing and validation
- `test_config.py`: Configuration constant validation
- `test_datasets.py`: Data loading and rotation
- `test_dual_metrics.py`: Geometric operations
- `test_engine.py`: Training loop and multi-seed experiments
- `test_evaluation.py`: Metrics computation
- `test_logging_utils.py`: Progress logging
- `test_main.py`: End-to-end integration
- `test_output_formatters.py`: Output formatting
- `test_param_matcher.py`: Parameter auto-tuning
- `test_tqf_ann.py`: TQF-ANN architecture
- `test_verification_features.py`: Duality and geometry verification

See [TESTS_README.md](TESTS_README.md) for detailed testing documentation.

---

## Performance Benchmarks (Updated v1.1.0)

### TQF-ANN Performance

See the project MEMORY.md for the latest benchmark results with optimal configuration.

---

## Related Documentation

- **[README.md](README.md)** - Project overview and quickstart
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design patterns
- **[CLI_PARAMETER_GUIDE.md](CLI_PARAMETER_GUIDE.md)** - Complete CLI reference
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Installation instructions
- **[QUICK_START.md](QUICK_START.md)** - Fast-track tutorial
- **[TESTS_README.md](TESTS_README.md)** - Testing guide

---

**`QED`**

**Last Updated:** February 26, 2026<br>
**Version:** 1.1.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
