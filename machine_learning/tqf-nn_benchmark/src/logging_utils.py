"""
logging_utils.py - Logging and Progress Tracking Utilities

This module provides utilities for experiment logging, console output formatting,
and system information tracking for the TQF-NN benchmark experiments.

Key Features:
- Console output formatting with separators and structured logging
- Comprehensive experiment configuration logging (hyperparameters, system info, TQF settings)
- Hardware detection and CUDA optimization reporting
- Parameter matching verification display
- ASCII-only output (no Unicode characters for maximum compatibility)

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Date: February 2026
"""

# MIT License
#
# Copyright (c) 2026 Nathan O. Schmidt, Cold Hammer Research & Development LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import platform
from typing import Any, Dict, List
import torch

# Import centralized configuration constants
from config import (
    BATCH_SIZE_DEFAULT,
    LEARNING_RATE_DEFAULT,
    MAX_EPOCHS_DEFAULT,
    MIN_DELTA_DEFAULT,
    PATIENCE_DEFAULT,
    WEIGHT_DECAY_DEFAULT,
    LABEL_SMOOTHING_DEFAULT,
    NUM_WORKERS_DEFAULT,
    PIN_MEMORY_DEFAULT,
    DROPOUT_DEFAULT,
    TARGET_PARAMS,
    TARGET_PARAMS_TOLERANCE_PERCENT,
    TARGET_PARAMS_TOLERANCE_ABSOLUTE,
    TIMESTEPS,
    NUM_TRAIN_DEFAULT,
    NUM_VAL_DEFAULT,
    NUM_TEST_ROT_DEFAULT,
    NUM_TEST_UNROT_DEFAULT,
    LEARNING_RATE_WARMUP_EPOCHS,
    SEED_DEFAULT,
    # TQF-specific constants
    TQF_TRUNCATION_R_DEFAULT,
    TQF_RADIUS_R_FIXED,
    TQF_HIDDEN_DIMENSION_DEFAULT,
    TQF_SYMMETRY_LEVEL_DEFAULT,
    TQF_FRACTAL_ITERATIONS_DEFAULT,
    TQF_FRACTAL_DIM_TOLERANCE_DEFAULT,
    TQF_SELF_SIMILARITY_WEIGHT_DEFAULT,
    TQF_FRACTAL_EPSILON_DEFAULT,
    TQF_BOX_COUNTING_WEIGHT_DEFAULT,
    TQF_BOX_COUNTING_SCALES_DEFAULT,
    TQF_GEOMETRY_REG_WEIGHT_DEFAULT,
    TQF_HOP_ATTENTION_TEMP_DEFAULT,
    TQF_DUALITY_TOLERANCE_DEFAULT,
    TQF_VERIFY_DUALITY_INTERVAL_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT
)

###################################################################################
# OUTPUT FORMATTING ###############################################################
###################################################################################

def print_single_separator(char: str = "=", width: int = 92) -> None:
    """
    Print a single separator line for console output.

    Args:
        char: Character to use for separator (default: '=')
        width: Width of separator line in characters (default: 92)

    Why: Provides visual structure in console logs. Width of 92 chosen to fit
         standard terminal widths (80-120 chars) while accommodating typical
         experiment output line lengths.
    """
    print(char * width)


def print_separator(title: str = "", width: int = 92) -> None:
    """
    Print a titled separator with centered text for section headers.

    Args:
        title: Title text to center in separator (default: empty)
        width: Total width of separator in characters (default: 92)

    Why: Creates clear visual boundaries between major sections in experiment logs
         (e.g., "EXPERIMENT CONFIGURATION", "TRAINING PROGRESS"). Centered title
         draws attention to section transitions, improving log readability when
         scanning long outputs.

    Example output:
        ========================================================================================
        ====================================== SECTION TITLE ====================================
        ========================================================================================
    """
    print_single_separator("=", width)
    if title:
        # Calculate padding for centered title (symmetric left/right)
        padding: int = (width - len(title) - 2) // 2
        # Print centered title line with proper padding balance
        print(f"{'=' * padding} {title} {'=' * (width - padding - len(title) - 2)}")
        print_single_separator("=", width)


###################################################################################
# EXPERIMENT CONFIGURATION LOGGING ################################################
###################################################################################

def log_experiment_config(
    args: argparse.Namespace,
    seeds: List[int],
    device: torch.device,
    output_path: str = ""
) -> None:
    """
    Log comprehensive experiment configuration to console.

    This function prints a structured overview of all experiment settings including:
    - Model selection and random seeds
    - Hardware configuration (GPU, CUDA, system info)
    - Dataset sizes and data loading settings
    - Training hyperparameters (learning rate, optimizer, regularization)
    - Parameter matching constraints
    - TQF-specific architecture settings (if applicable)
    - SNN parameters (reserved for TQF-SNN extension)

    Args:
        args: Command-line arguments namespace from argparse
        seeds: List of random seeds for multi-seed experiments
        device: PyTorch device (CPU or CUDA) for training

    Why: Comprehensive logging is essential for reproducibility and debugging.
         This single function call documents the entire experiment state at startup,
         ensuring all settings are recorded before training begins. Crucial for:
         1. Reproducing exact experiments from logs
         2. Debugging unexpected results (compare configs across runs)
         3. Peer review and publication (full transparency)
         4. Identifying configuration issues before wasting compute time
    """
    # -------------------------------------------------------------------------
    # Determine which models are being run
    # -------------------------------------------------------------------------
    # Available ANN models for benchmarking
    all_models: List[str] = ['FC-MLP', 'CNN-L5', 'ResNet-18-Scaled', 'TQF-ANN']

    # Parse model selection from CLI arguments
    if hasattr(args, 'model') and args.model:
        # Single model mode (--model X)
        models_to_run: List[str] = [args.model]
    elif hasattr(args, 'models') and args.models:
        # Multiple models mode (--models X Y Z)
        models_to_run: List[str] = args.models
    else:
        # Default: Run all models
        models_to_run: List[str] = all_models

    # Check if TQF model is involved (determines whether to show TQF config section)
    is_tqf_involved: bool = any('TQF' in m for m in models_to_run)

    # Format model list for display (truncate if too many)
    selected_models_str: str = ', '.join(models_to_run) if len(models_to_run) <= 4 else f'{len(models_to_run)} models'

    # -------------------------------------------------------------------------
    # Gather system and library version information
    # -------------------------------------------------------------------------
    python_version: str = platform.python_version()
    torch_version: str = torch.__version__
    cuda_version: str = torch.version.cuda if torch.cuda.is_available() else "N/A (CPU-only)"
    cudnn_version: str = str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A"
    gpu_name: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (CPU mode)"
    cuda_available: bool = torch.cuda.is_available()

    # -------------------------------------------------------------------------
    # Get GPU memory statistics (if CUDA available)
    # -------------------------------------------------------------------------
    if cuda_available:
        torch.cuda.synchronize()
        gpu_mem_total: float = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert bytes to GB
        gpu_mem_allocated: float = torch.cuda.memory_allocated(0) / 1e9
        gpu_mem_reserved: float = torch.cuda.memory_reserved(0) / 1e9
        gpu_mem_str: str = f"{gpu_mem_total:.2f} GB total, {gpu_mem_allocated:.2f} GB allocated, {gpu_mem_reserved:.2f} GB reserved"
    else:
        gpu_mem_str: str = "N/A (CPU mode)"

    # -------------------------------------------------------------------------
    # EXPERIMENT OVERVIEW SECTION
    # -------------------------------------------------------------------------
    print_separator("EXPERIMENT CONFIGURATION")
    print(f"\nModels                             : {selected_models_str}")
    print(f"Random Seeds                       : {len(seeds)} seeds: {seeds}")
    print(f"Seed Start                         : {getattr(args, 'seed_start', SEED_DEFAULT)}")
    print(f"Device                             : {device} ({'CUDA' if cuda_available else 'CPU'})")
    print(f"Batch Size                         : {getattr(args, 'batch_size', BATCH_SIZE_DEFAULT)}")
    print(f"Max Epochs                         : {getattr(args, 'num_epochs', MAX_EPOCHS_DEFAULT)}")
    compile_enabled = getattr(args, 'compile', False)
    print(f"Torch Compile                      : {'Enabled' if compile_enabled else 'Disabled'}")
    if output_path:
        print(f"Result Output                      : {output_path}")
    else:
        print(f"Result Output                      : Disabled (--no-save-results)")

    # -------------------------------------------------------------------------
    # SYSTEM INFORMATION SECTION
    # -------------------------------------------------------------------------
    print("\nSYSTEM INFORMATION:")
    print(f"  Python                           : {python_version}")
    print(f"  PyTorch                          : {torch_version}")
    print(f"  CUDA                             : {cuda_version}")
    print(f"  cuDNN                            : {cudnn_version}")
    print(f"  GPU                              : {gpu_name}")
    print(f"  GPU Memory                       : {gpu_mem_str}")
    print(f"  Platform                         : {platform.system()} {platform.release()}")
    print(f"  Processor                        : {platform.processor()}")

    # -------------------------------------------------------------------------
    # CUDA OPTIMIZATION FLAGS (only if CUDA is available)
    # -------------------------------------------------------------------------
    # These flags can significantly affect performance and reproducibility
    if cuda_available:
        print("\nCUDA OPTIMIZATIONS:")
        print(f"  cudnn.benchmark                  : {torch.backends.cudnn.benchmark}")
        print(f"  cuda.matmul.tf32                 : {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  cudnn.tf32                       : {torch.backends.cudnn.allow_tf32}")

    # -------------------------------------------------------------------------
    # DATASET CONFIGURATION SECTION
    # -------------------------------------------------------------------------
    print("\nDATASET CONFIGURATION:")
    print(f"  Training samples                 : {getattr(args, 'num_train', NUM_TRAIN_DEFAULT)}")
    print(f"  Validation samples               : {getattr(args, 'num_val', NUM_VAL_DEFAULT)}")
    print(f"  Test (unrotated)                 : {getattr(args, 'num_test_unrot', NUM_TEST_UNROT_DEFAULT)}")
    print(f"  Test (rotated)                   : {getattr(args, 'num_test_rot', NUM_TEST_ROT_DEFAULT)} base * 6 rotations")
    print(f"  Num workers                      : {getattr(args, 'num_workers', NUM_WORKERS_DEFAULT)}")
    pin_mem = getattr(args, 'pin_memory', PIN_MEMORY_DEFAULT)
    print(f"  Pin memory                       : {'Enabled' if pin_mem else 'Disabled'}")

    # -------------------------------------------------------------------------
    # OPTIMIZATION HYPERPARAMETERS SECTION
    # -------------------------------------------------------------------------
    print("\nOPTIMIZATION HYPERPARAMETERS:")
    print(f"  Learning rate                    : {getattr(args, 'learning_rate', LEARNING_RATE_DEFAULT)}")
    print(f"  LR warmup epochs                 : {getattr(args, 'learning_rate_warmup_epochs', LEARNING_RATE_WARMUP_EPOCHS)}")
    # Compute T_max for cosine annealing (excludes warmup epochs)
    t_max: int = getattr(args, 'num_epochs', MAX_EPOCHS_DEFAULT) - getattr(args, 'learning_rate_warmup_epochs', LEARNING_RATE_WARMUP_EPOCHS)
    print(f"  LR scheduler                     : CosineAnnealingLR (T_max={t_max}, eta_min=1e-6)")
    print(f"  Weight decay                     : {getattr(args, 'weight_decay', WEIGHT_DECAY_DEFAULT)}")
    print(f"  Label smoothing                  : {getattr(args, 'label_smoothing', LABEL_SMOOTHING_DEFAULT)}")
    print(f"  Patience (early stop)            : {getattr(args, 'patience', PATIENCE_DEFAULT)}")
    print(f"  Min delta                        : {getattr(args, 'min_delta', MIN_DELTA_DEFAULT)}")
    print(f"  Gradient clip norm               : 1.0")
    print(f"  Optimizer                        : Adam")
    print(f"  AMP (mixed precision)            : {'Enabled' if cuda_available else 'Disabled (CPU mode)'}")

    # -------------------------------------------------------------------------
    # PARAMETER MATCHING CONFIGURATION SECTION
    # -------------------------------------------------------------------------
    # Critical for ensuring fair "apples-to-apples" model comparison
    print("\nPARAMETER MATCHING:")
    print(f"  Target parameters                : {TARGET_PARAMS:,}")
    print(f"  Tolerance (percent)              : +/- {TARGET_PARAMS_TOLERANCE_PERCENT}%")
    print(f"  Tolerance (absolute)             : +/- {TARGET_PARAMS_TOLERANCE_ABSOLUTE:,} parameters")

    # -------------------------------------------------------------------------
    # TQF-SPECIFIC CONFIGURATION SECTION (only if TQF model is being run)
    # -------------------------------------------------------------------------
    if is_tqf_involved:
        print("\nTQF-SPECIFIC CONFIGURATION:")

        # Determine hidden dimension (may be auto-tuned or manual)
        tqf_hidden: int = getattr(args, 'tqf_hidden_dim', None) or TQF_HIDDEN_DIMENSION_DEFAULT
        auto_tuned: bool = getattr(args, 'tqf_hidden_dim', None) is None
        tqf_hidden_str: str = f"{tqf_hidden}" + (" (auto-tuned)" if auto_tuned else " (manual)")

        # Lattice-Graph Structure Parameters
        print("\n  Lattice-Graph Structure:")
        print(f"    Truncation radius (R)          : {getattr(args, 'tqf_R', TQF_TRUNCATION_R_DEFAULT)}")
        print(f"    Inversion radius (r) [FIXED]   : {TQF_RADIUS_R_FIXED}")
        print(f"    Hidden dimension               : {tqf_hidden_str}")
        sym_level = getattr(args, 'tqf_symmetry_level', TQF_SYMMETRY_LEVEL_DEFAULT)
        # Describe orbit pooling operations for each symmetry level
        sym_ops_map = {
            'none': '(no orbit pooling)',
            'Z6': '(6 rotation orbits)',
            'D6': '(12 D6 orbits: rotations + reflections)',
            'T24': '(24 T24 orbits: rotations + reflections + inversion)'
        }
        sym_ops = sym_ops_map.get(sym_level, '')
        print(f"    Symmetry level                 : {sym_level} {sym_ops}")

        # Fractal Parameters
        print("\n  Fractal Parameters:")
        print(f"    Fractal iterations             : {getattr(args, 'tqf_fractal_iterations', TQF_FRACTAL_ITERATIONS_DEFAULT)}")
        print(f"    Fractal dim tolerance          : {TQF_FRACTAL_DIM_TOLERANCE_DEFAULT} (internal default)")
        print(f"    Fractal epsilon                : {TQF_FRACTAL_EPSILON_DEFAULT}")
        print(f"    Self-similarity weight         : {getattr(args, 'tqf_self_similarity_weight', TQF_SELF_SIMILARITY_WEIGHT_DEFAULT)}")
        print(f"    Box-counting weight            : {getattr(args, 'tqf_box_counting_weight', TQF_BOX_COUNTING_WEIGHT_DEFAULT)}")
        print(f"    Box-counting scales            : {TQF_BOX_COUNTING_SCALES_DEFAULT} (internal default)")

        # Dual Metrics & Geometry Parameters
        print("\n  Dual Metrics & Geometry:")
        print(f"    Geometry regularization weight : {getattr(args, 'tqf_geometry_reg_weight', TQF_GEOMETRY_REG_WEIGHT_DEFAULT)}")
        # Inversion (duality) loss - enabled by providing weight value via CLI
        inv_weight = getattr(args, 'tqf_inversion_loss_weight', None)
        inv_status = f"Enabled (weight={inv_weight})" if inv_weight is not None else "Disabled"
        print(f"    Inversion (duality) loss       : {inv_status}")
        # Hop attention temperature: controls neighbor aggregation sharpness
        # - < 1.0: Sharp attention, prefer similar neighbors
        # - = 1.0: Uniform mean pooling (standard GNN)
        # - > 1.0: Smooth attention, more uniform weighting
        hop_temp = getattr(args, 'tqf_hop_attention_temp', TQF_HOP_ATTENTION_TEMP_DEFAULT)
        hop_mode = "sharp" if hop_temp < 1.0 else ("uniform" if hop_temp == 1.0 else "smooth")
        print(f"    Hop attention temperature      : {hop_temp} ({hop_mode})")

        print(f"    Duality tolerance              : {TQF_DUALITY_TOLERANCE_DEFAULT}")
        print(f"    Duality verify interval        : {getattr(args, 'tqf_verify_duality_interval', TQF_VERIFY_DUALITY_INTERVAL_DEFAULT)} epochs")
        verify_geom = getattr(args, 'tqf_verify_geometry', False)
        verify_geom_status = "Enabled (verifies fractal losses active)" if verify_geom else "Disabled"
        print(f"    Verify geometry flag           : {verify_geom_status}")

        # Symmetry/Invariance/Equivariance Losses
        # These losses are enabled by providing a weight value via CLI (None = disabled)
        print("\n  Symmetry/Invariance/Equivariance Losses:")
        z6_weight = getattr(args, 'tqf_z6_equivariance_weight', None)
        z6_status = f"Enabled (weight={z6_weight})" if z6_weight is not None else "Disabled"
        print(f"    Z6 equivariance loss           : {z6_status}")
        d6_weight = getattr(args, 'tqf_d6_equivariance_weight', None)
        d6_status = f"Enabled (weight={d6_weight})" if d6_weight is not None else "Disabled"
        print(f"    D6 equivariance loss           : {d6_status}")
        t24_weight = getattr(args, 'tqf_t24_orbit_invariance_weight', None)
        t24_status = f"Enabled (weight={t24_weight})" if t24_weight is not None else "Disabled"
        print(f"    T24 orbit invariance loss      : {t24_status}")

        # Orbit Mixing (evaluation-time ensemble)
        print("\n  Orbit Mixing (Evaluation-Time):")
        use_z6_om = getattr(args, 'tqf_use_z6_orbit_mixing', False)
        use_d6_om = getattr(args, 'tqf_use_d6_orbit_mixing', False)
        use_t24_om = getattr(args, 'tqf_use_t24_orbit_mixing', False)
        any_om = use_z6_om or use_d6_om or use_t24_om
        print(f"    Z6 orbit mixing                : {'Enabled' if use_z6_om else 'Disabled'}")
        print(f"    D6 orbit mixing                : {'Enabled' if use_d6_om else 'Disabled'}")
        print(f"    T24 orbit mixing               : {'Enabled' if use_t24_om else 'Disabled'}")
        if any_om:
            print(f"    Temp (rotation)                : {getattr(args, 'tqf_orbit_mixing_temp_rotation', TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT)}")
            print(f"    Temp (reflection)              : {getattr(args, 'tqf_orbit_mixing_temp_reflection', TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT)}")
            print(f"    Temp (inversion)               : {getattr(args, 'tqf_orbit_mixing_temp_inversion', TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT)}")

        # Memory Optimization
        print("\n  Memory Optimization:")
        use_grad_ckpt = getattr(args, 'tqf_use_gradient_checkpointing', False)
        print(f"    Gradient checkpointing         : {'Enabled' if use_grad_ckpt else 'Disabled'}")

    # -------------------------------------------------------------------------
    # SNN-SPECIFIC CONFIGURATION SECTION (reserved for TQF-SNN extension)
    # -------------------------------------------------------------------------
    print("\nSNN-SPECIFIC CONFIGURATION (Reserved for TQF-SNN):")
    print(f"  Timesteps                        : {TIMESTEPS} (reserved for spiking implementation)")

    # -------------------------------------------------------------------------
    # DATA AUGMENTATION & PREPROCESSING SECTION
    # -------------------------------------------------------------------------
    print("\nDATA AUGMENTATION & PREPROCESSING:")
    print(f"  Rotation angles                  : [0, 60, 120, 180, 240, 300] deg (Z6-aligned)")
    print(f"  Interpolation mode               : BICUBIC")
    print(f"  Normalization                    : mean=0.1307, std=0.3081 (MNIST standard)")
    print(f"  Stratified sampling              : Enabled (balanced classes)")
    use_z6_data_aug = getattr(args, 'z6_data_augmentation', False)
    print(f"  Z6 data augmentation             : {'Enabled' if use_z6_data_aug else 'Disabled'}")

    # -------------------------------------------------------------------------
    # REPRODUCIBILITY CONFIGURATION SECTION
    # -------------------------------------------------------------------------
    print("\nREPRODUCIBILITY:")
    print(f"  Fixed seeds                      : {seeds}")
    print(f"  Deterministic algos              : Enabled where possible")
    print(f"  Dropout                          : {getattr(args, 'dropout', DROPOUT_DEFAULT)}")

    # End of configuration logging
    print_single_separator(width=92)
    print()  # Extra newline for visual separation from subsequent output
