"""
cli.py - Command-Line Interface for TQF-NN Experiments

This module centralizes all argument parsing, validation, and help text for the
TQF-NN benchmark suite. Extracted from main.py to improve maintainability and
enable consistent validation across all CLI inputs.

Key Features:
- Comprehensive Argument Parser: 40+ CLI arguments for hyperparameters, architecture, datasets
- Model Selection: Support for 'all' keyword or individual model names (FC-MLP, CNN-L5, ResNet-18-Scaled, TQF-ANN)
- TQF Configuration: Lattice radius, hidden dimensions, symmetry level, fractal parameters
- Training Hyperparameters: Learning rate, batch size, epochs, weight decay, label smoothing
- Dataset Configuration: Configurable train/val/test split sizes for MNIST
- Validation: Range checking for all numeric parameters with clear error messages
- Default Values: Imported from config.py for consistency across the codebase
- Help Text: Detailed descriptions with valid ranges and scientific rationale
- Logging Setup: Configurable console-only logging (no file output)
- Reproducibility: Random seed configuration with multi-seed support

Validation Approach:
All numeric arguments have explicit MIN/MAX constants (e.g., TQF_R_MIN=2, TQF_R_MAX=100)
that are checked in _validate_args() to catch invalid inputs before training begins.
This prevents cryptic runtime errors and provides immediate feedback to users.

Usage:
    from cli import parse_args, setup_logging
    args = parse_args()  # Automatically validates all inputs
    setup_logging()       # Configure console logging

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
import sys
from typing import List
from config import (
    BATCH_SIZE_DEFAULT,
    DEFAULT_RESULTS_DIR,
    MAX_EPOCHS_DEFAULT,
    NUM_SEEDS_DEFAULT,
    SEED_DEFAULT,
    TQF_SYMMETRY_LEVEL_DEFAULT,
    TQF_TRUNCATION_R_DEFAULT,
    TQF_SELF_SIMILARITY_WEIGHT_DEFAULT,
    TQF_BOX_COUNTING_WEIGHT_DEFAULT,
    TQF_HOP_ATTENTION_TEMP_DEFAULT,
    LEARNING_RATE_DEFAULT,
    WEIGHT_DECAY_DEFAULT,
    LABEL_SMOOTHING_DEFAULT,
    PATIENCE_DEFAULT,
    MIN_DELTA_DEFAULT,
    LEARNING_RATE_WARMUP_EPOCHS,
    NUM_TRAIN_DEFAULT,
    NUM_VAL_DEFAULT,
    NUM_TEST_ROT_DEFAULT,
    NUM_TEST_UNROT_DEFAULT,
    TQF_GEOMETRY_REG_WEIGHT_DEFAULT,
    TQF_VERIFY_DUALITY_INTERVAL_DEFAULT,
    Z6_DATA_AUGMENTATION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT,
    # Numeric range constants (single source of truth in config.py)
    NUM_SEEDS_MIN, NUM_SEEDS_MAX,
    SEED_START_MIN,
    NUM_EPOCHS_MIN, NUM_EPOCHS_MAX,
    BATCH_SIZE_MIN, BATCH_SIZE_MAX,
    LEARNING_RATE_MIN, LEARNING_RATE_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX,
    LABEL_SMOOTHING_MIN, LABEL_SMOOTHING_MAX,
    PATIENCE_MIN, PATIENCE_MAX,
    MIN_DELTA_MIN, MIN_DELTA_MAX,
    LEARNING_RATE_WARMUP_EPOCHS_MIN, LEARNING_RATE_WARMUP_EPOCHS_MAX,
    NUM_TRAIN_MIN, NUM_TRAIN_MAX,
    NUM_VAL_MIN, NUM_VAL_MAX,
    NUM_TEST_ROT_MIN, NUM_TEST_ROT_MAX,
    NUM_TEST_UNROT_MIN, NUM_TEST_UNROT_MAX,
    TQF_R_MIN, TQF_R_MAX,
    TQF_HIDDEN_DIM_MIN, TQF_HIDDEN_DIM_MAX,
    TQF_FRACTAL_ITERATIONS_MIN, TQF_FRACTAL_ITERATIONS_MAX,
    TQF_SELF_SIMILARITY_WEIGHT_MIN, TQF_SELF_SIMILARITY_WEIGHT_MAX,
    TQF_BOX_COUNTING_WEIGHT_MIN, TQF_BOX_COUNTING_WEIGHT_MAX,
    TQF_HOP_ATTENTION_TEMP_MIN, TQF_HOP_ATTENTION_TEMP_MAX,
    TQF_ORBIT_MIXING_TEMP_MIN, TQF_ORBIT_MIXING_TEMP_MAX,
    TQF_GEOMETRY_REG_WEIGHT_MIN, TQF_GEOMETRY_REG_WEIGHT_MAX,
    TQF_INVERSION_LOSS_WEIGHT_MIN, TQF_INVERSION_LOSS_WEIGHT_MAX,
    TQF_Z6_EQUIVARIANCE_WEIGHT_MIN, TQF_Z6_EQUIVARIANCE_WEIGHT_MAX,
    TQF_D6_EQUIVARIANCE_WEIGHT_MIN, TQF_D6_EQUIVARIANCE_WEIGHT_MAX,
    TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN, TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX,
    TQF_VERIFY_DUALITY_INTERVAL_MIN, TQF_VERIFY_DUALITY_INTERVAL_MAX
)


###################################################################################
# HELPER FUNCTIONS ################################################################
###################################################################################

def get_all_model_names() -> List[str]:
    """
    Get list of all available model names.

    Returns:
        List of all model names (includes TQF-ANN via lazy loading)
    """
    # TQF-ANN is not in MODEL_REGISTRY to avoid slow import,
    # but is available via get_model() with lazy loading
    from models_baseline import MODEL_REGISTRY
    return list(MODEL_REGISTRY.keys()) + ['TQF-ANN']


###################################################################################
# ARGUMENT PARSING ################################################################
###################################################################################

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments with comprehensive error checking.

    Why: Provides user-friendly CLI with extensive help text, default values,
         validation for all hyperparameters, and TQF-specific settings.
         Enables reproducible experiments via argument logging.

    Returns:
        Validated arguments namespace

    Raises:
        SystemExit: If validation fails (with helpful error message)
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='TQF-NN: Tri-Quarter Framework Neural Networks Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models (default when --models not specified)
  python main.py

  # Train all models explicitly
  python main.py --models all

  # Train only TQF-ANN with Z6 symmetry
  python main.py --models TQF-ANN --num-seeds 3

  # Train specific baseline models in order
  python main.py --models FC-MLP CNN-L5 --batch-size 128

  # Train all models with custom epochs
  python main.py --models all --num-epochs 50

  # Ablation: TQF with no symmetry
  python main.py --models TQF-ANN --tqf-symmetry-level none

Result Output:
  Results are automatically saved to data/output/results_YYYYMMDD_HHMMSS.json
  after each seed completes (incremental saves survive crashes/interruptions).
  A human-readable .txt summary is also generated when the experiment finishes.
  The output path is displayed in the experiment configuration banner at startup.
  Use --no-save-results to disable, or --results-dir to change the output directory.
        """
    )

    # =========================================================================
    # GENERAL SETTINGS: Model and Device Selection
    # =========================================================================
    general_group = parser.add_argument_group(
        'General Settings',
        'Model selection and compute device configuration'
    )

    available_models: List[str] = get_all_model_names()
    general_group.add_argument(
        '--models',
        type=str,
        nargs='*',  # Changed from '+' to '*' to allow no arguments
        default=None,  # None means no argument provided -> use all
        help=f'Models to train and evaluate. '
             f'Choices: {available_models} or "all". '
             f'Default: all models (when --models not specified). '
             f'Use "all" to explicitly train all models. '
             f'Specify individual models to train only those in the given order.'
    )

    general_group.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (cuda, cpu, or auto). Default: auto (cuda if available, else cpu).'
    )

    general_group.add_argument(
        '--compile',
        action='store_true',
        default=False,
        help='Enable torch.compile for kernel fusion (PyTorch 2.0+). '
             'Provides additional speedup after initial compilation warmup. '
             'Requires Triton (Linux only; skipped on Windows). '
             'Default: disabled.'
    )

    # =========================================================================
    # RESULT OUTPUT SETTINGS
    # =========================================================================
    output_group = parser.add_argument_group(
        'Result Output',
        'Persistent result file saving configuration'
    )

    output_group.add_argument(
        '--no-save-results',
        action='store_true',
        default=False,
        help='Disable saving results to disk. By default, results are saved '
             'incrementally to data/output/ as JSON (per-seed) and TXT (final summary). '
             'Use this flag for quick test runs where persistent output is not needed.'
    )

    output_group.add_argument(
        '--results-dir',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f'Directory for result output files. '
             f'Default: {DEFAULT_RESULTS_DIR} (project root). '
             f'The directory will be created if it does not exist. '
             f'Ignored when --no-save-results is set.'
    )

    # =========================================================================
    # REPRODUCIBILITY SETTINGS
    # =========================================================================
    repro_group = parser.add_argument_group(
        'Reproducibility',
        'Random seed configuration for reproducible experiments'
    )

    repro_group.add_argument(
        '--num-seeds',
        type=int,
        default=NUM_SEEDS_DEFAULT,
        help=f'Number of random seeds to run. Range: [{NUM_SEEDS_MIN}, {NUM_SEEDS_MAX}]. '
             f'Default: {NUM_SEEDS_DEFAULT}. '
             f'Statistical significance requires >= 3 seeds.'
    )

    repro_group.add_argument(
        '--seed-start',
        type=int,
        default=SEED_DEFAULT,
        help=f'Starting random seed. Range: >={SEED_START_MIN}. '
             f'Default: {SEED_DEFAULT}. '
             f'Seeds will be consecutive from this value.'
    )

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    train_group = parser.add_argument_group(
        'Training Hyperparameters',
        'Core training configuration (epochs, batch size, optimizer settings)'
    )

    train_group.add_argument(
        '--num-epochs',
        type=int,
        default=MAX_EPOCHS_DEFAULT,
        help=f'Maximum training epochs. Range: [{NUM_EPOCHS_MIN}, {NUM_EPOCHS_MAX}]. '
             f'Default: {MAX_EPOCHS_DEFAULT}. '
             f'Early stopping may terminate before this.'
    )

    train_group.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f'Batch size for training. Range: [{BATCH_SIZE_MIN}, {BATCH_SIZE_MAX}]. '
             f'Default: {BATCH_SIZE_DEFAULT}. '
             f'Powers of 2 recommended for GPU efficiency.'
    )

    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=LEARNING_RATE_DEFAULT,
        help=f'Learning rate for optimizer. Range: ({LEARNING_RATE_MIN}, {LEARNING_RATE_MAX}]. '
             f'Default: {LEARNING_RATE_DEFAULT}. '
             f'Cosine annealing with warmup applied.'
    )

    train_group.add_argument(
        '--learning-rate-warmup-epochs',
        type=int,
        default=LEARNING_RATE_WARMUP_EPOCHS,
        help=f'Number of warmup epochs for learning rate. Range: [{LEARNING_RATE_WARMUP_EPOCHS_MIN}, {LEARNING_RATE_WARMUP_EPOCHS_MAX}]. '
             f'Default: {LEARNING_RATE_WARMUP_EPOCHS}. '
             f'Linear warmup from 0 to learning_rate over this many epochs.'
    )

    train_group.add_argument(
        '--weight-decay',
        type=float,
        default=WEIGHT_DECAY_DEFAULT,
        help=f'L2 regularization weight decay. Range: [{WEIGHT_DECAY_MIN}, {WEIGHT_DECAY_MAX}]. '
             f'Default: {WEIGHT_DECAY_DEFAULT}.'
    )

    train_group.add_argument(
        '--label-smoothing',
        type=float,
        default=LABEL_SMOOTHING_DEFAULT,
        help=f'Label smoothing factor. Range: [{LABEL_SMOOTHING_MIN}, {LABEL_SMOOTHING_MAX}]. '
             f'Default: {LABEL_SMOOTHING_DEFAULT}. '
             f'0.0 = no smoothing, 0.1 = 10%% smoothing.'
    )

    # =========================================================================
    # EARLY STOPPING
    # =========================================================================
    early_stop_group = parser.add_argument_group(
        'Early Stopping',
        'Validation-based early stopping configuration'
    )

    early_stop_group.add_argument(
        '--patience',
        type=int,
        default=PATIENCE_DEFAULT,
        help=f'Early stopping patience (epochs). Range: [{PATIENCE_MIN}, {PATIENCE_MAX}]. '
             f'Default: {PATIENCE_DEFAULT}. '
             f'Stops if validation loss does not improve for this many epochs.'
    )

    early_stop_group.add_argument(
        '--min-delta',
        type=float,
        default=MIN_DELTA_DEFAULT,
        help=f'Early stopping minimum improvement. Range: [{MIN_DELTA_MIN}, {MIN_DELTA_MAX}]. '
             f'Default: {MIN_DELTA_DEFAULT}. '
             f'Improvement must exceed this to reset patience counter.'
    )

    # =========================================================================
    # DATASET SIZE CONFIGURATION
    # =========================================================================
    data_group = parser.add_argument_group(
        'Dataset Configuration',
        'Training, validation, and test set sizes'
    )

    data_group.add_argument(
        '--num-train',
        type=int,
        default=NUM_TRAIN_DEFAULT,
        help=f'Number of training samples. Range: [{NUM_TRAIN_MIN}, {NUM_TRAIN_MAX}]. '
             f'Default: {NUM_TRAIN_DEFAULT}. '
             f'Must be divisible by 10 for balanced class distribution.'
    )

    data_group.add_argument(
        '--num-val',
        type=int,
        default=NUM_VAL_DEFAULT,
        help=f'Number of validation samples. Range: [{NUM_VAL_MIN}, {NUM_VAL_MAX}]. '
             f'Default: {NUM_VAL_DEFAULT}. '
             f'Must be divisible by 10 for balanced class distribution.'
    )

    data_group.add_argument(
        '--num-test-rot',
        type=int,
        default=NUM_TEST_ROT_DEFAULT,
        help=f'Number of rotated test samples. Range: [{NUM_TEST_ROT_MIN}, {NUM_TEST_ROT_MAX}]. '
             f'Default: {NUM_TEST_ROT_DEFAULT}. '
             f'Primary metric for rotation invariance evaluation.'
    )

    data_group.add_argument(
        '--num-test-unrot',
        type=int,
        default=NUM_TEST_UNROT_DEFAULT,
        help=f'Number of unrotated test samples. Range: [{NUM_TEST_UNROT_MIN}, {NUM_TEST_UNROT_MAX}]. '
             f'Default: {NUM_TEST_UNROT_DEFAULT}. '
             f'Baseline metric for standard accuracy.'
    )

    # =========================================================================
    # TQF CORE ARCHITECTURE PARAMETERS
    # =========================================================================
    tqf_arch_group = parser.add_argument_group(
        'TQF Architecture (TQF-ANN only)',
        'Core TQF-ANN architecture settings: lattice size and symmetry'
    )

    tqf_arch_group.add_argument(
        '--tqf-R',
        type=int,
        default=TQF_TRUNCATION_R_DEFAULT,
        help=f'Truncation radius R (radial lattice size). '
             f'Range: [{TQF_R_MIN}, {TQF_R_MAX}]. '
             f'Default: {TQF_TRUNCATION_R_DEFAULT}. '
             f'Number of hexagonal nodes approximately proportional to R^2.'
    )

    tqf_arch_group.add_argument(
        '--tqf-hidden-dim',
        type=int,
        default=None,  # Auto-tuned during parameter matching if None
        help=f'Hidden dimension. '
             f'Range: [{TQF_HIDDEN_DIM_MIN}, {TQF_HIDDEN_DIM_MAX}]. '
             f'Default: None (auto-tuned to match target parameter count). '
             f'Manually set to override auto-tuning.'
    )

    tqf_arch_group.add_argument(
        '--tqf-symmetry-level',
        type=str,
        default=TQF_SYMMETRY_LEVEL_DEFAULT,
        choices=['none', 'Z6', 'D6', 'T24'],
        help=f'Symmetry group for equivariant orbit pooling. '
             f'Choices: none (fastest, no pooling), Z6 (6x cost, rotation invariance), '
             f'D6 (12x cost, +reflections), T24 (24x cost, full symmetry). '
             f'Default: {TQF_SYMMETRY_LEVEL_DEFAULT}.'
    )

    tqf_arch_group.add_argument(
        '--z6-data-augmentation',
        action='store_true',
        dest='z6_data_augmentation',
        default=Z6_DATA_AUGMENTATION_DEFAULT,
        help='Enable Z6-aligned rotation augmentation during training for all models. '
             'When enabled, random rotations at 60-degree intervals (with +/-15 deg jitter) '
             'are applied to training images to teach rotation robustness. '
             'Note: conflicts with orbit mixing features; avoid using both together. '
             'Default: augmentation disabled (False).'
    )

    # =========================================================================
    # TQF EQUIVARIANCE LOSSES
    # =========================================================================
    # Equivariance: f(g·x) = g·f(x) - features transform correctly with input
    # These losses are DISABLED by default. Providing a weight value enables
    # the corresponding loss feature.
    tqf_equiv_group = parser.add_argument_group(
        'TQF Equivariance Losses (TQF-ANN only)',
        'Z6/D6 equivariance losses enforce f(transform(x)) = transform(f(x)). '
        'Disabled by default. Provide a weight value to enable.'
    )

    tqf_equiv_group.add_argument(
        '--tqf-z6-equivariance-weight',
        type=float,
        default=None,
        help=f'Enable and set weight for Z6 rotation equivariance loss. '
             f'Enforces f(rotate(x)) = rotate(f(x)) for 60-degree rotations. '
             f'Disabled by default. Provide a value in range '
             f'[{TQF_Z6_EQUIVARIANCE_WEIGHT_MIN}, {TQF_Z6_EQUIVARIANCE_WEIGHT_MAX}] to enable.'
    )

    tqf_equiv_group.add_argument(
        '--tqf-d6-equivariance-weight',
        type=float,
        default=None,
        help=f'Enable and set weight for D6 reflection equivariance loss. '
             f'Enforces f(reflect(x)) = reflect(f(x)) for reflections. '
             f'Disabled by default. Provide a value in range '
             f'[{TQF_D6_EQUIVARIANCE_WEIGHT_MIN}, {TQF_D6_EQUIVARIANCE_WEIGHT_MAX}] to enable.'
    )

    # =========================================================================
    # TQF INVARIANCE LOSSES
    # =========================================================================
    # Invariance: f(g·x) = f(x) - predictions unchanged under transformation
    # These losses are DISABLED by default. Providing a weight value enables
    # the corresponding loss feature.
    tqf_inv_group = parser.add_argument_group(
        'TQF Invariance Losses (TQF-ANN only)',
        'T24 orbit invariance enforces f(transform(x)) = f(x) for all 24 T24 operations. '
        'Disabled by default. Provide a weight value to enable.'
    )

    tqf_inv_group.add_argument(
        '--tqf-t24-orbit-invariance-weight',
        type=float,
        default=None,
        help=f'Enable and set weight for T24 orbit invariance loss. '
             f'Enforces prediction invariance across all 24 T24 symmetry operations. '
             f'Strongest symmetry constraint. Recommended for optimal symmetry enforcement. '
             f'Disabled by default. Provide a value in range '
             f'[{TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN}, {TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX}] to enable.'
    )

    # =========================================================================
    # TQF DUALITY LOSSES
    # =========================================================================
    # Duality: circle inversion bijection consistency (primal <-> dual lattice)
    # This loss is DISABLED by default. Providing a weight value enables it.
    tqf_duality_group = parser.add_argument_group(
        'TQF Duality Losses (TQF-ANN only)',
        'Circle inversion duality consistency loss (opt-in). '
        'Disabled by default. Provide a weight value to enable.'
    )

    tqf_duality_group.add_argument(
        '--tqf-inversion-loss-weight',
        type=float,
        default=None,
        help=f'Enable and set weight for circle inversion duality consistency loss. '
             f'Penalizes inconsistency between primal and dual lattice representations. '
             f'Disabled by default. Provide a value in range '
             f'[{TQF_INVERSION_LOSS_WEIGHT_MIN}, {TQF_INVERSION_LOSS_WEIGHT_MAX}] to enable.'
    )

    tqf_duality_group.add_argument(
        '--tqf-verify-duality-interval',
        type=int,
        default=TQF_VERIFY_DUALITY_INTERVAL_DEFAULT,
        help=f'Frequency (in epochs) for self-duality verification. '
             f'Range: [{TQF_VERIFY_DUALITY_INTERVAL_MIN}, num_epochs]. '
             f'Default: {TQF_VERIFY_DUALITY_INTERVAL_DEFAULT}. '
             f'Verifies circle inversion preserves mathematical duality.'
    )

    # =========================================================================
    # TQF ORBIT MIXING (Evaluation-Time)
    # =========================================================================
    # Orbit mixing averages predictions across symmetry-transformed inputs/features
    # at evaluation time. Z6 uses input-space rotation (6 full forward passes),
    # D6 adds feature-space reflections, T24 adds zone-swap (inner <-> outer).
    # All disabled by default — evaluation uses single forward pass.
    tqf_orbit_group = parser.add_argument_group(
        'TQF Orbit Mixing (TQF-ANN only, evaluation-time)',
        'Evaluation-time prediction averaging over symmetry orbits. '
        'Z6: 6 input-space rotations. D6: +6 feature-space reflections. '
        'T24: +12 zone-swap variants. All disabled by default.'
    )

    tqf_orbit_group.add_argument(
        '--tqf-use-z6-orbit-mixing',
        action='store_true',
        default=False,
        help='Enable Z6 orbit mixing at evaluation time. '
             'Averages predictions over 6 input-space rotations (0, 60, ..., 300 deg). '
             'Cost: ~6x inference time. Default: False.'
    )

    tqf_orbit_group.add_argument(
        '--tqf-use-d6-orbit-mixing',
        action='store_true',
        default=False,
        help='Enable D6 orbit mixing at evaluation time. '
             'Includes Z6 rotations plus 6 feature-space reflections. '
             'Cost: ~6x full forward + 6 lightweight head passes. Default: False.'
    )

    tqf_orbit_group.add_argument(
        '--tqf-use-t24-orbit-mixing',
        action='store_true',
        default=False,
        help='Enable T24 orbit mixing at evaluation time. '
             'Includes D6 operations plus inner/outer zone-swap variants. '
             'Cost: ~6x full forward + 18 lightweight head passes. Default: False.'
    )

    tqf_orbit_group.add_argument(
        '--tqf-orbit-mixing-temp-rotation',
        type=float,
        default=TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
        help='Temperature for Z6 rotation averaging. '
             'Lower = sharper (most confident rotation dominates). '
             f'Range: [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]. '
             f'Default: {TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT}.'
    )

    tqf_orbit_group.add_argument(
        '--tqf-orbit-mixing-temp-reflection',
        type=float,
        default=TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
        help='Temperature for D6 reflection averaging. '
             'Softer than rotation because some digits are asymmetric under reflection. '
             f'Range: [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]. '
             f'Default: {TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT}.'
    )

    tqf_orbit_group.add_argument(
        '--tqf-orbit-mixing-temp-inversion',
        type=float,
        default=TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT,
        help='Temperature for T24 inversion (zone-swap) averaging. '
             'Softest because circle inversion is the most abstract symmetry. '
             f'Range: [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]. '
             f'Default: {TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT}.'
    )

    # =========================================================================
    # TQF GEOMETRY REGULARIZATION
    # =========================================================================
    tqf_geom_group = parser.add_argument_group(
        'TQF Geometry Regularization (TQF-ANN only)',
        'Geometry verification and regularization settings'
    )

    tqf_geom_group.add_argument(
        '--tqf-verify-geometry',
        action='store_true',
        default=False,
        help='Enable geometry verification and regularization. '
             'Adds self-similarity and box-counting loss terms.'
    )

    tqf_geom_group.add_argument(
        '--tqf-geometry-reg-weight',
        type=float,
        default=TQF_GEOMETRY_REG_WEIGHT_DEFAULT,
        help=f'Overall geometry regularization weight. '
             f'Range: [{TQF_GEOMETRY_REG_WEIGHT_MIN}, {TQF_GEOMETRY_REG_WEIGHT_MAX}]. '
             f'Default: {TQF_GEOMETRY_REG_WEIGHT_DEFAULT} (disabled). '
             f'Recommended: 0.001-0.01 when enabled.'
    )

    # =========================================================================
    # TQF FRACTAL GEOMETRY PARAMETERS
    # =========================================================================
    tqf_fractal_group = parser.add_argument_group(
        'TQF Fractal Geometry (TQF-ANN only)',
        'Fractal dimension estimation and self-similarity settings'
    )

    tqf_fractal_group.add_argument(
        '--tqf-fractal-iterations',
        type=int,
        default=None,
        help=f'Enable and set iterations for fractal dimension estimation. '
             f'DISABLED by default (opt-in feature). '
             f'Provide a value in range [{TQF_FRACTAL_ITERATIONS_MIN}, {TQF_FRACTAL_ITERATIONS_MAX}] to enable. '
             f'Creates N fractal mixer layers and enables multi-scale self-similarity. '
             f'Recommended starting value: 5 (balanced). Higher values improve accuracy but slow training.'
    )

    # NOTE: --tqf-fractal-dim-tolerance removed (consolidated as internal constant
    # TQF_FRACTAL_DIM_TOLERANCE_DEFAULT=0.08 in config.py, not user-tunable)

    tqf_fractal_group.add_argument(
        '--tqf-self-similarity-weight',
        type=float,
        default=TQF_SELF_SIMILARITY_WEIGHT_DEFAULT,
        help=f'Weight for self-similarity fractal loss. '
             f'Range: [{TQF_SELF_SIMILARITY_WEIGHT_MIN}, {TQF_SELF_SIMILARITY_WEIGHT_MAX}]. '
             f'Default: {TQF_SELF_SIMILARITY_WEIGHT_DEFAULT} (disabled). '
             f'Recommended: 0.0001-0.001 when enabled.'
    )

    tqf_fractal_group.add_argument(
        '--tqf-box-counting-weight',
        type=float,
        default=TQF_BOX_COUNTING_WEIGHT_DEFAULT,
        help=f'Weight for box-counting fractal loss. '
             f'Range: [{TQF_BOX_COUNTING_WEIGHT_MIN}, {TQF_BOX_COUNTING_WEIGHT_MAX}]. '
             f'Default: {TQF_BOX_COUNTING_WEIGHT_DEFAULT} (disabled). '
             f'Recommended: 0.0001-0.001 when enabled.'
    )

    # NOTE: --tqf-box-counting-scales removed (consolidated as internal constant
    # TQF_BOX_COUNTING_SCALES_DEFAULT=10 in config.py, not user-tunable)

    # =========================================================================
    # TQF MEMORY OPTIMIZATION PARAMETERS
    # =========================================================================
    tqf_mem_group = parser.add_argument_group(
        'TQF Memory Optimization (TQF-ANN only)',
        'Settings for reducing GPU memory usage with large R values'
    )

    tqf_mem_group.add_argument(
        '--tqf-use-gradient-checkpointing',
        action='store_true',
        default=False,
        help='Enable gradient checkpointing to reduce memory usage during training. '
             'Trades more compute time for less activation memory. '
             'Recommended for large R values (R >= 15) on memory-constrained GPUs. '
             'Default: False.'
    )

    # =========================================================================
    # TQF ATTENTION PARAMETERS
    # =========================================================================
    tqf_attn_group = parser.add_argument_group(
        'TQF Attention (TQF-ANN only)',
        'Temperature settings'
    )

    tqf_attn_group.add_argument(
        '--tqf-hop-attention-temp',
        type=float,
        default=TQF_HOP_ATTENTION_TEMP_DEFAULT,
        help=f'Temperature for hop-distance attention. '
             f'Range: [{TQF_HOP_ATTENTION_TEMP_MIN}, {TQF_HOP_ATTENTION_TEMP_MAX}]. '
             f'Default: {TQF_HOP_ATTENTION_TEMP_DEFAULT}.'
    )

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # =========================================================================
    # POST-PROCESSING: Handle models argument
    # =========================================================================
    # If --models not provided (None) or --models all, use all models
    if args.models is None or (len(args.models) == 1 and args.models[0] == 'all'):
        args.models = get_all_model_names()
    # If --models provided with empty list (e.g., just --models with no values)
    elif len(args.models) == 0:
        args.models = get_all_model_names()
    # Otherwise, args.models contains the user-specified list in order

    # =========================================================================
    # VALIDATION AND SANITY CHECKS
    # =========================================================================
    _validate_args(args)

    return args


def _validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments and exit with informative error on violations.

    Why: Provides early detection of invalid inputs before expensive training
         begins, with actionable error messages referencing specific ranges.

    Args:
        args: Parsed command-line arguments
    Raises:
        SystemExit: If any validation check fails
    """
    errors: List[str] = []

    # Model selection
    available_models: List[str] = get_all_model_names()
    for model in args.models:
        if model not in available_models:
            errors.append(
                f"Invalid model '{model}'. "
                f"Available models: {available_models}"
            )

    # Device
    if args.device not in ['cuda', 'cpu', 'auto']:
        errors.append(f"Invalid device '{args.device}'. Must be 'cuda', 'cpu', or 'auto'.")

    # Result output: resolve results_dir to absolute path
    if not args.no_save_results:
        import os
        args.results_dir = os.path.abspath(args.results_dir)

    # Reproducibility
    if not (NUM_SEEDS_MIN <= args.num_seeds <= NUM_SEEDS_MAX):
        errors.append(
            f"--num-seeds={args.num_seeds} outside valid range "
            f"[{NUM_SEEDS_MIN}, {NUM_SEEDS_MAX}]"
        )

    if args.seed_start < SEED_START_MIN:
        errors.append(
            f"--seed-start={args.seed_start} must be >= {SEED_START_MIN}"
        )

    # Training hyperparameters
    if not (NUM_EPOCHS_MIN <= args.num_epochs <= NUM_EPOCHS_MAX):
        errors.append(
            f"--num-epochs={args.num_epochs} outside valid range "
            f"[{NUM_EPOCHS_MIN}, {NUM_EPOCHS_MAX}]"
        )

    if not (BATCH_SIZE_MIN <= args.batch_size <= BATCH_SIZE_MAX):
        errors.append(
            f"--batch-size={args.batch_size} outside valid range "
            f"[{BATCH_SIZE_MIN}, {BATCH_SIZE_MAX}]"
        )

    if not (LEARNING_RATE_MIN < args.learning_rate <= LEARNING_RATE_MAX):
        errors.append(
            f"--learning-rate={args.learning_rate} outside valid range "
            f"({LEARNING_RATE_MIN}, {LEARNING_RATE_MAX}]"
        )

    if not (WEIGHT_DECAY_MIN <= args.weight_decay <= WEIGHT_DECAY_MAX):
        errors.append(
            f"--weight-decay={args.weight_decay} outside valid range "
            f"[{WEIGHT_DECAY_MIN}, {WEIGHT_DECAY_MAX}]"
        )

    if not (LABEL_SMOOTHING_MIN <= args.label_smoothing <= LABEL_SMOOTHING_MAX):
        errors.append(
            f"--label-smoothing={args.label_smoothing} outside valid range "
            f"[{LABEL_SMOOTHING_MIN}, {LABEL_SMOOTHING_MAX}]"
        )

    if not (PATIENCE_MIN <= args.patience <= PATIENCE_MAX):
        errors.append(
            f"--patience={args.patience} outside valid range "
            f"[{PATIENCE_MIN}, {PATIENCE_MAX}]"
        )

    if not (MIN_DELTA_MIN <= args.min_delta <= MIN_DELTA_MAX):
        errors.append(
            f"--min-delta={args.min_delta} outside valid range "
            f"[{MIN_DELTA_MIN}, {MIN_DELTA_MAX}]"
        )

    if not (LEARNING_RATE_WARMUP_EPOCHS_MIN <= args.learning_rate_warmup_epochs <= LEARNING_RATE_WARMUP_EPOCHS_MAX):
        errors.append(
            f"--learning-rate-warmup-epochs={args.learning_rate_warmup_epochs} outside valid range "
            f"[{LEARNING_RATE_WARMUP_EPOCHS_MIN}, {LEARNING_RATE_WARMUP_EPOCHS_MAX}]"
        )

    # Dataset sizes
    if not (NUM_TRAIN_MIN <= args.num_train <= NUM_TRAIN_MAX):
        errors.append(
            f"--num-train={args.num_train} outside valid range "
            f"[{NUM_TRAIN_MIN}, {NUM_TRAIN_MAX}]"
        )

    if args.num_train % 10 != 0:
        errors.append(
            f"--num-train={args.num_train} must be divisible by 10 "
            f"for balanced class distribution"
        )

    if not (NUM_VAL_MIN <= args.num_val <= NUM_VAL_MAX):
        errors.append(
            f"--num-val={args.num_val} outside valid range "
            f"[{NUM_VAL_MIN}, {NUM_VAL_MAX}]"
        )

    if not (NUM_TEST_ROT_MIN <= args.num_test_rot <= NUM_TEST_ROT_MAX):
        errors.append(
            f"--num-test-rot={args.num_test_rot} outside valid range "
            f"[{NUM_TEST_ROT_MIN}, {NUM_TEST_ROT_MAX}]"
        )

    if not (NUM_TEST_UNROT_MIN <= args.num_test_unrot <= NUM_TEST_UNROT_MAX):
        errors.append(
            f"--num-test-unrot={args.num_test_unrot} outside valid range "
            f"[{NUM_TEST_UNROT_MIN}, {NUM_TEST_UNROT_MAX}]"
        )

    # TQF architecture
    if not (TQF_R_MIN <= args.tqf_R <= TQF_R_MAX):
        errors.append(
            f"--tqf-R={args.tqf_R} outside valid range "
            f"[{TQF_R_MIN}, {TQF_R_MAX}]"
        )

    if args.tqf_hidden_dim is not None:
        if not (TQF_HIDDEN_DIM_MIN <= args.tqf_hidden_dim <= TQF_HIDDEN_DIM_MAX):
            errors.append(
                f"--tqf-hidden-dim={args.tqf_hidden_dim} outside valid range "
                f"[{TQF_HIDDEN_DIM_MIN}, {TQF_HIDDEN_DIM_MAX}]"
            )

    if args.tqf_symmetry_level not in ['Z6', 'D6', 'T24', 'none']:
        errors.append(
            f"--tqf-symmetry-level={args.tqf_symmetry_level} invalid. "
            f"Must be one of: Z6, D6, T24, none"
        )

    # TQF fractal geometry - only validate when provided (not None)
    if args.tqf_fractal_iterations is not None:
        if not (TQF_FRACTAL_ITERATIONS_MIN <= args.tqf_fractal_iterations <= TQF_FRACTAL_ITERATIONS_MAX):
            errors.append(
                f"--tqf-fractal-iterations={args.tqf_fractal_iterations} outside valid range "
                f"[{TQF_FRACTAL_ITERATIONS_MIN}, {TQF_FRACTAL_ITERATIONS_MAX}]"
            )

    # NOTE: --tqf-fractal-dim-tolerance validation removed (internal constant)

    if not (TQF_SELF_SIMILARITY_WEIGHT_MIN <= args.tqf_self_similarity_weight <= TQF_SELF_SIMILARITY_WEIGHT_MAX):
        errors.append(
            f"--tqf-self-similarity-weight={args.tqf_self_similarity_weight} outside valid range "
            f"[{TQF_SELF_SIMILARITY_WEIGHT_MIN}, {TQF_SELF_SIMILARITY_WEIGHT_MAX}]"
        )

    if not (TQF_BOX_COUNTING_WEIGHT_MIN <= args.tqf_box_counting_weight <= TQF_BOX_COUNTING_WEIGHT_MAX):
        errors.append(
            f"--tqf-box-counting-weight={args.tqf_box_counting_weight} outside valid range "
            f"[{TQF_BOX_COUNTING_WEIGHT_MIN}, {TQF_BOX_COUNTING_WEIGHT_MAX}]"
        )

    # NOTE: --tqf-box-counting-scales validation removed (internal constant)

    # TQF attention
    if not (TQF_HOP_ATTENTION_TEMP_MIN <= args.tqf_hop_attention_temp <= TQF_HOP_ATTENTION_TEMP_MAX):
        errors.append(
            f"--tqf-hop-attention-temp={args.tqf_hop_attention_temp} outside valid range "
            f"[{TQF_HOP_ATTENTION_TEMP_MIN}, {TQF_HOP_ATTENTION_TEMP_MAX}]"
        )

    # TQF loss weights
    if not (TQF_GEOMETRY_REG_WEIGHT_MIN <= args.tqf_geometry_reg_weight <= TQF_GEOMETRY_REG_WEIGHT_MAX):
        errors.append(
            f"--tqf-geometry-reg-weight={args.tqf_geometry_reg_weight} outside valid range "
            f"[{TQF_GEOMETRY_REG_WEIGHT_MIN}, {TQF_GEOMETRY_REG_WEIGHT_MAX}]"
        )

    # TQF loss weights - only validate when provided (not None)
    # These weights enable the corresponding loss when provided with a value
    if args.tqf_inversion_loss_weight is not None:
        if not (TQF_INVERSION_LOSS_WEIGHT_MIN <= args.tqf_inversion_loss_weight <= TQF_INVERSION_LOSS_WEIGHT_MAX):
            errors.append(
                f"--tqf-inversion-loss-weight={args.tqf_inversion_loss_weight} outside valid range "
                f"[{TQF_INVERSION_LOSS_WEIGHT_MIN}, {TQF_INVERSION_LOSS_WEIGHT_MAX}]"
            )

    if args.tqf_z6_equivariance_weight is not None:
        if not (TQF_Z6_EQUIVARIANCE_WEIGHT_MIN <= args.tqf_z6_equivariance_weight <= TQF_Z6_EQUIVARIANCE_WEIGHT_MAX):
            errors.append(
                f"--tqf-z6-equivariance-weight={args.tqf_z6_equivariance_weight} outside valid range "
                f"[{TQF_Z6_EQUIVARIANCE_WEIGHT_MIN}, {TQF_Z6_EQUIVARIANCE_WEIGHT_MAX}]"
            )

    if args.tqf_d6_equivariance_weight is not None:
        if not (TQF_D6_EQUIVARIANCE_WEIGHT_MIN <= args.tqf_d6_equivariance_weight <= TQF_D6_EQUIVARIANCE_WEIGHT_MAX):
            errors.append(
                f"--tqf-d6-equivariance-weight={args.tqf_d6_equivariance_weight} outside valid range "
                f"[{TQF_D6_EQUIVARIANCE_WEIGHT_MIN}, {TQF_D6_EQUIVARIANCE_WEIGHT_MAX}]"
            )

    if args.tqf_t24_orbit_invariance_weight is not None:
        if not (TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN <= args.tqf_t24_orbit_invariance_weight <= TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX):
            errors.append(
                f"--tqf-t24-orbit-invariance-weight={args.tqf_t24_orbit_invariance_weight} outside valid range "
                f"[{TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN}, {TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX}]"
            )

    # TQF orbit mixing temperatures
    for temp_name, temp_val in [
        ('--tqf-orbit-mixing-temp-rotation', args.tqf_orbit_mixing_temp_rotation),
        ('--tqf-orbit-mixing-temp-reflection', args.tqf_orbit_mixing_temp_reflection),
        ('--tqf-orbit-mixing-temp-inversion', args.tqf_orbit_mixing_temp_inversion),
    ]:
        if not (TQF_ORBIT_MIXING_TEMP_MIN <= temp_val <= TQF_ORBIT_MIXING_TEMP_MAX):
            errors.append(
                f"{temp_name}={temp_val} outside valid range "
                f"[{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]"
            )

    # Orbit mixing + orbit pooling conflict warning
    any_orbit_mixing: bool = (
        args.tqf_use_z6_orbit_mixing
        or args.tqf_use_d6_orbit_mixing
        or args.tqf_use_t24_orbit_mixing
    )
    if any_orbit_mixing and args.tqf_symmetry_level != 'none':
        print(
            f"WARNING: Both orbit mixing and orbit pooling "
            f"(--tqf-symmetry-level={args.tqf_symmetry_level}) are enabled. "
            f"These mechanisms may conflict — orbit pooling destroys rotation-specific "
            f"information that orbit mixing needs.",
            file=sys.stdout
        )

    # Orbit mixing + equivariance loss conflict warning
    any_equiv_loss: bool = (
        args.tqf_z6_equivariance_weight is not None
        or args.tqf_d6_equivariance_weight is not None
    )
    if any_orbit_mixing and any_equiv_loss:
        print(
            "WARNING: Both orbit mixing and equivariance loss are enabled. "
            "These features conflict — equivariance loss constrains training-time "
            "representations that orbit mixing needs to vary at evaluation time. "
            "Experimental results show this combination reduces rotation accuracy "
            "(62.83% vs 67.42% with orbit mixing alone). "
            "Recommendation: use orbit mixing OR equivariance loss, not both.",
            file=sys.stdout
        )

    # TQF verification - validate minimum only, auto-correct maximum
    if args.tqf_verify_duality_interval < TQF_VERIFY_DUALITY_INTERVAL_MIN:
        errors.append(
            f"--tqf-verify-duality-interval={args.tqf_verify_duality_interval} must be >= "
            f"{TQF_VERIFY_DUALITY_INTERVAL_MIN}"
        )

    # Cross-parameter checks with auto-correction
    if args.tqf_verify_duality_interval > args.num_epochs:
        original_verify_interval: int = args.tqf_verify_duality_interval
        args.tqf_verify_duality_interval = args.num_epochs
        print(f"WARNING: --tqf-verify-duality-interval={original_verify_interval} > --num-epochs={args.num_epochs}. "
              f"Overriding tqf-verify-duality-interval to {args.tqf_verify_duality_interval}.", file=sys.stdout)
    if args.patience >= args.num_epochs:
        original_patience: int = args.patience
        args.patience = args.num_epochs
        print(f"WARNING: --patience={original_patience} >= --num-epochs={args.num_epochs}. "
              f"Overriding patience to {args.patience}.", file=sys.stdout)

    if args.learning_rate_warmup_epochs >= args.num_epochs:
        original_warmup: int = args.learning_rate_warmup_epochs
        args.learning_rate_warmup_epochs = args.num_epochs
        print(f"WARNING: --learning-rate-warmup-epochs={original_warmup} >= --num-epochs={args.num_epochs}. "
              f"Overriding learning-rate-warmup-epochs to {args.learning_rate_warmup_epochs}.", file=sys.stdout)

    # If validation errors found, print and exit
    if errors:
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("ERROR: Invalid command-line arguments detected:", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}", file=sys.stderr)
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("Run with --help for valid ranges and usage examples.", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)


def setup_logging() -> None:
    """
    Setup console-only logging with timestamps.

    Why: Provides clear, timestamped output to console for immediate visibility.
         As per best practice, all output goes to terminal for immediate
         visibility and can be redirected via shell if needed.
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ],
        force=True  # Override any existing configuration
    )


###################################################################################
# MAIN ############################################################################
###################################################################################

if __name__ == "__main__":
    # Test parser creation
    args = parse_args()
    print("Argument parsing successful!")
    print(f"Models: {args.models}")
    print(f"TQF Symmetry Level: {args.tqf_symmetry_level}")
