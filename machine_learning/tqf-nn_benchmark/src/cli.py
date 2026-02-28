"""
cli.py - Command-Line Interface for TQF-NN Experiments

This module centralizes all argument parsing, validation, and help text for the
TQF-NN benchmark suite. Extracted from main.py to improve maintainability and
enable consistent validation across all CLI inputs.

Key Features:
- Comprehensive Argument Parser: 40+ CLI arguments for hyperparameters, architecture, datasets
- Model Selection: Support for 'all' keyword or individual model names (FC-MLP, CNN-L5, ResNet-18-Scaled, TQF-ANN)
- TQF Configuration: Lattice radius, hidden dimensions, fractal parameters
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
    TQF_TRUNCATION_R_DEFAULT,
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
    Z6_DATA_AUGMENTATION_DEFAULT,
    TQF_Z6_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    TQF_D6_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    TQF_T24_ORBIT_MIXING_TEMP_INVERSION_DEFAULT,
    TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ROTATION_PAD_DEFAULT,
    NON_ROTATION_DATA_AUGMENTATION_DEFAULT,
    TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_DEFAULT,
    TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_DEFAULT,
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
    TQF_ORBIT_MIXING_TEMP_MIN, TQF_ORBIT_MIXING_TEMP_MAX,
    TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN, TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX,
    TQF_Z6_ORBIT_MIXING_TOP_K_MIN, TQF_Z6_ORBIT_MIXING_TOP_K_MAX,
    TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MIN, TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MAX,
    TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MIN, TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MAX,
    TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MIN, TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MAX,
    TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MIN, TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MAX
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
  # Best TQF-ANN config: Z6 orbit mixing for maximum rotation invariance
  python main.py --models TQF-ANN --tqf-use-z6-orbit-mixing

  # Statistical validation: TQF-ANN with Z6 orbit mixing, 3 seeds
  python main.py --models TQF-ANN --tqf-use-z6-orbit-mixing --num-seeds 3

  # Full benchmark: all models (TQF-ANN benefits most from orbit mixing)
  python main.py --models all --tqf-use-z6-orbit-mixing

  # Baselines only (FC-MLP, CNN-L5, ResNet-18-Scaled)
  python main.py --models FC-MLP CNN-L5 ResNet-18-Scaled

  # Train all models with defaults
  python main.py

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

    output_group.add_argument(
        '--experiment-label',
        type=str,
        default=None,
        help='Human-readable label stored in the JSON config section (e.g. '
             '"02 Z6 orbit mixing T=0.5"). Used by experiment scripts to '
             'identify what each results file represents. Has no effect on '
             'training behaviour.'
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
        'Training, validation, and test set sizes; training augmentation (all models)'
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

    data_group.add_argument(
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

    data_group.add_argument(
        '--non-rotation-data-augmentation',
        action='store_true',
        dest='non_rotation_data_augmentation',
        default=NON_ROTATION_DATA_AUGMENTATION_DEFAULT,
        help='Enable non-rotation training augmentation (random crop + brightness/contrast jitter) '
             'for all models. Applies random 28×28 crop with 2-pixel padding and ±10%% '
             'brightness/contrast jitter. Composable with --z6-data-augmentation. '
             'Default: disabled (False).'
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

    # =========================================================================
    # TQF Z6 ORBIT MIXING (Evaluation-Time) — PRIMARY FEATURE
    # =========================================================================
    # Z6 orbit mixing averages predictions over 6 input-space rotations at
    # evaluation time. All disabled by default — evaluation uses single pass.
    tqf_z6_orbit_group = parser.add_argument_group(
        'TQF Z6 Orbit Mixing (TQF-ANN only, evaluation-time)',
        'Primary evaluation-time ensemble: average predictions over 6 Z6 input-space rotations '
        '(0, 60, 120, 180, 240, 300 deg). Cost: ~6x inference. Disabled by default.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-use-z6-orbit-mixing',
        action='store_true',
        default=False,
        help='Enable Z6 orbit mixing at evaluation time. '
             'Averages predictions over 6 input-space rotations (0, 60, ..., 300 deg). '
             'Cost: ~6x inference time. Default: False.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-temp-rotation',
        type=float,
        default=TQF_Z6_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
        help='Temperature for Z6 rotation averaging. '
             'Lower = sharper (most confident rotation dominates). '
             f'Range: [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]. '
             f'Default: {TQF_Z6_ORBIT_MIXING_TEMP_ROTATION_DEFAULT}.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-confidence-mode',
        type=str,
        default=TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT,
        choices=['max_logit', 'margin'],
        help='Confidence signal used to weight each Z6 rotation variant. '
             'max_logit (default): maximum logit value. '
             'margin: top-1 minus top-2 logit (decision margin). '
             f'Default: {TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT}.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-aggregation-mode',
        type=str,
        default=TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT,
        choices=['logits', 'probs', 'log_probs'],
        help='Space in which weighted averaging is performed for Z6 orbit mixing. '
             'logits (default): raw logit space. '
             'probs: probability space (softmax then average). '
             'log_probs: log-probability space (geometric mean / product-of-experts). '
             f'Default: {TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT}.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-top-k',
        type=int,
        default=TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT,
        help=f'If set, keep only the top-K most confident Z6 rotation variants before averaging. '
             f'None (default) uses all 6. '
             f'Range: [{TQF_Z6_ORBIT_MIXING_TOP_K_MIN}, {TQF_Z6_ORBIT_MIXING_TOP_K_MAX}]. '
             f'Requires --tqf-use-z6-orbit-mixing.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-adaptive-temp',
        action='store_true',
        dest='tqf_z6_orbit_mixing_adaptive_temp',
        default=TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT,
        help='Enable per-sample adaptive temperature for Z6 orbit mixing. '
             'Scales temperature up when all rotation variants have similar confidence '
             '(high entropy), allowing smoother averaging in ambiguous cases. '
             'Default: disabled (False).'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-adaptive-temp-alpha',
        type=float,
        default=TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT,
        help=f'Sensitivity of adaptive temperature scaling. '
             f'0 = no adaptation; higher values strengthen entropy scaling. '
             f'Range: [{TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MIN}, {TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MAX}]. '
             f'Default: {TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT}. '
             f'Only used when --tqf-z6-orbit-mixing-adaptive-temp is set.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-rotation-mode',
        type=str,
        default=TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT,
        choices=['bilinear', 'bicubic'],
        help='Interpolation mode used when rotating images for Z6 orbit mixing. '
             'bilinear (default): faster, slightly smoother edges. '
             'bicubic: higher-order interpolation, may reduce aliasing artefacts. '
             f'Default: {TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT}.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-rotation-padding-mode',
        type=str,
        default=TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT,
        choices=['zeros', 'border'],
        help='Padding mode for rotated image corners. '
             'zeros (default): black (zero) fill for out-of-bounds regions. '
             'border: replicate edge pixels; avoids zero-corner artefacts. '
             f'Default: {TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT}.'
    )

    tqf_z6_orbit_group.add_argument(
        '--tqf-z6-orbit-mixing-rotation-pad',
        type=int,
        default=TQF_Z6_ORBIT_MIXING_ROTATION_PAD_DEFAULT,
        help=f'Pixels to pad before rotating then crop back to 28×28. '
             f'0 (default) = no padding (standard rotation). '
             f'>0 = reflect-pad to (28+2*pad)×(28+2*pad), rotate in padded space, '
             f'then center-crop back to 28×28. Eliminates zero corner artefacts. '
             f'Range: [{TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MIN}, {TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MAX}]. '
             f'Suggested value: 4.'
    )

    # =========================================================================
    # TQF EQUIVARIANCE LOSSES
    # =========================================================================
    # Equivariance: f(g·x) = g·f(x) - features transform correctly with input
    # These losses are DISABLED by default. Providing a weight value enables
    # the corresponding loss feature.
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

    tqf_inv_group.add_argument(
        '--tqf-z6-orbit-consistency-weight',
        type=float,
        default=TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_DEFAULT,
        help=f'Enable and set weight for orbit consistency self-distillation loss (TQF-ANN only). '
             f'Creates a Z6 orbit ensemble as soft target and penalises each rotation for '
             f'diverging from it (KL divergence). Trains the model to be consistent across '
             f'rotations without requiring rotation labels. Adds extra forward passes per batch. '
             f'Disabled by default (None). Provide a value in range '
             f'[{TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MIN}, {TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MAX}] to enable. '
             f'Suggested starting value: 0.01.'
    )

    tqf_inv_group.add_argument(
        '--tqf-z6-orbit-consistency-rotations',
        type=int,
        default=TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_DEFAULT,
        help=f'Number of extra Z6 rotations sampled per batch for orbit consistency loss. '
             f'Range: [{TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MIN}, {TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MAX}]. '
             f'Default: {TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_DEFAULT}. '
             f'Higher values = larger ensemble but more compute per batch.'
    )

    # =========================================================================
    # TQF D6/T24 ORBIT MIXING (Evaluation-Time) — SECONDARY FEATURES
    # =========================================================================
    # D6 adds feature-space reflections on top of Z6 rotations.
    # T24 further adds inner/outer zone-swap variants.
    # Both disabled by default; each requires --tqf-use-z6-orbit-mixing or
    # produces a superset evaluation independently.
    tqf_d6t24_orbit_group = parser.add_argument_group(
        'TQF D6/T24 Orbit Mixing (TQF-ANN only, evaluation-time)',
        'Secondary evaluation-time ensembles extending Z6 with reflections (D6) and '
        'zone-swap variants (T24). D6: +6 lightweight head passes. T24: +18. Disabled by default.'
    )

    tqf_d6t24_orbit_group.add_argument(
        '--tqf-use-d6-orbit-mixing',
        action='store_true',
        default=False,
        help='Enable D6 orbit mixing at evaluation time. '
             'Includes Z6 rotations plus 6 feature-space reflections. '
             'Cost: ~6x full forward + 6 lightweight head passes. Default: False.'
    )

    tqf_d6t24_orbit_group.add_argument(
        '--tqf-use-t24-orbit-mixing',
        action='store_true',
        default=False,
        help='Enable T24 orbit mixing at evaluation time. '
             'Includes D6 operations plus inner/outer zone-swap variants. '
             'Cost: ~6x full forward + 18 lightweight head passes. Default: False.'
    )

    tqf_d6t24_orbit_group.add_argument(
        '--tqf-d6-orbit-mixing-temp-reflection',
        type=float,
        default=TQF_D6_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
        help='Temperature for D6 reflection averaging. '
             'Softer than rotation because some digits are asymmetric under reflection. '
             f'Range: [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]. '
             f'Default: {TQF_D6_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT}.'
    )

    tqf_d6t24_orbit_group.add_argument(
        '--tqf-t24-orbit-mixing-temp-inversion',
        type=float,
        default=TQF_T24_ORBIT_MIXING_TEMP_INVERSION_DEFAULT,
        help='Temperature for T24 zone-swap averaging. '
             'Softest because circle inversion is the most abstract symmetry. '
             f'Range: [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]. '
             f'Default: {TQF_T24_ORBIT_MIXING_TEMP_INVERSION_DEFAULT}.'
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

    # TQF loss weights - only validate when provided (not None)
    if args.tqf_t24_orbit_invariance_weight is not None:
        if not (TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN <= args.tqf_t24_orbit_invariance_weight <= TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX):
            errors.append(
                f"--tqf-t24-orbit-invariance-weight={args.tqf_t24_orbit_invariance_weight} outside valid range "
                f"[{TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN}, {TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX}]"
            )

    if args.tqf_z6_orbit_consistency_weight is not None:
        if not (TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MIN <= args.tqf_z6_orbit_consistency_weight <= TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MAX):
            errors.append(
                f"--tqf-z6-orbit-consistency-weight={args.tqf_z6_orbit_consistency_weight} outside valid range "
                f"[{TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MIN}, {TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MAX}]"
            )

    if not (TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MIN <= args.tqf_z6_orbit_consistency_rotations <= TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MAX):
        errors.append(
            f"--tqf-z6-orbit-consistency-rotations={args.tqf_z6_orbit_consistency_rotations} outside valid range "
            f"[{TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MIN}, {TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MAX}]"
        )

    # TQF orbit mixing temperatures
    for temp_name, temp_val in [
        ('--tqf-z6-orbit-mixing-temp-rotation', args.tqf_z6_orbit_mixing_temp_rotation),
        ('--tqf-d6-orbit-mixing-temp-reflection', args.tqf_d6_orbit_mixing_temp_reflection),
        ('--tqf-t24-orbit-mixing-temp-inversion', args.tqf_t24_orbit_mixing_temp_inversion),
    ]:
        if not (TQF_ORBIT_MIXING_TEMP_MIN <= temp_val <= TQF_ORBIT_MIXING_TEMP_MAX):
            errors.append(
                f"{temp_name}={temp_val} outside valid range "
                f"[{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]"
            )

    # TQF orbit mixing quality params
    if args.tqf_z6_orbit_mixing_top_k is not None:
        if not (TQF_Z6_ORBIT_MIXING_TOP_K_MIN <= args.tqf_z6_orbit_mixing_top_k <= TQF_Z6_ORBIT_MIXING_TOP_K_MAX):
            errors.append(
                f"--tqf-z6-orbit-mixing-top-k={args.tqf_z6_orbit_mixing_top_k} outside valid range "
                f"[{TQF_Z6_ORBIT_MIXING_TOP_K_MIN}, {TQF_Z6_ORBIT_MIXING_TOP_K_MAX}]"
            )
        if not args.tqf_use_z6_orbit_mixing:
            errors.append(
                f"--tqf-z6-orbit-mixing-top-k requires --tqf-use-z6-orbit-mixing"
            )

    if not (TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MIN <= args.tqf_z6_orbit_mixing_adaptive_temp_alpha <= TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MAX):
        errors.append(
            f"--tqf-z6-orbit-mixing-adaptive-temp-alpha={args.tqf_z6_orbit_mixing_adaptive_temp_alpha} outside valid range "
            f"[{TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MIN}, {TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MAX}]"
        )

    if not (TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MIN <= args.tqf_z6_orbit_mixing_rotation_pad <= TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MAX):
        errors.append(
            f"--tqf-z6-orbit-mixing-rotation-pad={args.tqf_z6_orbit_mixing_rotation_pad} outside valid range "
            f"[{TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MIN}, {TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MAX}]"
        )

    # Cross-parameter checks with auto-correction
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
