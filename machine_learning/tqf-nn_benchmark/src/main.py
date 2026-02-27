"""
main.py - Main Entry Point for TQF-NN Benchmark Experiments

This is the primary executable script for running TQF-ANN vs baseline model comparison
experiments on MNIST. It orchestrates the complete experimental pipeline from argument
parsing through dataset loading, model initialization, training, evaluation, and
statistical comparison with publication-ready formatted output.

Key Features:
- CLI Integration: Argument parsing via cli.py with validation and help text
- Model Configuration: Automatic parameter matching (~650K params) for fair comparison
- GPU Optimization: CUDA optimization (cuDNN benchmark, TF32, mixed precision)
- Experiment Orchestration: Multi-seed training with statistical aggregation
- Parameter Verification: Strict ±1% tolerance checking before training
- Dataset Loading: Stratified MNIST splits with rotational test sets
- Logging: Comprehensive experiment configuration logging (reproducibility info)
- Results Aggregation: Statistical comparison tables with p-values
- TQF-Specific Handling: Auto-tuned hidden dimensions, fractal loss configuration

Execution Flow:
    1. Parse CLI arguments and validate ranges
    2. Configure CUDA optimization and logging
    3. Generate random seeds for multi-seed experiments
    4. Log comprehensive experiment configuration
    5. Configure all models (FC-MLP, CNN-L5, ResNet-18-Scaled, TQF-ANN)
    6. Verify parameter matching within tolerance
    7. Load MNIST datasets with stratified splits
    8. Run multi-seed experiments for selected models
    9. Aggregate results across seeds (mean ± std)
    10. Perform statistical significance tests (paired t-tests)
    11. Print final comparison table with rankings

Usage:
    python main.py --models TQF-ANN FC-MLP --num-seeds 5 --max-epochs 50
    python main.py --models all --tqf-verify-geometry --batch-size 128

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

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from cli import parse_args, setup_logging


def verify_parameter_matching(model_configs: Dict[str, Dict]) -> bool:
    """
    Verify all models meet +/- percentage parameter tolerance.

    Args:
        model_configs: Dict mapping model_name -> config dict
    Returns:
        True if all models within tolerance
    """
    from models_baseline import get_model
    from logging_utils import print_separator, print_single_separator
    from config import TARGET_PARAMS, TARGET_PARAMS_TOLERANCE_ABSOLUTE, TARGET_PARAMS_TOLERANCE_PERCENT

    print_separator("PARAMETER MATCHING VERIFICATION")
    print(f"Target: {TARGET_PARAMS:,} parameters (+/- {TARGET_PARAMS_TOLERANCE_PERCENT}% = {TARGET_PARAMS_TOLERANCE_ABSOLUTE:,})")
    print_single_separator("-")

    all_pass: bool = True

    for model_name, config in model_configs.items():
        model = get_model(model_name, **config)
        params: int = model.count_parameters()
        deviation: float = abs(params - TARGET_PARAMS) / TARGET_PARAMS * 100
        status: str = 'PASS' if deviation <= TARGET_PARAMS_TOLERANCE_PERCENT else 'FAIL'

        # For TQF-ANN, indicate auto-tuned status (details shown in experiment config above)
        extra_info: str = ""
        if model_name == 'TQF-ANN' and hasattr(model, 'hidden_dim'):
            extra_info = " (auto-tuned)"

        print(f"{model_name:<20} {params:>12,}  ({deviation:>5.2f}%)  {status}{extra_info}")

        if status == 'FAIL':
            all_pass = False

    print_single_separator("=")

    return all_pass

def main():
    """Main execution function with argument-driven configuration."""
    args = parse_args()

    # Deferred heavy imports (keep out of top-level so `-h` stays fast)
    import torch
    from engine import (
        run_multi_seed_experiment,
        compare_models_statistical,
        print_final_comparison_table
    )
    from output_formatters import format_time_seconds, save_final_summary_to_disk
    from logging_utils import log_experiment_config, print_separator, print_single_separator
    try:
        from prepare_datasets import get_dataloaders
    except ImportError:
        print("Warning: prepare_datasets.py not found. Using dummy loaders.")
        def get_dataloaders(*args, **kwargs):
            raise NotImplementedError("prepare_datasets.py not available")

    # Startup message
    print_separator("TQF-NN Benchmark Experiment Suite >>> It's time for Cold Hammer L33T 0WN4G3")
    print()

    # Resolve --device auto
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # GPU optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup onsole-only logging
    setup_logging()

    device = torch.device(args.device)

    # Generate seeds from starting seed
    # Always generate from range to respect args.num_seeds
    seeds: List[int] = list(range(args.seed_start, args.seed_start + args.num_seeds))

    # Generate timestamped output path for persistent result logging
    output_path: str = ""
    if not args.no_save_results:
        timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.results_dir, f'results_{timestamp}.json')
        os.makedirs(args.results_dir, exist_ok=True)

    # Log comprehensive experiment configuration (replaces separate log_reproducibility_info call)
    log_experiment_config(args, seeds, device, output_path=output_path)

    # Model configurations with argument-driven TQF params
    # Note: TQF hidden_dim=None triggers auto-tuning in TQFANN.__init__()

    all_model_configs: Dict[str, Dict] = {
        'FC-MLP': {},
        'CNN-L5': {},
        'ResNet-18-Scaled': {},
        'TQF-ANN': {
            # Lattice-Graph Structure
            'R': args.tqf_R,
            'hidden_dim': args.tqf_hidden_dim,  # None = auto-tune
            'symmetry_level': args.tqf_symmetry_level,

            # Geometry/Regularization
            'verify_geometry': args.tqf_verify_geometry,
            'geometry_reg_weight': args.tqf_geometry_reg_weight,

            # Dual Metrics (hardcoded - not exposed as CLI params per design)
            'use_dual_output': True,
            'use_dual_metric': True,

            # Memory Optimization
            'use_gradient_checkpointing': args.tqf_use_gradient_checkpointing,
        }
    }

    # Filter to selected models and preserve user-specified order
    # args.models is already processed by CLI (expanded "all" or specific list)
    model_configs: Dict[str, Dict] = {
        k: all_model_configs[k] for k in args.models if k in all_model_configs
    }

    # Verify parameter matching (will show auto-tuned hidden_dim for TQF)
    if not verify_parameter_matching(model_configs):
        logging.warning("Some models failed parameter matching check!")

    print()
    logging.info(f"Loading datasets (num_train={args.num_train}, num_val={args.num_val}, "
                 f"num_test_rot={args.num_test_rot}, num_test_unrot={args.num_test_unrot})...")
    any_orbit_mixing: bool = (
        args.tqf_use_z6_orbit_mixing or args.tqf_use_d6_orbit_mixing or args.tqf_use_t24_orbit_mixing
    )
    if any_orbit_mixing:
        logging.info(
            f"Orbit mixing: Z6={args.tqf_use_z6_orbit_mixing}, "
            f"D6={args.tqf_use_d6_orbit_mixing}, T24={args.tqf_use_t24_orbit_mixing} "
            f"(temps: rot={args.tqf_orbit_mixing_temp_rotation:.2f}, "
            f"refl={args.tqf_orbit_mixing_temp_reflection:.2f}, "
            f"inv={args.tqf_orbit_mixing_temp_inversion:.2f})"
        )
    train_loader, val_loader, test_loader_rot, test_loader_unrot = get_dataloaders(
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test_rot=args.num_test_rot,
        num_test_unrot=args.num_test_unrot,
        augment_train=args.z6_data_augmentation,
        augment_z6_non_rotation=args.tqf_z6_non_rotation_augmentation
    )

    # Warm up image caches so first training epoch isn't penalized by disk I/O
    print("Getting fired up! Loading dataset! Let's cache these puppies...", end=" ", flush=True)
    warmup_start: float = time.time()
    for loader in [train_loader, val_loader, test_loader_rot, test_loader_unrot]:
        ds = loader.dataset
        if hasattr(ds, 'warmup_cache'):
            ds.warmup_cache()
        elif hasattr(ds, 'dataset') and hasattr(ds.dataset, 'warmup_cache'):
            ds.dataset.warmup_cache()
    warmup_sec: float = time.time() - warmup_start
    print(f"done ({format_time_seconds(warmup_sec)})")

    dataloaders: Dict[str, Any] = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader_unrot,
        'test_rot': test_loader_rot
    }

    # Run experiments with args-driven hyperparameters
    # Note: Loss features (Z6/D6 equivariance, T24 invariance, inversion duality) are
    # enabled by providing a weight value via CLI. Weight=None means feature is disabled.
    logging.info("Starting experiments...")
    results = run_multi_seed_experiment(
        model_names=list(model_configs.keys()),
        model_configs=model_configs,
        dataloaders=dataloaders,
        device=device,
        seeds=seeds,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        min_delta=args.min_delta,
        warmup_epochs=args.learning_rate_warmup_epochs,
        verify_duality_interval=args.tqf_verify_duality_interval,
        inversion_loss_weight=args.tqf_inversion_loss_weight,
        z6_equivariance_weight=args.tqf_z6_equivariance_weight,
        d6_equivariance_weight=args.tqf_d6_equivariance_weight,
        t24_orbit_invariance_weight=args.tqf_t24_orbit_invariance_weight,
        z6_orbit_consistency_weight=args.tqf_z6_orbit_consistency_weight,
        z6_orbit_consistency_rotations=args.tqf_z6_orbit_consistency_rotations,
        use_compile=args.compile,
        output_path=output_path,
        use_z6_orbit_mixing=args.tqf_use_z6_orbit_mixing,
        use_d6_orbit_mixing=args.tqf_use_d6_orbit_mixing,
        use_t24_orbit_mixing=args.tqf_use_t24_orbit_mixing,
        orbit_mixing_temp_rotation=args.tqf_orbit_mixing_temp_rotation,
        orbit_mixing_temp_reflection=args.tqf_orbit_mixing_temp_reflection,
        orbit_mixing_temp_inversion=args.tqf_orbit_mixing_temp_inversion,
        z6_orbit_mixing_confidence_mode=args.tqf_z6_orbit_mixing_confidence_mode,
        z6_orbit_mixing_aggregation_mode=args.tqf_z6_orbit_mixing_aggregation_mode,
        z6_orbit_mixing_top_k=args.tqf_z6_orbit_mixing_top_k,
        z6_orbit_mixing_adaptive_temp=args.tqf_z6_orbit_mixing_adaptive_temp,
        z6_orbit_mixing_adaptive_temp_alpha=args.tqf_z6_orbit_mixing_adaptive_temp_alpha,
        z6_orbit_mixing_rotation_mode=args.tqf_z6_orbit_mixing_rotation_mode,
        z6_orbit_mixing_rotation_padding_mode=args.tqf_z6_orbit_mixing_rotation_padding_mode,
        z6_orbit_mixing_rotation_pad=args.tqf_z6_orbit_mixing_rotation_pad
    )

    # Statistical comparison
    logging.info("Computing statistical comparisons...")
    summary = compare_models_statistical(results)

    # Save final summary to disk (JSON + human-readable .txt)
    if output_path:
        save_final_summary_to_disk(summary, output_path)

    # Print results
    print_final_comparison_table(summary)

    if output_path:
        logging.info(f"Experiment complete! Results saved to {output_path}")
    else:
        logging.info("Experiment complete! (result saving disabled)")

if __name__ == "__main__":
    main()
