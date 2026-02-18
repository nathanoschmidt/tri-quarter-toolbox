"""
output_formatters.py - Console Output Formatting and Results Presentation

This module provides centralized formatting utilities for experiment output, progress
tracking, and results visualization. All output is ASCII-only for LaTeX compatibility
and maximum terminal portability.

Key Features:
- Separator Utilities: Consistent section headers (=== major, --- minor)
- Typography Constants: Standardized widths (80/92/120/175 chars) and column alignments
- Numeric Formatters: Accuracy (XX.XX%), loss (X.XXXX), confidence intervals (±X.XX)
- Progress Reporting: Per-epoch training summaries with learning rate tracking
- Results Tables: Multi-model comparison tables with rankings and statistical significance
- Early Stopping Messages: Formatted convergence notifications with best epoch info
- Seed Headers: Per-seed experiment section separators
- Label-Value Pairs: Consistent key-value formatting with alignment
- Time Formatting: Human-readable elapsed time (HH:MM:SS) and timestamps

Design Philosophy:
ASCII-only characters ensure output can be:
  - Copy-pasted directly into LaTeX documents without Unicode issues
  - Displayed correctly in any terminal (Windows CMD, PowerShell, Unix shells)
  - Parsed reliably for automated result extraction
  - Printed to file-based logs without encoding problems

Usage:
    from output_formatters import (
        print_section_header,
        format_accuracy,
        print_epoch_progress,
        print_final_comparison_table
    )

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

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import time
from datetime import datetime
import numpy as np

# =============================================================================
# SECTION 1: Typography Constants (ASCII-Only)
# =============================================================================

# Separator styles (ASCII only)
MAJOR_SEP_CHAR: str = "="  # Major sections
MINOR_SEP_CHAR: str = "-"  # Subsections

# Standard widths
WIDTH_NARROW: int = 80    # Simple output
WIDTH_STANDARD: int = 92  # Most outputs
WIDTH_WIDE: int = 120     # Wide tables
WIDTH_EXTRA: int = 175    # Final results table

# Column widths
COL_LABEL: int = 30       # Left column for labels
COL_VALUE: int = 15       # Right column for values

# =============================================================================
# SECTION 2: Basic Formatting Utilities
# =============================================================================

def make_separator(char: str = MAJOR_SEP_CHAR, width: int = WIDTH_STANDARD) -> str:
    """
    Create a separator line of specified character and width.

    Why: Centralized separator creation ensures consistent formatting across
         all output functions. ASCII-only for LaTeX compatibility.

    Args:
        char: Character to use (typically '=' or '-')
        width: Line width in characters
    Returns:
        Separator string
    """
    return char * width

def print_section_header(
    title: str,
    char: str = MAJOR_SEP_CHAR,
    width: int = WIDTH_STANDARD
) -> None:
    """
    Print a titled section header with separators.

    Why: Provides visual structure and hierarchy to console output,
         improving readability of multi-stage experiments.

    Args:
        title: Section title
        char: Separator character
        width: Section width
    """
    sep: str = make_separator(char, width)
    print(f"\n{sep}")
    print(f" {title}")
    print(f"{sep}")

def format_labeled_value(
    label: str,
    value: Any,
    label_width: int = COL_LABEL,
    value_width: int = COL_VALUE,
    value_fmt: Optional[str] = None
) -> str:
    """
    Format a label-value pair with consistent alignment.

    Why: Ensures all key-value outputs have consistent spacing and
         alignment, improving readability of configuration dumps.

    Args:
        label: Label text
        value: Value to format
        label_width: Width for label column
        value_width: Width for value column
        value_fmt: Optional format string (e.g., '.2f', ',d')
    Returns:
        Formatted string

    Example:
        >>> format_labeled_value("Learning rate", 0.0003, value_fmt='.2e')
        "  Learning rate              : 3.00e-04"
    """
    if value_fmt:
        if isinstance(value, (int, float)):
            value_str: str = f"{value:{value_fmt}}"
        else:
            value_str: str = str(value)
    else:
        value_str: str = str(value)

    return f"  {label:<{label_width}} : {value_str}"

# =============================================================================
# SECTION 3: Numeric Formatting Standards
# =============================================================================

def format_accuracy(value: float, width: int = 6) -> str:
    """Format accuracy as XX.XX%"""
    return f"{value * 100:{width}.2f}"

def format_loss(value: float) -> str:
    """Format loss with 6 decimal places for precision"""
    return f"{value:.6f}"

def format_time_ms(value: float) -> str:
    """Format time in milliseconds"""
    return f"{value:.2f} ms"

def format_time_seconds(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Duration in seconds
    Returns:
        Formatted string (e.g., "1.5 min", "2.3 hr")
    """
    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:02.1f} min"
    else:
        return f"{seconds / 3600:.2f} hr"

def format_params(params: int) -> str:
    """
    Format parameter count with thousands separator or K/M suffix.

    Args:
        params: Number of parameters
    Returns:
        Formatted string (e.g., "650,000" or "650k")
    """
    if params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    elif params >= 1_000:
        return f"{params / 1_000:.0f}k"
    else:
        return f"{params:,}"

def format_flops(flops: float) -> str:
    """
    Format FLOPs with appropriate units (M, G, T).

    Args:
        flops: Number of FLOPs
    Returns:
        Formatted string with unit
    """
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.1f}M"
    elif flops >= 1e3:
        return f"{flops / 1e3:.1f}k"
    else:
        return f"{flops:.0f}"

def format_scientific(value: float) -> str:
    """Format in scientific notation for small values"""
    return f"{value:.2e}"

def format_mean_std(mean: float, std: float, fmt: str = ".2f") -> str:
    """
    Format mean +/- std deviation.

    Args:
        mean: Mean value
        std: Standard deviation
        fmt: Format string for both values
    Returns:
        Formatted string "mean +/- std"
    """
    return f"{mean:{fmt}} +/- {std:{fmt}}"

# =============================================================================
# SECTION 4: Experiment Progress Formatting
# =============================================================================

def print_seed_header(
    seed: int,
    total_seeds: int,
    model_name: str,
    seed_idx: int = 1,
    width: int = WIDTH_STANDARD
) -> None:
    """
    Print header for a single seed experiment.

    Why: Clearly demarcates each seed run for easier log navigation
         during multi-seed experiments.

    Args:
        seed: Current seed number
        total_seeds: Total number of seeds
        model_name: Name of model being evaluated
        seed_idx: Seed index (1-based)
        width: Header width
    """
    print_section_header(
        f"{model_name} >>> SEED {seed_idx} (#{seed} of {total_seeds}): {seed} >>> EXECUTING",
        char=MAJOR_SEP_CHAR,
        width=width
    )

def print_model_training_start(
    model_name: str,
    total_seeds: int,
    width: int = WIDTH_STANDARD
) -> None:
    """
    Print separator indicating start of model training.

    Why: Clearly demarcates when a new model's training begins across
         all seeds, improving readability when training multiple models.

    Args:
        model_name: Name of model being evaluated
        total_seeds: Total number of seeds for this model
        width: Separator width
    """
    print(f"\n{make_separator('#', width)}")
    print(f"# BEGIN MODEL TRAINING: {model_name}")
    print(f"{make_separator('#', width)}")

def print_model_training_end(
    model_name: str,
    width: int = WIDTH_STANDARD
) -> None:
    """
    Print separator indicating end of model training.

    Why: Clearly demarcates when a model's training completes across
         all seeds, improving readability when training multiple models.

    Args:
        model_name: Name of model that finished
        width: Separator width
    """
    print(f"{make_separator('#', width)}")
    print(f"# END MODEL TRAINING: {model_name}")
    print(f"{make_separator('#', width)}\n")

def print_epoch_progress(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    lr: float,
    elapsed: float,
    geom_loss: Optional[float] = None,
    z6_equiv_loss: Optional[float] = None,
    d6_equiv_loss: Optional[float] = None,
    t24_orbit_loss: Optional[float] = None
) -> None:
    """
    Print compact epoch progress line with timestamp and expanded terminology.

    Why: Provides real-time feedback during training without overwhelming
         the console. Uses clear terminology for better user understanding.
         Timestamp allows tracking when each epoch completes.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total epochs
        train_loss: Training loss
        val_loss: Validation loss
        val_acc: Validation accuracy
        lr: Current learning rate
        elapsed: Elapsed time in seconds
        geom_loss: Optional geometry loss (TQF models)
        z6_equiv_loss: Optional Z6 rotation equivariance loss (TQF models)
        d6_equiv_loss: Optional D6 reflection equivariance loss (TQF models)
        t24_orbit_loss: Optional T24 orbit invariance loss (TQF models)
    """
    from datetime import datetime

    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress: str = f"Epoch [{epoch + 1:3d}/{total_epochs}]"
    acc_str: str = f"Val Accuracy: {val_acc:5.2f}%"
    lr_str: str = f"Learning Rate: {lr:.2e}"
    time_str: str = f"Elapsed: {format_time_seconds(elapsed)}"

    output: str = f"{timestamp} - {progress} | {acc_str} | {lr_str} | {time_str}"

    if geom_loss is not None:
        geom_str: str = f"Geom: {geom_loss:.4f}"
        output += f" | {geom_str}"

    # Add symmetry/equivariance losses (compact format for console)
    if z6_equiv_loss is not None:
        output += f" | Z6: {z6_equiv_loss:.4f}"

    if d6_equiv_loss is not None:
        output += f" | D6: {d6_equiv_loss:.4f}"

    if t24_orbit_loss is not None:
        output += f" | T24: {t24_orbit_loss:.4f}"

    print(output)

def print_early_stopping_message(
    patience: int,
    best_loss_epoch: int,
    best_acc_epoch: int,
    total_epochs: int,
    best_val_acc_at_best_loss: float,
    best_val_acc_overall: float,
    best_val_loss: float,
    width: int = WIDTH_STANDARD
) -> None:
    """
    Print formatted early stopping message with separate tracking of best loss and best accuracy.

    Why: Clearly communicates why training stopped and disambiguates between
         the epoch with best validation loss (used for early stopping) and
         the epoch with highest validation accuracy (best performance).
         This distinction is critical since they're often different epochs.

    Args:
        patience: Patience threshold that was exceeded
        best_loss_epoch: Epoch with lowest validation loss (model restored to this)
        best_acc_epoch: Epoch with highest validation accuracy
        total_epochs: Total epochs planned
        best_val_acc_at_best_loss: Accuracy at the epoch with best (lowest) loss
        best_val_acc_overall: Highest accuracy achieved across all epochs
        best_val_loss: Best (lowest) validation loss achieved
        width: Message width
    """
    print(f"\n{make_separator(MINOR_SEP_CHAR, width)}")
    print(f"  EARLY STOPPING TRIGGERED")
    print(f"{make_separator(MINOR_SEP_CHAR, width)}")
    print(format_labeled_value("Reason", f"No improvement for {patience} consecutive epochs"))
    print(format_labeled_value("Best Loss Epoch", f"{best_loss_epoch + 1}/{total_epochs} (model restored to this)"))
    print(format_labeled_value("Best Validation Loss", f"{best_val_loss:.6f}"))
    print(format_labeled_value("Val Accuracy at Best Loss", f"{best_val_acc_at_best_loss * 100:.4f}%"))
    print(f"{make_separator(MINOR_SEP_CHAR, width)}")
    print(format_labeled_value("Best Accuracy Epoch", f"{best_acc_epoch + 1}/{total_epochs} (highest accuracy)"))
    print(format_labeled_value("Best Val Accuracy Overall", f"{best_val_acc_overall * 100:.4f}%"))
    print(f"{make_separator(MINOR_SEP_CHAR, width)}\n")

# =============================================================================
# SECTION 5: Results Formatting
# =============================================================================

def print_seed_results_summary(
    seed: int,
    model_name: str,
    results: Dict[str, Any],
    seed_idx: int = 1,
    total_seeds: int = 1,
    width: int = WIDTH_STANDARD,
    show_per_class: bool = False,
    show_per_angle: bool = False
) -> None:
    """
    Print comprehensive per-seed results summary with improved single-line accuracy display.

    Why: Provides immediate feedback after each seed completes with all key accuracies
         visible at a glance. Expanded terminology makes output clearer for users.

    Args:
        seed: Seed number
        model_name: Model name
        results: Results dictionary
        seed_idx: Seed index (1-based)
        total_seeds: Total number of seeds
        width: Output width
        show_per_class: Whether to show per-class accuracy breakdown
        show_per_angle: Whether to show per-angle accuracy breakdown
    """
    print(f"\n{make_separator(MINOR_SEP_CHAR, width)}")
    print(f" {model_name} >>> SEED {seed_idx} (#{seed} of {total_seeds}): {seed} >>> RESULTS")
    print(f"{make_separator(MINOR_SEP_CHAR, width)}")

    # =========================================================================
    # SECTION 1: Overall Accuracy Summary
    # =========================================================================
    print("ACCURACY SUMMARY:")

    # Extract accuracies with defaults
    # NOTE: Accuracies are already percentages from engine.evaluate()
    # DO NOT multiply by 100 again!
    val_acc: float = results.get('best_val_acc', 0.0)
    test_unrot_acc: float = results.get('test_unrot_acc', 0.0)
    test_rot_acc: float = results.get('test_rot_acc', 0.0)

    # Single-line compact format with expanded terminology
    print(f"  Validation: {val_acc:6.2f}%  |  "
          f"Test (Unrotated): {test_unrot_acc:6.2f}%  |  "
          f"Test (Rotated): {test_rot_acc:6.2f}%")

    # =========================================================================
    # SECTION 2: Per-Class Accuracy (Optional - can be verbose)
    # =========================================================================
    if show_per_class and 'per_class_acc' in results:
        per_class: Dict[int, float] = results['per_class_acc']
        if per_class:
            print("\nPER-CLASS ACCURACY (unrotated test set):")
            # Show in a compact format: 5 classes per line
            for i in range(0, 10, 5):
                line_parts: List[str] = []
                for c in range(i, min(i + 5, 10)):
                    if c in per_class:
                        line_parts.append(f"Class {c}: {per_class[c] * 100:5.2f}%")
                print("  " + "  |  ".join(line_parts))

    # =========================================================================
    # SECTION 3: Per-Angle Average Accuracy (Optional)
    # =========================================================================
    if show_per_angle and 'per_class_acc' in results:
        # Compute per-angle averages if we have per_class_rotated_accuracy
        # This is stored as Dict[Tuple[int, int], float] where tuple is (class, angle)
        # For now, just note this feature for future implementation
        print("\n[Per-angle accuracy breakdown available in detailed report]")

    # =========================================================================
    # SECTION 4: Invariance Metrics
    # =========================================================================
    print("\nGEOMETRIC INVARIANCE METRICS:")

    if 'rotation_inv_error' in results:
        print(format_labeled_value("Rotation Invariance Error", f"{results['rotation_inv_error']:.4f}"))
    if 'invariance_l2_error' in results:
        err = results['invariance_l2_error']
        if err is not None:
            print(format_labeled_value("Inversion L2 Error", format_scientific(err)))
        else:
            print(format_labeled_value("Inversion L2 Error", "N/A (non-TQF model)"))

    # =========================================================================
    # SECTION 5: Model Specifications
    # =========================================================================
    print("\nMODEL SPECIFICATIONS:")

    if 'params' in results:
        print(format_labeled_value("Total Parameters", format_params(results['params'])))
    if 'flops' in results:
        print(format_labeled_value("FLOPs (Forward Pass)", format_flops(results['flops'])))

    # =========================================================================
    # SECTION 6: Performance Metrics
    # =========================================================================
    print("\nPERFORMANCE METRICS:")

    if 'inference_time_ms' in results:
        print(format_labeled_value("Mean Inference Time", format_time_ms(results['inference_time_ms'])))
    if 'train_time_total' in results:
        print(format_labeled_value("Total Training Time", format_time_seconds(results['train_time_total'])))
    if 'best_loss_epoch' in results:
        print(format_labeled_value("Epochs to Best Loss", str(results['best_loss_epoch'] + 1)))
    if 'best_acc_epoch' in results:
        print(format_labeled_value("Epochs to Best Accuracy", str(results['best_acc_epoch'] + 1)))
    if 'train_time_per_epoch' in results:
        print(format_labeled_value("Average Time per Epoch", format_time_seconds(results['train_time_per_epoch'])))

    print(f"{make_separator(MINOR_SEP_CHAR, width)}\n")

# =============================================================================
# SECTION 6: Persistent Result Logging (Disk I/O)
# =============================================================================

def _make_result_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single seed result dict to JSON-serializable form.

    Handles numpy types (float64, int64, ndarray) and per_class_acc dict
    keys that may be integer types.

    Args:
        result: Raw result dict from run_single_seed_experiment()
    Returns:
        Dict with all values converted to native Python types
    """
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if k == 'per_class_acc':
            out[k] = {str(cls): float(acc) for cls, acc in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def save_seed_result_to_disk(
    result: Dict[str, Any],
    output_path: str,
    experiment_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Incrementally save a seed result to a JSON file on disk.

    Called after each seed completes so partial results survive crashes
    or session expiry. The file accumulates results across seeds and models.

    Output format (data/output/results_YYYYMMDD_HHMMSS.json):
        {
            "status": "in_progress" | "completed",
            "started_at": "YYYY-MM-DD HH:MM:SS",
            "last_updated": "YYYY-MM-DD HH:MM:SS",
            "config": { ... },           // experiment configuration (first call only)
            "results": {
                "TQF-ANN": [ {seed_result}, ... ],
                "FC-MLP": [ ... ]
            },
            "summary": { ... }           // populated by save_final_summary_to_disk()
        }

    Args:
        result: Single seed result dict from run_single_seed_experiment()
        output_path: Path to the JSON results file
        experiment_config: Optional experiment configuration to store once
    """
    # Load existing data or initialize
    data: Dict[str, Any]
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'status': 'in_progress',
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': {},
            'summary': {}
        }
        if experiment_config:
            data['config'] = experiment_config

    model_name: str = result['model_name']
    if model_name not in data['results']:
        data['results'][model_name] = []

    data['results'][model_name].append(_make_result_serializable(result))
    data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logging.info(f"\nSaved seed {result['seed']} results to {output_path}")


def save_final_summary_to_disk(
    summary: Dict[str, Dict[str, Tuple[float, float]]],
    output_path: str
) -> None:
    """
    Save final aggregated summary (mean +/- std) to the results JSON file.

    Also writes a human-readable .txt companion file with the same basename.

    Args:
        summary: Aggregated statistics from compare_models_statistical()
                 Format: {model_name: {metric: (mean, std), ...}, ...}
        output_path: Path to the JSON results file
    """
    data: Dict[str, Any]
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        data = {'results': {}, 'summary': {}}

    # Convert summary tuples to serializable dicts
    serializable_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, stats in summary.items():
        serializable_summary[model_name] = {}
        for metric, (mean, std) in stats.items():
            serializable_summary[model_name][metric] = {
                'mean': float(mean),
                'std': float(std)
            }

    data['summary'] = serializable_summary
    data['status'] = 'completed'
    data['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Also write a human-readable text summary
    txt_path: str = output_path.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write("FINAL MODEL COMPARISON (Mean +/- Std)\n")
        f.write("=" * 120 + "\n")
        f.write(
            f"{'Model':<20}  "
            f"{'Val Acc (%)':<15} "
            f"{'Test Acc (%)':<15} "
            f"{'Rot Acc (%)':<15}  "
            f"{'Params (k)':<15}   "
            f"{'FLOPs (M)':<15}"
            f"{'Inf Time (ms)':<15}\n"
        )
        f.write("-" * 120 + "\n")
        for model_name, stats in summary.items():
            val_mean, val_std = stats['val_acc']
            test_mean, test_std = stats['test_unrot_acc']
            rot_mean, rot_std = stats['test_rot_acc']
            params_mean, params_std = stats['params']
            flops_mean, flops_std = stats['flops']
            time_mean, time_std = stats['inference_time_ms']
            f.write(
                f"{model_name:<20} "
                f"{val_mean:6.2f}+/-{val_std:4.2f}   "
                f"{test_mean:6.2f}+/-{test_std:4.2f}   "
                f"{rot_mean:6.2f}+/-{rot_std:4.2f}   "
                f"{params_mean/1e3:7.1f}+/-{params_std/1e3:4.1f}   "
                f"{flops_mean/1e6:7.1f}+/-{flops_std/1e6:4.1f}   "
                f"{time_mean:6.2f}+/-{time_std:4.2f}\n"
            )
        f.write("=" * 120 + "\n")

    logging.info(f"Final summary saved to {output_path} and {txt_path}")
