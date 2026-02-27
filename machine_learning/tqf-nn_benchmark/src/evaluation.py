"""
evaluation.py - Evaluation Metrics and Statistical Analysis

This module implements comprehensive model evaluation metrics, statistical tests,
and performance analysis tools for TQF-NN and baseline model comparison. It provides
both standard classification metrics and TQF-specific geometric verification measures.

Key Features:
- Basic Metrics: Accuracy, loss, per-class accuracy, trainable parameter counting
- Rotation Invariance Error: Measure of prediction consistency across Z6 orbit
- Statistical Tests: Paired t-tests with p-values and confidence intervals
- TQF Verification: Inversion consistency metrics, self-duality compliance checks
- Computational Efficiency: FLOPs estimation (via ptflops), inference time measurement
- Aggregation Utilities: Multi-seed mean/std computation with NaN handling
- Comprehensive Reporting: Formatted metric tables with statistical significance markers

Scientific Rationale:
The rotation invariance error quantifies how much a model's predictions change under
rotations from the Z6 symmetry group (0, 60, 120, 180, 240, 300 degrees). For TQF-ANN,
which has inherent hexagonal symmetry, lower rotation invariance error indicates better
exploitation of the geometric structure.

Usage:
    from evaluation import (
        compute_per_class_accuracy,
        compute_rotation_invariance_error,
        compute_statistical_significance,
        estimate_model_flops
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

import math
import os
import time
import logging
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import (
    TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT,
    TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT,
    TQF_Z6_ORBIT_MIXING_ROTATION_PAD_DEFAULT
)

# =============================================================================
# SECTION 1: Performance Metrics
# =============================================================================

def estimate_model_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 784)
) -> int:
    """
    Estimate FLOPs (floating point operations) for one forward pass.

    Simplified estimation based on layer types when ptflops is unavailable.

    Args:
        model: Neural network model
        input_shape: Input tensor shape
    Returns:
        Estimated FLOPs
    """
    total_flops: int = 0

    def count_linear_flops(in_features: int, out_features: int) -> int:
        # FLOPs = 2 * in * out (multiply-add)
        return 2 * in_features * out_features

    def count_conv2d_flops(
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        output_size: Tuple[int, int]
    ) -> int:
        # FLOPs = 2 * Cin * Kh * Kw * Hout * Wout * Cout
        return (2 * in_channels * kernel_size[0] * kernel_size[1] *
                output_size[0] * output_size[1] * out_channels)

    # Traverse model layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_flops += count_linear_flops(
                module.in_features,
                module.out_features
            )
        elif isinstance(module, nn.Conv2d):
            # Estimate output size (simplified)
            out_h: int = 28 // 2  # Assume pooling
            out_w: int = 28 // 2
            total_flops += count_conv2d_flops(
                module.in_channels,
                module.out_channels,
                (module.kernel_size[0], module.kernel_size[1]),
                (out_h, out_w)
            )

    return total_flops

def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_trials: int = 100
) -> Dict[str, float]:
    """
    Measure inference time statistics.

    Args:
        model: Neural network model
        input_shape: Input tensor shape (batch_size, features)
        device: Computation device
        num_trials: Number of timing trials
    Returns:
        Dict with mean, std, min, max times (ms)
    """
    model.eval()
    times: List[float] = []

    # Check model capability once (avoid repeated introspection inside loops)
    has_inv_loss: bool = (
        hasattr(model, 'forward') and
        'return_inv_loss' in model.forward.__code__.co_varnames
    )

    # Create dummy input once and reuse for all trials.
    # Timing measures model compute, not input allocation.
    dummy_input: torch.Tensor = torch.randn(input_shape, device=device)

    # Warm-up (ensures CUDA kernels are compiled/cached before timing)
    with torch.no_grad():
        for _ in range(10):
            if has_inv_loss:
                _ = model(dummy_input, return_inv_loss=False)
            else:
                _ = model(dummy_input)

    # Timed trials
    with torch.no_grad():
        for _ in range(num_trials):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start: float = time.time()

            if has_inv_loss:
                _ = model(dummy_input, return_inv_loss=False)
            else:
                _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed: float = (time.time() - start) * 1000  # ms

            times.append(elapsed)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times)
    }

# =============================================================================
# SECTION 3: Accuracy Metrics
# =============================================================================

def compute_per_class_accuracy_from_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 10
) -> Dict[int, float]:
    """
    Compute per-class accuracy from prediction and target tensors.

    Helper function for unit testing and analysis of pre-computed predictions.

    Args:
        predictions: Predicted class indices (1D tensor)
        targets: Ground truth class indices (1D tensor)
        num_classes: Number of classes
    Returns:
        Dict mapping class_id -> accuracy
    """
    class_correct: np.ndarray = np.zeros(num_classes)
    class_total: np.ndarray = np.zeros(num_classes)

    predictions_np: np.ndarray = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    targets_np: np.ndarray = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    for label, pred in zip(targets_np, predictions_np):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    # Compute per-class accuracy
    per_class_acc: Dict[int, float] = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc[c] = class_correct[c] / class_total[c]
        else:
            per_class_acc[c] = 0.0

    return per_class_acc

def compute_per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10
) -> Dict[int, float]:
    """
    Compute per-class accuracy on standard (unrotated) dataset.

    Args:
        model: Neural network model
        loader: Data loader
        device: Computation device
        num_classes: Number of classes
    Returns:
        Dict mapping class_id -> accuracy
    """
    model.eval()

    class_correct: np.ndarray = np.zeros(num_classes)
    class_total: np.ndarray = np.zeros(num_classes)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)

            # Forward pass
            if hasattr(model, 'forward') and 'return_inv_loss' in model.forward.__code__.co_varnames:
                outputs = model(inputs, return_inv_loss=False)
            else:
                outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            # Transfer both tensors to CPU once per batch (not per sample)
            labels_np: np.ndarray = labels.cpu().numpy()
            predicted_np: np.ndarray = predicted.cpu().numpy()

            # Per-class stats
            for label, pred in zip(labels_np, predicted_np):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    # Compute per-class accuracy
    per_class_acc: Dict[int, float] = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc[c] = class_correct[c] / class_total[c]
        else:
            per_class_acc[c] = 0.0

    return per_class_acc

def compute_rotation_invariance_error_from_outputs(
    outputs_dict: Dict[int, torch.Tensor]
) -> float:
    """
    Compute rotation invariance error from pre-computed outputs dictionary.

    Helper function for unit testing rotation invariance metrics.
    Computes variance in softmax probabilities across different rotation angles.

    Args:
        outputs_dict: Dict mapping rotation_angle -> output logits tensor (B, num_classes)
    Returns:
        Mean variance in softmax outputs across rotations (lower is better)
    """
    if not outputs_dict:
        return 0.0

    # Convert all outputs to softmax probabilities
    probs_dict: Dict[int, np.ndarray] = {}
    for angle, logits in outputs_dict.items():
        if isinstance(logits, torch.Tensor):
            probs: torch.Tensor = F.softmax(logits, dim=1)
            probs_dict[angle] = probs.detach().cpu().numpy()
        else:
            # Assume numpy array
            probs_dict[angle] = logits

    # Compute variance across rotations for each sample
    angles: List[int] = sorted(probs_dict.keys())
    batch_size: int = probs_dict[angles[0]].shape[0]
    num_classes: int = probs_dict[angles[0]].shape[1]

    variances: List[float] = []

    for sample_idx in range(batch_size):
        # Collect probabilities for this sample across all rotations
        sample_probs: List[np.ndarray] = []
        for angle in angles:
            sample_probs.append(probs_dict[angle][sample_idx])

        # Compute variance across rotations for each class
        sample_probs_array: np.ndarray = np.array(sample_probs)  # (num_rotations, num_classes)
        sample_variance: float = float(np.mean(np.var(sample_probs_array, axis=0)))
        variances.append(sample_variance)

    # Return mean variance across all samples
    return float(np.mean(variances))

def compute_rotation_invariance_error(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rotation_angles: List[int] = [0, 60, 120, 180, 240, 300]
) -> float:
    """
    Compute rotation invariance error: variance in predictions across rotations.

    Lower is better (perfect invariance = 0 variance).

    Extracts rotation metadata from filenames in the dataset, making it robust
    to shuffled data and maintaining scientific rigor per TQF specification.

    Args:
        model: Neural network model
        loader: Data loader (rotated test set)
        device: Computation device
        rotation_angles: List of expected rotation angles (for validation)
    Returns:
        Mean variance in softmax outputs across rotations
    """
    model.eval()

    # Group samples by (base_image_id, class) to compare across rotations
    sample_outputs: Dict[Tuple[int, int], List[np.ndarray]] = {}

    # Access the dataset to extract metadata from filenames
    dataset = loader.dataset
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices'):
        # Using SubsetRandomSampler - get actual indices
        indices: List[int] = list(loader.sampler.indices)
    else:
        # Sequential access
        indices: List[int] = list(range(len(dataset)))

    with torch.no_grad():
        batch_start_idx: int = 0
        for inputs, labels in loader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)
            batch_size: int = inputs.size(0)

            # Forward pass
            if hasattr(model, 'forward') and 'return_inv_loss' in model.forward.__code__.co_varnames:
                outputs = model(inputs, return_inv_loss=False)
            else:
                outputs = model(inputs)

            probs: torch.Tensor = F.softmax(outputs, dim=1)

            # Transfer the full batch to CPU once (avoids per-sample GPU sync).
            probs_np: np.ndarray = probs.cpu().numpy()       # (B, C)
            labels_np: np.ndarray = labels.cpu().numpy()     # (B,)

            # Use pre-parsed metadata from CustomMNIST._metadata when available
            # (avoids 3 string-split calls per sample in this hot loop).
            has_metadata: bool = hasattr(dataset, 'get_metadata')

            for i in range(batch_size):
                dataset_idx: int = indices[batch_start_idx + i]

                if has_metadata:
                    base_img_id, _ = dataset.get_metadata(dataset_idx)
                else:
                    # Fallback: parse filename on the fly (legacy datasets)
                    img_path, _ = dataset.samples[dataset_idx]
                    filename: str = os.path.basename(img_path)
                    base_img_id = int(filename.split('_')[1])

                label_val: int = int(labels_np[i])
                key: Tuple[int, int] = (base_img_id, label_val)

                if key not in sample_outputs:
                    sample_outputs[key] = []
                sample_outputs[key].append(probs_np[i])

            batch_start_idx += batch_size

    # Compute variance across rotations for each base image
    variances: List[float] = []
    for key, probs_list in sample_outputs.items():
        if len(probs_list) < 2:
            continue  # Need multiple rotations to compute variance

        # Stack and compute variance across rotations
        probs_array: np.ndarray = np.array(probs_list)  # (num_rotations, num_classes)
        # Variance across rotations for each class, then mean across classes
        var: float = np.var(probs_array, axis=0).mean()
        variances.append(var)

    mean_variance: float = np.mean(variances) if variances else 0.0
    return mean_variance

# =============================================================================
# SECTION 4: TQF-Specific Metrics
# =============================================================================

def compute_inversion_consistency_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute inversion consistency metrics for TQF models.

    Measures:
    - Mean inv loss: Average MSE between outer/inner logits
    - Max inv loss: Worst-case disagreement
    - Inv accuracy: % samples where outer and inner agree on class

    Args:
        model: TQF model with dual outputs
        loader: Data loader
        device: Computation device
    Returns:
        Dict of metrics
    """
    if not hasattr(model, 'dual_output'):
        return {'inv_loss_mean': 0.0, 'inv_loss_max': 0.0, 'inv_accuracy': 0.0}

    model.eval()

    inv_losses: List[float] = []
    agreements: List[bool] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs: torch.Tensor = inputs.to(device)

            # Flatten input if needed (handle (B, 784), (B, 1, 28, 28), or (B, 28, 28))
            if inputs.dim() == 4:  # (B, C, H, W)
                batch_size: int = inputs.size(0)
                inputs = inputs.view(batch_size, -1)
            elif inputs.dim() == 3:  # (B, H, W)
                batch_size: int = inputs.size(0)
                inputs = inputs.view(batch_size, -1)

            # Forward to get features
            pre_feats: torch.Tensor = model.pre_encoder(inputs)
            boundary_feats: torch.Tensor = model.boundary_encoder(pre_feats)

            # T24 binner returns sector features directly (B, 6, H)
            outer_sector_feats, inner_sector_feats = model.radial_binner(
                boundary_feats,
                use_hop_attention=model.use_dual_metric
            )

            # Get ensemble logits and inversion loss from dual output
            ensemble_logits, inv_loss_batch = model.dual_output(
                outer_sector_feats,
                return_inversion_loss=True
            )

            # Compute per-sample inversion loss
            # outer_sector_feats is already (batch, 6, hidden_dim) from aggregate_features_by_sector

            # Outer logits (sector-wise) using pre-aggregated sector features
            sector_weights_normalized: torch.Tensor = F.softmax(model.dual_output.sector_weights, dim=0)
            outer_logits_per_sector: torch.Tensor = model.dual_output.classification_head(outer_sector_feats)
            outer_logits: torch.Tensor = torch.einsum('bsc,s->bc', outer_logits_per_sector, sector_weights_normalized)

            # Inner logits (apply circle inversion bijection)
            if hasattr(model.dual_output, 'apply_circle_inversion_bijection'):
                inner_sector_feats: torch.Tensor = model.dual_output.apply_circle_inversion_bijection(outer_sector_feats)
                inner_logits_per_sector: torch.Tensor = model.dual_output.classification_head(inner_sector_feats)
                inner_logits: torch.Tensor = torch.einsum('bsc,s->bc', inner_logits_per_sector, sector_weights_normalized)
            elif hasattr(model.dual_output, 'apply_inversion'):
                # Old DualOutputHead: has apply_inversion and inner_head
                inner_feats: torch.Tensor = model.dual_output.apply_inversion(outer_sector_feats)
                inner_feats_pooled: torch.Tensor = inner_feats.mean(dim=1)
                inner_logits: torch.Tensor = model.dual_output.inner_head(inner_feats_pooled)
            else:
                # Fallback: use outer logits as inner logits (perfect consistency)
                inner_logits: torch.Tensor = outer_logits.clone()

            # Compute per-sample loss
            inv_loss: torch.Tensor = F.mse_loss(
                outer_logits, inner_logits, reduction='none'
            ).mean(dim=1)

            # Transfer both result tensors to CPU in a single operation each
            inv_losses.extend(inv_loss.cpu().numpy())

            # Agreement: do outer and inner predict same class?
            outer_pred: torch.Tensor = torch.argmax(outer_logits, dim=1)
            inner_pred: torch.Tensor = torch.argmax(inner_logits, dim=1)
            agreements.extend((outer_pred == inner_pred).cpu().numpy())

    metrics: Dict[str, float] = {
        'inv_loss_mean': np.mean(inv_losses) if inv_losses else 0.0,
        'inv_loss_max': np.max(inv_losses) if inv_losses else 0.0,
        'inv_accuracy': np.mean(agreements) if agreements else 0.0
    }

    return metrics

# =============================================================================
# SECTION 5: Statistical Analysis
# =============================================================================

def compute_statistical_significance(
    scores_a: List[float],
    scores_b: List[float],
    test: str = 'ttest'
) -> Dict[str, float]:
    """
    Compute statistical significance between two sets of scores.

    Args:
        scores_a: Scores from model A (across seeds)
        scores_b: Scores from model B (across seeds)
        test: Statistical test ('ttest', 'wilcoxon', 'mannwhitneyu')
    Returns:
        Dict with keys: 'p_value', 't_statistic', 'effect_size' (Cohen's d)
    """
    from scipy import stats

    if test == 'ttest':
        stat, p_val = stats.ttest_ind(scores_a, scores_b)
    elif test == 'wilcoxon':
        stat, p_val = stats.wilcoxon(scores_a, scores_b)
    elif test == 'mannwhitneyu':
        stat, p_val = stats.mannwhitneyu(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test}")

    # Compute Cohen's d effect size
    mean_a: float = np.mean(scores_a)
    mean_b: float = np.mean(scores_b)
    std_a: float = np.std(scores_a, ddof=1) if len(scores_a) > 1 else 0.0
    std_b: float = np.std(scores_b, ddof=1) if len(scores_b) > 1 else 0.0

    # Pooled standard deviation
    n_a: int = len(scores_a)
    n_b: int = len(scores_b)
    if n_a > 1 and n_b > 1:
        pooled_std: float = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        cohens_d: float = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
    else:
        cohens_d: float = 0.0

    return {
        'p_value': float(p_val),
        't_statistic': float(stat),
        'effect_size': float(cohens_d)
    }

# =============================================================================
# SECTION 5: Orbit Mixing Evaluation
# =============================================================================

def adaptive_orbit_mixing(
    logits_per_variant: List[torch.Tensor],
    temperature: float = TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    confidence_mode: str = TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT,
    aggregation_mode: str = TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT,
    top_k: Optional[int] = TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT,
    adaptive_temp: bool = TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT,
    adaptive_temp_alpha: float = TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT
) -> torch.Tensor:
    """
    Combine multiple prediction variants using confidence-weighted averaging.

    Each variant's contribution is weighted by a per-variant confidence score.
    Lower temperatures sharpen the weights toward the most confident variant;
    higher temperatures approach uniform averaging.

    Args:
        logits_per_variant: List of logit tensors, each (batch, num_classes)
        temperature: Softmax temperature for confidence weighting.
            Lower = sharper (most confident variant dominates).
            Higher = more uniform averaging.
        confidence_mode: Signal used to score each variant's confidence.
            'max_logit' (default): maximum logit value.
            'margin': top-1 minus top-2 logit (decision margin).
        aggregation_mode: Space in which weighted averaging is performed.
            'logits' (default): raw logit space.
            'probs': probability space (softmax before averaging).
            'log_probs': log-probability space (log-softmax; geometric mean).
        top_k: If set, keep only the top-K most confident variants before
            averaging. None (default) uses all N variants.
        adaptive_temp: If True, scale temperature per-sample by the entropy of
            preliminary weights. High-entropy samples (all variants similarly
            confident) get a higher temperature for smoother averaging.
        adaptive_temp_alpha: Controls sensitivity of adaptive temperature.
            0 = no adaptation; larger values → stronger entropy scaling.

    Returns:
        Weighted average in the chosen aggregation space (batch, num_classes)
    """
    if len(logits_per_variant) == 1:
        return logits_per_variant[0]

    stacked: torch.Tensor = torch.stack(logits_per_variant, dim=0)  # (N, B, C)
    N, B, C = stacked.shape

    # ── Confidence score per variant ──
    if confidence_mode == 'margin':
        top2: torch.Tensor = stacked.topk(2, dim=2).values  # (N, B, 2)
        confidence: torch.Tensor = top2[..., 0] - top2[..., 1]  # (N, B)
    else:  # 'max_logit' (default)
        confidence: torch.Tensor = stacked.max(dim=2).values  # (N, B)

    # ── Top-K filtering (select most confident K variants) ──
    if top_k is not None and top_k < N:
        _, topk_idx = confidence.topk(top_k, dim=0)  # (top_k, B)
        expand_idx = topk_idx.unsqueeze(2).expand(-1, -1, C)
        stacked = stacked.gather(0, expand_idx)  # (top_k, B, C)
        confidence = confidence.gather(0, topk_idx)  # (top_k, B)
        N = top_k

    # ── Temperature (adaptive or fixed) ──
    if adaptive_temp and N > 1:
        # Preliminary weights at base temperature to estimate per-sample entropy
        w_base: torch.Tensor = F.softmax(confidence / temperature, dim=0)  # (N, B)
        # Entropy ∈ [0, log(N)]; high = all variants similarly confident
        entropy: torch.Tensor = -(w_base * (w_base + 1e-10).log()).sum(dim=0)  # (B,)
        # Scale T up when variants are similar (high entropy → more averaging)
        adaptive_T: torch.Tensor = temperature * (
            1.0 + adaptive_temp_alpha * entropy / math.log(N)
        )  # (B,)
        weights: torch.Tensor = F.softmax(
            confidence / adaptive_T.unsqueeze(0), dim=0
        )  # (N, B)
    else:
        weights: torch.Tensor = F.softmax(confidence / temperature, dim=0)  # (N, B)

    # ── Aggregation ──
    if aggregation_mode == 'probs':
        values: torch.Tensor = F.softmax(stacked, dim=2)  # (N, B, C)
    elif aggregation_mode == 'log_probs':
        values: torch.Tensor = F.log_softmax(stacked, dim=2)  # (N, B, C)
    else:  # 'logits' (default)
        values: torch.Tensor = stacked

    return (values * weights.unsqueeze(2)).sum(dim=0)  # (B, C)


def classify_from_sector_features(
    model: nn.Module,
    outer_sector_feats: torch.Tensor,
    inner_sector_feats: torch.Tensor,
    swap_weights: bool = False,
) -> torch.Tensor:
    """
    Run the TQF-ANN classification pipeline on pre-computed zone features.

    This mirrors the classification steps in TQFANN.forward() — applying the
    shared classification_head per sector, weighting by learned sector_weights,
    and combining zones via confidence_weighted_ensemble. No gradient tracking.

    Args:
        model: TQFANN model instance (must have dual_output attribute)
        outer_sector_feats: Outer zone features (batch, 6, hidden_dim)
        inner_sector_feats: Inner zone features (batch, 6, hidden_dim)
        swap_weights: If True, flip the confidence weighting so the normally-
            less-confident zone dominates. Used by T24 zone-swap to produce
            a genuinely different prediction (the default ensemble is symmetric
            in outer/inner, so simply swapping args is a no-op).

    Returns:
        Ensemble logits (batch, num_classes)
    """
    sector_weights: torch.Tensor = F.softmax(model.dual_output.sector_weights, dim=0)

    outer_logits_per_sector: torch.Tensor = model.dual_output.classification_head(outer_sector_feats)
    outer_logits: torch.Tensor = torch.einsum('bsc,s->bc', outer_logits_per_sector, sector_weights)

    inner_logits_per_sector: torch.Tensor = model.dual_output.classification_head(inner_sector_feats)
    inner_logits: torch.Tensor = torch.einsum('bsc,s->bc', inner_logits_per_sector, sector_weights)

    if swap_weights:
        return model.dual_output.confidence_weighted_ensemble(
            outer_logits, inner_logits, swap_weights=True
        )
    return model.dual_output.confidence_weighted_ensemble(outer_logits, inner_logits)


def evaluate_with_orbit_mixing(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_z6: bool = False,
    use_d6: bool = False,
    use_t24: bool = False,
    temp_rotation: float = TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    temp_reflection: float = TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    temp_inversion: float = TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT,
    confidence_mode: str = TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT,
    aggregation_mode: str = TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT,
    top_k: Optional[int] = TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT,
    adaptive_temp: bool = TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT,
    adaptive_temp_alpha: float = TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT,
    rotation_mode: str = TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT,
    rotation_padding_mode: str = TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT,
    pad_before_rotate: int = TQF_Z6_ORBIT_MIXING_ROTATION_PAD_DEFAULT,
    use_amp: bool = True,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Evaluate TQF-ANN with orbit mixing over independent symmetry levels.

    Three independently toggleable levels of symmetry exploitation:
    - Level 1 (Z6): Input-space rotation — 6 full forward passes (0-300 deg, step 60)
    - Level 2 (D6): Feature-space reflection — sector permutation on both zones
    - Level 3 (T24): Zone-swap — flip inner/outer confidence weighting in dual ensemble

    Each level can be enabled independently or in any combination. When multiple
    levels are active, averaging is applied hierarchically:
    - Inner: D6 reflection averaging (temp_reflection)
    - Middle: T24 zone-swap averaging (temp_inversion)
    - Outer: Z6 rotation averaging (temp_rotation)

    Args:
        model: TQFANN model instance
        loader: DataLoader for evaluation
        device: Torch device
        use_z6: Enable Z6 rotation orbit mixing (6 input rotations)
        use_d6: Enable D6 reflection orbit mixing (sector permutation)
        use_t24: Enable T24 zone-swap orbit mixing (flip zone confidence weights)
        temp_rotation: Temperature for Z6 rotation averaging
        temp_reflection: Temperature for D6 reflection averaging
        temp_inversion: Temperature for T24 zone-swap averaging
        confidence_mode: Confidence signal mode for Z6 weighting ('max_logit' or 'margin')
        aggregation_mode: Aggregation space for Z6 weighting ('logits', 'probs', 'log_probs')
        top_k: Top-K variant selection for Z6 (None = all 6)
        adaptive_temp: Enable per-sample adaptive temperature for Z6
        adaptive_temp_alpha: Sensitivity of adaptive temperature scaling
        rotation_mode: Interpolation mode for image rotation ('bilinear' or 'bicubic')
        rotation_padding_mode: Padding mode for rotated corners ('zeros' or 'border')
        pad_before_rotate: Pixels to pad before rotating then crop back (0=disabled)
        use_amp: Enable automatic mixed precision
        verbose: Log per-batch progress

    Returns:
        Tuple of (average_loss, accuracy_percent)
    """
    # Lazy imports to avoid circular dependency (engine.py imports evaluation)
    from engine import rotate_batch_images
    from symmetry_ops import apply_d6_reflection_to_sectors

    model.eval()

    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    # Z6: iterate over 6 rotation angles, or just [0] (no rotation) if disabled
    z6_angles: List[int] = [0, 60, 120, 180, 240, 300] if use_z6 else [0]

    if verbose:
        levels: List[str] = []
        if use_z6:
            levels.append("Z6")
        if use_d6:
            levels.append("D6")
        if use_t24:
            levels.append("T24")
        logging.info(f"Orbit mixing evaluation: {'+'.join(levels) if levels else 'none'}")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size: int = inputs.size(0)

            # ── Level 1: Z6 input-space rotations ──
            rotation_logits: List[torch.Tensor] = []

            if use_z6:
                # Rotate all 6 angles upfront (grid is cached after the first call
                # for each angle, so subsequent batches incur only a GPU memcpy).
                rotated_all: List[torch.Tensor] = [
                    rotate_batch_images(
                        inputs, angle,
                        mode=rotation_mode,
                        padding_mode=rotation_padding_mode,
                        pad_before_rotate=pad_before_rotate
                    )
                    for angle in z6_angles
                ]
                # Stack to (6B, ...) and run a single forward pass.
                # TQF-ANN uses LayerNorm (per-sample), so per-image results are
                # bit-identical to running each rotation in a separate pass.
                batched_input: torch.Tensor = torch.cat(rotated_all, dim=0)

                with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                    batched_logits: torch.Tensor = model(batched_input, return_inv_loss=False)

                # Split (6B, C) → list of 6 × (B, C)
                split_logits: List[torch.Tensor] = list(
                    batched_logits.split(batch_size, dim=0)
                )

                # If D6 or T24 are active, retrieve and split cached sector features.
                if use_d6 or use_t24:
                    outer_all: torch.Tensor = model.get_cached_sector_features()   # (6B, 6, H)
                    inner_all: torch.Tensor = model.get_cached_inner_sector_features()  # (6B, 6, H)

                for i in range(len(z6_angles)):
                    current: torch.Tensor = split_logits[i]

                    if use_d6 or use_t24:
                        outer_feats: torch.Tensor = outer_all[i * batch_size:(i + 1) * batch_size]
                        inner_feats: torch.Tensor = inner_all[i * batch_size:(i + 1) * batch_size]

                    # ── Level 2: D6 feature-space reflection ──
                    if use_d6:
                        reflected_outer: torch.Tensor = apply_d6_reflection_to_sectors(
                            outer_feats, reflection_axis=0
                        )
                        reflected_inner: torch.Tensor = apply_d6_reflection_to_sectors(
                            inner_feats, reflection_axis=0
                        )
                        reflected_logits: torch.Tensor = classify_from_sector_features(
                            model, reflected_outer, reflected_inner
                        )
                        refl_stack = torch.stack([current, reflected_logits], dim=0)
                        refl_conf = refl_stack.max(dim=2).values
                        refl_w = F.softmax(refl_conf / temp_reflection, dim=0)
                        current = (refl_stack * refl_w.unsqueeze(2)).sum(dim=0)

                    # ── Level 3: T24 zone-swap (flip confidence weighting) ──
                    if use_t24:
                        swapped_logits: torch.Tensor = classify_from_sector_features(
                            model, outer_feats, inner_feats, swap_weights=True
                        )
                        if use_d6:
                            swapped_refl_logits: torch.Tensor = classify_from_sector_features(
                                model, reflected_outer, reflected_inner, swap_weights=True
                            )
                            sw_stack = torch.stack([swapped_logits, swapped_refl_logits], dim=0)
                            sw_conf = sw_stack.max(dim=2).values
                            sw_w = F.softmax(sw_conf / temp_reflection, dim=0)
                            swapped_logits = (sw_stack * sw_w.unsqueeze(2)).sum(dim=0)

                        inv_stack = torch.stack([current, swapped_logits], dim=0)
                        inv_conf = inv_stack.max(dim=2).values
                        inv_w = F.softmax(inv_conf / temp_inversion, dim=0)
                        current = (inv_stack * inv_w.unsqueeze(2)).sum(dim=0)

                    rotation_logits.append(current)

            else:
                # No Z6: single forward pass for the unrotated input (angle = 0).
                with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                    logits: torch.Tensor = model(inputs, return_inv_loss=False)

                current = logits

                if use_d6 or use_t24:
                    outer_feats = model.get_cached_sector_features()
                    inner_feats = model.get_cached_inner_sector_features()

                # ── Level 2: D6 feature-space reflection ──
                if use_d6:
                    reflected_outer = apply_d6_reflection_to_sectors(
                        outer_feats, reflection_axis=0
                    )
                    reflected_inner = apply_d6_reflection_to_sectors(
                        inner_feats, reflection_axis=0
                    )
                    reflected_logits = classify_from_sector_features(
                        model, reflected_outer, reflected_inner
                    )
                    refl_stack = torch.stack([current, reflected_logits], dim=0)
                    refl_conf = refl_stack.max(dim=2).values
                    refl_w = F.softmax(refl_conf / temp_reflection, dim=0)
                    current = (refl_stack * refl_w.unsqueeze(2)).sum(dim=0)

                # ── Level 3: T24 zone-swap (flip confidence weighting) ──
                if use_t24:
                    swapped_logits = classify_from_sector_features(
                        model, outer_feats, inner_feats, swap_weights=True
                    )
                    if use_d6:
                        swapped_refl_logits = classify_from_sector_features(
                            model, reflected_outer, reflected_inner, swap_weights=True
                        )
                        sw_stack = torch.stack([swapped_logits, swapped_refl_logits], dim=0)
                        sw_conf = sw_stack.max(dim=2).values
                        sw_w = F.softmax(sw_conf / temp_reflection, dim=0)
                        swapped_logits = (sw_stack * sw_w.unsqueeze(2)).sum(dim=0)

                    inv_stack = torch.stack([current, swapped_logits], dim=0)
                    inv_conf = inv_stack.max(dim=2).values
                    inv_w = F.softmax(inv_conf / temp_inversion, dim=0)
                    current = (inv_stack * inv_w.unsqueeze(2)).sum(dim=0)

                rotation_logits.append(current)

            # ── Final Z6 rotation averaging ──
            if use_z6:
                ensemble_logits: torch.Tensor = adaptive_orbit_mixing(
                    rotation_logits,
                    temperature=temp_rotation,
                    confidence_mode=confidence_mode,
                    aggregation_mode=aggregation_mode,
                    top_k=top_k,
                    adaptive_temp=adaptive_temp,
                    adaptive_temp_alpha=adaptive_temp_alpha
                )
            else:
                ensemble_logits = rotation_logits[0]

            # Standard evaluation metrics
            loss = F.cross_entropy(ensemble_logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(ensemble_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if verbose and batch_idx % 10 == 0:
                logging.info(f"  Batch {batch_idx}/{len(loader)}: "
                             f"acc={100.0 * correct / total:.2f}%")

    avg_loss: float = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy: float = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def compute_orbit_consistency_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    base_logits: torch.Tensor,
    num_rotations: int = 2
) -> Optional[torch.Tensor]:
    """
    Compute orbit consistency self-distillation loss (training-time only).

    Creates a consensus ensemble from base logits + extra rotated forward passes,
    then penalises each rotation (including base) for diverging from the ensemble
    via KL divergence. This encourages the model to produce consistent predictions
    across Z6 orbit members without requiring labelled data for the rotations.

    Only runs for TQF models (model must accept ``return_inv_loss`` kwarg).
    Returns None for non-TQF models so callers can skip without branching.

    Args:
        model: Neural network (TQF-ANN expected)
        inputs: Batch inputs currently being trained on (B, 784)
        base_logits: Logits from the current forward pass (B, num_classes).
            Must still be in the computation graph (not detached).
        num_rotations: Number of additional Z6 rotation passes (1-5).
            These are randomly sampled from {60, 120, 180, 240, 300}.

    Returns:
        Scalar KL-divergence loss averaged over all variants, or None.
    """
    # Only applies to TQF models
    if not hasattr(model, 'dual_output'):
        return None

    # Lazy import to avoid circular dependency
    from engine import rotate_batch_images

    import random
    extra_angles: List[int] = random.sample(
        [60, 120, 180, 240, 300], k=min(num_rotations, 5)
    )

    # Rotate all extra angles, then run a single batched forward pass.
    # TQF-ANN uses LayerNorm (per-sample), so results are identical to running
    # each rotation separately. This replaces num_rotations serial passes with one.
    batch_size: int = inputs.size(0)
    rotated_list: List[torch.Tensor] = [
        rotate_batch_images(inputs, angle) for angle in extra_angles
    ]
    batched_rotated: torch.Tensor = torch.cat(rotated_list, dim=0)   # (num_rot*B, 784)
    batched_rot_logits: torch.Tensor = model(batched_rotated, return_inv_loss=False)
    extra_logits: List[torch.Tensor] = list(batched_rot_logits.split(batch_size, dim=0))

    # Collect all logit tensors (base + extra rotations, all with gradients)
    all_logits: List[torch.Tensor] = [base_logits] + extra_logits

    # Build stop-gradient ensemble soft target
    stacked_detached: torch.Tensor = torch.stack(
        [l.detach() for l in all_logits], dim=0
    ).mean(dim=0)  # (B, C)
    ensemble_probs: torch.Tensor = F.softmax(stacked_detached, dim=1)  # (B, C)

    # KL divergence of each variant against ensemble target
    total_kl: torch.Tensor = torch.zeros(1, device=inputs.device)
    for logits in all_logits:
        log_probs: torch.Tensor = F.log_softmax(logits, dim=1)
        kl: torch.Tensor = F.kl_div(log_probs, ensemble_probs, reduction='batchmean')
        total_kl = total_kl + kl

    return total_kl / len(all_logits)
