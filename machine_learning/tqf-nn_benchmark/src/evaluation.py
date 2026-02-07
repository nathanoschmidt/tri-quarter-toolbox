"""
evaluation.py - Evaluation Metrics and Statistical Analysis

This module implements comprehensive model evaluation metrics, statistical tests,
and performance analysis tools for TQF-NN and baseline model comparison. It provides
both standard classification metrics and TQF-specific geometric verification measures.

Key Features:
- Basic Metrics: Accuracy, loss, per-class accuracy, trainable parameter counting
- Rotational Robustness: Z6 orbit mixing (averaging predictions over 60-degree rotations)
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
exploitation of the geometric structure. Z6 orbit mixing aggregates predictions across
all 6 rotations to leverage symmetry for improved robustness.

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
import statistics
from typing import Any, Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats

# Try to import ptflops for FLOPs measurement (graceful fallback)
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE: bool = True
except ImportError:
    PTFLOPS_AVAILABLE: bool = False
    logging.warning("ptflops not available, using simplified FLOPs estimation")

# =============================================================================
# SECTION 1: Basic Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Why: Essential for verifying that models are parameter-matched (~650k) in
         apples-to-apples benchmarks, ensuring fair comparisons of efficiency
         and capacity across architectures.

    Args:
        model: PyTorch neural network module
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def safe_mean_std(values: List[Any]) -> Tuple[float, float]:
    """
    Safely compute mean and std dev, filtering None/NaN values.

    Why: Handles incomplete or erroneous data in experiment aggregations
         (e.g., failed seeds); returns NaN for empty lists to propagate
         errors gracefully in scientific reporting.

    Args:
        values: List of numeric values (may contain None or NaN)
    Returns:
        Tuple of (mean, std_dev)
    """
    clean: List[float] = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return (float('nan'), float('nan'))
    return (statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0)

def safe_mean_std_seeds(values: List[Any]) -> Tuple[float, float]:
    """
    Safely compute mean and std dev across seeds, filtering None/NaN.

    Why: Aggregates multi-seed results robustly for statistical reliability;
         std=0 for single values aligns with scientific practice for
         low-sample variance estimation.

    Args:
        values: List of values across different random seeds
    Returns:
        Tuple of (mean, std_dev)
    """
    clean: List[float] = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return (float('nan'), float('nan'))
    mean: float = statistics.mean(clean)
    std: float = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return (mean, std)

def custom_scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Custom scatter-mean operation for aggregating messages in graph neural networks.

    Why: Efficiently computes means over indexed groups (e.g., node neighbors
         in TQF lattice); avoids built-in scatter_mean for compatibility and
         control in symmetry-reduced graphs.

    Args:
        src: Source tensor to scatter
        index: Index tensor for grouping
        dim_size: Output dimension size
    Returns:
        Scattered and averaged tensor
    """
    out: torch.Tensor = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    count: torch.Tensor = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(1).repeat(1, src.size(1)), src)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp(min=1.0).unsqueeze(1)

def rotate_image_tensor(
    x: torch.Tensor,
    angle_deg: float,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Rotate image tensor by specified angle in degrees.

    Why: Essential for Z6 orbit mixing during TQF-ANN inference. Applies
         rotation to input images at 60-degree increments (0, 60, 120, 180,
         240, 300) to leverage hexagonal symmetry for improved rotation
         invariance.

    Scientific rationale: TQF's hexagonal lattice has natural 60-degree
    rotational symmetry (Z6 group). Evaluating at all 6 orientations and
    combining predictions exploits this symmetry for robust classification.

    Args:
        x: Input tensor of shape (B, C, H, W) or (B, H, W) or (B, 784)
        angle_deg: Rotation angle in degrees (positive = counter-clockwise)
        mode: Interpolation mode ('bilinear' or 'nearest')
    Returns:
        Rotated tensor with same shape as input
    """
    original_shape: Tuple[int, ...] = x.shape
    needs_reshape: bool = False

    # Handle flattened input (B, 784)
    if x.dim() == 2 and x.size(1) == 784:
        batch_size: int = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        needs_reshape = True
    # Handle (B, H, W)
    elif x.dim() == 3:
        x = x.unsqueeze(1)  # Add channel dimension

    # Now x is (B, C, H, W)
    batch_size: int = x.size(0)
    num_channels: int = x.size(1)
    height: int = x.size(2)
    width: int = x.size(3)

    # Convert angle to radians
    angle_rad: float = angle_deg * np.pi / 180.0

    # Create rotation matrix
    cos_theta: float = np.cos(angle_rad)
    sin_theta: float = np.sin(angle_rad)

    # Affine transformation matrix for rotation around center
    # [cos -sin tx]
    # [sin  cos ty]
    theta: torch.Tensor = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0]
    ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # Generate affine grid
    grid: torch.Tensor = F.affine_grid(
        theta,
        x.size(),
        align_corners=False
    )

    # Apply rotation
    rotated: torch.Tensor = F.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=False
    )

    # Reshape back if needed
    if needs_reshape:
        rotated = rotated.view(batch_size, 784)

    return rotated

def evaluate_with_z6_orbit_mixing(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 0.3,
    use_amp: bool = True,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Evaluate model using Z6 orbit mixing with adaptive temperature weighting.

    Why: Implements TQF-ANN's full inference pipeline by evaluating input at
         all 6 rotations (Z6 orbit: 0deg, 60deg, 120deg, 180deg, 240deg, 300deg) and
         combining predictions using confidence-weighted mixing. This exploits
         hexagonal symmetry for superior rotation invariance.

    Scientific rationale (per TQF spec):
    - Low temperature (0.1-0.3): Sharp mixing, emphasizes most confident rotation
      Benefits: Handles partially-visible digits, leverages TQF equivariance
      Expected gain: +1-2% accuracy vs uniform averaging
    - High temperature (0.8-1.0): Uniform averaging (ensemble voting)
      Benefits: Robust to individual rotation errors
      Risk: Dilutes signal from correct rotation

    Computational cost: 6x forward passes per sample (unavoidable for orbit mixing)
    Memory: Batch size should be reduced 6x to maintain same VRAM usage

    Args:
        model: Neural network model (should be TQF-ANN for best results)
        loader: Data loader
        device: Computation device
        temperature: Temperature for softmax weighting (default: 0.3)
                    Lower = sharper (emphasize best), higher = uniform
        use_amp: Whether to use automatic mixed precision
        verbose: Whether to print progress
    Returns:
        Tuple of (average_loss, accuracy_percent)
    """
    model.eval()

    # Import adaptive_orbit_mixing from models_tqf
    try:
        from models_tqf import adaptive_orbit_mixing
    except ImportError:
        logging.error("Could not import adaptive_orbit_mixing from models_tqf")
        raise

    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    # Z6 rotation angles (60-degree increments)
    z6_angles: List[float] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]

    # For loss computation, we'll use the ensemble logits
    from models_tqf import LabelSmoothingCrossEntropy
    criterion: LabelSmoothingCrossEntropy = LabelSmoothingCrossEntropy(smoothing=0.0)

    with torch.no_grad():
        batch_count: int = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Collect logits from all 6 rotations
            logits_per_rotation: List[torch.Tensor] = []

            for angle in z6_angles:
                # Rotate input
                rotated_input: torch.Tensor = rotate_image_tensor(inputs, angle)

                # Forward pass with AMP
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Handle different model signatures
                    if hasattr(model, 'forward') and 'return_inv_loss' in model.forward.__code__.co_varnames:
                        logits = model(rotated_input, return_inv_loss=False)
                    else:
                        logits = model(rotated_input)

                logits_per_rotation.append(logits)

            # Apply adaptive orbit mixing with temperature
            ensemble_logits: torch.Tensor = adaptive_orbit_mixing(
                logits_per_rotation,
                temperature=temperature
            )

            # Compute loss on ensemble
            loss: torch.Tensor = criterion(ensemble_logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(ensemble_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_count += 1
            if verbose and batch_count % 10 == 0:
                logging.info(f"  Processed {batch_count} batches with Z6 orbit mixing...")

    avg_loss: float = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy: float = 100.0 * correct / total if total > 0 else 0.0

    if verbose:
        logging.info(
            f"Z6 Orbit Mixing Evaluation: "
            f"Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, "
            f"Temperature={temperature:.2f}"
        )

    return avg_loss, accuracy

# =============================================================================
# SECTION 2: Performance Metrics
# =============================================================================

def measure_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 1, 28, 28)
) -> Union[int, float]:
    """
    Measure FLOPs (floating-point operations) for ANN models.

    Why: Quantifies computational complexity as a proxy for energy/inference
         efficiency; crucial for comparing ANNs to SNNs in scientific benchmarks,
         with graceful error handling.

    Args:
        model: Neural network model
        input_shape: Input tensor shape (batch, channels, height, width)
    Returns:
        FLOPs count (0 if estimation fails)
    """
    if PTFLOPS_AVAILABLE:
        try:
            with torch.no_grad():
                flops, _ = get_model_complexity_info(
                    model, input_shape[1:], as_strings=False,
                    print_per_layer_stat=False, verbose=False,
                    input_constructor=lambda s: torch.randn(1, *s).to(next(model.parameters()).device)
                )
            return flops
        except Exception as e:
            logging.info(f"FLOPs estimation skipped ({type(e).__name__}): {e}")
            return 0.0
    else:
        # Fallback to simplified estimation
        return estimate_model_flops(model, input_shape)

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

    # Warm-up
    dummy_input: torch.Tensor = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            if hasattr(model, 'forward') and 'return_inv_loss' in model.forward.__code__.co_varnames:
                _ = model(dummy_input, return_inv_loss=False)
            else:
                _ = model(dummy_input)

    # Timed trials
    with torch.no_grad():
        for _ in range(num_trials):
            dummy_input = torch.randn(input_shape).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start: float = time.time()

            if hasattr(model, 'forward') and 'return_inv_loss' in model.forward.__code__.co_varnames:
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

            # Per-class stats
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
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

def compute_per_class_rotated_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    rotation_angles: List[int] = [0, 60, 120, 180, 240, 300]
) -> Dict[Tuple[int, int], float]:
    """
    Compute per-class accuracy for each rotation angle.

    Spec requirement: Report accuracy breakdown by digit class AND rotation.

    Args:
        model: Neural network model
        loader: Data loader
        device: Computation device
        num_classes: Number of classes
        rotation_angles: List of rotation angles (degrees)
    Returns:
        Dict mapping (class_id, rotation_angle) -> accuracy
    """
    model.eval()

    # Stats: [class, rotation]
    class_rot_correct: np.ndarray = np.zeros((num_classes, len(rotation_angles)))
    class_rot_total: np.ndarray = np.zeros((num_classes, len(rotation_angles)))

    with torch.no_grad():
        for inputs, labels, rotations in loader:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)
            rotations: torch.Tensor = rotations.to(device)

            # Forward pass
            if hasattr(model, 'forward') and 'return_inv_loss' in model.forward.__code__.co_varnames:
                outputs = model(inputs, return_inv_loss=False)
            else:
                outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            # Per-class, per-rotation stats
            for label, pred, rot in zip(
                labels.cpu().numpy(),
                predicted.cpu().numpy(),
                rotations.cpu().numpy()
            ):
                try:
                    rot_idx: int = rotation_angles.index(int(rot))
                except ValueError:
                    continue  # Skip unknown rotations

                class_rot_total[label, rot_idx] += 1
                if label == pred:
                    class_rot_correct[label, rot_idx] += 1

    # Compute per-class, per-rotation accuracy
    per_class_rot_acc: Dict[Tuple[int, int], float] = {}
    for c in range(num_classes):
        for r_idx, angle in enumerate(rotation_angles):
            if class_rot_total[c, r_idx] > 0:
                acc: float = class_rot_correct[c, r_idx] / class_rot_total[c, r_idx]
            else:
                acc: float = 0.0
            per_class_rot_acc[(c, angle)] = acc

    return per_class_rot_acc

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

            # Process each sample in batch
            for i in range(batch_size):
                # Get dataset index for this sample
                dataset_idx: int = indices[batch_start_idx + i]

                # Extract metadata from filename
                img_path, label_from_path = dataset.samples[dataset_idx]
                img_path: str
                label_from_path: int
                filename: str = os.path.basename(img_path)

                # Parse filename: test_00042_label_3_rot_120.png
                # Extract base image ID (e.g., 00042)
                base_img_id: int = int(filename.split('_')[1])

                # Extract rotation angle (e.g., 120)
                rot_angle: int = int(filename.split('_rot_')[1].split('.')[0])

                # Use actual label from forward pass (more robust)
                label_val: int = int(labels[i].item())

                # Group by (base_image_id, class)
                key: Tuple[int, int] = (base_img_id, label_val)

                if key not in sample_outputs:
                    sample_outputs[key] = []
                sample_outputs[key].append(probs[i].cpu().numpy())

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

            inv_losses.extend(inv_loss.cpu().numpy().tolist())

            # Agreement: do outer and inner predict same class?
            outer_pred: torch.Tensor = torch.argmax(outer_logits, dim=1)
            inner_pred: torch.Tensor = torch.argmax(inner_logits, dim=1)
            agree: torch.Tensor = (outer_pred == inner_pred)
            agreements.extend(agree.cpu().numpy().tolist())

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

def report_statistical_tests(
    raw_results: Dict[str, Any],
    symmetry_level: str = 'Z6'
) -> None:
    """
    Perform and log statistical tests (Welch's t-test) on rotated accuracies.

    Why: Rigorously tests if TQF models significantly outperform baselines
         (p < 0.01); uses Welch's for unequal variances, aligning with
         scientific best practices in benchmarks.

    Args:
        raw_results: Dict mapping model_name -> list of result dicts
        symmetry_level: TQF symmetry level for reporting
    """
    if 'TQF-ANN' in raw_results:
        tqf_vals: List[float] = [r['test_rot_acc'][0] for r in raw_results['TQF-ANN']]
        for base in ['FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']:
            if base in raw_results:
                base_vals: List[float] = [r['test_rot_acc'][0] for r in raw_results[base]]
                t_stat, p_val = stats.ttest_ind(tqf_vals, base_vals, equal_var=False)
                logging.info(
                    f"Welch's t-test (TQF-ANN {symmetry_level} vs {base}): "
                    f"t={t_stat:+.3f}, p={p_val:.2e} "
                    f"(significant if p < 0.01 per scientific best practice)"
                )

def print_comprehensive_metrics(
    model_name: str,
    metrics: Dict[str, any]
) -> None:
    """
    Print comprehensive evaluation metrics in formatted table.

    Args:
        model_name: Model name
        metrics: Dict of all metrics
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE METRICS: {model_name}")
    print(f"{'='*70}")

    # Standard metrics
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 70)

    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"{key:<30} {val:>15.4f}")
        elif isinstance(val, int):
            print(f"{key:<30} {val:>15,}")
        elif isinstance(val, dict):
            # Nested dict (e.g., per-class accuracies)
            if len(val) <= 10:
                for sub_key, sub_val in val.items():
                    print(f"  {str(sub_key):<28} {sub_val:>15.4f}")
        else:
            print(f"{key:<30} {str(val):>15}")

    print("=" * 70)
