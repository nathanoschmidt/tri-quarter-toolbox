"""
engine.py - Training Engine and Experiment Orchestration

This module provides the core training loop, validation, and multi-seed experiment
orchestration for TQF-NN benchmark experiments. It handles the complete lifecycle
of model training from initialization through convergence, with robust early stopping,
learning rate scheduling, and statistical aggregation across random seeds.

Key Features:
- TrainingEngine class with mixed precision training (AMP) and torch.compile support
- Comprehensive learning rate scheduling (warmup + cosine annealing)
- Early stopping with validation loss plateau detection
- TQF-specific geometric regularization losses (fractal self-similarity, box-counting)
- Periodic self-duality verification for TQF-ANN compliance
- Multi-seed experiment runner with statistical aggregation (mean ± std)
- Paired t-test statistical comparison between models
- Rotation invariance testing (0, 60, 120, 180, 240, 300 degrees for Z6 symmetry)
- Per-epoch progress tracking with comprehensive metrics logging
- Final comparison table generation with confidence intervals

Scientific Rationale:
Multi-seed experiments (typically 5-10 seeds) provide statistical reliability for
performance comparisons. The early stopping criterion (validation loss) prevents
overfitting while the patience parameter allows transient plateaus during training.
TQF geometric losses enforce lattice structure preservation and fractal properties.

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

import sys
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Suppress PyTorch scheduler deprecation warnings (PyTorch 2.8+)
# This is a known issue with SequentialLR internally passing epoch parameter
warnings.filterwarnings('ignore', message='.*epoch parameter.*scheduler.step.*')

# Import output formatting utilities
try:
    from output_formatters import (
        print_epoch_progress,
        print_early_stopping_message,
        print_seed_header,
        print_seed_results_summary,
        save_seed_result_to_disk
    )
    OUTPUT_FORMATTERS_AVAILABLE: bool = True
except ImportError:
    OUTPUT_FORMATTERS_AVAILABLE: bool = False
    logging.warning("output_formatters not available, using basic output")

    def save_seed_result_to_disk(*args, **kwargs) -> None:
        """Fallback no-op when output_formatters not available."""
        pass

from config import (
    LEARNING_RATE_DEFAULT,
    PATIENCE_DEFAULT,
    MAX_EPOCHS_DEFAULT,
    SEED_DEFAULT,
    WEIGHT_DECAY_DEFAULT,
    LABEL_SMOOTHING_DEFAULT,
    LEARNING_RATE_WARMUP_EPOCHS,
    TQF_GEOMETRY_REG_WEIGHT_DEFAULT,
    TQF_VERIFY_DUALITY_INTERVAL_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT
)
from models_baseline import get_model, MODEL_REGISTRY
from evaluation import (
    compute_per_class_accuracy,
    compute_per_class_rotated_accuracy,
    compute_rotation_invariance_error,
    estimate_model_flops,
    compute_inversion_consistency_metrics,
    measure_inference_time,
    compute_statistical_significance,
    print_comprehensive_metrics,
    evaluate_with_orbit_mixing
)
from param_matcher import TARGET_PARAMS, TARGET_PARAMS_TOLERANCE_PERCENT

def rotate_batch_images(
    images: torch.Tensor,
    angle_degrees: int,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Rotate a batch of MNIST images by a specified angle.

    Args:
        images: Input tensor of shape (B, 784), (B, 28, 28), or (B, 1, 28, 28)
        angle_degrees: Rotation angle in degrees (typically 60, 120, 180, 240, 300)
        mode: Interpolation mode ('bilinear' or 'nearest')
    Returns:
        Rotated images in same shape as input (except 3D becomes 4D)
    """
    # Store original shape for restoration
    original_shape: Tuple = images.shape
    was_flattened: bool = (len(original_shape) == 2)
    was_3d: bool = (len(original_shape) == 3 and original_shape[1] == 28)

    # Reshape to (B, 1, 28, 28) if needed
    if was_flattened:
        batch_size: int = images.size(0)
        images = images.view(batch_size, 1, 28, 28)
    elif was_3d:
        batch_size: int = images.size(0)
        images = images.unsqueeze(1)  # (B, 28, 28) -> (B, 1, 28, 28)

    # Convert angle to radians
    angle_rad: float = angle_degrees * (3.14159265358979 / 180.0)

    # Create rotation matrix
    cos_theta: float = np.cos(angle_rad)
    sin_theta: float = np.sin(angle_rad)

    # Affine transformation matrix for rotation around center
    # [cos  -sin  0]
    # [sin   cos  0]
    theta: torch.Tensor = torch.tensor([
        [cos_theta, -sin_theta, 0.0],
        [sin_theta, cos_theta, 0.0]
    ], dtype=images.dtype, device=images.device).unsqueeze(0)

    # Expand for batch
    batch_size: int = images.size(0)
    theta = theta.expand(batch_size, 2, 3)

    # Create affine grid and apply transformation
    grid: torch.Tensor = F.affine_grid(
        theta,
        images.size(),
        align_corners=False
    )
    rotated: torch.Tensor = F.grid_sample(
        images,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=False
    )

    # Restore original shape if needed
    if was_flattened:
        rotated = rotated.view(batch_size, -1)
    # Note: 3D inputs become 4D outputs as per test expectations

    return rotated

class TrainingEngine:
    """
    Training engine with dual metrics integration for TQF-ANN.

    - Supports geometric regularization loss (including fractal self-similarity)
    - Periodic self-duality verification
    - Enhanced logging for dual metrics
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = LEARNING_RATE_DEFAULT,
        weight_decay: float = WEIGHT_DECAY_DEFAULT,
        label_smoothing: float = LABEL_SMOOTHING_DEFAULT,
        use_geometry_reg: bool = False,
        geometry_weight: float = TQF_GEOMETRY_REG_WEIGHT_DEFAULT,
        self_similarity_weight: float = 0.0,
        box_counting_weight: float = 0.0,
        use_amp: bool = True,
        warmup_epochs: int = LEARNING_RATE_WARMUP_EPOCHS,
        num_epochs: int = MAX_EPOCHS_DEFAULT,
        use_compile: bool = False
    ):
        """
        Args:
            model: Neural network model
            device: Computation device
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
            label_smoothing: Label smoothing factor (0=hard labels, 0.1=standard)
            use_geometry_reg: Enable geometric preservation loss (TQF only)
            geometry_weight: Weight for geometric regularization
            self_similarity_weight: Weight for fractal self-similarity loss (TQF only)
            box_counting_weight: Weight for box-counting dimension loss (TQF only)
            use_amp: Enable automatic mixed precision (faster on RTX GPUs)
            warmup_epochs: Number of epochs for linear LR warmup (0 disables warmup)
            num_epochs: Total number of training epochs (for scheduler T_max)
            use_compile: Enable torch.compile for kernel fusion (PyTorch 2.0+)
        """
        self.model: nn.Module = model.to(device)

        # Apply torch.compile for kernel fusion and reduced Python overhead
        # Only available in PyTorch 2.0+ and requires Triton (Linux only)
        if use_compile and hasattr(torch, 'compile'):
            try:
                # Check if Triton is available (required for inductor backend)
                import importlib.util
                triton_available = importlib.util.find_spec('triton') is not None

                if triton_available:
                    # Full optimization with Triton backend
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    logging.info("torch.compile enabled with 'reduce-overhead' mode")
                else:
                    # Triton not available (common on Windows)
                    # Skip compilation - the vectorized symmetry ops still provide speedup
                    logging.warning(
                        "torch.compile requested but Triton not available (Windows limitation). "
                        "Continuing without compilation. Install Triton on Linux for full optimization."
                    )
            except Exception as e:
                # Fallback if compilation fails (e.g., unsupported ops)
                logging.warning(f"torch.compile failed, continuing without compilation: {e}")

        self.device: torch.device = device
        self.use_geometry_reg: bool = use_geometry_reg
        self.geometry_weight: float = geometry_weight
        self.self_similarity_weight: float = self_similarity_weight
        self.box_counting_weight: float = box_counting_weight
        self.use_fractal_reg: bool = (self_similarity_weight > 0.0 or box_counting_weight > 0.0)
        self.use_amp: bool = use_amp and torch.cuda.is_available()

        # Optimizer setup with L2 regularization (weight decay)
        # Weight decay adds L2 penalty to loss: Total_Loss = Loss + (weight_decay/2) * ||W||^2
        # This encourages smaller weights, reducing overfitting by penalizing model complexity.
        # Default 1e-4 is standard for Adam; lower values (e.g., 5e-5) recommended when
        # TQF regularizations (geometry, self-similarity, box-counting) are active.
        # CLI override: --weight-decay
        self.optimizer: optim.Adam = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Calculate eta_min based on actual learning_rate (not hardcoded default)
        # Maintains 100x decay ratio as documented in config.py
        eta_min: float = learning_rate / 100

        # Create scheduler with optional warmup
        if warmup_epochs > 0 and warmup_epochs < num_epochs:
            # Warmup phase: LinearLR from near-zero to base LR
            warmup_scheduler: optim.lr_scheduler.LinearLR = \
                optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1e-10,  # Start near zero to prevent gradient explosion
                    end_factor=1.0,      # End at base learning_rate
                    total_iters=warmup_epochs
                )

            # Main training phase: CosineAnnealingLR after warmup
            # Ensure T_max is at least 1 to prevent ZeroDivisionError
            cosine_t_max: int = max(1, num_epochs - warmup_epochs)
            cosine_scheduler: optim.lr_scheduler.CosineAnnealingLR = \
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_t_max,
                    eta_min=eta_min
                )

            # Sequential scheduler: warmup -> cosine
            self.scheduler: optim.lr_scheduler.SequentialLR = \
                optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
        else:
            # No warmup or warmup >= num_epochs: Use CosineAnnealingLR directly
            # Ensure T_max is at least 1 to prevent ZeroDivisionError
            cosine_t_max: int = max(1, num_epochs)
            self.scheduler: optim.lr_scheduler.CosineAnnealingLR = \
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_t_max,
                    eta_min=eta_min
                )

        # Loss function with label smoothing (using PyTorch built-in for performance)
        # PyTorch's native CrossEntropyLoss with label_smoothing is CUDA-optimized
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

        # PyTorch 2.8+ GradScaler (new API)
        self.scaler: torch.amp.GradScaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Loss history for tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.geometry_losses: List[float] = []
        self.fractal_losses: List[float] = []

        # Cache model capability checks (performance optimization)
        # Avoids repeated hasattr/introspection calls in training loop
        self._supports_geometry_loss: bool = (
            hasattr(self.model, 'forward') and
            'return_geometry_loss' in self.model.forward.__code__.co_varnames
        )
        self._supports_inv_loss: bool = (
            hasattr(self.model, 'forward') and
            'return_inv_loss' in self.model.forward.__code__.co_varnames
        )
        self._has_sector_features: bool = hasattr(self.model, 'get_cached_sector_features')
        self._has_dual_output: bool = hasattr(self.model, 'dual_output')
        self._has_fractal_loss: bool = hasattr(self.model, 'compute_fractal_loss')
        self._has_verify_self_duality: bool = hasattr(self.model, 'verify_self_duality')

    def train_epoch(
        self,
        train_loader: DataLoader,
        inversion_loss_weight: Optional[float] = None,
        z6_equivariance_weight: Optional[float] = None,
        d6_equivariance_weight: Optional[float] = None,
        t24_orbit_invariance_weight: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Loss features are enabled by providing a weight value (not None).
        A weight of None means the corresponding loss is disabled.

        Args:
            train_loader: Training data loader
            inversion_loss_weight: Weight for inversion loss (None=disabled, TQF only)
            z6_equivariance_weight: Weight for Z6 equivariance loss (None=disabled, TQF only)
            d6_equivariance_weight: Weight for D6 equivariance loss (None=disabled, TQF only)
            t24_orbit_invariance_weight: Weight for T24 orbit invariance loss (None=disabled, TQF only)
        Returns:
            Dict of loss metrics
        """
        # Determine which losses are enabled based on weight being provided
        use_inversion_loss: bool = inversion_loss_weight is not None
        use_z6_equivariance_loss: bool = z6_equivariance_weight is not None
        use_d6_equivariance_loss: bool = d6_equivariance_weight is not None
        use_t24_orbit_invariance_loss: bool = t24_orbit_invariance_weight is not None
        self.model.train()
        total_cls_loss: float = 0.0
        total_inversion_loss: float = 0.0
        total_geom_loss: float = 0.0
        total_fractal_loss: float = 0.0
        total_z6_equiv_loss: float = 0.0
        total_d6_equiv_loss: float = 0.0
        total_t24_orbit_loss: float = 0.0
        num_batches: int = 0

        for inputs, labels in train_loader:
            inputs: torch.Tensor = inputs.to(self.device)
            labels: torch.Tensor = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP (PyTorch 2.8+ API)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # Use cached capability checks (performance optimization)
                if self._supports_geometry_loss:
                    # TQF model with geometry support
                    if self.use_geometry_reg and use_inversion_loss:
                        # Request both geometry and inversion losses
                        outputs: Union[torch.Tensor, Tuple] = self.model(
                            inputs,
                            return_inv_loss=True,
                            return_geometry_loss=True
                        )
                        logits, inversion_loss, geom_loss = outputs
                    elif self.use_geometry_reg:
                        # Request geometry loss (requires both flags per DualOutput API)
                        # Note: inversion_loss is computed but not added to total loss here
                        outputs: Union[torch.Tensor, Tuple] = self.model(
                            inputs,
                            return_inv_loss=True,
                            return_geometry_loss=True
                        )
                        logits, inversion_loss, geom_loss = outputs
                        # Explicitly ignore inversion_loss in this path (not weighted in loss)
                        inversion_loss = None
                    elif use_inversion_loss:
                        # Request only inversion loss (not geometry)
                        outputs: Union[torch.Tensor, Tuple] = self.model(
                            inputs,
                            return_inv_loss=True
                        )
                        logits, inversion_loss = outputs
                        geom_loss = None
                    else:
                        # Standard forward pass (no additional losses)
                        logits: torch.Tensor = self.model(inputs)
                        inversion_loss = None
                        geom_loss = None
                elif self._supports_inv_loss:
                    # TQF model with inversion loss support only (legacy compatibility path)
                    outputs = self.model(inputs, return_inv_loss=use_inversion_loss)
                    if use_inversion_loss:
                        logits, inversion_loss = outputs
                    else:
                        logits = outputs
                        inversion_loss = None
                    geom_loss = None
                else:
                    # Non-TQF model
                    logits = self.model(inputs)
                    inversion_loss = None
                    geom_loss = None

                # Classification loss
                cls_loss: torch.Tensor = self.criterion(logits, labels)

                # Z6 Rotation Equivariance Loss (TQF only)
                z6_equiv_loss: Optional[torch.Tensor] = None
                if use_z6_equivariance_loss and self._has_sector_features:
                    from symmetry_ops import compute_z6_rotation_equivariance_loss
                    sector_feats: Optional[torch.Tensor] = self.model.get_cached_sector_features()
                    if sector_feats is not None:
                        z6_equiv_loss = compute_z6_rotation_equivariance_loss(
                            self.model, inputs, sector_feats, num_rotations=3
                        )

                # D6 Reflection Equivariance Loss (TQF only)
                d6_equiv_loss: Optional[torch.Tensor] = None
                if use_d6_equivariance_loss and self._has_sector_features:
                    from symmetry_ops import compute_d6_reflection_equivariance_loss
                    sector_feats: Optional[torch.Tensor] = self.model.get_cached_sector_features()
                    if sector_feats is not None:
                        d6_equiv_loss = compute_d6_reflection_equivariance_loss(
                            self.model, inputs, sector_feats, num_reflections=3
                        )

                # T24 Orbit Invariance Loss (TQF only)
                t24_orbit_loss: Optional[torch.Tensor] = None
                if use_t24_orbit_invariance_loss and self._has_sector_features:
                    from symmetry_ops import compute_t24_orbit_invariance_loss
                    sector_feats: Optional[torch.Tensor] = self.model.get_cached_sector_features()
                    if sector_feats is not None:
                        inversion_fn = self.model.dual_output.apply_circle_inversion_bijection if self._has_dual_output else None
                        t24_orbit_loss = compute_t24_orbit_invariance_loss(
                            sector_feats, logits, num_samples=8, inversion_fn=inversion_fn
                        )

                # Total loss
                loss: torch.Tensor = cls_loss
                if use_inversion_loss and inversion_loss is not None:
                    loss = loss + inversion_loss_weight * inversion_loss
                if self.use_geometry_reg and geom_loss is not None:
                    loss = loss + self.geometry_weight * geom_loss

                # Fractal regularization losses (self-similarity + box-counting)
                fractal_loss: Optional[torch.Tensor] = None
                if self.use_fractal_reg and self._has_fractal_loss:
                    fractal_loss = self.model.compute_fractal_loss()
                    if fractal_loss is not None and fractal_loss.item() > 0:
                        loss = loss + fractal_loss

                if use_z6_equivariance_loss and z6_equiv_loss is not None:
                    loss = loss + z6_equivariance_weight * z6_equiv_loss
                if use_d6_equivariance_loss and d6_equiv_loss is not None:
                    loss = loss + d6_equivariance_weight * d6_equiv_loss
                if use_t24_orbit_invariance_loss and t24_orbit_loss is not None:
                    loss = loss + t24_orbit_invariance_weight * t24_orbit_loss

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if use_inversion_loss and inversion_loss is not None:
                total_inversion_loss += inversion_loss.item()
            if self.use_geometry_reg and geom_loss is not None:
                total_geom_loss += geom_loss.item()
            if self.use_fractal_reg and fractal_loss is not None:
                total_fractal_loss += fractal_loss.item()
            if use_z6_equivariance_loss and z6_equiv_loss is not None:
                total_z6_equiv_loss += z6_equiv_loss.item()
            if use_d6_equivariance_loss and d6_equiv_loss is not None:
                total_d6_equiv_loss += d6_equiv_loss.item()
            if use_t24_orbit_invariance_loss and t24_orbit_loss is not None:
                total_t24_orbit_loss += t24_orbit_loss.item()

            total_cls_loss += cls_loss.item()
            num_batches += 1

        # Compute averages
        metrics: Dict[str, float] = {
            'cls_loss': total_cls_loss / num_batches if num_batches > 0 else 0.0,
        }

        if use_inversion_loss:
            metrics['inversion_loss'] = total_inversion_loss / num_batches if num_batches > 0 else 0.0

        if self.use_geometry_reg:
            geom_avg: float = total_geom_loss / num_batches if num_batches > 0 else 0.0
            metrics['geom_loss'] = geom_avg
            self.geometry_losses.append(geom_avg)

        if self.use_fractal_reg:
            fractal_avg: float = total_fractal_loss / num_batches if num_batches > 0 else 0.0
            metrics['fractal_loss'] = fractal_avg
            self.fractal_losses.append(fractal_avg)

        if use_z6_equivariance_loss:
            metrics['z6_equiv_loss'] = total_z6_equiv_loss / num_batches if num_batches > 0 else 0.0

        if use_d6_equivariance_loss:
            metrics['d6_equiv_loss'] = total_d6_equiv_loss / num_batches if num_batches > 0 else 0.0

        if use_t24_orbit_invariance_loss:
            metrics['t24_orbit_loss'] = total_t24_orbit_loss / num_batches if num_batches > 0 else 0.0

        # Store total training loss
        self.train_losses.append(metrics['cls_loss'])

        return metrics

    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs: torch.Tensor = inputs.to(self.device)
                labels: torch.Tensor = labels.to(self.device)

                # Forward pass with AMP (PyTorch 2.8+ API)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    # Forward pass (handle different model types, using cached check)
                    if self._supports_inv_loss:
                        outputs = self.model(inputs, return_inv_loss=False)
                    else:
                        outputs = self.model(inputs)

                    # Loss
                    loss: torch.Tensor = self.criterion(outputs, labels)

                total_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss: float = total_loss / len(val_loader)
        accuracy: float = 100.0 * correct / total

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)

        return avg_loss, accuracy

    def verify_self_duality_periodic(
        self,
        val_loader: DataLoader,
        epoch: int,
        verify_interval: int = 10
    ) -> Optional[Dict[str, float]]:
        """
        Periodically verify self-duality (TQF-ANN only).

        Performance note: Verification processes ~5 batches (320 samples with batch_size=64)
        and adds ~1-2 seconds per check. Default interval of 10 epochs minimizes overhead
        while providing periodic validation of theoretical properties.

        Output behavior: COMPLETELY SILENT on success.
        Only logs warnings if duality violations exceed tolerance.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            verify_interval: Epochs between verification checks
        Returns:
            Dict of duality metrics if checked, None otherwise
        """
        # Only verify at specified intervals
        if epoch % verify_interval != 0:
            return None

        # Only verify if model has duality verification (using cached check)
        if not self._has_verify_self_duality:
            return None

        # Run verification (silent on success)
        duality_metrics: Dict[str, float] = self.model.verify_self_duality(
            val_loader, self.device, num_batches=5
        )

        # Only log if violations exceed tolerance
        from config import TQF_DUALITY_TOLERANCE_DEFAULT
        if duality_metrics.get('max_error', 0.0) > TQF_DUALITY_TOLERANCE_DEFAULT:
            logging.warning(
                f"Epoch {epoch}: Duality violation detected! "
                f"Max error: {duality_metrics['max_error']:.6f} "
                f"(tolerance: {TQF_DUALITY_TOLERANCE_DEFAULT:.6f})"
            )

        return duality_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = MAX_EPOCHS_DEFAULT,
        early_stopping_patience: int = PATIENCE_DEFAULT,
        min_delta: float = 0.0005,
        inversion_loss_weight: Optional[float] = None,
        z6_equivariance_weight: Optional[float] = None,
        d6_equivariance_weight: Optional[float] = None,
        t24_orbit_invariance_weight: Optional[float] = None,
        verify_duality_interval: int = TQF_VERIFY_DUALITY_INTERVAL_DEFAULT,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model with early stopping based on validation loss.

        Loss features are enabled by providing a weight value (not None).
        A weight of None means the corresponding loss is disabled.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement to count as progress
            inversion_loss_weight: Weight for inversion loss (None=disabled, TQF only)
            z6_equivariance_weight: Weight for Z6 equivariance loss (None=disabled, TQF only)
            d6_equivariance_weight: Weight for D6 equivariance loss (None=disabled, TQF only)
            t24_orbit_invariance_weight: Weight for T24 orbit invariance loss (None=disabled, TQF only)
            verify_duality_interval: Epochs between self-duality checks (TQF only)
            verbose: Whether to print progress
        Returns:
            Dict with training history and best model info
        """
        best_val_loss: float = float('inf')
        best_val_acc: float = 0.0
        best_acc_epoch: int = 0
        best_loss_epoch: int = 0
        patience_counter: int = 0
        best_model_state: Dict = None
        best_acc_value: float = 0.0

        for epoch in range(num_epochs):
            epoch_start_time: float = time.time()

            # Train one epoch
            train_metrics: Dict[str, float] = self.train_epoch(
                train_loader,
                inversion_loss_weight=inversion_loss_weight,
                z6_equivariance_weight=z6_equivariance_weight,
                d6_equivariance_weight=d6_equivariance_weight,
                t24_orbit_invariance_weight=t24_orbit_invariance_weight
            )

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)

            # Update best accuracy tracking
            if val_acc > best_acc_value:
                best_acc_value = val_acc
                best_acc_epoch = epoch

            # Early stopping based on validation loss (with min_delta)
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_loss_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            # Periodic self-duality verification (TQF only, silent on success)
            _ = self.verify_self_duality_periodic(
                val_loader, epoch, verify_duality_interval
            )

            # Learning rate scheduling
            self.scheduler.step()

            # Epoch time
            epoch_time: float = time.time() - epoch_start_time

            # Print progress
            if verbose and OUTPUT_FORMATTERS_AVAILABLE:
                print_epoch_progress(
                    epoch=epoch,
                    total_epochs=num_epochs,
                    train_loss=train_metrics.get('cls_loss', 0.0),
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=self.optimizer.param_groups[0]['lr'],
                    elapsed=epoch_time,
                    geom_loss=train_metrics.get('geom_loss'),
                    z6_equiv_loss=train_metrics.get('z6_equiv_loss'),
                    d6_equiv_loss=train_metrics.get('d6_equiv_loss'),
                    t24_orbit_loss=train_metrics.get('t24_orbit_loss')
                )

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                if verbose and OUTPUT_FORMATTERS_AVAILABLE:
                    print_early_stopping_message(
                        patience=early_stopping_patience,
                        best_loss_epoch=best_loss_epoch,
                        best_acc_epoch=best_acc_epoch,
                        total_epochs=num_epochs,
                        best_val_acc_at_best_loss=best_val_acc / 100.0,  # Convert to decimal
                        best_val_acc_overall=best_acc_value / 100.0,      # Convert to decimal
                        best_val_loss=best_val_loss
                    )
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_acc_value': best_acc_value,
            'best_loss_epoch': best_loss_epoch,
            'best_acc_epoch': best_acc_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'geometry_losses': self.geometry_losses if self.use_geometry_reg else []
        }

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms for scientific reproducibility
    # benchmark=True allows cuDNN to auto-tune among deterministic algorithms
    # This provides both reproducibility AND performance optimization
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def run_single_seed_experiment(
    model_name: str,
    model_config: Dict,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    num_epochs: int,
    seed: int,
    learning_rate: float = LEARNING_RATE_DEFAULT,
    weight_decay: float = WEIGHT_DECAY_DEFAULT,
    label_smoothing: float = LABEL_SMOOTHING_DEFAULT,
    patience: int = PATIENCE_DEFAULT,
    min_delta: float = 0.0005,
    warmup_epochs: int = LEARNING_RATE_WARMUP_EPOCHS,
    verify_duality_interval: int = TQF_VERIFY_DUALITY_INTERVAL_DEFAULT,
    inversion_loss_weight: Optional[float] = None,
    z6_equivariance_weight: Optional[float] = None,
    d6_equivariance_weight: Optional[float] = None,
    t24_orbit_invariance_weight: Optional[float] = None,
    total_seeds: int = 1,
    seed_idx: int = 1,
    verbose: bool = True,
    use_compile: bool = False,
    use_z6_orbit_mixing: bool = False,
    use_d6_orbit_mixing: bool = False,
    use_t24_orbit_mixing: bool = False,
    orbit_mixing_temp_rotation: float = TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    orbit_mixing_temp_reflection: float = TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    orbit_mixing_temp_inversion: float = TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT
) -> Dict:
    """
    Run single-seed experiment for a model.

    Loss features are enabled by providing a weight value (not None).
    A weight of None means the corresponding loss is disabled.

    Args:
        model_name: Name of model
        model_config: Model configuration dict
        dataloaders: Dict of dataloaders
        device: Computation device
        num_epochs: Training epochs
        seed: Random seed
        learning_rate: Learning rate
        weight_decay: Weight decay
        label_smoothing: Label smoothing factor
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        warmup_epochs: Number of epochs for linear LR warmup
        verify_duality_interval: Epochs between self-duality checks (TQF only)
        inversion_loss_weight: Weight for inversion loss (None=disabled, TQF only)
        z6_equivariance_weight: Weight for Z6 equivariance loss (None=disabled, TQF only)
        d6_equivariance_weight: Weight for D6 equivariance loss (None=disabled, TQF only)
        t24_orbit_invariance_weight: Weight for T24 orbit invariance loss (None=disabled, TQF only)
        total_seeds: Total number of seeds (for display)
        seed_idx: Current seed index (for display)
        verbose: Whether to print progress
        use_z6_orbit_mixing: Enable Z6 rotation orbit mixing at evaluation
        use_d6_orbit_mixing: Enable D6 reflection orbit mixing at evaluation
        use_t24_orbit_mixing: Enable T24 zone-swap orbit mixing at evaluation
        orbit_mixing_temp_rotation: Temperature for Z6 rotation averaging
        orbit_mixing_temp_reflection: Temperature for D6 reflection averaging
        orbit_mixing_temp_inversion: Temperature for T24 zone-swap averaging
    Returns:
        Dict with results
    """
    # Set seed
    set_seed(seed)

    # Print seed header
    if verbose and OUTPUT_FORMATTERS_AVAILABLE:
        print_seed_header(model_name, seed_idx, total_seeds, seed)

    # Create model
    model: nn.Module = get_model(model_name, **model_config)
    params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Verbose parameter matching feedback
    if verbose:
        tolerance_abs: int = int(TARGET_PARAMS * TARGET_PARAMS_TOLERANCE_PERCENT / 100)
        deviation_pct: float = 100 * abs(params - TARGET_PARAMS) / TARGET_PARAMS
        logging.info(
            f"Model parameters: {params:,} "
            f"(deviation: {deviation_pct:.2f}%, target: +/-{TARGET_PARAMS_TOLERANCE_PERCENT}%)"
        )

    # ===== Enable geometry regularization for TQF models =====
    # Geometry regularization is controlled by geometry_reg_weight (independent of verify_geometry)
    # verify_geometry controls validation/debugging features, not training loss
    geometry_weight: float = model_config.get('geometry_reg_weight', TQF_GEOMETRY_REG_WEIGHT_DEFAULT)
    use_geometry_reg: bool = 'TQF' in model_name and geometry_weight > 0.0

    # ===== Enable fractal regularization for TQF models =====
    # Fractal losses: self-similarity (multi-scale correlation) + box-counting (dimension matching)
    self_similarity_weight: float = model_config.get('self_similarity_weight', 0.0) if 'TQF' in model_name else 0.0
    box_counting_weight: float = model_config.get('box_counting_weight', 0.0) if 'TQF' in model_name else 0.0

    # Training engine with specified learning rate, weight decay, label smoothing, warmup, and geometry reg
    engine: TrainingEngine = TrainingEngine(
        model,
        device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        use_geometry_reg=use_geometry_reg,
        geometry_weight=geometry_weight,
        self_similarity_weight=self_similarity_weight,
        box_counting_weight=box_counting_weight,
        warmup_epochs=warmup_epochs,
        num_epochs=num_epochs,
        use_compile=use_compile
    )

    # Determine TQF-specific loss weights (only applied for TQF models)
    # Weights are passed through to engine.train(); non-TQF models ignore them
    # A weight value enables the loss; None means disabled
    is_tqf_model: bool = 'TQF' in model_name
    apply_inversion_weight: Optional[float] = inversion_loss_weight if is_tqf_model else None
    apply_z6_weight: Optional[float] = z6_equivariance_weight if is_tqf_model else None
    apply_d6_weight: Optional[float] = d6_equivariance_weight if is_tqf_model else None
    apply_t24_weight: Optional[float] = t24_orbit_invariance_weight if is_tqf_model else None

    # Train with specified patience
    start_time: float = time.time()
    history: Dict = engine.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=num_epochs,
        early_stopping_patience=patience,
        min_delta=min_delta,
        inversion_loss_weight=apply_inversion_weight,
        z6_equivariance_weight=apply_z6_weight,
        d6_equivariance_weight=apply_d6_weight,
        t24_orbit_invariance_weight=apply_t24_weight,
        verify_duality_interval=verify_duality_interval,
        verbose=verbose
    )
    train_time: float = time.time() - start_time

    # Evaluate on test sets
    # Standard evaluation on unrotated test set
    _, test_acc = engine.evaluate(dataloaders['test'])

    any_orbit_mixing: bool = use_z6_orbit_mixing or use_d6_orbit_mixing or use_t24_orbit_mixing
    if 'TQF' in model_name and any_orbit_mixing:
        _, test_rot_acc = evaluate_with_orbit_mixing(
            model, dataloaders['test_rot'], device,
            use_z6=True,
            use_d6=use_d6_orbit_mixing or use_t24_orbit_mixing,
            use_t24=use_t24_orbit_mixing,
            temp_rotation=orbit_mixing_temp_rotation,
            temp_reflection=orbit_mixing_temp_reflection,
            temp_inversion=orbit_mixing_temp_inversion
        )
    else:
        _, test_rot_acc = engine.evaluate(dataloaders['test_rot'])

    # Per-class accuracy
    per_class_acc: Dict = compute_per_class_accuracy(
        model, dataloaders['test'], device
    )

    # Rotation invariance
    rot_inv_error: float = compute_rotation_invariance_error(
        model, dataloaders['test_rot'], device
    )

    # FLOPs
    flops: int = estimate_model_flops(model)

    # Inference time
    time_stats: Dict = measure_inference_time(
        model, (64, 784), device, num_trials=100
    )

    # Inversion consistency (TQF only)
    inv_metrics: Dict = {}
    if 'TQF' in model_name:
        inv_metrics = compute_inversion_consistency_metrics(
            model, dataloaders['test'], device
        )

    results: Dict[str, Any] = {
        'model_name': model_name,
        'seed': seed,
        'params': params,
        'best_val_acc': history['best_val_acc'],
        'best_acc_value': history['best_acc_value'],
        'test_unrot_acc': test_acc,
        'test_rot_acc': test_rot_acc,
        'per_class_acc': per_class_acc,
        'rotation_inv_error': rot_inv_error,
        'flops': flops,
        'inference_time_ms': time_stats['mean_ms'],
        'train_time_total': train_time,
        'train_time_per_epoch': train_time / (history['best_loss_epoch'] + 1),
        'best_loss_epoch': history['best_loss_epoch'],
        'best_acc_epoch': history['best_acc_epoch'],
        **inv_metrics
    }

    # Print per-seed summary
    if verbose and OUTPUT_FORMATTERS_AVAILABLE:
        print_seed_results_summary(seed, model_name, results, seed_idx, total_seeds)

    return results

def run_multi_seed_experiment(
    model_names: List[str],
    model_configs: Dict[str, Dict],
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    seeds: List[int],
    num_epochs: int = MAX_EPOCHS_DEFAULT,
    learning_rate: float = LEARNING_RATE_DEFAULT,
    weight_decay: float = WEIGHT_DECAY_DEFAULT,
    label_smoothing: float = LABEL_SMOOTHING_DEFAULT,
    patience: int = PATIENCE_DEFAULT,
    min_delta: float = 0.0005,
    warmup_epochs: int = LEARNING_RATE_WARMUP_EPOCHS,
    verify_duality_interval: int = TQF_VERIFY_DUALITY_INTERVAL_DEFAULT,
    inversion_loss_weight: Optional[float] = None,
    z6_equivariance_weight: Optional[float] = None,
    d6_equivariance_weight: Optional[float] = None,
    t24_orbit_invariance_weight: Optional[float] = None,
    use_compile: bool = False,
    output_path: Optional[str] = None,
    use_z6_orbit_mixing: bool = False,
    use_d6_orbit_mixing: bool = False,
    use_t24_orbit_mixing: bool = False,
    orbit_mixing_temp_rotation: float = TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
    orbit_mixing_temp_reflection: float = TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
    orbit_mixing_temp_inversion: float = TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT
) -> Dict[str, List[Dict]]:
    """
    Run multi-seed experiments for multiple models.

    Loss features are enabled by providing a weight value (not None).
    A weight of None means the corresponding loss is disabled.

    Args:
        model_names: List of model names to evaluate
        model_configs: Dict mapping model_name -> config
        dataloaders: Dict of dataloaders
        device: Computation device
        seeds: List of random seeds
        num_epochs: Training epochs per seed
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 regularization) for optimizer
        label_smoothing: Label smoothing factor for loss function
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        warmup_epochs: Number of epochs for linear LR warmup
        verify_duality_interval: Epochs between self-duality checks (TQF only)
        inversion_loss_weight: Weight for inversion loss (None=disabled, TQF only)
        z6_equivariance_weight: Weight for Z6 equivariance loss (None=disabled, TQF only)
        d6_equivariance_weight: Weight for D6 equivariance loss (None=disabled, TQF only)
        t24_orbit_invariance_weight: Weight for T24 orbit invariance loss (None=disabled, TQF only)
        output_path: Path to JSON file for incremental result saving (None=no disk save)
        use_z6_orbit_mixing: Enable Z6 rotation orbit mixing at evaluation
        use_d6_orbit_mixing: Enable D6 reflection orbit mixing at evaluation
        use_t24_orbit_mixing: Enable T24 zone-swap orbit mixing at evaluation
        orbit_mixing_temp_rotation: Temperature for Z6 rotation averaging
        orbit_mixing_temp_reflection: Temperature for D6 reflection averaging
        orbit_mixing_temp_inversion: Temperature for T24 zone-swap averaging
    Returns:
        Dict mapping model_name -> list of results (one per seed)
    """
    all_results: Dict[str, List[Dict]] = {}
    total_seeds: int = len(seeds)

    for model_name in model_names:
        logging.info(f"Evaluating {model_name} across {total_seeds} seeds...")

        # Print model training start separator
        if OUTPUT_FORMATTERS_AVAILABLE:
            from output_formatters import print_model_training_start
            print_model_training_start(model_name, total_seeds)

        model_results: List[Dict] = []

        for seed_idx, seed in enumerate(seeds, start=1):
            result: Dict = run_single_seed_experiment(
                model_name,
                model_configs.get(model_name, {}),
                dataloaders,
                device,
                num_epochs,
                seed,
                learning_rate,
                weight_decay,
                label_smoothing,
                patience,
                min_delta,
                warmup_epochs,
                verify_duality_interval,
                inversion_loss_weight,
                z6_equivariance_weight,
                d6_equivariance_weight,
                t24_orbit_invariance_weight,
                total_seeds=total_seeds,
                seed_idx=seed_idx,
                verbose=True,
                use_compile=use_compile,
                use_z6_orbit_mixing=use_z6_orbit_mixing,
                use_d6_orbit_mixing=use_d6_orbit_mixing,
                use_t24_orbit_mixing=use_t24_orbit_mixing,
                orbit_mixing_temp_rotation=orbit_mixing_temp_rotation,
                orbit_mixing_temp_reflection=orbit_mixing_temp_reflection,
                orbit_mixing_temp_inversion=orbit_mixing_temp_inversion
            )
            model_results.append(result)

            # Incrementally save to disk so results survive crashes/session expiry
            if output_path:
                save_seed_result_to_disk(result, output_path)

        all_results[model_name] = model_results

        # Print model training end separator
        if OUTPUT_FORMATTERS_AVAILABLE:
            from output_formatters import print_model_training_end
            print_model_training_end(model_name)

    return all_results


def compare_models_statistical(
    results: Dict[str, List[Dict]]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compare models statistically and compute significance.

    Args:
        results: Dict mapping model_name -> list of seed results
    Returns:
        Dict of aggregated statistics
    """
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for model_name, model_results in results.items():
        # Extract metrics across seeds
        val_accs: List[float] = [r['best_val_acc'] for r in model_results]
        test_unrot: List[float] = [r['test_unrot_acc'] for r in model_results]
        test_rot: List[float] = [r['test_rot_acc'] for r in model_results]
        params: List[int] = [r['params'] for r in model_results]
        flops: List[int] = [r['flops'] for r in model_results]
        infer_times: List[float] = [r['inference_time_ms'] for r in model_results]

        summary[model_name] = {
            'val_acc': (np.mean(val_accs), np.std(val_accs)),
            'test_unrot_acc': (np.mean(test_unrot), np.std(test_unrot)),
            'test_rot_acc': (np.mean(test_rot), np.std(test_rot)),
            'params': (np.mean(params), np.std(params)),
            'flops': (np.mean(flops), np.std(flops)),
            'inference_time_ms': (np.mean(infer_times), np.std(infer_times))
        }

    return summary

def print_final_comparison_table(
    summary: Dict[str, Dict[str, Tuple[float, float]]]
) -> None:
    """
    Print final comparison table.

    Args:
        summary: Aggregated statistics dict
    """
    print("\n" + "=" * 120)
    print("FINAL MODEL COMPARISON (Mean +/- Std)")
    print("=" * 120)
    print(
        f"{'Model':<20}  "
        f"{'Val Acc (%)':<15} "
        f"{'Test Acc (%)':<15} "
        f"{'Rot Acc (%)':<15}  "
        f"{'Params (k)':<15}   "
        f"{'FLOPs (M)':<15}"
        f"{'Inf Time (ms)':<15}"
    )
    print("-" * 120)

    for model_name, stats in summary.items():
        val_mean, val_std = stats['val_acc']
        test_mean, test_std = stats['test_unrot_acc']
        rot_mean, rot_std = stats['test_rot_acc']
        params_mean, params_std = stats['params']
        flops_mean, flops_std = stats['flops']
        time_mean, time_std = stats['inference_time_ms']

        print(
            f"{model_name:<20} "
            f"{val_mean:6.2f}+/-{val_std:4.2f}   "
            f"{test_mean:6.2f}+/-{test_std:4.2f}   "
            f"{rot_mean:6.2f}+/-{rot_std:4.2f}   "
            f"{params_mean/1e3:7.1f}+/-{params_std/1e3:4.1f}   "
            f"{flops_mean/1e6:7.1f}+/-{flops_std/1e6:4.1f}   "
            f"{time_mean:6.2f}+/-{time_std:4.2f}"
        )

    print("=" * 120)
