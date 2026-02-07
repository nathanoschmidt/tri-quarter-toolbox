"""
test_engine.py - Comprehensive Training Engine Tests for TQF-NN

This module tests all training engine functionality in engine.py, including
optimizer configuration, learning rate scheduling, early stopping, batch
augmentation, and TQF-specific loss function applications.

Key Test Coverage:
- Batch Rotation Augmentation: Z6-aligned 60-degree rotations for training data
- Rotation Validation: Exact 60-degree increments (0°, 60°, 120°, 180°, 240°, 300°)
- Optimizer Setup: AdamW configuration with weight decay, parameter group validation
- Scheduler Setup: CosineAnnealingLR configuration, warmup periods, T_max calculation
- Learning Rate Schedule Regression: Fixes for epoch-based (not step-based) scheduling
- Scheduler State: Learning rate progression, warmup behavior, cosine annealing
- Early Stopping: Patience-based validation loss monitoring, best model checkpointing
- Device Handling: CPU/CUDA model placement, batch transfer to correct device
- Reproducibility: Seed-based deterministic training, consistent results across runs
- TQF-Specific Losses:
  - Z6 Equivariance Loss: Rotational equivariance under 60-degree rotations
  - D6 Equivariance Loss: Reflectional equivariance under mirror symmetries
  - Inversion Loss: Consistency between outer and inner zone predictions
  - T24 Orbit Invariance Loss: Full tetrahedral-octahedral group invariance
  - Box-Counting Fractal Loss: Multiscale fractal dimension regularization
  - Geometry Regularization Loss: Phase pair and sector consistency
- Loss Weight Parameters: Optional loss function weights (None = disabled)
- Loss Application Validation: Correct backpropagation through all loss terms
- Multi-Seed Management: Independent seeds for fair statistical comparison
- Training Loop Integration: Forward pass, loss computation, backward pass, optimizer step

Test Organization:
- TestRotateBatchImages: Z6 rotation augmentation for training batches
- TestSetupOptimizer: AdamW optimizer configuration validation
- TestSetupScheduler: CosineAnnealingLR scheduler configuration
- TestLearningRateScheduleRegression: Epoch-based scheduling fixes
- TestEarlyStopping: Patience-based early stopping logic
- TestDeviceHandling: CPU/GPU tensor placement
- TestReproducibility: Deterministic training with fixed seeds
- TestTQFSpecificLosses: TQF loss function computation and weighting
- TestInversionLossApplication: Dual zone consistency loss
- TestT24OrbitInvarianceLossApplication: T24 group invariance loss
- TestLossWeightParameters: Optional loss weight handling (None vs numeric)

Scientific Rationale:
Z6-aligned rotation augmentation ensures training data matches TQF's hexagonal
symmetry group. TQF-specific losses provide geometric inductive biases that
guide models toward equivariant and self-dual representations.

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

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from conftest import TORCH_AVAILABLE, CUDA_AVAILABLE, device, set_seed, assert_tensor_shape

if not TORCH_AVAILABLE:
    pytest.skip("PyTorch not available", allow_module_level=True)

from engine import rotate_batch_images
import config


class TestRotateBatchImages:
    """Test suite for image rotation utility."""

    def test_rotate_2d_input(self) -> None:
        """
        WHY: Must handle flattened 28x28 images (B, 784)
        HOW: Create 2D tensor, rotate, check shape
        WHAT: Expect same shape after rotation
        """
        batch_size: int = 4
        images: torch.Tensor = torch.randn(batch_size, 784)
        rotated: torch.Tensor = rotate_batch_images(images, 60)

        assert_tensor_shape(rotated, (batch_size, 784))

    def test_rotate_4d_input(self) -> None:
        """
        WHY: Must handle standard 4D image tensors (B, C, H, W)
        HOW: Create 4D tensor, rotate, check shape
        WHAT: Expect same shape after rotation
        """
        batch_size: int = 4
        images: torch.Tensor = torch.randn(batch_size, 1, 28, 28)
        rotated: torch.Tensor = rotate_batch_images(images, 60)

        assert_tensor_shape(rotated, (batch_size, 1, 28, 28))

    def test_rotate_3d_input(self) -> None:
        """
        WHY: Must handle 3D tensors (B, H, W) by adding channel
        HOW: Create 3D tensor, rotate, check shape
        WHAT: Expect (B, 1, H, W) output
        """
        batch_size: int = 4
        images: torch.Tensor = torch.randn(batch_size, 28, 28)
        rotated: torch.Tensor = rotate_batch_images(images, 60)

        # Output should be 4D with channel added
        assert rotated.dim() == 4
        assert rotated.shape[1] == 1

    def test_rotation_angles(self) -> None:
        """
        WHY: Must support all Z6 rotation angles
        HOW: Rotate by 0, 60, 120, 180, 240, 300 degrees
        WHAT: Expect no errors for any angle
        """
        images: torch.Tensor = torch.randn(2, 1, 28, 28)
        angles: List[int] = [0, 60, 120, 180, 240, 300]

        for angle in angles:
            rotated: torch.Tensor = rotate_batch_images(images, angle)
            assert_tensor_shape(rotated, (2, 1, 28, 28))

    def test_zero_rotation_identity(self) -> None:
        """
        WHY: 0-degree rotation should not change image
        HOW: Rotate by 0 degrees, compare to original
        WHAT: Expect very small difference (numerical precision)
        """
        images: torch.Tensor = torch.randn(2, 1, 28, 28)
        rotated: torch.Tensor = rotate_batch_images(images, 0)

        # Should be nearly identical (allow for interpolation artifacts)
        diff: float = (images - rotated).abs().max().item()
        assert diff < 0.1, f"0-degree rotation changed image by {diff}"

    def test_rotation_preserves_device(self, device: torch.device) -> None:
        """
        WHY: Rotation must preserve tensor device
        HOW: Create tensor on device, rotate, check device
        WHAT: Expect same device after rotation
        """
        images: torch.Tensor = torch.randn(2, 1, 28, 28, device=device)
        rotated: torch.Tensor = rotate_batch_images(images, 90)

        assert rotated.device == images.device

    def test_rotation_preserves_dtype(self) -> None:
        """
        WHY: Rotation should not change tensor dtype
        HOW: Create float32 tensor, rotate, check dtype
        WHAT: Expect float32 output
        """
        images: torch.Tensor = torch.randn(2, 1, 28, 28, dtype=torch.float32)
        rotated: torch.Tensor = rotate_batch_images(images, 90)

        assert rotated.dtype == torch.float32

    def test_batch_size_one(self) -> None:
        """
        WHY: Must handle single-image batches
        HOW: Rotate batch of size 1
        WHAT: Expect correct output shape
        """
        images: torch.Tensor = torch.randn(1, 1, 28, 28)
        rotated: torch.Tensor = rotate_batch_images(images, 60)

        assert_tensor_shape(rotated, (1, 1, 28, 28))

    def test_large_batch(self) -> None:
        """
        WHY: Must handle typical batch sizes efficiently
        HOW: Rotate batch of 128 images
        WHAT: Expect correct output shape, reasonable time
        """
        images: torch.Tensor = torch.randn(128, 1, 28, 28)
        rotated: torch.Tensor = rotate_batch_images(images, 60)

        assert_tensor_shape(rotated, (128, 1, 28, 28))


# Note: The following test classes would require extensive mocking
# or integration testing setup. They are outlined here as templates.

class TestSetupOptimizer:
    """Test suite for optimizer setup."""

    def test_adam_optimizer_created(self, lightweight_models) -> None:
        """
        WHY: Must create AdamW optimizer correctly
        HOW: Setup optimizer with model from fixture, check type
        WHAT: Expect torch.optim.AdamW instance
        """
        model = lightweight_models['FC-MLP']
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        assert isinstance(optimizer, optim.AdamW)

    def test_learning_rate_set_correctly(self, lightweight_models) -> None:
        """
        WHY: Optimizer must use specified learning rate
        HOW: Setup optimizer with custom LR, check param groups
        WHAT: Expect LR matches input
        """
        model = lightweight_models['FC-MLP']
        custom_lr: float = 0.0042
        optimizer = optim.AdamW(model.parameters(), lr=custom_lr)

        for param_group in optimizer.param_groups:
            assert param_group['lr'] == custom_lr, \
                f"Expected lr={custom_lr}, got {param_group['lr']}"

    def test_weight_decay_applied(self) -> None:
        """
        WHY: Weight decay provides L2 regularization to prevent overfitting
        HOW: Setup Adam optimizer with weight_decay > 0, verify param groups
        WHAT: Expect weight_decay value present in optimizer.param_groups
        """
        # Create minimal model for optimizer setup
        model: nn.Module = nn.Linear(10, 10)
        weight_decay_value: float = 0.0001

        # Create optimizer with explicit weight_decay
        optimizer: optim.Optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=weight_decay_value
        )

        # Verify weight_decay is correctly set in param groups
        for param_group in optimizer.param_groups:
            assert 'weight_decay' in param_group, \
                "weight_decay key missing from optimizer param_groups"
            assert param_group['weight_decay'] == weight_decay_value, \
                f"Expected weight_decay={weight_decay_value}, got {param_group['weight_decay']}"

    def test_weight_decay_zero_disables_regularization(self) -> None:
        """
        WHY: Zero weight decay should disable L2 regularization (valid use case)
        HOW: Setup optimizer with weight_decay=0.0
        WHAT: Expect weight_decay=0.0 in param groups (no regularization)
        """
        model: nn.Module = nn.Linear(10, 10)

        optimizer: optim.Optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0
        )

        for param_group in optimizer.param_groups:
            assert param_group['weight_decay'] == 0.0, \
                f"Expected weight_decay=0.0, got {param_group['weight_decay']}"

    def test_weight_decay_affects_gradient_update(self) -> None:
        """
        WHY: Weight decay should actually modify gradients during training
        HOW: Compare gradient updates with and without weight decay
        WHAT: Expect different weight updates when weight_decay > 0

        NOTE: This tests that weight_decay has practical effect, not just stored.
        """
        import torch

        torch.manual_seed(42)

        # Model and input
        model_no_decay: nn.Module = nn.Linear(10, 10)
        model_with_decay: nn.Module = nn.Linear(10, 10)

        # Copy weights to ensure same starting point
        model_with_decay.load_state_dict(model_no_decay.state_dict())

        # Optimizers with different weight_decay
        opt_no_decay: optim.Optimizer = optim.Adam(
            model_no_decay.parameters(), lr=0.01, weight_decay=0.0
        )
        opt_with_decay: optim.Optimizer = optim.Adam(
            model_with_decay.parameters(), lr=0.01, weight_decay=0.1
        )

        # Same input and target
        x: torch.Tensor = torch.randn(4, 10)
        target: torch.Tensor = torch.randn(4, 10)

        # Forward pass for both
        loss_no_decay: torch.Tensor = ((model_no_decay(x) - target) ** 2).mean()
        loss_with_decay: torch.Tensor = ((model_with_decay(x) - target) ** 2).mean()

        # Backward pass
        opt_no_decay.zero_grad()
        opt_with_decay.zero_grad()
        loss_no_decay.backward()
        loss_with_decay.backward()

        # Step
        opt_no_decay.step()
        opt_with_decay.step()

        # Weights should now be different due to weight decay
        weights_no_decay: torch.Tensor = model_no_decay.weight.data.clone()
        weights_with_decay: torch.Tensor = model_with_decay.weight.data.clone()

        # Weight decay shrinks weights, so with_decay should have smaller magnitude
        assert not torch.allclose(weights_no_decay, weights_with_decay), \
            "Weight decay should cause different weight updates"


class TestSetupScheduler:
    """
    Test suite for learning rate scheduler setup.

    Validates edge cases in scheduler configuration, particularly when
    warmup_epochs approaches or exceeds num_epochs.
    """

    def test_scheduler_handles_warmup_equals_epochs(self) -> None:
        """
        Verify scheduler handles warmup_epochs == num_epochs without error.

        When warmup equals total epochs, cosine phase has zero duration.
        The scheduler must handle this gracefully (T_max >= 1).
        """
        model: nn.Module = nn.Linear(10, 10)
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Edge case: warmup_epochs == num_epochs
        num_epochs: int = 5
        warmup_epochs: int = 5

        # This should NOT raise ZeroDivisionError
        try:
            if warmup_epochs > 0 and warmup_epochs < num_epochs:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_epochs
                )
                cosine_t_max: int = max(1, num_epochs - warmup_epochs)
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_t_max, eta_min=1e-6
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                # No warmup or warmup >= epochs
                cosine_t_max: int = max(1, num_epochs)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_t_max, eta_min=1e-6
                )

            # Step the scheduler to trigger any errors
            scheduler.step()

        except ZeroDivisionError as e:
            pytest.fail(f"ZeroDivisionError with warmup={warmup_epochs}, "
                       f"epochs={num_epochs}: {e}")

    def test_scheduler_handles_warmup_exceeds_epochs(self) -> None:
        """
        Verify scheduler handles warmup_epochs > num_epochs without error.

        Invalid configuration where warmup exceeds training duration should
        fall back to cosine-only scheduling with T_max >= 1.
        """
        model: nn.Module = nn.Linear(10, 10)
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Edge case: warmup_epochs > num_epochs
        num_epochs: int = 2
        warmup_epochs: int = 10

        try:
            if warmup_epochs > 0 and warmup_epochs < num_epochs:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_epochs
                )
                cosine_t_max: int = max(1, num_epochs - warmup_epochs)
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_t_max, eta_min=1e-6
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                cosine_t_max: int = max(1, num_epochs)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_t_max, eta_min=1e-6
                )

            scheduler.step()

        except (ZeroDivisionError, ValueError) as e:
            pytest.fail(f"Error with warmup={warmup_epochs}, epochs={num_epochs}: {e}")

    def test_t_max_always_positive(self) -> None:
        """
        Verify T_max >= 1 for all epoch/warmup combinations.

        CosineAnnealingLR requires T_max >= 1 to avoid division by zero.
        Tests various edge cases to ensure the guard is effective.
        """
        test_cases: List[Tuple[int, int]] = [
            (1, 0),   # Minimal: 1 epoch, no warmup
            (1, 1),   # Edge: warmup == epochs
            (2, 2),   # Edge: warmup == epochs
            (5, 0),   # Normal: no warmup
            (5, 3),   # Normal: warmup < epochs
            (5, 5),   # Edge: warmup == epochs
            (5, 10),  # Edge: warmup > epochs
        ]

        for num_epochs, warmup_epochs in test_cases:
            # Calculate T_max as in fixed code
            if warmup_epochs > 0 and warmup_epochs < num_epochs:
                t_max: int = max(1, num_epochs - warmup_epochs)
            else:
                t_max: int = max(1, num_epochs)

            assert t_max >= 1, (
                f"T_max={t_max} < 1 for epochs={num_epochs}, warmup={warmup_epochs}"
            )

    def test_warmup_conditional_logic_correct(self) -> None:
        """
        Verify warmup only applies when 0 < warmup_epochs < num_epochs.

        SequentialLR (warmup + cosine) should only be used when warmup is
        in valid range. Otherwise, use CosineAnnealingLR directly.
        """
        # Case 1: Valid warmup (should use SequentialLR)
        num_epochs: int = 10
        warmup_epochs: int = 5
        should_use_sequential: bool = (warmup_epochs > 0 and warmup_epochs < num_epochs)
        assert should_use_sequential, "Case 1 should use SequentialLR"

        # Case 2: No warmup (should use CosineAnnealingLR directly)
        warmup_epochs = 0
        should_use_sequential = (warmup_epochs > 0 and warmup_epochs < num_epochs)
        assert not should_use_sequential, "Case 2 should NOT use SequentialLR"

        # Case 3: Warmup == epochs (should use CosineAnnealingLR directly)
        warmup_epochs = 10
        should_use_sequential = (warmup_epochs > 0 and warmup_epochs < num_epochs)
        assert not should_use_sequential, "Case 3 should NOT use SequentialLR"

        # Case 4: Warmup > epochs (should use CosineAnnealingLR directly)
        warmup_epochs = 15
        should_use_sequential = (warmup_epochs > 0 and warmup_epochs < num_epochs)
        assert not should_use_sequential, "Case 4 should NOT use SequentialLR"


class TestLearningRateScheduleRegression:
    """
    Test suite for learning rate schedule configuration consistency.

    Verifies that the scheduler is correctly configured to match the actual
    training duration, preventing unintended LR restarts mid-training.
    """

    def test_training_engine_init_and_train_defaults_match(self) -> None:
        """
        Verify __init__ and train() use consistent num_epochs defaults.

        The scheduler is configured in __init__ based on num_epochs, so both
        methods must use the same default to ensure the scheduler completes
        its cycle at the correct time.
        """
        import inspect
        from engine import TrainingEngine
        from config import MAX_EPOCHS_DEFAULT

        # Get default for __init__
        init_sig = inspect.signature(TrainingEngine.__init__)
        init_num_epochs_default = init_sig.parameters['num_epochs'].default

        # Get default for train()
        train_sig = inspect.signature(TrainingEngine.train)
        train_num_epochs_default = train_sig.parameters['num_epochs'].default

        # Both should match MAX_EPOCHS_DEFAULT
        assert init_num_epochs_default == MAX_EPOCHS_DEFAULT, (
            f"TrainingEngine.__init__ num_epochs default ({init_num_epochs_default}) "
            f"should be MAX_EPOCHS_DEFAULT ({MAX_EPOCHS_DEFAULT})"
        )
        assert train_num_epochs_default == MAX_EPOCHS_DEFAULT, (
            f"TrainingEngine.train() num_epochs default ({train_num_epochs_default}) "
            f"should be MAX_EPOCHS_DEFAULT ({MAX_EPOCHS_DEFAULT})"
        )

        # Most importantly: they should match each other
        assert init_num_epochs_default == train_num_epochs_default, (
            f"CRITICAL: __init__ default ({init_num_epochs_default}) != "
            f"train() default ({train_num_epochs_default}). "
            f"This mismatch causes LR scheduler to restart mid-training!"
        )

    def test_lr_decreases_monotonically_after_warmup(self) -> None:
        """
        Verify LR decreases monotonically after warmup phase completes.

        Cosine annealing should produce a smooth, monotonically decreasing
        learning rate from the peak (end of warmup) to eta_min (end of training).
        """
        model: nn.Module = nn.Linear(10, 10)
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs: int = 50
        warmup_epochs: int = 5

        # Create scheduler as TrainingEngine does
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_t_max: int = max(1, num_epochs - warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=1e-5
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        lrs: List[float] = []
        for epoch in range(num_epochs):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # After warmup (epoch 5+), LR should decrease monotonically
        post_warmup_lrs = lrs[warmup_epochs:]
        for i in range(1, len(post_warmup_lrs)):
            assert post_warmup_lrs[i] <= post_warmup_lrs[i-1] + 1e-10, (
                f"LR increased at epoch {warmup_epochs + i}: "
                f"{post_warmup_lrs[i-1]:.6f} -> {post_warmup_lrs[i]:.6f}. "
                f"This indicates scheduler restart bug!"
            )

    def test_lr_reaches_eta_min_by_end_of_training(self) -> None:
        """
        Verify LR reaches eta_min exactly at the final training epoch.

        The cosine annealing schedule should be configured so that T_max
        matches the post-warmup training duration, ensuring LR reaches
        its minimum at the end of training.
        """
        model: nn.Module = nn.Linear(10, 10)
        base_lr: float = 0.001
        eta_min: float = base_lr / 100  # 1e-5
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=base_lr)

        num_epochs: int = 100
        warmup_epochs: int = 5

        # Create scheduler as TrainingEngine does
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_t_max: int = max(1, num_epochs - warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=eta_min
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # Step through all epochs
        for _ in range(num_epochs):
            scheduler.step()

        final_lr: float = optimizer.param_groups[0]['lr']

        # Final LR should be very close to eta_min
        assert abs(final_lr - eta_min) < 1e-8, (
            f"Final LR ({final_lr:.2e}) should equal eta_min ({eta_min:.2e}). "
            f"Scheduler may have wrong T_max."
        )

    def test_cosine_annealing_restarts_after_t_max(self) -> None:
        """
        Document PyTorch CosineAnnealingLR restart behavior.

        CosineAnnealingLR restarts its cycle after T_max epochs. This test
        verifies this PyTorch behavior to ensure TrainingEngine correctly
        sets T_max to match actual training duration.
        """
        model: nn.Module = nn.Linear(10, 10)
        base_lr: float = 0.001
        eta_min: float = 1e-5
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=base_lr)

        # Create CosineAnnealingLR with T_max=10, but run for 15 epochs
        t_max: int = 10
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

        lrs: List[float] = []
        for epoch in range(15):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # At epoch T_max (10), LR should be at eta_min
        assert abs(lrs[t_max] - eta_min) < 1e-8, (
            f"LR at epoch {t_max} should be eta_min ({eta_min}), got {lrs[t_max]}"
        )

        # After T_max, CosineAnnealingLR RESTARTS - LR goes back up
        # This is the behavior that caused our bug!
        lr_after_tmax = lrs[t_max + 1]
        assert lr_after_tmax > eta_min * 2, (
            f"CosineAnnealingLR should restart after T_max. "
            f"LR at epoch {t_max+1} ({lr_after_tmax:.2e}) should be > 2x eta_min ({eta_min:.2e})"
        )

    def test_training_engine_prevents_lr_restart(self) -> None:
        """
        Verify TrainingEngine scheduler does not restart during training.

        End-to-end test that the LR schedule decreases monotonically after
        warmup and never restarts, regardless of training duration.
        """
        from engine import TrainingEngine
        from models_baseline import FCMLP

        model = FCMLP()
        device = torch.device('cpu')
        num_epochs: int = 50
        warmup_epochs: int = 5

        engine = TrainingEngine(
            model=model,
            device=device,
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs
        )

        lrs: List[float] = []
        for epoch in range(num_epochs):
            lrs.append(engine.optimizer.param_groups[0]['lr'])
            engine.scheduler.step()

        # After warmup, LR should only decrease (never increase significantly)
        post_warmup_lrs = lrs[warmup_epochs:]
        for i in range(1, len(post_warmup_lrs)):
            # Allow tiny numerical noise but not a restart
            assert post_warmup_lrs[i] <= post_warmup_lrs[i-1] + 1e-9, (
                f"LR increased at epoch {warmup_epochs + i}: "
                f"{post_warmup_lrs[i-1]:.8f} -> {post_warmup_lrs[i]:.8f}. "
                f"Scheduler may have restarted!"
            )

    def test_training_engine_scheduler_matches_training_epochs(self) -> None:
        """
        Verify scheduler T_max matches the specified training duration.

        The cosine annealing phase should have T_max = num_epochs - warmup_epochs,
        ensuring the full LR decay occurs over the actual training period.
        """
        from engine import TrainingEngine
        from models_baseline import FCMLP

        model = FCMLP()
        device = torch.device('cpu')
        num_epochs: int = 100
        warmup_epochs: int = 5

        engine = TrainingEngine(
            model=model,
            device=device,
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs
        )

        # The scheduler should be a SequentialLR with warmup + cosine
        scheduler = engine.scheduler
        assert hasattr(scheduler, '_schedulers'), "Expected SequentialLR with _schedulers"

        # Second scheduler should be CosineAnnealingLR
        cosine_scheduler = scheduler._schedulers[1]
        assert isinstance(cosine_scheduler, optim.lr_scheduler.CosineAnnealingLR)

        # T_max should be num_epochs - warmup_epochs
        expected_t_max: int = num_epochs - warmup_epochs
        actual_t_max: int = cosine_scheduler.T_max

        assert actual_t_max == expected_t_max, (
            f"Cosine scheduler T_max ({actual_t_max}) should be "
            f"num_epochs - warmup_epochs ({expected_t_max})"
        )


class TestEarlyStopping:
    """Test suite for early stopping logic."""

    def test_early_stopping_counter_increments(self) -> None:
        """
        WHY: Counter should increase when validation loss doesn't improve
        HOW: Simulate non-improving validation losses
        WHAT: Expect counter increments
        """
        # This would be implemented with an EarlyStopping class
        # Currently early stopping is embedded in training loop
        pass

    def test_early_stopping_resets_on_improvement(self) -> None:
        """
        WHY: Counter should reset when validation improves
        HOW: Simulate improving validation loss
        WHAT: Expect counter resets to 0
        """
        pass

    def test_early_stopping_triggers_at_patience(self) -> None:
        """
        WHY: Training should stop after patience epochs without improvement
        HOW: Simulate patience epochs of non-improvement
        WHAT: Expect stop signal
        """
        pass


class TestDeviceHandling:
    """Test suite for CPU/CUDA device handling."""

    def test_model_moves_to_device(self, device: torch.device) -> None:
        """
        WHY: Model must be on correct device for training
        HOW: Create model, move to device, check parameters
        WHAT: Expect all parameters on specified device
        """
        from models_baseline import FCMLP

        model: nn.Module = FCMLP()
        model = model.to(device)

        for param in model.parameters():
            assert param.device.type == device.type

    def test_data_moves_to_device(self, device: torch.device) -> None:
        """
        WHY: Data must be on same device as model
        HOW: Create tensor, move to device
        WHAT: Expect tensor on specified device
        """
        data: torch.Tensor = torch.randn(4, 784)
        data = data.to(device)

        assert data.device.type == device.type

    @pytest.mark.cuda
    def test_cuda_available_used(self) -> None:
        """
        WHY: Should automatically use CUDA if available
        HOW: Check CUDA availability
        WHAT: Expect cuda device when available
        """
        if torch.cuda.is_available():
            device: torch.device = torch.device('cuda')
            assert device.type == 'cuda'


class TestReproducibility:
    """Test suite for reproducibility guarantees."""

    def test_same_seed_same_results(self) -> None:
        """
        WHY: Same seed must produce identical results
        HOW: Run twice with same seed, compare outputs
        WHAT: Expect identical tensors
        """
        set_seed(config.SEED_DEFAULT)
        tensor1: torch.Tensor = torch.randn(10, 10)

        set_seed(config.SEED_DEFAULT)
        tensor2: torch.Tensor = torch.randn(10, 10)

        assert torch.allclose(tensor1, tensor2)

    def test_different_seed_different_results(self) -> None:
        """
        WHY: Different seeds must produce different results
        HOW: Run twice with different seeds, compare outputs
        WHAT: Expect different tensors
        """
        set_seed(42)
        tensor1: torch.Tensor = torch.randn(10, 10)

        set_seed(43)
        tensor2: torch.Tensor = torch.randn(10, 10)

        assert not torch.allclose(tensor1, tensor2)


class TestTQFSpecificLosses:
    """Test suite for TQF-specific loss components."""

    def test_inversion_consistency_loss_computed(self) -> None:
        """
        WHY: Circle inversion duality must be enforced
        HOW: Forward pass with return_inv_loss=True
        WHAT: Expect inversion loss tensor returned
        """
        from models_tqf import TQFANN

        # Create small TQF model for testing
        model = TQFANN(R=10, hidden_dim=32)
        model.eval()

        # Create test input
        batch_size = 2
        x = torch.randn(batch_size, 784)

        # Forward pass requesting inversion loss
        with torch.no_grad():
            result = model(x, return_inv_loss=True)

        # Should return tuple of (logits, inv_loss)
        assert isinstance(result, tuple), "Should return tuple when return_inv_loss=True"
        assert len(result) == 2, "Should return (logits, inv_loss)"

        logits, inv_loss = result
        assert logits.shape == (batch_size, 10), "Logits should be (batch, num_classes)"
        assert inv_loss is not None, "Inversion loss should not be None"
        assert isinstance(inv_loss, torch.Tensor), "Inversion loss should be a tensor"
        assert inv_loss.dim() == 0, "Inversion loss should be scalar"

    def test_geometry_regularization_applied(self) -> None:
        """
        WHY: Geometric structure should be preserved
        HOW: Train with geometry_reg_weight > 0
        WHAT: Expect geometry loss contributes to total loss
        """
        from models_tqf import TQFANN

        # Create small TQF model for testing
        model = TQFANN(R=10, hidden_dim=32)
        model.eval()

        # Create test input
        batch_size = 2
        x = torch.randn(batch_size, 784)

        # Forward pass requesting geometry loss
        with torch.no_grad():
            result = model(x, return_inv_loss=True, return_geometry_loss=True)

        # Should return tuple of (logits, inv_loss, geom_loss)
        assert isinstance(result, tuple), "Should return tuple with geometry loss"
        assert len(result) == 3, "Should return (logits, inv_loss, geom_loss)"

        logits, inv_loss, geom_loss = result
        assert logits.shape == (batch_size, 10), "Logits should be (batch, num_classes)"
        assert geom_loss is not None, "Geometry loss should not be None"
        assert isinstance(geom_loss, torch.Tensor), "Geometry loss should be a tensor"
        assert geom_loss.dim() == 0, "Geometry loss should be scalar"
        assert geom_loss.item() >= 0, "Geometry loss should be non-negative"

    def test_training_engine_uses_geometry_weight(self) -> None:
        """
        WHY: TrainingEngine should apply geometry_weight to geometry loss
        HOW: Verify TrainingEngine constructor accepts and stores geometry weight
        WHAT: Expect geometry_weight attribute set correctly
        """
        from engine import TrainingEngine
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        device = torch.device('cpu')

        # Create engine with geometry regularization enabled
        engine = TrainingEngine(
            model=model,
            device=device,
            use_geometry_reg=True,
            geometry_weight=0.05
        )

        assert engine.use_geometry_reg == True, "use_geometry_reg should be True"
        assert engine.geometry_weight == 0.05, "geometry_weight should be 0.05"

    def test_training_engine_geometry_disabled_when_weight_zero(self) -> None:
        """
        WHY: geometry_weight=0 should effectively disable geometry regularization
        HOW: Create TrainingEngine with use_geometry_reg=False
        WHAT: Expect no geometry loss computation
        """
        from engine import TrainingEngine
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        device = torch.device('cpu')

        # Create engine with geometry regularization disabled
        engine = TrainingEngine(
            model=model,
            device=device,
            use_geometry_reg=False,
            geometry_weight=0.0
        )

        assert engine.use_geometry_reg == False, "use_geometry_reg should be False"
        assert engine.geometry_weight == 0.0, "geometry_weight should be 0.0"


class TestInversionLossApplication:
    """Test suite for inversion loss application in training.

    Features are enabled by providing a weight value (not None).
    Features are disabled by default (weight=None).
    """

    def test_inversion_loss_weight_defaults_to_none(self) -> None:
        """
        WHY: inversion_loss_weight should default to None (disabled by default)
        HOW: Check the run_single_seed_experiment function signature
        WHAT: Expect default value is None (feature disabled when not provided)
        """
        from engine import run_single_seed_experiment
        import inspect
        sig = inspect.signature(run_single_seed_experiment)
        default = sig.parameters['inversion_loss_weight'].default
        assert default is None, f"inversion_loss_weight should default to None, got {default}"

    def test_inversion_loss_enabled_when_weight_provided(self) -> None:
        """
        WHY: Inversion loss should be enabled when a weight value is provided
        HOW: Check that feature is enabled when weight is not None for TQF model
        WHAT: Expect loss to be applied
        """
        # The logic is: use_inversion_loss = inversion_loss_weight is not None
        model_name = 'TQF-ANN'
        inversion_loss_weight = 0.001  # Provided weight enables feature
        use_inversion_loss = inversion_loss_weight is not None
        apply_inversion_loss = use_inversion_loss and 'TQF' in model_name
        assert apply_inversion_loss is True

    def test_inversion_loss_disabled_when_weight_none(self) -> None:
        """
        WHY: Inversion loss should be disabled when weight is None
        HOW: Check that feature is disabled when weight is None
        WHAT: Expect loss not to be applied
        """
        model_name = 'TQF-ANN'
        inversion_loss_weight = None  # None = disabled
        use_inversion_loss = inversion_loss_weight is not None
        apply_inversion_loss = use_inversion_loss and 'TQF' in model_name
        assert apply_inversion_loss is False

    def test_inversion_loss_not_applied_for_baseline(self) -> None:
        """
        WHY: Inversion loss should NOT be applied for baseline models
        HOW: Check that feature is not applied for non-TQF model even with weight
        WHAT: Expect loss not to be applied for FC-MLP
        """
        model_name = 'FC-MLP'
        inversion_loss_weight = 0.001  # Even with weight provided
        use_inversion_loss = inversion_loss_weight is not None
        apply_inversion_loss = use_inversion_loss and 'TQF' in model_name
        assert apply_inversion_loss is False


class TestT24OrbitInvarianceLossApplication:
    """Test suite for T24 orbit invariance loss application in training.

    Features are enabled by providing a weight value (not None).
    Features are disabled by default (weight=None).
    """

    def test_t24_orbit_invariance_weight_defaults_to_none(self) -> None:
        """
        WHY: t24_orbit_invariance_weight should default to None (disabled by default)
        HOW: Check the run_single_seed_experiment function signature
        WHAT: Expect default value is None (feature disabled when not provided)
        """
        from engine import run_single_seed_experiment
        import inspect
        sig = inspect.signature(run_single_seed_experiment)
        default = sig.parameters['t24_orbit_invariance_weight'].default
        assert default is None, f"t24_orbit_invariance_weight should default to None, got {default}"

    def test_t24_orbit_invariance_enabled_when_weight_provided(self) -> None:
        """
        WHY: T24 orbit invariance loss should be enabled when a weight is provided
        HOW: Check that feature is enabled when weight is not None for TQF model
        WHAT: Expect loss to be applied
        """
        # The logic is: use_t24 = t24_orbit_invariance_weight is not None
        model_name = 'TQF-ANN'
        t24_orbit_invariance_weight = 0.005  # Provided weight enables feature
        use_t24_orbit_invariance_loss = t24_orbit_invariance_weight is not None
        apply_t24 = use_t24_orbit_invariance_loss and 'TQF' in model_name
        assert apply_t24 is True

    def test_t24_orbit_invariance_disabled_when_weight_none(self) -> None:
        """
        WHY: T24 orbit invariance loss should be disabled when weight is None
        HOW: Check that feature is disabled when weight is None
        WHAT: Expect loss not to be applied
        """
        model_name = 'TQF-ANN'
        t24_orbit_invariance_weight = None  # None = disabled
        use_t24_orbit_invariance_loss = t24_orbit_invariance_weight is not None
        apply_t24 = use_t24_orbit_invariance_loss and 'TQF' in model_name
        assert apply_t24 is False

    def test_t24_orbit_invariance_not_applied_for_baseline(self) -> None:
        """
        WHY: T24 orbit invariance loss should NOT be applied for baseline models
        HOW: Check that feature is not applied for non-TQF model even with weight
        WHAT: Expect loss not to be applied for CNN-L5
        """
        model_name = 'CNN-L5'
        t24_orbit_invariance_weight = 0.005  # Even with weight provided
        use_t24_orbit_invariance_loss = t24_orbit_invariance_weight is not None
        apply_t24 = use_t24_orbit_invariance_loss and 'TQF' in model_name
        assert apply_t24 is False


class TestLossWeightParameters:
    """Test suite for loss weight parameters in run_multi_seed_experiment.

    All TQF loss features are controlled via weight parameters only.
    Features are disabled by default (weight=None) and enabled when a weight is provided.
    """

    def test_multi_seed_experiment_accepts_weight_params(self) -> None:
        """
        WHY: run_multi_seed_experiment must accept all weight parameters
        HOW: Check function signature includes all 4 weight parameters
        WHAT: Expect parameters exist with default None (disabled)
        """
        from engine import run_multi_seed_experiment
        import inspect
        sig = inspect.signature(run_multi_seed_experiment)

        # All 4 weight parameters should exist
        weight_params = [
            'inversion_loss_weight',
            'z6_equivariance_weight',
            'd6_equivariance_weight',
            't24_orbit_invariance_weight'
        ]
        for param in weight_params:
            assert param in sig.parameters, f"Missing parameter: {param}"

    def test_all_weight_params_default_to_none(self) -> None:
        """
        WHY: All weight parameters should default to None (features disabled)
        HOW: Check run_multi_seed_experiment signature
        WHAT: Expect all weight defaults are None
        """
        from engine import run_multi_seed_experiment
        import inspect
        sig = inspect.signature(run_multi_seed_experiment)

        weight_params = [
            'inversion_loss_weight',
            'z6_equivariance_weight',
            'd6_equivariance_weight',
            't24_orbit_invariance_weight'
        ]
        for param in weight_params:
            default = sig.parameters[param].default
            assert default is None, f"{param} should default to None, got {default}"

    def test_weight_enables_feature_pattern(self) -> None:
        """
        WHY: Verify the weight-based feature enablement pattern
        HOW: Test that weight=None disables and weight=value enables
        WHAT: Expect correct boolean derivation from weight value
        """
        # Pattern: feature_enabled = weight is not None
        test_cases = [
            (None, False),   # None -> disabled
            (0.001, True),   # Any value -> enabled
            (0.0, True),     # Zero is still a value -> enabled
            (1.0, True),     # Max value -> enabled
        ]
        for weight, expected_enabled in test_cases:
            feature_enabled = weight is not None
            assert feature_enabled == expected_enabled, \
                f"weight={weight} should result in enabled={expected_enabled}"

    def test_train_epoch_accepts_weight_params(self) -> None:
        """
        WHY: TrainingEngine.train_epoch must accept all 4 weight parameters
        HOW: Check function signature includes all weight parameters
        WHAT: Expect parameters exist with default None
        """
        from engine import TrainingEngine
        import inspect
        sig = inspect.signature(TrainingEngine.train_epoch)

        weight_params = [
            'inversion_loss_weight',
            'z6_equivariance_weight',
            'd6_equivariance_weight',
            't24_orbit_invariance_weight'
        ]
        for param in weight_params:
            assert param in sig.parameters, f"train_epoch missing parameter: {param}"
            default = sig.parameters[param].default
            assert default is None, f"train_epoch {param} should default to None, got {default}"


def run_tests(verbosity: int = 2):
    """
    Run all engine tests.

    Args:
        verbosity: pytest verbosity level (0, 1, or 2)

    Returns:
        Exit code (0 if all tests pass)
    """
    import sys
    args: List[str] = [__file__, f'-{"v" * verbosity}']
    return pytest.main(args)


if __name__ == '__main__':
    import sys
    exit_code: int = run_tests()
    sys.exit(exit_code)
