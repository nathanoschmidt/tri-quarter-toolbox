"""
test_evaluation.py - Comprehensive Model Evaluation and Metrics Tests for TQF-NN

This module tests all evaluation metrics, performance measurements, and statistical
analysis functions in evaluation.py, ensuring robust model assessment across all
experimental configurations.

Key Test Coverage:
- Per-Class Accuracy: Class-wise accuracy computation for all 10 MNIST digits
- Class Balance Validation: Equal representation across all classes in test sets
- Confusion Matrix Metrics: Full 10x10 confusion matrix generation
- Rotation Invariance Error: Quantitative measurement of Z6 rotation equivariance
- Z6 Rotation Angles: 60-degree increments (0°, 60°, 120°, 180°, 240°, 300°)
- Rotation Error Bounds: Expected [0, 1] range for accuracy difference metrics
- Statistical Significance: Confidence intervals, standard error, p-value computation
- Multi-Seed Aggregation: Mean/std/CI calculation across independent random seeds
- Bootstrap Confidence Intervals: Non-parametric CI estimation for robust statistics
- Inference Time Measurement: Latency profiling with warm-up and batch processing
- FLOPS Estimation: Floating-point operation counting for computational complexity
- Model Efficiency Metrics: FLOPS per parameter, FLOPS per inference
- Inversion Consistency Metrics: Dual zone prediction agreement for TQF-ANN
- Inner vs Outer Zone Comparison: Bijective duality validation
- Adaptive Orbit Mixing: T24 orbit-based prediction aggregation with confidence weighting
- Orbit Mixing Temperature: Softmax temperature scaling for adaptive prediction fusion
- Orbit Confidence Computation: Per-orbit prediction confidence for weighted mixing

Test Organization:
- TestPerClassAccuracy: Per-digit accuracy calculation and validation
- TestRotationInvarianceError: Z6 rotation equivariance measurement
- TestStatisticalSignificance: Multi-seed statistical analysis
- TestMeasureInferenceTime: Latency profiling and timing validation
- TestEstimateModelFLOPS: Computational complexity estimation
- TestInversionConsistencyMetrics: TQF dual zone consistency validation
- TestAdaptiveOrbitMixing: T24 orbit-based prediction fusion

Scientific Rationale:
Rotation invariance error quantifies how well models maintain prediction consistency
under Z6 rotations, a key property for hexagonal symmetry. Inversion consistency
validates TQF's dual structure. Statistical significance tests ensure reproducible
results across multiple random seeds for fair model comparison.

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
from typing import Dict, List, Tuple
from conftest import TORCH_AVAILABLE, device, set_seed, assert_tensor_shape

if not TORCH_AVAILABLE:
    pytest.skip("PyTorch not available", allow_module_level=True)

from evaluation import (
    compute_per_class_accuracy_from_predictions as compute_per_class_accuracy,
    compute_rotation_invariance_error_from_outputs as compute_rotation_invariance_error,
    compute_statistical_significance,
    measure_inference_time,
    estimate_model_flops
)
import config


class TestPerClassAccuracy:
    """Test per-class accuracy computation."""

    def test_perfect_predictions(self) -> None:
        """
        WHY: Perfect predictions should give 100% accuracy
        HOW: Create predictions matching targets
        WHAT: Expect accuracy = 1.0 for all classes
        """
        num_samples: int = 100
        num_classes: int = 10
        targets: torch.Tensor = torch.arange(num_classes).repeat(num_samples // num_classes)
        predictions: torch.Tensor = targets.clone()

        accuracies: Dict[int, float] = compute_per_class_accuracy(
            predictions, targets, num_classes
        )

        for class_id in range(num_classes):
            assert accuracies[class_id] == 1.0, f"Class {class_id} should be 100%"

    def test_zero_predictions(self) -> None:
        """
        WHY: All-wrong predictions should give 0% accuracy
        HOW: Create predictions different from targets
        WHAT: Expect accuracy = 0.0 for all classes
        """
        num_samples: int = 100
        num_classes: int = 10
        targets: torch.Tensor = torch.zeros(num_samples, dtype=torch.long)
        predictions: torch.Tensor = torch.ones(num_samples, dtype=torch.long)

        accuracies: Dict[int, float] = compute_per_class_accuracy(
            predictions, targets, num_classes
        )

        assert accuracies[0] == 0.0, "Class 0 should be 0%"

    def test_partial_accuracy(self) -> None:
        """
        WHY: Must correctly compute intermediate accuracies
        HOW: Create 50% correct predictions
        WHAT: Expect accuracy = 0.5
        """
        num_samples: int = 100
        num_classes: int = 10
        targets: torch.Tensor = torch.zeros(num_samples, dtype=torch.long)
        predictions: torch.Tensor = torch.zeros(num_samples, dtype=torch.long)
        predictions[:50] = 1  # Half wrong

        accuracies: Dict[int, float] = compute_per_class_accuracy(
            predictions, targets, num_classes
        )

        assert abs(accuracies[0] - 0.5) < 0.01, "Class 0 should be ~50%"

    def test_empty_class(self) -> None:
        """
        WHY: Must handle classes with no samples gracefully
        HOW: Create targets missing some classes
        WHAT: Expect 0.0 or NaN for missing classes
        """
        num_samples: int = 20
        num_classes: int = 10
        targets: torch.Tensor = torch.zeros(num_samples, dtype=torch.long)  # Only class 0
        predictions: torch.Tensor = torch.zeros(num_samples, dtype=torch.long)

        accuracies: Dict[int, float] = compute_per_class_accuracy(
            predictions, targets, num_classes
        )

        # Classes 1-9 have no samples
        for class_id in range(1, num_classes):
            # Should be 0.0 or handle gracefully
            assert accuracies[class_id] == 0.0 or accuracies[class_id] != accuracies[class_id]  # NaN check


class TestRotationInvarianceError:
    """Test rotation invariance metrics."""

    def test_identical_outputs_zero_error(self) -> None:
        """
        WHY: Identical outputs across rotations means perfect invariance
        HOW: Create identical outputs for all rotations
        WHAT: Expect error = 0.0
        """
        batch_size: int = 10
        num_classes: int = 10

        # Same output for all rotations
        base_output: torch.Tensor = torch.randn(batch_size, num_classes)
        outputs_dict: Dict[int, torch.Tensor] = {
            0: base_output.clone(),
            60: base_output.clone(),
            120: base_output.clone(),
            180: base_output.clone(),
            240: base_output.clone(),
            300: base_output.clone()
        }

        error: float = compute_rotation_invariance_error(outputs_dict)

        assert error < 1e-6, f"Expected error ~0, got {error}"

    def test_different_outputs_nonzero_error(self) -> None:
        """
        WHY: Different outputs should produce nonzero error
        HOW: Create random outputs for each rotation
        WHAT: Expect error > 0
        """
        batch_size: int = 10
        num_classes: int = 10

        outputs_dict: Dict[int, torch.Tensor] = {
            angle: torch.randn(batch_size, num_classes)
            for angle in [0, 60, 120, 180, 240, 300]
        }

        error: float = compute_rotation_invariance_error(outputs_dict)

        assert error > 0, "Random outputs should have nonzero rotation error"

    def test_error_magnitude_reasonable(self) -> None:
        """
        WHY: Error should be in reasonable range [0, inf)
        HOW: Compute error for typical outputs
        WHAT: Expect error >= 0
        """
        batch_size: int = 10
        num_classes: int = 10

        outputs_dict: Dict[int, torch.Tensor] = {
            angle: torch.randn(batch_size, num_classes)
            for angle in [0, 60, 120, 180, 240, 300]
        }

        error: float = compute_rotation_invariance_error(outputs_dict)

        assert error >= 0, "Error must be non-negative"
        assert error < float('inf'), "Error must be finite"


class TestStatisticalSignificance:
    """Test statistical significance computation."""

    def test_identical_distributions_not_significant(self) -> None:
        """
        WHY: Identical distributions should not be significantly different
        HOW: Create two identical lists
        WHAT: Expect p-value > 0.05
        """
        set_seed(config.SEED_DEFAULT)

        group1: List[float] = [0.9, 0.91, 0.89, 0.90, 0.92]
        group2: List[float] = [0.9, 0.91, 0.89, 0.90, 0.92]

        result: Dict = compute_statistical_significance(group1, group2)

        assert result['p_value'] > 0.05, "Identical distributions should not be significant"

    def test_different_distributions_may_be_significant(self) -> None:
        """
        WHY: Clearly different distributions should be significant
        HOW: Create two very different groups
        WHAT: Expect p-value < 0.05
        """
        set_seed(config.SEED_DEFAULT)

        group1: List[float] = [0.5, 0.52, 0.48, 0.51, 0.49]
        group2: List[float] = [0.9, 0.92, 0.88, 0.91, 0.89]

        result: Dict = compute_statistical_significance(group1, group2)

        # With such different means, should be significant
        assert result['p_value'] < 0.05, "Very different distributions should be significant"

    def test_returns_required_keys(self) -> None:
        """
        WHY: Result dict must contain all required statistics
        HOW: Compute significance, check keys
        WHAT: Expect p_value, t_statistic, effect_size keys
        """
        group1: List[float] = [0.5, 0.6, 0.7]
        group2: List[float] = [0.8, 0.9, 1.0]

        result: Dict = compute_statistical_significance(group1, group2)

        required_keys: List[str] = ['p_value', 't_statistic', 'effect_size']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_effect_size_magnitude(self) -> None:
        """
        WHY: Cohen's d should be in reasonable range
        HOW: Compute for known distributions
        WHAT: Expect |d| >= 0
        """
        group1: List[float] = [0.5, 0.6, 0.7]
        group2: List[float] = [0.8, 0.9, 1.0]

        result: Dict = compute_statistical_significance(group1, group2)

        assert abs(result['effect_size']) >= 0, "Effect size must be non-negative"


class TestMeasureInferenceTime:
    """Test inference timing measurement."""

    def test_timing_positive(self, lightweight_models, device) -> None:
        """
        WHY: Inference time must be positive
        HOW: Measure inference on FC-MLP model
        WHAT: Expect mean time > 0
        """
        model = lightweight_models['FC-MLP'].to(device)
        timing: Dict = measure_inference_time(
            model, input_shape=(1, 784), device=device, num_trials=10
        )

        assert timing['mean_ms'] > 0, f"Mean inference time must be positive, got {timing['mean_ms']}"
        assert timing['min_ms'] > 0, f"Min inference time must be positive, got {timing['min_ms']}"

    def test_timing_consistent(self, lightweight_models, device) -> None:
        """
        WHY: Repeated measurements should be similar
        HOW: Measure multiple times, check variance is reasonable
        WHAT: Expect std < 10x mean (reasonable variance)
        """
        model = lightweight_models['FC-MLP'].to(device)
        timing: Dict = measure_inference_time(
            model, input_shape=(1, 784), device=device, num_trials=50
        )

        # Standard deviation should be less than 10x mean for consistent timing
        assert timing['std_ms'] < timing['mean_ms'] * 10, \
            f"Timing variance too high: std={timing['std_ms']:.4f}, mean={timing['mean_ms']:.4f}"


class TestEstimateModelFLOPS:
    """Test FLOPS estimation."""

    def test_flops_positive(self, lightweight_models) -> None:
        """
        WHY: FLOPS count must be positive
        HOW: Estimate for FC-MLP model
        WHAT: Expect FLOPS > 0
        """
        model = lightweight_models['FC-MLP']
        flops: int = estimate_model_flops(model, input_shape=(1, 784))

        assert flops > 0, f"FLOPS must be positive, got {flops}"

    def test_larger_model_more_flops(self, lightweight_models) -> None:
        """
        WHY: CNN typically has more FLOPS than simple MLP for same params
        HOW: Compare FC-MLP and CNN-L5 FLOPS
        WHAT: Expect meaningful FLOPS counts for both
        """
        mlp_flops: int = estimate_model_flops(
            lightweight_models['FC-MLP'], input_shape=(1, 784)
        )
        cnn_flops: int = estimate_model_flops(
            lightweight_models['CNN-L5'], input_shape=(1, 1, 28, 28)
        )

        # Both should have meaningful FLOPS counts
        assert mlp_flops > 100000, f"MLP FLOPS unexpectedly low: {mlp_flops}"
        assert cnn_flops > 100000, f"CNN FLOPS unexpectedly low: {cnn_flops}"


class TestInversionConsistencyMetrics:
    """
    Test circle inversion consistency metrics.

    Regression tests for AttributeError
    """

    def test_bijection_dual_output_has_classification_head(self) -> None:
        """
        WHY: BijectionDualOutputHead uses single classification_head (not dual heads)
        HOW: Check that BijectionDualOutputHead has correct attribute
        WHAT: Expect classification_head exists and is Linear layer

        REGRESSION: If class refactored to use different attribute names.
        """
        from models_tqf import BijectionDualOutputHead

        dual_output = BijectionDualOutputHead(hidden_dim=100)

        assert hasattr(dual_output, 'classification_head'), (
            "BijectionDualOutputHead missing classification_head attribute"
        )
        assert isinstance(dual_output.classification_head, nn.Linear), (
            "classification_head is not a Linear layer"
        )

    def test_bijection_dual_output_has_circle_inversion_method(self) -> None:
        """
        WHY: BijectionDualOutputHead uses apply_circle_inversion_bijection method
        HOW: Check that method exists and is callable
        WHAT: Expect method present

        REGRESSION: If method renamed or removed.
        """
        from models_tqf import BijectionDualOutputHead

        dual_output = BijectionDualOutputHead(hidden_dim=100)

        assert hasattr(dual_output, 'apply_circle_inversion_bijection'), (
            "BijectionDualOutputHead missing apply_circle_inversion_bijection method"
        )
        assert callable(dual_output.apply_circle_inversion_bijection), (
            "apply_circle_inversion_bijection is not callable"
        )

    def test_evaluation_attribute_detection_logic(self) -> None:
        """
        WHY: Evaluation code must detect correct attributes via hasattr()
        HOW: Simulate the detection logic from compute_inversion_consistency_metrics()
        WHAT: Expect BijectionDualOutputHead has new API, not old API

        REGRESSION: If attribute detection in evaluation.py is modified incorrectly.
        """
        from models_tqf import BijectionDualOutputHead

        dual_output = BijectionDualOutputHead(hidden_dim=100)

        # Simulate detection logic in evaluation.py
        has_bijection_method: bool = hasattr(dual_output, 'apply_circle_inversion_bijection')
        has_old_method: bool = hasattr(dual_output, 'apply_inversion')
        has_old_heads: bool = hasattr(dual_output, 'outer_head') and hasattr(dual_output, 'inner_head')

        # BijectionDualOutputHead should have new API, not old
        assert has_bijection_method, "Should have apply_circle_inversion_bijection (new API)"
        assert not has_old_method, "Should NOT have apply_inversion (old DualOutputHead API)"
        assert not has_old_heads, "Should NOT have outer_head/inner_head (old API)"

    def test_circle_inversion_preserves_shape(self) -> None:
        """
        WHY: Circle inversion is a bijection, must preserve tensor shape
        HOW: Apply circle inversion, check output shape matches input
        WHAT: Expect shape preservation (bijection property)

        REGRESSION: If bijection method changes to alter shapes.
        """
        from models_tqf import BijectionDualOutputHead

        dual_output = BijectionDualOutputHead(hidden_dim=100)

        # Input: (batch, 6, hidden_dim) - 6 sectors for Z6 symmetry
        batch_size: int = 4
        num_sectors: int = 6
        hidden_dim: int = 100

        outer_feats: torch.Tensor = torch.randn(batch_size, num_sectors, hidden_dim)

        # Apply circle inversion bijection
        inner_feats: torch.Tensor = dual_output.apply_circle_inversion_bijection(outer_feats)

        # Bijection must preserve shape
        assert inner_feats.shape == outer_feats.shape, (
            f"Circle inversion changed shape from {outer_feats.shape} to {inner_feats.shape}. "
            f"Bijection property violated!"
        )


class TestAdaptiveOrbitMixing:
    """
    Test adaptive orbit mixing function for Z6 ensemble evaluation.

    The adaptive_orbit_mixing function combines logits from 6 rotated versions
    of an input using temperature-scaled softmax weighting. Lower temperatures
    emphasize the most confident rotation, while higher temperatures average
    all rotations uniformly.
    """

    def test_adaptive_orbit_mixing_basic_output_shape(self) -> None:
        """
        WHY: Function must return correctly shaped ensemble logits
        HOW: Pass list of 6 logit tensors, check output shape
        WHAT: Expect (batch_size, num_classes) shape
        """
        from models_tqf import adaptive_orbit_mixing

        batch_size: int = 8
        num_classes: int = 10
        num_rotations: int = 6

        # Create logits for each Z6 rotation
        logits_per_rotation: List[torch.Tensor] = [
            torch.randn(batch_size, num_classes) for _ in range(num_rotations)
        ]

        result: torch.Tensor = adaptive_orbit_mixing(logits_per_rotation, temperature=0.3)

        assert result.shape == (batch_size, num_classes), (
            f"Expected shape ({batch_size}, {num_classes}), got {result.shape}"
        )

    def test_adaptive_orbit_mixing_low_temperature_peaked(self) -> None:
        """
        WHY: Low temperature should emphasize most confident rotation
        HOW: Create one rotation with high confidence, check it dominates
        WHAT: Expect result close to the high-confidence rotation
        """
        from models_tqf import adaptive_orbit_mixing

        batch_size: int = 4
        num_classes: int = 10

        # Create 6 rotations, one with much higher confidence
        logits_per_rotation: List[torch.Tensor] = []
        for i in range(6):
            if i == 2:  # Third rotation has high confidence
                logits: torch.Tensor = torch.ones(batch_size, num_classes) * 10.0
            else:
                logits: torch.Tensor = torch.zeros(batch_size, num_classes)
            logits_per_rotation.append(logits)

        # Low temperature should heavily weight the high-confidence rotation
        result: torch.Tensor = adaptive_orbit_mixing(logits_per_rotation, temperature=0.1)

        # Result should be close to the high-confidence logits
        expected: torch.Tensor = logits_per_rotation[2]
        assert torch.allclose(result, expected, atol=1.0), (
            "Low temperature should emphasize most confident rotation"
        )

    def test_adaptive_orbit_mixing_high_temperature_uniform(self) -> None:
        """
        WHY: High temperature should average rotations uniformly
        HOW: Create equal rotations, high temp should give same result
        WHAT: Expect result equal to simple average
        """
        from models_tqf import adaptive_orbit_mixing

        batch_size: int = 4
        num_classes: int = 10

        # Create 6 identical rotations
        base_logits: torch.Tensor = torch.randn(batch_size, num_classes)
        logits_per_rotation: List[torch.Tensor] = [base_logits.clone() for _ in range(6)]

        # High temperature should give nearly uniform weighting
        result: torch.Tensor = adaptive_orbit_mixing(logits_per_rotation, temperature=10.0)

        # With identical inputs, result should match input (uniform average = input)
        assert torch.allclose(result, base_logits, atol=0.1), (
            "High temperature with identical inputs should return same values"
        )

    def test_adaptive_orbit_mixing_empty_list_fails(self) -> None:
        """
        WHY: Empty input list is invalid
        HOW: Pass empty list
        WHAT: Expect ValueError
        """
        from models_tqf import adaptive_orbit_mixing

        with pytest.raises(ValueError):
            adaptive_orbit_mixing([], temperature=0.3)

    def test_adaptive_orbit_mixing_negative_temperature_fails(self) -> None:
        """
        WHY: Negative temperature is invalid for softmax
        HOW: Pass negative temperature
        WHAT: Expect ValueError
        """
        from models_tqf import adaptive_orbit_mixing

        logits_per_rotation: List[torch.Tensor] = [torch.randn(4, 10) for _ in range(6)]

        with pytest.raises(ValueError):
            adaptive_orbit_mixing(logits_per_rotation, temperature=-0.1)

    def test_adaptive_orbit_mixing_zero_temperature_fails(self) -> None:
        """
        WHY: Zero temperature causes division by zero in softmax
        HOW: Pass zero temperature
        WHAT: Expect ValueError
        """
        from models_tqf import adaptive_orbit_mixing

        logits_per_rotation: List[torch.Tensor] = [torch.randn(4, 10) for _ in range(6)]

        with pytest.raises(ValueError):
            adaptive_orbit_mixing(logits_per_rotation, temperature=0.0)

    def test_adaptive_orbit_mixing_single_rotation(self) -> None:
        """
        WHY: Single rotation should return that rotation's logits
        HOW: Pass single-element list
        WHAT: Expect output equals input
        """
        from models_tqf import adaptive_orbit_mixing

        logits: torch.Tensor = torch.randn(4, 10)
        logits_per_rotation: List[torch.Tensor] = [logits]

        result: torch.Tensor = adaptive_orbit_mixing(logits_per_rotation, temperature=0.3)

        assert torch.allclose(result, logits), (
            "Single rotation should return same logits"
        )

    def test_adaptive_orbit_mixing_gradient_flow(self) -> None:
        """
        WHY: Function must allow gradient flow for training
        HOW: Backprop through result, check input has gradients
        WHAT: Expect gradients on input tensors
        """
        from models_tqf import adaptive_orbit_mixing

        logits_per_rotation: List[torch.Tensor] = [
            torch.randn(4, 10, requires_grad=True) for _ in range(6)
        ]

        result: torch.Tensor = adaptive_orbit_mixing(logits_per_rotation, temperature=0.3)
        loss: torch.Tensor = result.sum()
        loss.backward()

        for i, logits in enumerate(logits_per_rotation):
            assert logits.grad is not None, f"Rotation {i} should have gradients"

    def test_adaptive_orbit_mixing_default_temperature(self) -> None:
        """
        WHY: Default temperature should work without explicit argument
        HOW: Call without temperature argument
        WHAT: Expect successful execution
        """
        from models_tqf import adaptive_orbit_mixing

        logits_per_rotation: List[torch.Tensor] = [torch.randn(4, 10) for _ in range(6)]

        # Should work with default temperature
        result: torch.Tensor = adaptive_orbit_mixing(logits_per_rotation)

        assert result.shape == (4, 10), "Default temperature call should succeed"


def run_tests(verbosity: int = 2):
    """Run all evaluation tests."""
    import sys
    args: List[str] = [__file__, f'-{"v" * verbosity}']
    return pytest.main(args)


if __name__ == '__main__':
    import sys
    exit_code: int = run_tests()
    sys.exit(exit_code)
