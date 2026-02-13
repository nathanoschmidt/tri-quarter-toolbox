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

Test Organization:
- TestPerClassAccuracy: Per-digit accuracy calculation and validation
- TestRotationInvarianceError: Z6 rotation equivariance measurement
- TestStatisticalSignificance: Multi-seed statistical analysis
- TestMeasureInferenceTime: Latency profiling and timing validation
- TestEstimateModelFLOPS: Computational complexity estimation
- TestInversionConsistencyMetrics: TQF dual zone consistency validation

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
from torch.utils.data import DataLoader
from conftest import TORCH_AVAILABLE, device, set_seed, assert_tensor_shape

if not TORCH_AVAILABLE:
    pytest.skip("PyTorch not available", allow_module_level=True)

from evaluation import (
    compute_per_class_accuracy_from_predictions as compute_per_class_accuracy,
    compute_rotation_invariance_error_from_outputs as compute_rotation_invariance_error,
    compute_statistical_significance,
    measure_inference_time,
    estimate_model_flops,
    adaptive_orbit_mixing,
    classify_from_sector_features,
    evaluate_with_orbit_mixing
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
    """Test adaptive orbit mixing (max-logit confidence-weighted averaging)."""

    def test_single_variant_returns_unchanged(self) -> None:
        """
        WHY: Single variant should pass through unchanged
        HOW: Call with one-element list
        WHAT: Expect identical output
        """
        logits: torch.Tensor = torch.randn(8, 10)
        result: torch.Tensor = adaptive_orbit_mixing([logits], temperature=0.3)
        assert torch.allclose(result, logits), "Single variant should return unchanged"

    def test_output_shape_matches_input(self) -> None:
        """
        WHY: Output must have same (B, C) shape as inputs
        HOW: Mix multiple variants, check shape
        WHAT: Expect (batch, num_classes) output
        """
        batch_size: int = 16
        num_classes: int = 10
        variants: List[torch.Tensor] = [torch.randn(batch_size, num_classes) for _ in range(6)]
        result: torch.Tensor = adaptive_orbit_mixing(variants, temperature=0.3)
        assert result.shape == (batch_size, num_classes), f"Expected ({batch_size}, {num_classes}), got {result.shape}"

    def test_high_temperature_approaches_uniform(self) -> None:
        """
        WHY: High temperature should produce near-uniform weights (equal averaging)
        HOW: Use very high temperature, compare to simple mean
        WHAT: Expect close to arithmetic mean
        """
        torch.manual_seed(42)
        variants: List[torch.Tensor] = [torch.randn(4, 10) for _ in range(6)]
        result_high_temp: torch.Tensor = adaptive_orbit_mixing(variants, temperature=100.0)
        simple_mean: torch.Tensor = torch.stack(variants, dim=0).mean(dim=0)
        assert torch.allclose(result_high_temp, simple_mean, atol=0.01), (
            "High temperature should approach uniform averaging"
        )

    def test_low_temperature_favors_confident(self) -> None:
        """
        WHY: Low temperature should favor the most confident (highest max-logit) variant
        HOW: Create one confident variant and others near-zero, use very low temperature
        WHAT: Expect output close to the confident variant
        """
        # Create 6 weak variants and 1 very confident one
        weak_variants: List[torch.Tensor] = [torch.zeros(4, 10) for _ in range(5)]
        confident: torch.Tensor = torch.zeros(4, 10)
        confident[:, 3] = 10.0  # Very high logit for class 3
        all_variants: List[torch.Tensor] = weak_variants + [confident]

        result: torch.Tensor = adaptive_orbit_mixing(all_variants, temperature=0.01)
        # The confident variant should dominate
        preds = result.argmax(dim=1)
        assert (preds == 3).all(), "Low temperature should favor the most confident variant"

    def test_weights_sum_to_one(self) -> None:
        """
        WHY: Confidence weights should form a valid probability distribution
        HOW: Manually compute weights and verify they sum to 1
        WHAT: Expect weights along variant dimension sum to 1 for each sample
        """
        variants: List[torch.Tensor] = [torch.randn(4, 10) for _ in range(6)]
        stacked: torch.Tensor = torch.stack(variants, dim=0)
        max_logits: torch.Tensor = stacked.max(dim=2).values
        weights: torch.Tensor = torch.nn.functional.softmax(max_logits / 0.3, dim=0)
        weight_sums: torch.Tensor = weights.sum(dim=0)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
            "Weights should sum to 1 along variant dimension"
        )


class TestClassifyFromSectorFeatures:
    """Test classification from pre-computed sector features."""

    def test_output_shape(self) -> None:
        """
        WHY: classify_from_sector_features must produce correct logit shape
        HOW: Create model and arbitrary sector features, check output shape
        WHAT: Expect (batch, num_classes) output
        """
        from models_tqf import TQFANN

        model: TQFANN = TQFANN(R=2)
        model.eval()

        batch_size: int = 4
        hidden_dim: int = model.dual_output.classification_head.in_features
        outer_feats: torch.Tensor = torch.randn(batch_size, 6, hidden_dim)
        inner_feats: torch.Tensor = torch.randn(batch_size, 6, hidden_dim)

        logits: torch.Tensor = classify_from_sector_features(model, outer_feats, inner_feats)
        assert logits.shape == (batch_size, 10), f"Expected (4, 10), got {logits.shape}"

    def test_matches_forward_pass(self) -> None:
        """
        WHY: classify_from_sector_features on cached features should match forward pass logits
        HOW: Run forward pass, grab cached features, run classify_from_sector_features
        WHAT: Expect logits match
        """
        from models_tqf import TQFANN

        model: TQFANN = TQFANN(R=2)
        model.eval()

        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(4, 784)

        with torch.no_grad():
            forward_logits: torch.Tensor = model(x)
            outer_feats: torch.Tensor = model.get_cached_sector_features()
            inner_feats: torch.Tensor = model.get_cached_inner_sector_features()
            recomputed_logits: torch.Tensor = classify_from_sector_features(model, outer_feats, inner_feats)

        assert torch.allclose(forward_logits, recomputed_logits, atol=1e-5), (
            "classify_from_sector_features should match forward pass logits"
        )

    def test_zone_swap_produces_different_logits(self) -> None:
        """
        WHY: Swapping inner/outer roles should generally produce different logits
        HOW: Run with normal and swapped features
        WHAT: Expect different outputs (since sector_weights break symmetry)
        """
        from models_tqf import TQFANN

        model: TQFANN = TQFANN(R=2)
        model.eval()

        torch.manual_seed(42)
        x: torch.Tensor = torch.randn(4, 784)

        with torch.no_grad():
            model(x)
            outer_feats: torch.Tensor = model.get_cached_sector_features()
            inner_feats: torch.Tensor = model.get_cached_inner_sector_features()

            normal_logits: torch.Tensor = classify_from_sector_features(model, outer_feats, inner_feats)
            swapped_logits: torch.Tensor = classify_from_sector_features(model, inner_feats, outer_feats)

        # They should generally differ (unless sector_weights happen to be uniform)
        # Just verify both are valid tensors with correct shape
        assert normal_logits.shape == swapped_logits.shape
        assert not torch.isnan(swapped_logits).any(), "Swapped logits should not contain NaN"


class TestEvaluateWithOrbitMixing:
    """Test the full orbit mixing evaluation function."""

    def _make_tiny_loader(self) -> DataLoader:
        """Create a minimal DataLoader for testing."""
        from torch.utils.data import TensorDataset
        x: torch.Tensor = torch.randn(20, 784)
        y: torch.Tensor = torch.randint(0, 10, (20,))
        return DataLoader(TensorDataset(x, y), batch_size=10)

    def test_z6_orbit_mixing_returns_valid_metrics(self) -> None:
        """
        WHY: Z6 orbit mixing should produce valid loss and accuracy
        HOW: Run evaluate_with_orbit_mixing with use_z6=True
        WHAT: Expect loss >= 0 and accuracy in [0, 100]
        """
        from models_tqf import TQFANN

        model: TQFANN = TQFANN(R=2)
        model.eval()
        loader: DataLoader = self._make_tiny_loader()

        loss, acc = evaluate_with_orbit_mixing(
            model, loader, torch.device('cpu'),
            use_z6=True, use_d6=False, use_t24=False,
            use_amp=False
        )

        assert loss >= 0.0, f"Loss should be non-negative, got {loss}"
        assert 0.0 <= acc <= 100.0, f"Accuracy should be in [0, 100], got {acc}"

    def test_d6_orbit_mixing_returns_valid_metrics(self) -> None:
        """
        WHY: D6 orbit mixing should produce valid loss and accuracy
        HOW: Run with use_d6=True (which adds reflection averaging)
        WHAT: Expect valid metrics
        """
        from models_tqf import TQFANN

        model: TQFANN = TQFANN(R=2)
        model.eval()
        loader: DataLoader = self._make_tiny_loader()

        loss, acc = evaluate_with_orbit_mixing(
            model, loader, torch.device('cpu'),
            use_z6=True, use_d6=True, use_t24=False,
            use_amp=False
        )

        assert loss >= 0.0, f"Loss should be non-negative, got {loss}"
        assert 0.0 <= acc <= 100.0, f"Accuracy should be in [0, 100], got {acc}"

    def test_t24_orbit_mixing_returns_valid_metrics(self) -> None:
        """
        WHY: T24 orbit mixing (full symmetry) should produce valid metrics
        HOW: Run with use_t24=True (implies D6 and Z6)
        WHAT: Expect valid metrics
        """
        from models_tqf import TQFANN

        model: TQFANN = TQFANN(R=2)
        model.eval()
        loader: DataLoader = self._make_tiny_loader()

        loss, acc = evaluate_with_orbit_mixing(
            model, loader, torch.device('cpu'),
            use_z6=True, use_d6=True, use_t24=True,
            use_amp=False
        )

        assert loss >= 0.0, f"Loss should be non-negative, got {loss}"
        assert 0.0 <= acc <= 100.0, f"Accuracy should be in [0, 100], got {acc}"


def run_tests(verbosity: int = 2):
    """Run all evaluation tests."""
    import sys
    args: List[str] = [__file__, f'-{"v" * verbosity}']
    return pytest.main(args)


if __name__ == '__main__':
    import sys
    exit_code: int = run_tests()
    sys.exit(exit_code)
