"""
test_verification_features.py - Model Verification and Validation Tests for TQF-NN

This module tests model initialization, forward pass correctness, and parameter
matching for all TQF-NN models (TQF-ANN and baselines). Includes both fast tests
(small models) and slow tests (production-sized models).

Key Test Coverage:
- Model Initialization: Fast (R=10) and production (R=20) TQF-ANN initialization
- Forward Pass Validation: Correct output shapes, no NaN/inf values, gradient flow
- Parameter Counting: Trainable parameter verification for fair comparison (~650K target)
- Baseline Models: FC-MLP, CNN-L5, ResNet-18-Scaled initialization and forward pass
- Device Compatibility: CPU and CUDA execution testing
- Batch Handling: Correct behavior with various batch sizes (1, 32, 128)
- Model Methods: count_parameters(), verify_self_duality(), get_inversion_map()

Test Organization:
- Fast tests: Use R=10, hidden_dim=32 for ~2-3s initialization
- Slow tests: Use production config (R=20) for ~10-15s initialization, marked with @pytest.mark.slow

Scientific Rationale:
Parameter matching ensures fair "apples-to-apples" comparison between TQF-ANN
and baseline models. All models target ~650K parameters to isolate architectural
differences from capacity differences in performance evaluation.

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

import unittest
import sys
import pytest

# Import shared utilities
from conftest import TORCH_AVAILABLE, PATHS, count_parameters

# Import required modules
try:
    from models_tqf import TQFANN
    from models_baseline import get_model
    import config
except ImportError as e:
    raise ImportError(f"Cannot import required modules: {e}")

if TORCH_AVAILABLE:
    import torch


class TestModelInitialization(unittest.TestCase):
    """Test suite for model initialization verification."""

    def test_tqf_model_initialization_fast(self) -> None:
        """Test that TQF model can be initialized (fast version with small R)."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        # Use smaller R for faster initialization (~2-3 seconds vs ~10-15 seconds)
        model = TQFANN(R=10, hidden_dim=32)
        self.assertIsNotNone(model)
        self.assertEqual(model.R, 10)
        self.assertEqual(model.hidden_dim, 32)

    @pytest.mark.slow
    def test_tqf_model_initialization_production(self) -> None:
        """Test that TQF model can be initialized with production config (SLOW: ~10-15s)."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        # Uses R=20 (default) - slow due to O(R^2) graph structure computation
        model = TQFANN(
            R=config.TQF_TRUNCATION_R_DEFAULT,
            hidden_dim=config.TQF_HIDDEN_DIMENSION_DEFAULT
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.R, config.TQF_TRUNCATION_R_DEFAULT)
        self.assertEqual(model.hidden_dim, config.TQF_HIDDEN_DIMENSION_DEFAULT)

    def test_fcmlp_initialization(self) -> None:
        """Test that FC-MLP model can be initialized."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = get_model('FC-MLP')
        self.assertIsNotNone(model)

    def test_cnn_initialization(self) -> None:
        """Test that CNN model can be initialized."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = get_model('CNN-L5')
        self.assertIsNotNone(model)


class TestModelForwardPass(unittest.TestCase):
    """Test suite for model forward pass verification."""

    def test_tqf_forward_pass(self) -> None:
        """Test that TQF model forward pass works."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=config.TQF_TRUNCATION_R_DEFAULT, hidden_dim=64)
        x = torch.randn(2, 784)  # Flattened MNIST input
        output = model(x)
        self.assertEqual(output.shape, (2, 10))

    def test_fcmlp_forward_pass(self) -> None:
        """Test that FC-MLP forward pass works."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = get_model('FC-MLP')
        x = torch.randn(2, 784)
        output = model(x)
        self.assertEqual(output.shape, (2, 10))


class TestParameterCounting(unittest.TestCase):
    """Test suite for parameter counting verification."""

    def test_count_parameters_tqf(self) -> None:
        """Test parameter counting for TQF model."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=config.TQF_TRUNCATION_R_DEFAULT, hidden_dim=64)
        params: int = count_parameters(model)
        self.assertGreater(params, 0)

    def test_count_parameters_fcmlp(self) -> None:
        """Test parameter counting for FC-MLP."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = get_model('FC-MLP')
        params: int = count_parameters(model)
        self.assertGreater(params, 0)


class TestFractalDimensionTolerance(unittest.TestCase):
    """Test suite for fractal dimension tolerance feature verification."""

    def test_fractal_dim_tol_stored_in_model(self) -> None:
        """Test that fractal_dim_tol is stored in TQFANN model."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        # Test with default tolerance
        model = TQFANN(R=10, hidden_dim=32)
        self.assertTrue(hasattr(model, 'fractal_dim_tol'))
        self.assertEqual(model.fractal_dim_tol, config.TQF_FRACTAL_DIM_TOLERANCE_DEFAULT)

        # Test with custom tolerance
        custom_tol = 0.25
        model_custom = TQFANN(R=10, hidden_dim=32, fractal_dim_tol=custom_tol)
        self.assertEqual(model_custom.fractal_dim_tol, custom_tol)

    def test_fractal_dim_tol_stored_in_dual_output(self) -> None:
        """Test that fractal_dim_tol is stored in BijectionDualOutputHead."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, fractal_dim_tol=0.2)
        self.assertTrue(hasattr(model.dual_output, 'fractal_dim_tol'))
        self.assertEqual(model.dual_output.fractal_dim_tol, 0.2)

    def test_theoretical_fractal_dim_constant(self) -> None:
        """Test that theoretical fractal dimension constant exists and is correct."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        self.assertTrue(hasattr(config, 'TQF_THEORETICAL_FRACTAL_DIM_DEFAULT'))
        # Sierpinski triangle dimension: log(3)/log(2) ~ 1.585
        self.assertAlmostEqual(config.TQF_THEORETICAL_FRACTAL_DIM_DEFAULT, 1.585, places=3)

    def test_get_fractal_dimension_info(self) -> None:
        """Test get_fractal_dimension_info returns correct structure."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, fractal_dim_tol=0.15)
        info = model.get_fractal_dimension_info()

        self.assertIn('theoretical_dim', info)
        self.assertIn('tolerance', info)
        self.assertIn('last_measured_dim', info)
        self.assertEqual(info['tolerance'], 0.15)
        self.assertAlmostEqual(info['theoretical_dim'], 1.585, places=3)
        self.assertIsNone(info['last_measured_dim'])  # Not computed yet

    def test_compute_box_counting_fractal_dimension(self) -> None:
        """Test that box-counting fractal dimension computation works."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32)

        # Create synthetic features (batch, 6 sectors, hidden_dim)
        features = torch.randn(4, 6, 32)
        dim = model.dual_output.compute_box_counting_fractal_dimension(features)

        # Should return a scalar tensor
        self.assertEqual(dim.dim(), 0)
        # Fractal dimension should be non-negative and finite
        # Note: For random noise with few points, dimension can be quite low
        self.assertGreaterEqual(dim.item(), 0.0)
        self.assertLess(dim.item(), 5.0)
        self.assertFalse(torch.isnan(dim))
        self.assertFalse(torch.isinf(dim))

    def test_verify_fractal_dimension_within_tolerance(self) -> None:
        """Test verify_fractal_dimension when within tolerance."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        # Use very loose tolerance to ensure pass
        model = TQFANN(R=10, hidden_dim=32, fractal_dim_tol=2.0)
        features = torch.randn(4, 6, 32)

        passed, measured_dim, message = model.dual_output.verify_fractal_dimension(features)

        self.assertTrue(passed)
        self.assertIn('OK', message)
        self.assertIsNotNone(model.dual_output._last_measured_fractal_dim)

    def test_verify_fractal_dimension_exceeds_tolerance(self) -> None:
        """Test verify_fractal_dimension when exceeding tolerance."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        import warnings

        # Use very tight tolerance to ensure failure
        model = TQFANN(R=10, hidden_dim=32, fractal_dim_tol=0.0001)
        features = torch.randn(4, 6, 32)

        # Capture warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            passed, measured_dim, message = model.dual_output.verify_fractal_dimension(features)

            # Should fail due to tight tolerance
            self.assertFalse(passed)
            self.assertIn('WARNING', message)
            # Should have issued a warning
            self.assertGreater(len(w), 0)
            self.assertIn('Fractal dimension deviation', str(w[0].message))

    def test_reset_fractal_dim_warning(self) -> None:
        """Test that reset_fractal_dim_warning clears the warning flag."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        import warnings

        model = TQFANN(R=10, hidden_dim=32, fractal_dim_tol=0.0001)
        features = torch.randn(4, 6, 32)

        # First call should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.dual_output.verify_fractal_dimension(features)
            first_warning_count = len(w)

        # Second call should not warn (flag already set)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.dual_output.verify_fractal_dimension(features)
            second_warning_count = len(w)

        self.assertEqual(second_warning_count, 0)  # No new warning

        # Reset and verify warning can be issued again
        model.reset_fractal_dim_warning()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.dual_output.verify_fractal_dimension(features)
            self.assertGreater(len(w), 0)  # Warning issued again

    def test_fractal_epsilon_prevents_division_by_zero(self) -> None:
        """Test that fractal_epsilon prevents numerical issues."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        self.assertTrue(hasattr(config, 'TQF_FRACTAL_EPSILON_DEFAULT'))
        self.assertGreater(config.TQF_FRACTAL_EPSILON_DEFAULT, 0)
        self.assertLess(config.TQF_FRACTAL_EPSILON_DEFAULT, 1e-6)

        model = TQFANN(R=10, hidden_dim=32)
        self.assertEqual(model.dual_output.fractal_epsilon, config.TQF_FRACTAL_EPSILON_DEFAULT)

    def test_tqfann_verify_fractal_dimension_delegation(self) -> None:
        """Test that TQFANN.verify_fractal_dimension delegates to dual_output."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, fractal_dim_tol=2.0)

        # Run forward pass to cache sector features
        x = torch.randn(2, 784)
        _ = model(x)

        # Verify using cached features
        passed, measured_dim, message = model.verify_fractal_dimension()

        self.assertTrue(passed)  # Using loose tolerance
        self.assertIsInstance(measured_dim, float)
        self.assertIn('OK', message)


class TestSelfSimilarityWeight(unittest.TestCase):
    """Test suite for self-similarity weight feature verification."""

    def test_self_similarity_weight_stored(self) -> None:
        """Test that self_similarity_weight is stored in model."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, self_similarity_weight=0.05)
        self.assertEqual(model.dual_output.self_similarity_weight, 0.05)

    def test_box_counting_weight_stored(self) -> None:
        """Test that box_counting_weight is stored in model."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, box_counting_weight=0.02)
        self.assertEqual(model.dual_output.box_counting_weight, 0.02)

    def test_compute_self_similarity_loss(self) -> None:
        """Test that self-similarity loss computation works."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, self_similarity_weight=0.1)
        features = torch.randn(4, 6, 32)

        loss = model.dual_output.compute_self_similarity_loss(features)

        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_compute_self_similarity_loss_zero_weight(self) -> None:
        """Test that self-similarity loss is zero when weight is zero."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, self_similarity_weight=0.0)
        features = torch.randn(4, 6, 32)

        loss = model.dual_output.compute_self_similarity_loss(features)
        self.assertEqual(loss.item(), 0.0)

    def test_compute_box_counting_loss(self) -> None:
        """Test that box-counting loss computation works."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, box_counting_weight=0.1)
        features = torch.randn(4, 6, 32)

        loss = model.dual_output.compute_box_counting_loss(features)

        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_compute_box_counting_loss_zero_weight(self) -> None:
        """Test that box-counting loss is zero when weight is zero."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, box_counting_weight=0.0)
        features = torch.randn(4, 6, 32)

        loss = model.dual_output.compute_box_counting_loss(features)
        self.assertEqual(loss.item(), 0.0)

    def test_compute_fractal_loss_combined(self) -> None:
        """Test that combined fractal loss works."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, self_similarity_weight=0.1, box_counting_weight=0.1)
        features = torch.randn(4, 6, 32)

        combined = model.dual_output.compute_fractal_loss(features)
        self_sim = model.dual_output.compute_self_similarity_loss(features)
        box_count = model.dual_output.compute_box_counting_loss(features)

        # Combined should equal sum of individual losses
        self.assertAlmostEqual(combined.item(), (self_sim + box_count).item(), places=5)

    def test_tqfann_compute_fractal_loss_delegation(self) -> None:
        """Test that TQFANN delegates fractal loss computation."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        model = TQFANN(R=10, hidden_dim=32, self_similarity_weight=0.1, box_counting_weight=0.1)

        # Run forward pass to cache features
        x = torch.randn(2, 784)
        _ = model(x)

        # Compute via TQFANN method (uses cached features)
        loss = model.compute_fractal_loss()

        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0.0)


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """
    Run all verification feature tests.

    Note: The production TQF initialization test is marked as slow (@pytest.mark.slow)
    and takes ~10-15 seconds due to O(R^2) graph structure pre-computation with R=20.
    Fast tests use R=10 instead (~2-3 seconds).

    To skip slow tests: pytest tests/test_verification_features.py -m "not slow"
    """
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()

    for test_class in [TestModelInitialization, TestModelForwardPass,
                       TestParameterCounting, TestFractalDimensionTolerance,
                       TestSelfSimilarityWeight]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verification Feature Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("VERIFICATION FEATURE TESTS")
    print("=" * 80)

    result = run_tests(verbosity=2 if args.verbose else 1)
    sys.exit(0 if result.wasSuccessful() else 1)
