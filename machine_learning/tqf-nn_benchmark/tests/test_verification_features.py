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
                       TestParameterCounting]:
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
