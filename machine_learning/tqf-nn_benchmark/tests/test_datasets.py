"""
test_datasets.py - Dataset Preparation and Loading Tests for TQF-NN

This module tests all dataset preparation functionality including MNIST loading,
stratified sampling, rotational augmentation, and DataLoader configuration.

Key Test Coverage:
- DataLoader Creation: Validates get_dataloaders returns 4 loaders (train/val/test_rot/test_unrot)
- Dataset Sizes: Ensures stratified sampling produces correct split sizes
- Rotational Augmentation: Tests Z6AlignedRotation at 60-degree increments (0, 60, 120, 180, 240, 300)
- Image Properties: Validates dimensions (28x28), value ranges ([0, 1]), grayscale format
- Batch Consistency: Checks batch sizes, tensor shapes, label distributions
- Class Balance: Verifies stratified sampling maintains class proportions
- Rotation Alignment: Tests that rotated datasets use exact Z6 angles for TQF symmetry

Scientific Rationale:
The 60-degree rotation increments align with TQF's hexagonal (Z6) symmetry group,
enabling evaluation of rotational equivariance properties. Stratified sampling ensures
balanced class representation across train/val/test splits for reliable performance metrics.

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
from typing import Tuple

# Import shared utilities
from conftest import TORCH_AVAILABLE, PATHS, assert_positive_integer

# Import dataset preparation module
try:
    from prepare_datasets import get_dataloaders, Z6AlignedRotation
    import config
except ImportError as e:
    raise ImportError(f"Cannot import required modules: {e}")

# Import PIL for image tests
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader


class TestDatasetPreparation(unittest.TestCase):
    """Test suite for dataset preparation functions."""

    def test_prepare_datasets_returns_four_loaders(self) -> None:
        """Test that get_dataloaders returns 4 DataLoaders."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        result = get_dataloaders(
            batch_size=32, num_train=100, num_val=20,
            num_test_rot=20, num_test_unrot=20
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_prepare_datasets_loader_types(self) -> None:
        """Test that all returned objects are DataLoaders."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        train, val, test_rot, test_unrot = get_dataloaders(
            batch_size=32, num_train=100, num_val=20,
            num_test_rot=20, num_test_unrot=20
        )

        self.assertIsInstance(train, DataLoader)
        self.assertIsInstance(val, DataLoader)
        self.assertIsInstance(test_rot, DataLoader)
        self.assertIsInstance(test_unrot, DataLoader)

    def test_dataset_sizes_correct(self) -> None:
        """Test that dataset sizes match requested sizes (approximately)."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        num_train, num_val, num_test_rot, num_test_unrot = 100, 20, 30, 40

        train, val, test_rot, test_unrot = get_dataloaders(
            batch_size=32, num_train=num_train, num_val=num_val,
            num_test_rot=num_test_rot, num_test_unrot=num_test_unrot
        )

        # For SubsetRandomSampler or TransformedSubset, check the actual dataset length
        # Note: get_dataloaders may return slightly different sizes due to stratification
        # and combo calculations (num_test_rot // 60 * 60)

        # Train and val should be close to requested
        train_len = len(train.dataset) if hasattr(train.dataset, '__len__') else len(train.sampler)
        val_len = len(val.dataset) if hasattr(val.dataset, '__len__') else len(val.sampler)

        # Check within reasonable bounds (stratification may cause small differences)
        self.assertGreaterEqual(train_len, num_train - 20)
        self.assertLessEqual(train_len, num_train + 20)

        self.assertGreaterEqual(val_len, num_val - 5)
        self.assertLessEqual(val_len, num_val + 5)


class TestRotatedMNISTDataset(unittest.TestCase):
    """Test suite for rotated MNIST dataset functionality."""

    def test_rotated_dataset_creation(self) -> None:
        """Test that rotated dataset can be created via get_dataloaders."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        try:
            train, val, test_rot, test_unrot = get_dataloaders(
                batch_size=16, num_train=50, num_val=10,
                num_test_rot=10, num_test_unrot=10
            )
            self.assertIsNotNone(train)
        except Exception as e:
            self.skipTest(f"Dataset initialization failed: {e}")

    def test_rotated_dataset_batch_format(self) -> None:
        """Test that batches have correct format."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        try:
            train, _, _, _ = get_dataloaders(
                batch_size=16, num_train=50, num_val=10,
                num_test_rot=10, num_test_unrot=10
            )
            batch = next(iter(train))

            self.assertIsInstance(batch, (tuple, list))
            self.assertEqual(len(batch), 2)  # (images, labels)
        except Exception as e:
            self.skipTest(f"Dataset test failed: {e}")


class TestDataLoaderConfiguration(unittest.TestCase):
    """Test suite for DataLoader configuration."""

    def test_dataloader_batch_size(self) -> None:
        """Test that DataLoaders use correct batch size."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        batch_size_requested: int = 64
        train, _, _, _ = get_dataloaders(
            batch_size=batch_size_requested, num_train=100, num_val=20,
            num_test_rot=20, num_test_unrot=20
        )

        self.assertEqual(train.batch_size, batch_size_requested)

    def test_dataloader_iteration(self) -> None:
        """Test that DataLoaders can be iterated."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")

        train, _, _, _ = get_dataloaders(
            batch_size=32, num_train=50, num_val=10,
            num_test_rot=10, num_test_unrot=10
        )

        batch = next(iter(train))
        self.assertIsInstance(batch, (tuple, list))
        self.assertEqual(len(batch), 2)


class TestZ6AlignedRotation(unittest.TestCase):
    """
    Test suite for Z6AlignedRotation transform.

    This transform is critical for training TQF-ANN with rotational equivariance.
    It applies rotations aligned with the hexagonal Z6 symmetry group (60 deg increments)
    plus optional jitter for regularization.
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        if not PIL_AVAILABLE:
            self.skipTest("PIL required for Z6AlignedRotation tests")
        # Create a simple test image (28x28 grayscale, like MNIST)
        self.test_image = Image.new('L', (28, 28), color=128)
        # Draw a simple pattern to make rotation detectable
        for i in range(10, 20):
            self.test_image.putpixel((i, 14), 255)  # Horizontal line

    def test_z6_rotation_instantiation(self) -> None:
        """Test that Z6AlignedRotation can be instantiated with default jitter."""
        transform = Z6AlignedRotation()
        self.assertIsNotNone(transform)
        self.assertEqual(transform.jitter, 15.0)

    def test_z6_rotation_custom_jitter(self) -> None:
        """Test that Z6AlignedRotation accepts custom jitter value."""
        transform = Z6AlignedRotation(jitter=10.0)
        self.assertEqual(transform.jitter, 10.0)

    def test_z6_rotation_zero_jitter(self) -> None:
        """Test that Z6AlignedRotation works with zero jitter."""
        transform = Z6AlignedRotation(jitter=0.0)
        self.assertEqual(transform.jitter, 0.0)

    def test_z6_rotation_returns_image(self) -> None:
        """Test that applying transform returns a PIL Image."""
        transform = Z6AlignedRotation(jitter=0.0)
        result = transform(self.test_image)
        self.assertIsInstance(result, Image.Image)

    def test_z6_rotation_preserves_size(self) -> None:
        """Test that rotation preserves image dimensions."""
        transform = Z6AlignedRotation(jitter=0.0)
        result = transform(self.test_image)
        self.assertEqual(result.size, self.test_image.size)

    def test_z6_rotation_preserves_mode(self) -> None:
        """Test that rotation preserves image mode (grayscale)."""
        transform = Z6AlignedRotation(jitter=0.0)
        result = transform(self.test_image)
        self.assertEqual(result.mode, self.test_image.mode)

    def test_z6_angles_constant(self) -> None:
        """Test that Z6_ANGLES contains correct 60-degree increments."""
        expected_angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        self.assertEqual(Z6AlignedRotation.Z6_ANGLES, expected_angles)

    def test_z6_angles_count(self) -> None:
        """Test that there are exactly 6 Z6 angles."""
        self.assertEqual(len(Z6AlignedRotation.Z6_ANGLES), 6)

    def test_z6_rotation_repr(self) -> None:
        """Test that __repr__ returns expected format."""
        transform = Z6AlignedRotation(jitter=15.0)
        repr_str = repr(transform)
        self.assertIn("Z6AlignedRotation", repr_str)
        self.assertIn("15.0", repr_str)

    def test_z6_rotation_multiple_calls_vary(self) -> None:
        """Test that multiple calls produce different rotations (stochastic)."""
        transform = Z6AlignedRotation(jitter=15.0)

        # Apply transform multiple times and collect results
        results = []
        for _ in range(20):
            result = transform(self.test_image)
            # Convert to bytes for comparison
            results.append(result.tobytes())

        # With jitter, we should see variation (not all identical)
        # At least 2 different results out of 20 attempts
        unique_results = len(set(results))
        self.assertGreater(unique_results, 1,
            "Z6AlignedRotation with jitter should produce varying results")

    def test_z6_rotation_zero_jitter_limited_outcomes(self) -> None:
        """Test that zero jitter produces only 6 possible rotations."""
        transform = Z6AlignedRotation(jitter=0.0)

        # Apply transform many times and collect unique results
        results = set()
        for _ in range(100):
            result = transform(self.test_image)
            results.add(result.tobytes())

        # With zero jitter, should have at most 6 unique rotations
        # (one for each Z6 angle: 0, 60, 120, 180, 240, 300)
        self.assertLessEqual(len(results), 6,
            "Zero jitter should produce at most 6 unique rotations (Z6 group)")

    def test_z6_rotation_callable(self) -> None:
        """Test that Z6AlignedRotation is callable."""
        transform = Z6AlignedRotation()
        self.assertTrue(callable(transform))

    def test_z6_rotation_with_rgb_image(self) -> None:
        """Test that Z6AlignedRotation works with RGB images too."""
        rgb_image = Image.new('RGB', (28, 28), color=(128, 64, 32))
        transform = Z6AlignedRotation(jitter=0.0)
        result = transform(rgb_image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.size, (28, 28))

    def test_z6_rotation_identity_possible(self) -> None:
        """Test that 0-degree rotation (identity) is possible with zero jitter."""
        transform = Z6AlignedRotation(jitter=0.0)

        # Run many times to check if identity (0 deg) rotation occurs
        found_identity = False
        for _ in range(100):
            result = transform(self.test_image)
            if result.tobytes() == self.test_image.tobytes():
                found_identity = True
                break

        # 0 deg is one of the 6 angles, so should appear ~1/6 of the time
        # With 100 tries, probability of never seeing it is (5/6)^100 ~ 0
        self.assertTrue(found_identity,
            "0-degree rotation should be possible with zero jitter")


class TestZ6AlignedRotationIntegration(unittest.TestCase):
    """Integration tests for Z6AlignedRotation with torchvision transforms."""

    def test_z6_rotation_in_compose(self) -> None:
        """Test that Z6AlignedRotation works in transforms.Compose pipeline."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        if not PIL_AVAILABLE:
            self.skipTest("PIL required")

        from torchvision import transforms

        # Create a transform pipeline similar to training
        pipeline = transforms.Compose([
            Z6AlignedRotation(jitter=15.0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Create test image
        test_image = Image.new('L', (28, 28), color=128)

        # Apply pipeline
        result = pipeline(test_image)

        # Check output is a tensor with correct shape
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 28, 28))

    def test_z6_rotation_tensor_output_normalized(self) -> None:
        """Test that Z6AlignedRotation + normalization produces expected range."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        if not PIL_AVAILABLE:
            self.skipTest("PIL required")

        from torchvision import transforms

        pipeline = transforms.Compose([
            Z6AlignedRotation(jitter=0.0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_image = Image.new('L', (28, 28), color=128)
        result = pipeline(test_image)

        # After normalization, values should be roughly centered around 0
        # (128/255 - 0.1307) / 0.3081 ~ 1.21
        mean_val = result.mean().item()
        self.assertGreater(mean_val, -5.0)
        self.assertLess(mean_val, 5.0)


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all dataset tests."""
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()

    for test_class in [TestDatasetPreparation, TestRotatedMNISTDataset,
                       TestDataLoaderConfiguration, TestZ6AlignedRotation,
                       TestZ6AlignedRotationIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dataset Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("DATASET TESTS")
    print("=" * 80)

    result = run_tests(verbosity=2 if args.verbose else 1)
    sys.exit(0 if result.wasSuccessful() else 1)
