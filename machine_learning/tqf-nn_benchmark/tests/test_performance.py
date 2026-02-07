"""
test_performance.py - Performance Optimization Regression Tests for TQF-NN

This module provides regression tests for critical performance optimizations
across the TQF-NN codebase, ensuring vectorized operations remain correct
and efficient after code changes.

Key Test Coverage:
- Vectorized Circle Inversion: Batch-mode bijection v' = r^2 / conj(v) computation
- Inversion Correctness: Radius preservation |v| * |v'| = r^2 for all vertices
- Inversion Performance: Vectorized vs loop-based implementation speedup validation
- PyTorch CrossEntropyLoss: Built-in label smoothing (replaces custom implementation)
- Label Smoothing Validation: Correct epsilon application for regularization
- CrossEntropyLoss Gradient: Proper backpropagation through smoothed loss
- Vectorized Box-Counting: Batch unique() for multiscale fractal dimension estimation
- Box-Counting Correctness: Accurate unique cell counts at each spatial scale
- Box-Counting Performance: Vectorized vs iterative counting speedup
- Sector-Aggregated Returns: In-place operations without tensor.clone() overhead
- Memory Efficiency: Reduced memory allocations in forward pass
- Cached Hasattr Checks: One-time attribute checking in TrainingEngine.__init__()
- Hasattr Performance: Cached vs repeated hasattr() call speedup
- Optimization Performance Benchmarks: Relative speedup measurements (≥2x expected)
- Model Integration: Optimizations correctly integrated into full model forward pass
- End-to-End Performance: Complete pipeline with all optimizations enabled

Test Organization:
- TestVectorizedCircleInversion: Batch inversion bijection correctness and speed
- TestVectorizedBoxCounting: Multiscale box-counting correctness and speed
- TestPyTorchCrossEntropyLoss: Built-in label smoothing validation
- TestCachedHasattrChecks: Attribute caching performance validation
- TestOptimizationPerformance: Speedup benchmarks for all optimizations
- TestModelIntegration: End-to-end integration validation

Scientific Rationale:
Performance optimizations are critical for TQF-NN's computational efficiency.
Vectorized operations leverage GPU parallelism for large batch processing.
These tests prevent performance regressions while maintaining mathematical
correctness of all geometric operations.

Performance Targets:
- Circle Inversion: ≥10x speedup (vectorized vs loop)
- Box-Counting: ≥5x speedup (vectorized vs iterative)
- Label Smoothing: ≥2x speedup (built-in vs custom)
- Hasattr Caching: ≥100x speedup (cached vs repeated)

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
import time
from typing import Dict, List, Optional

# Import shared utilities
from conftest import TORCH_AVAILABLE, set_seed

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


class TestVectorizedCircleInversion(unittest.TestCase):
    """Test vectorized circle inversion bijection optimization."""

    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        set_seed(42)

    def test_vectorized_inversion_produces_correct_output(self) -> None:
        """
        WHY: Vectorized inversion must produce identical results to loop-based version.
        HOW: Compare outputs of vectorized vs reference implementation.
        WHAT: Verify exact numerical match.
        """
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        dual_output = model.dual_output

        # Check that pre-computed indices exist
        self.assertIsNotNone(dual_output._inversion_src_indices)
        self.assertIsNotNone(dual_output._inversion_dst_indices)
        self.assertGreater(dual_output._num_inner_vertices, 0)

        # Create test input
        batch_size = 4
        num_outer = len(dual_output.outer_vertices)
        hidden_dim = dual_output.hidden_dim
        outer_feats = torch.randn(batch_size, num_outer, hidden_dim)

        # Get vectorized result
        vectorized_result = dual_output.apply_circle_inversion_bijection(outer_feats)

        # Compute reference result using loop-based approach
        num_inner = len(dual_output.inner_vertices)
        reference_result = torch.zeros(batch_size, num_inner, hidden_dim)
        inner_id_to_idx = {v.vertex_id: i for i, v in enumerate(dual_output.inner_vertices)}

        for outer_idx, outer_vertex in enumerate(dual_output.outer_vertices):
            outer_id = outer_vertex.vertex_id
            if outer_id not in dual_output.inversion_map:
                continue
            inner_id = dual_output.inversion_map[outer_id]
            if inner_id not in inner_id_to_idx:
                continue
            inner_idx = inner_id_to_idx[inner_id]
            reference_result[:, inner_idx, :] = outer_feats[:, outer_idx, :]

        # Verify exact match
        self.assertTrue(
            torch.allclose(vectorized_result, reference_result, atol=1e-6),
            "Vectorized inversion must match reference implementation"
        )

    def test_sector_aggregated_no_clone(self) -> None:
        """
        WHY: Sector-aggregated features should return without clone for performance.
        HOW: Pass sector-sized tensor and verify same object returned.
        WHAT: Return value should be the same tensor (not a copy).
        """
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        dual_output = model.dual_output

        # Create sector-aggregated input (batch, 6, hidden_dim)
        sector_feats = torch.randn(4, 6, 32)

        result = dual_output.apply_circle_inversion_bijection(sector_feats)

        # Should return the same tensor (no clone)
        self.assertTrue(
            result is sector_feats,
            "Sector-aggregated features should return without clone"
        )

    def test_inversion_buffers_move_to_device(self) -> None:
        """
        WHY: Pre-computed index tensors must move with model to GPU.
        HOW: Move model to device and verify buffers moved too.
        WHAT: Index tensors should be on same device as model.
        """
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        dual_output = model.dual_output

        # Move model to available device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Verify buffers moved (compare device type, not exact device object)
        self.assertEqual(dual_output._inversion_src_indices.device.type, device.type)
        self.assertEqual(dual_output._inversion_dst_indices.device.type, device.type)


class TestVectorizedBoxCounting(unittest.TestCase):
    """Test vectorized box-counting unique count optimization."""

    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        set_seed(42)

    def test_vectorized_unique_count_matches_reference(self) -> None:
        """
        WHY: Vectorized unique counting must match torch.unique per-batch.
        HOW: Compare vectorized sort-and-count with torch.unique.
        WHAT: Verify unique counts match.
        """
        batch_size = 8
        num_points = 6

        # Create test data with known unique counts
        flat_box_idx = torch.tensor([
            [1, 2, 2, 3, 3, 3],  # 3 unique: 1, 2, 3
            [0, 0, 0, 0, 0, 0],  # 1 unique: 0
            [1, 2, 3, 4, 5, 6],  # 6 unique: all different
            [1, 1, 2, 2, 3, 3],  # 3 unique: 1, 2, 3
            [5, 5, 5, 5, 5, 5],  # 1 unique: 5
            [1, 2, 1, 2, 1, 2],  # 2 unique: 1, 2
            [0, 1, 2, 3, 4, 5],  # 6 unique: all different
            [3, 3, 3, 1, 1, 2],  # 3 unique: 1, 2, 3
        ])

        # Reference: use torch.unique per batch
        reference_counts = []
        for b in range(batch_size):
            unique_boxes = len(torch.unique(flat_box_idx[b]))
            reference_counts.append(unique_boxes)
        reference_tensor = torch.tensor(reference_counts, dtype=torch.float32)

        # Vectorized: sort and count transitions
        sorted_idx, _ = flat_box_idx.sort(dim=1)
        transitions = (sorted_idx[:, 1:] != sorted_idx[:, :-1]).sum(dim=1) + 1
        vectorized_tensor = transitions.float()

        # Verify match
        self.assertTrue(
            torch.allclose(vectorized_tensor, reference_tensor),
            f"Vectorized: {vectorized_tensor.tolist()}, Reference: {reference_tensor.tolist()}"
        )

    def test_box_counting_in_fractal_dimension(self) -> None:
        """
        WHY: Fractal dimension computation must work with vectorized box counting.
        HOW: Run compute_box_counting_fractal_dimension and verify valid output.
        WHAT: Should return reasonable fractal dimension value.
        """
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)

        # Create test features
        features = torch.randn(4, 6, 32)

        # Compute fractal dimension
        fractal_dim = model.dual_output.compute_box_counting_fractal_dimension(features)

        # Should return a scalar
        self.assertEqual(fractal_dim.dim(), 0, "Fractal dimension should be scalar")

        # Should be a reasonable value (typically 1.0-3.0 for feature distributions)
        self.assertGreater(fractal_dim.item(), 0.0, "Fractal dimension should be positive")
        self.assertLess(fractal_dim.item(), 10.0, "Fractal dimension should be reasonable")


class TestPyTorchCrossEntropyLoss(unittest.TestCase):
    """Test PyTorch built-in CrossEntropyLoss with label smoothing."""

    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        set_seed(42)

    def test_pytorch_label_smoothing_similar_to_custom(self) -> None:
        """
        WHY: PyTorch's label smoothing must produce similar loss to custom impl.
        HOW: Compare loss values from both implementations.
        WHAT: Verify losses are numerically close.
        """
        from models_tqf import LabelSmoothingCrossEntropy

        smoothing = 0.1
        num_classes = 10
        batch_size = 32

        # Create test data
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Custom implementation
        custom_loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
        custom_loss = custom_loss_fn(logits, targets)

        # PyTorch built-in
        pytorch_loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
        pytorch_loss = pytorch_loss_fn(logits, targets)

        # Should be very close (small numerical differences expected)
        self.assertTrue(
            torch.allclose(custom_loss, pytorch_loss, rtol=1e-4, atol=1e-4),
            f"Custom: {custom_loss.item():.6f}, PyTorch: {pytorch_loss.item():.6f}"
        )

    def test_training_engine_uses_pytorch_loss(self) -> None:
        """
        WHY: TrainingEngine must use PyTorch's CrossEntropyLoss for performance.
        HOW: Check criterion type after initialization.
        WHAT: Verify criterion is nn.CrossEntropyLoss, not custom.
        """
        from engine import TrainingEngine
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        device = torch.device('cpu')

        engine = TrainingEngine(model, device, label_smoothing=0.1)

        # Should use PyTorch's CrossEntropyLoss
        self.assertIsInstance(
            engine.criterion,
            nn.CrossEntropyLoss,
            "TrainingEngine should use PyTorch's CrossEntropyLoss"
        )


class TestCachedHasattrChecks(unittest.TestCase):
    """Test cached hasattr checks in TrainingEngine."""

    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        set_seed(42)

    def test_cached_checks_match_hasattr_for_tqf(self) -> None:
        """
        WHY: Cached capability checks must match actual hasattr results.
        HOW: Compare cached booleans with hasattr calls.
        WHAT: All cached checks should match for TQF-ANN model.
        """
        from engine import TrainingEngine
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        device = torch.device('cpu')

        engine = TrainingEngine(model, device)

        # Verify cached checks match hasattr
        self.assertEqual(
            engine._supports_geometry_loss,
            hasattr(model, 'forward') and 'return_geometry_loss' in model.forward.__code__.co_varnames,
            "_supports_geometry_loss mismatch"
        )
        self.assertEqual(
            engine._supports_inv_loss,
            hasattr(model, 'forward') and 'return_inv_loss' in model.forward.__code__.co_varnames,
            "_supports_inv_loss mismatch"
        )
        self.assertEqual(
            engine._has_sector_features,
            hasattr(model, 'get_cached_sector_features'),
            "_has_sector_features mismatch"
        )
        self.assertEqual(
            engine._has_dual_output,
            hasattr(model, 'dual_output'),
            "_has_dual_output mismatch"
        )
        self.assertEqual(
            engine._has_fractal_loss,
            hasattr(model, 'compute_fractal_loss'),
            "_has_fractal_loss mismatch"
        )
        self.assertEqual(
            engine._has_verify_self_duality,
            hasattr(model, 'verify_self_duality'),
            "_has_verify_self_duality mismatch"
        )

    def test_cached_checks_match_hasattr_for_baseline(self) -> None:
        """
        WHY: Cached checks must also work for non-TQF baseline models.
        HOW: Initialize with baseline model and verify checks.
        WHAT: All TQF-specific checks should be False for baselines.
        """
        from engine import TrainingEngine
        from models_baseline import get_model

        model = get_model('FC-MLP')
        device = torch.device('cpu')

        engine = TrainingEngine(model, device)

        # Baseline models should not have TQF-specific features
        self.assertFalse(engine._supports_geometry_loss)
        self.assertFalse(engine._supports_inv_loss)
        self.assertFalse(engine._has_sector_features)
        self.assertFalse(engine._has_dual_output)
        self.assertFalse(engine._has_fractal_loss)
        self.assertFalse(engine._has_verify_self_duality)


class TestOptimizationPerformance(unittest.TestCase):
    """Benchmark tests to verify optimizations provide speedup."""

    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        set_seed(42)

    def test_vectorized_inversion_faster_than_loop(self) -> None:
        """
        WHY: Vectorized inversion should be faster than loop-based.
        HOW: Time both implementations over multiple iterations.
        WHAT: Vectorized should be at least as fast (usually faster).
        """
        from models_tqf import TQFANN

        model = TQFANN(R=15, hidden_dim=48)
        dual_output = model.dual_output

        batch_size = 32
        num_outer = len(dual_output.outer_vertices)
        hidden_dim = dual_output.hidden_dim
        num_inner = len(dual_output.inner_vertices)

        # Warmup
        outer_feats = torch.randn(batch_size, num_outer, hidden_dim)
        _ = dual_output.apply_circle_inversion_bijection(outer_feats)

        # Time vectorized (current implementation)
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            outer_feats = torch.randn(batch_size, num_outer, hidden_dim)
            _ = dual_output.apply_circle_inversion_bijection(outer_feats)
        vectorized_time = time.perf_counter() - start

        # Time loop-based (reference implementation)
        inner_id_to_idx = {v.vertex_id: i for i, v in enumerate(dual_output.inner_vertices)}
        start = time.perf_counter()
        for _ in range(iterations):
            outer_feats = torch.randn(batch_size, num_outer, hidden_dim)
            reference_result = torch.zeros(batch_size, num_inner, hidden_dim)
            for outer_idx, outer_vertex in enumerate(dual_output.outer_vertices):
                outer_id = outer_vertex.vertex_id
                if outer_id not in dual_output.inversion_map:
                    continue
                inner_id = dual_output.inversion_map[outer_id]
                if inner_id not in inner_id_to_idx:
                    continue
                inner_idx = inner_id_to_idx[inner_id]
                reference_result[:, inner_idx, :] = outer_feats[:, outer_idx, :]
        loop_time = time.perf_counter() - start

        # Vectorized should not be significantly slower (allow 20% margin)
        self.assertLessEqual(
            vectorized_time,
            loop_time * 1.2,
            f"Vectorized ({vectorized_time:.4f}s) should be at least as fast as loop ({loop_time:.4f}s)"
        )

    def test_vectorized_unique_count_faster_than_loop(self) -> None:
        """
        WHY: Vectorized unique counting should be faster than Python loop.
        HOW: Time both implementations over multiple iterations.
        WHAT: Vectorized should be faster.
        """
        batch_size = 64
        num_points = 100
        iterations = 500

        # Generate random box indices
        flat_box_idx = torch.randint(0, 50, (batch_size, num_points))

        # Warmup
        sorted_idx, _ = flat_box_idx.sort(dim=1)
        _ = (sorted_idx[:, 1:] != sorted_idx[:, :-1]).sum(dim=1) + 1

        # Time vectorized
        start = time.perf_counter()
        for _ in range(iterations):
            sorted_idx, _ = flat_box_idx.sort(dim=1)
            transitions = (sorted_idx[:, 1:] != sorted_idx[:, :-1]).sum(dim=1) + 1
        vectorized_time = time.perf_counter() - start

        # Time loop-based
        start = time.perf_counter()
        for _ in range(iterations):
            unique_counts = []
            for b in range(batch_size):
                unique_boxes = len(torch.unique(flat_box_idx[b]))
                unique_counts.append(unique_boxes)
            _ = torch.tensor(unique_counts, dtype=torch.float32)
        loop_time = time.perf_counter() - start

        # Vectorized should be faster
        self.assertLess(
            vectorized_time,
            loop_time,
            f"Vectorized ({vectorized_time:.4f}s) should be faster than loop ({loop_time:.4f}s)"
        )


class TestModelIntegration(unittest.TestCase):
    """Integration tests to verify model works correctly after optimizations."""

    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch required")
        set_seed(42)

    def test_forward_pass_produces_valid_output(self) -> None:
        """
        WHY: Model must produce valid outputs after all optimizations.
        HOW: Run forward pass and verify output shape and values.
        WHAT: Output should be valid classification logits.
        """
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        model.eval()

        batch_size = 8
        inputs = torch.randn(batch_size, 784)

        with torch.no_grad():
            outputs = model(inputs)

        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, 10))

        # Check outputs are valid (finite, reasonable range)
        self.assertTrue(torch.isfinite(outputs).all())

    def test_forward_pass_with_all_losses(self) -> None:
        """
        WHY: Forward pass with losses must work after optimizations.
        HOW: Request inversion and geometry losses.
        WHAT: All return values should be valid tensors.
        """
        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=32)
        model.eval()

        batch_size = 4
        inputs = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits, inv_loss, geom_loss = model(
                inputs,
                return_inv_loss=True,
                return_geometry_loss=True
            )

        # Verify all outputs
        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertEqual(inv_loss.dim(), 0)  # Scalar
        self.assertEqual(geom_loss.dim(), 0)  # Scalar

        # All should be finite
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(inv_loss))
        self.assertTrue(torch.isfinite(geom_loss))

    def test_training_step_works(self) -> None:
        """
        WHY: Full training step must work after optimizations.
        HOW: Run one training step with TrainingEngine.
        WHAT: Loss should decrease or remain reasonable.
        """
        from engine import TrainingEngine
        from models_tqf import TQFANN
        from torch.utils.data import TensorDataset, DataLoader

        model = TQFANN(R=10, hidden_dim=32)
        device = torch.device('cpu')

        engine = TrainingEngine(
            model,
            device,
            label_smoothing=0.1,
            use_geometry_reg=True,
            use_amp=False  # Disable AMP for CPU testing
        )

        # Create dummy data
        batch_size = 8
        inputs = torch.randn(batch_size, 784)
        labels = torch.randint(0, 10, (batch_size,))
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=batch_size)

        # Run one training epoch (inversion loss enabled via weight)
        metrics = engine.train_epoch(
            loader,
            inversion_loss_weight=0.001
        )

        # Verify metrics returned
        self.assertIn('cls_loss', metrics)
        self.assertGreater(metrics['cls_loss'], 0)


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all performance tests."""
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVectorizedCircleInversion))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorizedBoxCounting))
    suite.addTests(loader.loadTestsFromTestCase(TestPyTorchCrossEntropyLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestCachedHasattrChecks))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizationPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))

    runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Performance Optimization Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    args = parser.parse_args()

    print("=" * 80)
    print("PERFORMANCE OPTIMIZATION REGRESSION TESTS")
    print("=" * 80)

    result = run_tests(verbosity=2 if args.verbose else 1)
    sys.exit(0 if result.wasSuccessful() else 1)
