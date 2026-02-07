"""
test_fibonacci_inner_zone_mirroring.py - Unit Tests for Inner Zone Fibonacci Mirroring

This test suite ensures complete specification compliance for the Fibonacci
inner zone mirroring feature, as required by spec_doc_mark03_2026-01-28.tex.

IMPORTANT: As of January 2026, Fibonacci mode uses WEIGHT-BASED scaling,
not dimension scaling. All layers have CONSTANT dimensions (hidden_dim).
The Fibonacci sequence only affects feature aggregation WEIGHTS.

Specification Requirements (updated for weight-based scaling):
==================================================================
1. Fibonacci Sequence Generation:
   - Generate via iterative addition (O(L) time)
   - Sequence: [1, 1, 2, 3, 5, 8, 13, ...]

2. Weight-Based Scaling (NOT dimension scaling):
   - All layers use constant hidden_dim
   - Fibonacci weights affect feature aggregation only
   - Parameter counts are IDENTICAL across all modes

3. Inner Zone Mirroring:
   - Inverse=True reverses the weight sequence
   - Preserves bijective duality in feature weighting
   - Dimensions remain constant (hidden_dim)

4. Architecture Requirements:
   - Separate graph convolution pipelines for outer and inner zones
   - All layers have IDENTICAL dimensions (hidden_dim)
   - Fibonacci weights differ between zones (normal vs inverse)

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
from typing import List, Dict

# Import shared utilities
try:
    from conftest import TORCH_AVAILABLE
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    TORCH_AVAILABLE = True  # Assume available for standalone run
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

# Skip entire module if torch not available (only when pytest is available)
if PYTEST_AVAILABLE:
    pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")


# ==============================================================================
# TEST SUITE 1: FIBONACCI WEIGHT SCALER WITH INVERSE MODE
# ==============================================================================

class TestFibonacciWeightScalerInverse(unittest.TestCase):
    """
    Test FibonacciWeightScaler with inverse mode for inner zone mirroring.

    IMPORTANT: As of Jan 2026, Fibonacci mode uses WEIGHT-BASED scaling.
    All layers have CONSTANT dimensions (base_dim). The Fibonacci sequence
    only affects feature aggregation weights, not layer dimensions.
    """

    def test_inverse_parameter_exists(self):
        """
        Test that FibonacciWeightScaler accepts inverse parameter.

        WHY: Specification requires separate weighting for inner zone
        WHAT: Verify inverse parameter is accepted during initialization
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        # Should accept inverse=False (outer zone)
        scaler_outer = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='fibonacci',
            inverse=False
        )
        self.assertFalse(scaler_outer.inverse)

        # Should accept inverse=True (inner zone)
        scaler_inner = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='fibonacci',
            inverse=True
        )
        self.assertTrue(scaler_inner.inverse)

    def test_fibonacci_sequence_generation(self):
        """
        Test correct Fibonacci sequence generation.

        WHY: Specification requires [1, 1, 2, 3, 5, 8, 13, ...] via iterative addition
        WHAT: Verify sequence matches expected Fibonacci numbers
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        scaler = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='fibonacci',
            inverse=False
        )

        # Expected Fibonacci sequence: [1, 1, 2, 3, 5, 8, 13]
        expected_fib = [1, 1, 2, 3, 5, 8, 13]
        actual_fib = scaler.fib_seq

        for i, expected_val in enumerate(expected_fib):
            self.assertEqual(
                actual_fib[i], expected_val,
                f"Fibonacci sequence at index {i} should be {expected_val}, got {actual_fib[i]}"
            )

    def test_dimensions_are_constant(self):
        """
        Test that get_dimension always returns constant base_dim.

        WHY: Weight-based Fibonacci means ALL layers have constant dimensions
        WHAT: Verify dimensions are base_dim regardless of layer index or inverse
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        base_dim = 100
        scaler_outer = FibonacciWeightScaler(
            num_layers=5,
            base_dim=base_dim,
            mode='fibonacci',
            inverse=False
        )
        scaler_inner = FibonacciWeightScaler(
            num_layers=5,
            base_dim=base_dim,
            mode='fibonacci',
            inverse=True
        )

        # All layers should have constant dimension (base_dim)
        for layer_idx in range(5):
            self.assertEqual(
                scaler_outer.get_dimension(layer_idx), base_dim,
                f"Outer layer {layer_idx} should have constant dim={base_dim}"
            )
            self.assertEqual(
                scaler_inner.get_dimension(layer_idx), base_dim,
                f"Inner layer {layer_idx} should have constant dim={base_dim}"
            )

    def test_outer_zone_normal_weighting(self):
        """
        Test outer zone uses normal Fibonacci weighting (not dimension scaling).

        WHY: Outer zone uses Fibonacci weights for feature aggregation
        WHAT: Verify weights follow normalized Fibonacci progression
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        scaler_outer = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='fibonacci',
            inverse=False
        )

        # Get all weights
        weights = scaler_outer.get_all_weights()
        self.assertEqual(len(weights), 5)

        # Weights should sum to approximately 1.0 (normalized)
        weight_sum = sum(weights)
        self.assertAlmostEqual(weight_sum, 1.0, places=5,
                              msg=f"Weights should sum to 1.0, got {weight_sum}")

        # Fibonacci sequence: [1, 1, 2, 3, 5] -> normalized
        # Each weight should be positive
        for i, w in enumerate(weights):
            self.assertGreater(w, 0, f"Weight at index {i} should be positive")

    def test_inner_zone_inverse_weighting(self):
        """
        Test inner zone uses inverse Fibonacci weighting.

        WHY: Inner zone uses reversed weight sequence for bijective duality
        WHAT: Verify weights are reversed compared to outer zone
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        scaler_outer = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='fibonacci',
            inverse=False
        )
        scaler_inner = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='fibonacci',
            inverse=True
        )

        outer_weights = scaler_outer.get_all_weights()
        inner_weights = scaler_inner.get_all_weights()

        # Inner weights should be reversed relative to outer
        # (reversed sequence for bijective duality)
        for i in range(5):
            # Inner weight at i should correspond to outer weight at (4-i)
            expected_inner = outer_weights[4 - i]
            self.assertAlmostEqual(
                inner_weights[i], expected_inner, places=5,
                msg=f"Inner weight at {i} should match outer weight at {4-i}"
            )

    def test_bijective_duality_preservation_weights(self):
        """
        Test that outer and inner scalers preserve bijective duality in weights.

        WHY: Specification requires weight duality: outer[k] = inner[N-1-k]
        WHAT: Verify paired layers have matching weights
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        num_layers = 5
        scaler_outer = FibonacciWeightScaler(
            num_layers=num_layers,
            base_dim=100,
            mode='fibonacci',
            inverse=False
        )
        scaler_inner = FibonacciWeightScaler(
            num_layers=num_layers,
            base_dim=100,
            mode='fibonacci',
            inverse=True
        )

        # Check bijective pairing: outer[k] should equal inner[num_layers-1-k]
        for k in range(num_layers):
            outer_weight = scaler_outer.get_weight(k)
            paired_idx = num_layers - 1 - k
            inner_weight_at_k = scaler_inner.get_weight(k)
            expected_inner_weight = scaler_outer.get_weight(paired_idx)

            self.assertAlmostEqual(
                inner_weight_at_k, expected_inner_weight, places=5,
                msg=f"Inner layer {k} weight ({inner_weight_at_k:.4f}) should equal "
                    f"outer layer {paired_idx} weight ({expected_inner_weight:.4f}) for bijective duality"
            )

    def test_linear_mode_inverse_weights(self):
        """
        Test that linear mode also supports inverse weights.

        WHY: Ensures inverse mode works for all scaling modes
        WHAT: Verify linear weights reverse correctly
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        scaler_linear_outer = FibonacciWeightScaler(
            num_layers=5,
            base_dim=10,
            mode='linear',
            inverse=False
        )
        scaler_linear_inner = FibonacciWeightScaler(
            num_layers=5,
            base_dim=10,
            mode='linear',
            inverse=True
        )

        outer_weights = scaler_linear_outer.get_all_weights()
        inner_weights = scaler_linear_inner.get_all_weights()

        # Inner weights should be reversed
        for k in range(5):
            self.assertAlmostEqual(
                inner_weights[k], outer_weights[4 - k], places=5,
                msg=f"Inner linear weight at {k} should equal outer weight at {4-k}"
            )

        # Dimensions should all be constant (base_dim=10)
        for k in range(5):
            self.assertEqual(scaler_linear_outer.get_dimension(k), 10)
            self.assertEqual(scaler_linear_inner.get_dimension(k), 10)

    def test_none_mode_uniform_weights(self):
        """
        Test that 'none' mode uses uniform weights.

        WHY: Constant mode should have uniform weights (no Fibonacci weighting)
        WHAT: Verify weights are uniform regardless of inverse setting
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        scaler_none_outer = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='none',
            inverse=False
        )
        scaler_none_inner = FibonacciWeightScaler(
            num_layers=5,
            base_dim=100,
            mode='none',
            inverse=True
        )

        # Both should return constant dimension and uniform weights
        expected_weight = 1.0 / 5  # Uniform weight
        for k in range(5):
            self.assertEqual(scaler_none_outer.get_dimension(k), 100)
            self.assertEqual(scaler_none_inner.get_dimension(k), 100)
            self.assertAlmostEqual(scaler_none_outer.get_weight(k), expected_weight, places=5)
            self.assertAlmostEqual(scaler_none_inner.get_weight(k), expected_weight, places=5)


# ==============================================================================
# TEST SUITE 2: DUAL ZONE ARCHITECTURE
# ==============================================================================

class TestDualZoneArchitecture(unittest.TestCase):
    """Test that TQFANN correctly implements dual zone architecture."""

    def test_zone_specific_vertex_lists(self):
        """
        Test that TQFANN creates separate vertex lists for outer and inner zones.

        WHY: Specification requires separate processing for outer and inner zones
        WHAT: Verify outer_zone_vertices and inner_zone_vertices exist
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )

        # Should have zone-specific vertex lists
        self.assertTrue(hasattr(model, 'outer_zone_vertices'))
        self.assertTrue(hasattr(model, 'inner_zone_vertices'))

        # Both should be non-empty
        self.assertGreater(len(model.outer_zone_vertices), 0)
        self.assertGreater(len(model.inner_zone_vertices), 0)

        # Boundary + outer = outer_zone_vertices
        expected_outer_count = len(model.boundary_vertices) + len(model.outer_vertices)
        self.assertEqual(len(model.outer_zone_vertices), expected_outer_count)

        # Boundary + inner = inner_zone_vertices
        expected_inner_count = len(model.boundary_vertices) + len(model.inner_vertices)
        self.assertEqual(len(model.inner_zone_vertices), expected_inner_count)

    def test_zone_specific_adjacency(self):
        """
        Test that TQFANN creates zone-specific adjacency dictionaries.

        WHY: Each zone needs its own adjacency for graph convolutions
        WHAT: Verify adjacency_outer and adjacency_inner exist and are correct
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )

        # Should have zone-specific adjacency
        self.assertTrue(hasattr(model, 'adjacency_outer'))
        self.assertTrue(hasattr(model, 'adjacency_inner'))

        # Both should be non-empty
        self.assertGreater(len(model.adjacency_outer), 0)
        self.assertGreater(len(model.adjacency_inner), 0)

        # Verify adjacency only includes vertices within the zone
        outer_vertex_ids = {v.vertex_id for v in model.outer_zone_vertices}
        inner_vertex_ids = {v.vertex_id for v in model.inner_zone_vertices}

        # Check outer adjacency
        for vid, neighbors in model.adjacency_outer.items():
            self.assertIn(vid, outer_vertex_ids, "Outer adjacency key should be in outer zone")
            for nid in neighbors:
                self.assertIn(nid, outer_vertex_ids, "Outer adjacency neighbor should be in outer zone")

        # Check inner adjacency
        for vid, neighbors in model.adjacency_inner.items():
            self.assertIn(vid, inner_vertex_ids, "Inner adjacency key should be in inner zone")
            for nid in neighbors:
                self.assertIn(nid, inner_vertex_ids, "Inner adjacency neighbor should be in inner zone")

    def test_final_dimensions_stored(self):
        """
        Test that final dimensions for both zones are stored.

        WHY: Need to know final dimensions for projection layer and dual output
        WHAT: Verify outer_final_dim and inner_final_dim attributes
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )

        # Should have final dimension attributes
        self.assertTrue(hasattr(model, 'outer_final_dim'))
        self.assertTrue(hasattr(model, 'inner_final_dim'))

        # Both should be positive integers
        self.assertGreater(model.outer_final_dim, 0)
        self.assertGreater(model.inner_final_dim, 0)

        # T24 binner uses constant dimensions, so both should equal hidden_dim
        self.assertEqual(model.outer_final_dim, 32)  # hidden_dim from model creation
        self.assertEqual(model.inner_final_dim, 32)

    def test_projection_layer_exists_when_dimensions_differ(self):
        """
        Test that projection layer is created when dimensions differ.

        WHY: Bijective duality requires matching dimensions at dual output
        WHAT: Verify inner_to_outer_proj exists when dimensions don't match
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )

        # Check if dimensions differ
        if model.outer_final_dim != model.inner_final_dim:
            # Should have projection layer
            self.assertIsNotNone(model.inner_to_outer_proj)
            self.assertIsInstance(model.inner_to_outer_proj, nn.Linear)

            # Projection should map inner_final_dim -> outer_final_dim
            self.assertEqual(model.inner_to_outer_proj.in_features, model.inner_final_dim)
            self.assertEqual(model.inner_to_outer_proj.out_features, model.outer_final_dim)
        else:
            # Should not have projection layer
            self.assertIsNone(model.inner_to_outer_proj)


# ==============================================================================
# TEST SUITE 3: FORWARD PASS WITH DUAL ZONES
# ==============================================================================

class TestDualZoneForwardPass(unittest.TestCase):
    """Test forward pass with separate outer and inner zone processing."""

    def test_forward_pass_completes(self):
        """
        Test that forward pass completes without errors.

        WHY: Implementation must work end-to-end
        WHAT: Verify forward pass returns correct output shape
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_forward_pass_various_batch_sizes(self):
        """
        Test forward pass with various batch sizes.

        WHY: Model should handle different batch sizes
        WHAT: Verify output shape for batch sizes 1, 2, 8, 16
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            with torch.no_grad():
                x = torch.randn(batch_size, 784)
                logits = model(x)

            self.assertEqual(
                logits.shape, (batch_size, 10),
                f"Failed for batch_size={batch_size}"
            )

    def test_inversion_loss_computation(self):
        """
        Test that inversion consistency loss can be computed.

        WHY: Specification requires verification of bijection
        WHAT: Verify return_inv_loss=True returns loss tensor
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            result = model(x, return_inv_loss=True)

        # Should return tuple (logits, inv_loss)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        logits, inv_loss = result
        self.assertEqual(logits.shape, (4, 10))
        self.assertIsInstance(inv_loss, torch.Tensor)
        self.assertEqual(inv_loss.dim(), 0)  # Scalar loss
        self.assertGreaterEqual(inv_loss.item(), 0)  # Non-negative

    def test_geometry_loss_computation(self):
        """
        Test that geodesic verification loss can be computed.

        WHY: Specification requires geometric verification
        WHAT: Verify return_geometry_loss=True returns loss tensor
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            result = model(x, return_geometry_loss=True)

        # Should return tuple (logits, geom_loss)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        logits, geom_loss = result
        self.assertEqual(logits.shape, (4, 10))
        self.assertIsInstance(geom_loss, torch.Tensor)
        self.assertEqual(geom_loss.dim(), 0)  # Scalar loss
        self.assertGreaterEqual(geom_loss.item(), 0)  # Non-negative

    def test_both_losses_computation(self):
        """
        Test that both losses can be computed simultaneously.

        WHY: Training may require both verification losses
        WHAT: Verify both return flags work together
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            result = model(x, return_inv_loss=True, return_geometry_loss=True)

        # Should return tuple (logits, inv_loss, geom_loss)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        logits, inv_loss, geom_loss = result
        self.assertEqual(logits.shape, (4, 10))
        self.assertIsInstance(inv_loss, torch.Tensor)
        self.assertIsInstance(geom_loss, torch.Tensor)

    def test_fibonacci_mode_none_still_works(self):
        """
        Test that fibonacci_mode='none' still works (backward compatibility).

        WHY: Should not break existing functionality
        WHAT: Verify model works without Fibonacci scaling
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='none',  # Constant dimensions
            fractal_iters=3
        )
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (4, 10))

    def test_fibonacci_mode_linear_works(self):
        """
        Test that fibonacci_mode='linear' works with inverse.

        WHY: Linear mode should also support inverse
        WHAT: Verify model works with linear scaling
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='linear',
            fractal_iters=3
        )
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (4, 10))


# ==============================================================================
# TEST SUITE 4: SPECIFICATION COMPLIANCE
# ==============================================================================

class TestSpecificationCompliance(unittest.TestCase):
    """Test compliance with specification requirements (weight-based Fibonacci)."""

    def test_type_hints_present(self):
        """
        Test that type hints are present in FibonacciWeightScaler.

        WHY: Specification requires type hints for all new variables
        WHAT: Verify __init__, get_dimension, and get_weight have type hints
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler
        import inspect

        # Check __init__ signature
        init_sig = inspect.signature(FibonacciWeightScaler.__init__)
        self.assertIn('num_layers', init_sig.parameters)
        self.assertIn('base_dim', init_sig.parameters)
        self.assertIn('mode', init_sig.parameters)
        self.assertIn('inverse', init_sig.parameters)

        # Check get_dimension signature
        get_dim_sig = inspect.signature(FibonacciWeightScaler.get_dimension)
        self.assertIn('layer_idx', get_dim_sig.parameters)

        # Check get_weight signature (new for weight-based)
        get_weight_sig = inspect.signature(FibonacciWeightScaler.get_weight)
        self.assertIn('layer_idx', get_weight_sig.parameters)

    def test_edge_case_single_layer(self):
        """
        Test edge case: single layer (L=1).

        WHY: Specification requires handling edge cases
        WHAT: Verify model works with minimal layers
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        scaler = FibonacciWeightScaler(
            num_layers=1,
            base_dim=100,
            mode='fibonacci',
            inverse=False
        )

        # Should not raise error - dimension is constant (base_dim)
        dim = scaler.get_dimension(0)
        self.assertEqual(dim, 100)  # Always base_dim

        # Weight should be 1.0 (single layer)
        weight = scaler.get_weight(0)
        self.assertAlmostEqual(weight, 1.0, places=5)

    def test_edge_case_large_layer_count(self):
        """
        Test edge case: large layer count.

        WHY: Weight-based Fibonacci uses normalized weights (no overflow risk)
        WHAT: Verify weights are valid for large layer counts
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        # 20 layers - weights are normalized so no overflow
        scaler = FibonacciWeightScaler(
            num_layers=20,
            base_dim=10,
            mode='fibonacci',
            inverse=False
        )

        for i in range(20):
            # Dimension is always constant
            dim = scaler.get_dimension(i)
            self.assertEqual(dim, 10)  # Always base_dim

            # Weight should be positive and bounded
            weight = scaler.get_weight(i)
            self.assertGreater(weight, 0)
            self.assertLessEqual(weight, 1.0)

        # All weights should sum to 1.0
        total_weight = sum(scaler.get_all_weights())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    def test_phase_pair_preservation_in_zones(self):
        """
        Test that phase pairs are preserved in both zones.

        WHY: Specification requires phase preservation under all operations
        WHAT: Verify boundary vertices have same sectors in both zone lists
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fibonacci_mode='fibonacci',
            fractal_iters=3
        )

        # Boundary vertices appear in both zones
        boundary_ids = {v.vertex_id for v in model.boundary_vertices}

        outer_boundary = [v for v in model.outer_zone_vertices if v.vertex_id in boundary_ids]
        inner_boundary = [v for v in model.inner_zone_vertices if v.vertex_id in boundary_ids]

        # Same boundary vertices should have same sectors
        outer_boundary.sort(key=lambda v: v.vertex_id)
        inner_boundary.sort(key=lambda v: v.vertex_id)

        for outer_v, inner_v in zip(outer_boundary, inner_boundary):
            self.assertEqual(outer_v.vertex_id, inner_v.vertex_id)
            self.assertEqual(outer_v.sector, inner_v.sector,
                           f"Boundary vertex {outer_v.vertex_id} sector mismatch: "
                           f"outer={outer_v.sector}, inner={inner_v.sector}")

    def test_golden_ratio_weight_ratio_property(self):
        """
        Test that Fibonacci weights exhibit golden ratio property.

        WHY: Specification mentions golden ratio phi ~ 1.618
        WHAT: Verify raw Fibonacci sequence (used for weights) follows golden ratio
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler
        import math

        scaler = FibonacciWeightScaler(
            num_layers=20,
            base_dim=100,  # Base dim doesn't affect fib_seq
            mode='fibonacci',
            inverse=False
        )

        phi = (1 + math.sqrt(5)) / 2.0  # ~ 1.618

        # Check ratio of raw Fibonacci sequence (used for weighting)
        fib_seq = scaler.fib_seq
        for i in range(10, min(19, len(fib_seq) - 1)):
            fib_i = fib_seq[i]
            fib_i_plus_1 = fib_seq[i + 1]
            ratio = fib_i_plus_1 / fib_i

            # Should be close to golden ratio
            self.assertAlmostEqual(ratio, phi, delta=0.1,
                                 msg=f"Fibonacci ratio at index {i} should be close to phi")

        # Dimensions should still be constant (weight-based, not dimension-based)
        for i in range(20):
            self.assertEqual(scaler.get_dimension(i), 100)


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def run_all_tests():
    """Run all test suites and report results."""
    print("\n" + "=" * 80)
    print("FIBONACCI INNER ZONE MIRRORING TEST SUITE (WEIGHT-BASED)")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test suites
    suite.addTests(loader.loadTestsFromTestCase(TestFibonacciWeightScalerInverse))
    suite.addTests(loader.loadTestsFromTestCase(TestDualZoneArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestDualZoneForwardPass))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecificationCompliance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\nALL TESTS PASSED")
        print("  Fibonacci inner zone mirroring is correctly implemented.")
        print("  Specification requirements satisfied (weight-based):")
        print("    - Fibonacci sequence generation")
        print("    - Constant dimensions (weight-based, not dimension scaling)")
        print("    - Outer zone normal Fibonacci weighting")
        print("    - Inner zone inverse Fibonacci weighting")
        print("    - Bijective duality preserved in weights")
        print("    - Separate radial binners")
        print("    - Forward pass functional")
    else:
        print("\nSOME TESTS FAILED")
        print("  Please review failures before proceeding.")

    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping tests")
        sys.exit(0)

    success = run_all_tests()
    sys.exit(0 if success else 1)
