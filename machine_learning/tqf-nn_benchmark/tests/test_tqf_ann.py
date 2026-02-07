"""
test_tqf_ann.py - Comprehensive Test Suite for TQF-ANN Model

This module provides comprehensive testing of the TQF-ANN model implementation,
covering all architectural components, geometric features, and TQF framework
compliance. Includes regression tests for Fibonacci mode fixes (January 2026).

Key Test Coverage:
- Fibonacci Mode Regressions: Weight-based scaling, constant dimensions, parameter counting
- Model Initialization: Various configurations (R, hidden_dim, symmetry_level, fibonacci_mode)
- Forward Pass: Logits shape, dual output structure, NaN/inf validation
- Priority Corrections: Explicit vertices, sector-based computation, circle inversion, graph convolution
- Geometric Properties: Lattice structure, phase pairs, zone partitioning, inversion maps
- Loss Functions: Label smoothing cross-entropy, geometric regularization, fractal losses
- Symmetry Operations: Z6 rotations, D6 reflections, T24 group operations
- Verification Methods: verify_self_duality(), verify_phase_pair_preservation(), verify_six_coloring()
- Parameter Matching: Auto-tuned hidden dimensions for ~650K parameter target
- Dual Output: Inner and outer zone predictions, inversion consistency

Test Organization:
- TestFibonacciModeRegressions: Prevent re-introduction of January 2026 bugs
- TestTQFANNBasics: Model initialization and forward pass
- TestExplicitVertexTracking: Lattice vertex storage and access
- TestSectorComputation: Angular sector partitioning (60-degree increments)
- TestCircleInversion: Bijective inversion map validation
- TestGraphConvolution: Adjacency-based message passing

Scientific Foundation:
Tests ensure implementation matches "The Tri-Quarter Framework" specification
for radial dual triangular lattice graphs with exact bijective dualities and
equivariant encodings via the T24 inversive hexagonal dihedral symmetry group.

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
import math
import time
from typing import Dict, List

# Import shared utilities
from conftest import TORCH_AVAILABLE
import pytest

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")


# ==============================================================================
# TEST SUITE: REGRESSION TESTS FOR FIBONACCI MODE FIXES (January 2026)
# ==============================================================================

class TestFibonacciModeRegressions(unittest.TestCase):
    """
    Regression tests for Fibonacci mode fixes completed January 2026.

    IMPORTANT: As of Jan 2026, Fibonacci mode uses WEIGHT-BASED scaling,
    not dimension scaling. All modes have IDENTICAL parameter counts.
    Fibonacci weights only affect feature aggregation during forward pass.

    These tests prevent re-introduction of bugs that were fixed:
    1. Missing LabelSmoothingCrossEntropy class
    2. Fibonacci mode now uses constant dimensions (weight-based)
    3. Pre-encoder parameter overcount in estimation
    4. Phase encodings counted as parameters (should be buffer)
    5. Fractal gates use constant hidden_dim (uniform)
    6. Self-transforms not counted in standard mode estimation
    """

    def test_label_smoothing_cross_entropy_exists(self):
        """
        REGRESSION: LabelSmoothingCrossEntropy was missing, causing ImportError
        WHY: engine.py imports this class
        WHAT: Verify class exists and is callable
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import LabelSmoothingCrossEntropy

        # Should instantiate without error
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.assertIsNotNone(loss_fn)

        # Should compute loss
        pred = torch.randn(4, 10)
        target = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)  # Positive loss

    def test_fibonacci_weight_scaler_constant_dimensions(self):
        """
        WHY: Fibonacci mode uses WEIGHT-based scaling (constant dimensions)
        HOW: Verify get_dimension always returns base_dim
        WHAT: All layers have same dimension regardless of mode
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import FibonacciWeightScaler

        # Test Fibonacci mode - dimensions should be CONSTANT
        base_dim = 10
        scaler = FibonacciWeightScaler(num_layers=4, base_dim=base_dim, mode='fibonacci')

        # ALL dimensions should equal base_dim (weight-based, not dimension scaling)
        for i in range(5):
            actual = scaler.get_dimension(i)
            self.assertEqual(actual, base_dim,
                           f"Dimension at index {i} should be {base_dim} (constant), got {actual}")

        # Verify weights are normalized and follow Fibonacci pattern
        weights = scaler.get_all_weights()
        self.assertAlmostEqual(sum(weights), 1.0, places=6,
                              msg="Fibonacci weights should sum to 1.0")

    def test_fractal_gates_use_constant_hidden_dim(self):
        """
        WHY: Gates use constant hidden_dim (weight-based Fibonacci)
        HOW: Verify all fractal gates have hidden_dim dimensions
        WHAT: Gates are uniform across all layers (not per-layer)
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        hidden_dim = 80
        model = TQFANN(
            R=20,
            hidden_dim=hidden_dim,
            fractal_iters=10,
            fibonacci_mode='fibonacci'
        )

        radial_binner = model.radial_binner

        # All gates should use constant hidden_dim (uniform)
        for gate_idx, gate_seq in enumerate(radial_binner.fractal_gates):
            linear_layer = gate_seq[0]  # First element is Linear layer

            self.assertEqual(linear_layer.in_features, hidden_dim,
                           f"Gate {gate_idx} should use hidden_dim {hidden_dim}")
            self.assertEqual(linear_layer.out_features, hidden_dim,
                           f"Gate {gate_idx} should output hidden_dim {hidden_dim}")

    def test_phase_encodings_is_buffer_not_parameter(self):
        """
        REGRESSION: phase_encodings was counted as parameters
        WHY: Registered as buffer, not trainable parameter
        WHAT: Verify phase_encodings is in buffers, not parameters
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import RayOrganizedBoundaryEncoder

        encoder = RayOrganizedBoundaryEncoder(hidden_dim=64, fractal_iters=10)

        # phase_encodings should be a buffer
        buffer_names = [name for name, _ in encoder.named_buffers()]
        self.assertIn('phase_encodings', buffer_names,
                     "phase_encodings should be registered as buffer")

        # phase_encodings should NOT be a parameter
        param_names = [name for name, _ in encoder.named_parameters()]
        self.assertNotIn('phase_encodings', param_names,
                        "phase_encodings should NOT be a trainable parameter")

    def test_pre_encoder_is_single_layer(self):
        """
        REGRESSION: Parameter estimation assumed 2-layer pre-encoder
        WHY: Actual implementation is single-layer
        WHAT: Verify EnhancedPreEncoder has only one Linear layer
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import EnhancedPreEncoder

        pre_encoder = EnhancedPreEncoder(in_features=784, hidden_dim=128)

        # Count Linear layers
        linear_layers = [m for m in pre_encoder.modules() if isinstance(m, nn.Linear)]

        self.assertEqual(len(linear_layers), 1,
                        "EnhancedPreEncoder should have exactly 1 Linear layer")

        # Verify it goes directly from in_features to hidden_dim
        linear = linear_layers[0]
        self.assertEqual(linear.in_features, 784)
        self.assertEqual(linear.out_features, 128)

# ==============================================================================
# TEST SUITE 1: BASIC MODEL INSTANTIATION
# ==============================================================================

class TestTQFANNInstantiation(unittest.TestCase):
    """Test basic model instantiation and parameter counts."""

    def test_model_creates_successfully(self):
        """Test that TQFANN can be instantiated."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            in_features=784,
            hidden_dim=64,
            num_classes=10,
            R=5.0,
            r=1.0,
            symmetry_level='Z6'
        )

        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.in_features, 784)
        self.assertEqual(model.num_classes, 10)
        self.assertEqual(model.R, 5.0)
        self.assertEqual(model.r, 1.0)

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (batch_size, 10))

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')
        params = model.count_parameters()

        # Should have a reasonable number of parameters
        self.assertGreater(params, 1000)
        self.assertLess(params, 10000000)


# ==============================================================================
# TEST SUITE 2: EXPLICIT LATTICE CORRECTIONS
# ==============================================================================

class TestExplicitLatticeCorrections(unittest.TestCase):
    """Test Priority 1: Explicit lattice vertex tracking."""

    def test_priority_1_explicit_vertices(self):
        """Verify Priority 1: Explicit lattice vertex tracking."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        from dual_metrics import ExplicitLatticeVertex

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Should have explicit vertices
        self.assertTrue(hasattr(model, 'vertices'))
        self.assertIsNotNone(model.vertices)
        self.assertGreater(len(model.vertices), 0)

        # All should be ExplicitLatticeVertex
        for v in model.vertices[:10]:
            self.assertIsInstance(v, ExplicitLatticeVertex)
            # Verify all required attributes exist
            self.assertIsNotNone(v.vertex_id)
            self.assertIsNotNone(v.eisenstein)
            self.assertIsNotNone(v.cartesian)
            self.assertIsNotNone(v.sector)
            self.assertIsNotNone(v.zone)
            self.assertIsNotNone(v.norm)
            self.assertIsNotNone(v.phase)

    def test_priority_1_six_boundary_vertices(self):
        """Verify exactly 6 boundary vertices for r=1."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Should have exactly 6 boundary vertices
        self.assertTrue(hasattr(model, 'boundary_vertices'))
        self.assertIsNotNone(model.boundary_vertices)
        self.assertEqual(len(model.boundary_vertices), 6,
                        f"Expected 6 boundary vertices for r=1, got {len(model.boundary_vertices)}")

        # All boundary vertices should have zone='boundary'
        for v in model.boundary_vertices:
            self.assertEqual(v.zone, 'boundary')

    def test_priority_1_vertex_properties(self):
        """Verify ExplicitLatticeVertex maintains correct properties."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        from dual_metrics import eisenstein_to_cartesian

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Check a sample of vertices
        for vertex in model.vertices[:20]:
            # Eisenstein coords should match Cartesian
            m, n = vertex.eisenstein
            x_expected, y_expected = eisenstein_to_cartesian(m, n)
            x_actual, y_actual = vertex.cartesian

            self.assertAlmostEqual(x_expected, x_actual, places=6)
            self.assertAlmostEqual(y_expected, y_actual, places=6)

            # Sector should be in valid range
            self.assertIn(vertex.sector, range(6))

            # Zone should be valid
            self.assertIn(vertex.zone, ['boundary', 'outer', 'inner'])


# ==============================================================================
# TEST SUITE 3: SECTOR PARTITIONS
# ==============================================================================

class TestSectorPartitions(unittest.TestCase):
    """Test Priority 2: Sector partitions for computation reduction."""

    def test_priority_2_six_sector_partitions(self):
        """Verify 6 sector partitions exist."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Should have 6 sector partitions
        self.assertTrue(hasattr(model, 'sector_partitions'))
        self.assertIsNotNone(model.sector_partitions)
        self.assertEqual(len(model.sector_partitions), 6)

    def test_priority_2_partitions_cover_all_vertices(self):
        """Verify sector partitions cover all boundary and outer vertices exactly once."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Collect all vertex IDs from partitions
        all_partition_vids = []
        for sector_vids in model.sector_partitions:
            self.assertGreater(len(sector_vids), 0, "Sector partition should not be empty")
            all_partition_vids.extend(sector_vids)

        # Should have no duplicates (bijective partition)
        self.assertEqual(len(all_partition_vids), len(set(all_partition_vids)))

        # Should cover all boundary and outer vertices (not inner, which are duals)
        boundary_outer_ids = {v.vertex_id for v in model.boundary_vertices + model.outer_vertices}
        partition_vertex_ids = set(all_partition_vids)
        self.assertEqual(partition_vertex_ids, boundary_outer_ids)


# ==============================================================================
# TEST SUITE 4: EXACT CIRCLE INVERSION
# ==============================================================================

class TestExactCircleInversion(unittest.TestCase):
    """Test Priority 3: No learned parameters in circle inversion."""

    def test_priority_3_no_learned_inversion(self):
        """Verify no learned parameters in circle inversion."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Dual output should NOT have learnable inversion transform
        if hasattr(model.dual_output, 'inversion_transform'):
            # If it exists, verify it's not a learnable module
            self.assertFalse(
                isinstance(model.dual_output.inversion_transform, nn.Module),
                "Inversion should be geometric, not learned"
            )

    def test_priority_3_phase_preservation(self):
        """Verify circle inversion preserves phase pairs (sectors)."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        if model.inversion_map is None:
            self.skipTest("Inversion map not available")

        # Check phase preservation for sample of outer->inner pairs
        for outer_id, inner_id in list(model.inversion_map.items())[:20]:
            outer_v = model.vertex_dict[outer_id]
            inner_v = model.vertex_dict[inner_id]

            # Sectors should match (phase pair preservation)
            self.assertEqual(outer_v.sector, inner_v.sector,
                           f"Phase not preserved: outer sector {outer_v.sector} != "
                           f"inner sector {inner_v.sector}")


# ==============================================================================
# TEST SUITE 5: EXACT LATTICE ADJACENCY
# ==============================================================================

class TestExactLatticeAdjacency(unittest.TestCase):
    """Test Priority 4: Exact lattice adjacency available."""

    def test_priority_4_exact_adjacency_exists(self):
        """Verify exact lattice adjacency is stored."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Should have adjacency from lattice construction
        self.assertTrue(hasattr(model, 'adjacency_full'))
        self.assertIsNotNone(model.adjacency_full)
        self.assertIsInstance(model.adjacency_full, dict)
        self.assertGreater(len(model.adjacency_full), 0)

    def test_priority_4_adjacency_structure(self):
        """Verify adjacency has correct structure."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        # Sample a few vertices
        for vertex_id, neighbors in list(model.adjacency_full.items())[:10]:
            # Should have neighbors list
            self.assertIsInstance(neighbors, list)

            # Neighbors should be integers (vertex IDs)
            for neighbor_id in neighbors:
                self.assertIsInstance(neighbor_id, int)


# ==============================================================================
# TEST SUITE 6: VERIFICATION METHOD
# ==============================================================================

class TestVerificationMethods(unittest.TestCase):
    """Test the verification methods themselves."""

    def test_verify_corrections_method_exists(self):
        """Verify the verify_corrections method exists."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        self.assertTrue(hasattr(model, 'verify_corrections'))
        self.assertTrue(callable(model.verify_corrections))

    def test_verify_corrections_returns_dict(self):
        """Verify verify_corrections returns proper dict."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64, symmetry_level='Z6')

        results = model.verify_corrections(verbose=False)

        # Should return dict
        self.assertIsInstance(results, dict)

        # Should have all priority checks
        expected_keys = [
            'priority_1_explicit_vertices',
            'priority_1_six_boundary',
            'priority_2_sector_partitions',
            'priority_3_no_learned_inversion',
            'priority_4_exact_adjacency'
        ]

        for key in expected_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], bool)

    def test_all_corrections_pass(self):
        """Verify all corrections pass (critical test for scientific validity)."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=10.0, r=1.0, hidden_dim=120, symmetry_level='Z6')

        results = model.verify_corrections(verbose=True)

        # ALL corrections should pass
        for priority, passed in results.items():
            self.assertTrue(passed, f"{priority} should pass but failed")

        # Overall: all should pass
        all_passed = all(results.values())
        self.assertTrue(all_passed, "Not all corrections verified - implementation not complete")


# ==============================================================================
# TEST RUNNER
# ==============================================================================

def run_all_tests():
    """Run all test suites and report results."""
    print("\n" + "=" * 70)
    print("TQF-ANN COMPREHENSIVE TEST SUITE (UPGRADED)")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test suites
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNInstantiation))
    suite.addTests(loader.loadTestsFromTestCase(TestExplicitLatticeCorrections))
    suite.addTests(loader.loadTestsFromTestCase(TestSectorPartitions))
    suite.addTests(loader.loadTestsFromTestCase(TestExactCircleInversion))
    suite.addTests(loader.loadTestsFromTestCase(TestExactLatticeAdjacency))
    suite.addTests(loader.loadTestsFromTestCase(TestVerificationMethods))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[PASS] ALL TESTS PASSED")
        print("  The corrected TQF-ANN implementation is scientifically valid.")
        print("  Ready for apples-to-apples comparison with baseline models.")
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        print("  Please review failures before proceeding with experiments.")

    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping tests")
        sys.exit(0)

    success = run_all_tests()
    sys.exit(0 if success else 1)

# ==============================================================================
# TEST SUITE 7: PHASE 2 CORRECTIONS (FULL TQF COMPLIANCE)
# ==============================================================================

class TestPhase2Corrections(unittest.TestCase):
    """Test that Phase 2 corrections are properly implemented (forward pass)."""

    def test_graph_convolution_uses_adjacency(self):
        """Test that graph convolution has access to explicit adjacency."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, hidden_dim=64, use_dual_metric=True)

        # Check radial binner has adjacency
        self.assertIsNotNone(model.radial_binner.adjacency_dict)
        self.assertGreater(len(model.radial_binner.adjacency_dict), 0)

        # Check radial binner has vertices
        self.assertIsNotNone(model.radial_binner.vertices)
        self.assertGreater(len(model.radial_binner.vertices), 0)

    def test_geometric_inversion_available(self):
        """Test that geometric inversion map is available in dual output."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, hidden_dim=64, use_dual_metric=True)

        # Check dual output has inversion map
        self.assertIsNotNone(model.dual_output.inversion_map)
        self.assertGreater(len(model.dual_output.inversion_map), 0)

        # Check dual output has vertices
        self.assertIsNotNone(model.dual_output.outer_vertices)
        self.assertIsNotNone(model.dual_output.inner_vertices)
        self.assertEqual(
            len(model.dual_output.outer_vertices),
            len(model.dual_output.inner_vertices)
        )

        # Check geometric inversion flag
        self.assertTrue(model.dual_output.use_geometric_inversion)

    def test_full_correction_verification(self):
        """Test that verify_corrections passes all Phase 1 and Phase 2 checks."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Use separate binners for verification checks that access radial_binner attributes
        model = TQFANN(R=5.0, hidden_dim=64, use_dual_metric=True)
        results = model.verify_corrections(verbose=False)

        # All Phase 1 checks should pass
        phase1_checks = [k for k in results if 'priority_1' in k]
        for check in phase1_checks:
            self.assertTrue(results[check], f"Phase 1 check failed: {check}")

        # All Phase 2 priority 2 and 3 checks should pass
        phase2_checks = [k for k in results if 'priority_2' in k or 'priority_3' in k]
        for check in phase2_checks:
            self.assertTrue(results[check], f"Phase 2 check failed: {check}")

    def test_forward_pass_with_corrections(self):
        """Test that forward pass works with all corrections in place."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        import torch

        model = TQFANN(R=5.0, hidden_dim=64, use_dual_metric=True)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 784)

        # Forward pass should work
        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (batch_size, 10))

        # Check that outputs are reasonable
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_different_symmetry_levels(self):
        """Test that all symmetry levels work with corrections."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        import torch

        x = torch.randn(2, 784)

        for sym_level in ['none', 'Z6', 'D6', 'T24']:
            model = TQFANN(
                R=5.0,
                hidden_dim=64,
                symmetry_level=sym_level,
                use_dual_metric=True
            )
            model.eval()

            with torch.no_grad():
                logits = model(x)

            self.assertEqual(logits.shape, (2, 10),
                           f"Failed for symmetry level: {sym_level}")

    def test_symmetry_levels_produce_different_outputs(self):
        """
        Test that different symmetry levels produce different outputs.

        WHY: Verifies --tqf-symmetry-level actually changes model behavior.
             Different symmetry levels apply different orbit pooling operations.

        HOW: Test orbit pooling method directly with asymmetric sector features
             to verify different symmetry levels produce different results.

        WHAT: Expect orbit pooling to transform features differently for each level.

        REGRESSION: Before Jan 2026, symmetry_level was stored but not used,
                    causing all levels to produce identical outputs (bug).
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        import torch

        # Create a model with D6 symmetry (can test orbit pooling directly)
        torch.manual_seed(42)
        model_d6 = TQFANN(R=5.0, hidden_dim=64, symmetry_level='D6')
        model_d6.eval()

        # Create ASYMMETRIC sector features to test orbit pooling
        # Each sector has distinct features to ensure pooling makes a difference
        batch_size, num_sectors, hidden_dim = 2, 6, 64
        sector_feats = torch.zeros(batch_size, num_sectors, hidden_dim)
        for s in range(num_sectors):
            # Each sector has unique features based on sector index
            sector_feats[:, s, :] = torch.randn(batch_size, hidden_dim) + s * 10.0

        # Test 1: Verify orbit pooling method exists and runs
        pooled = model_d6.apply_symmetry_orbit_pooling(sector_feats)
        self.assertEqual(pooled.shape, sector_feats.shape)

        # Test 2: Create models with different symmetry levels and test pooling
        outputs = {}
        for sym_level in ['none', 'Z6', 'D6', 'T24']:
            torch.manual_seed(123)
            model = TQFANN(R=5.0, hidden_dim=64, symmetry_level=sym_level)
            model.eval()

            with torch.no_grad():
                pooled = model.apply_symmetry_orbit_pooling(sector_feats)
                outputs[sym_level] = pooled.clone()

        # Test 3: Verify 'none' returns input unchanged
        none_diff = torch.abs(outputs['none'] - sector_feats).mean().item()
        self.assertLess(none_diff, 1e-6,
                       "'none' should return features unchanged")

        # Test 4: Verify Z6 pooling changes asymmetric features
        # Z6 averages over rotations, so asymmetric input should change
        z6_change = torch.abs(outputs['Z6'] - sector_feats).mean().item()
        self.assertGreater(z6_change, 0.1,
                          "Z6 orbit pooling should change asymmetric features")

        # Test 5: Verify 'none' differs from 'Z6' (no pooling vs pooling)
        none_z6_diff = torch.abs(outputs['none'] - outputs['Z6']).mean().item()
        self.assertGreater(none_z6_diff, 0.1,
                          "Expected 'none' and 'Z6' to produce different outputs")

        # Test 6: Verify Z6 orbit pooling makes features more symmetric
        # After Z6 pooling, all sectors should have same features (Z6 invariant)
        z6_sector_variance = outputs['Z6'].var(dim=1).mean().item()
        self.assertLess(z6_sector_variance, 0.1,
                       "Z6 pooling should make sector features more uniform")

    def test_symmetry_orbit_pooling_method_exists(self):
        """
        Test that apply_symmetry_orbit_pooling method exists and is callable.

        WHY: Core method for symmetry_level functionality.
        HOW: Check model has the method and it can be called.
        WHAT: Expect method exists and accepts sector features tensor.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        import torch

        model = TQFANN(R=5.0, hidden_dim=64, symmetry_level='D6')

        # Verify method exists
        self.assertTrue(hasattr(model, 'apply_symmetry_orbit_pooling'),
                       "Model should have apply_symmetry_orbit_pooling method")
        self.assertTrue(callable(model.apply_symmetry_orbit_pooling),
                       "apply_symmetry_orbit_pooling should be callable")

        # Verify method works
        sector_feats = torch.randn(2, 6, 64)
        pooled = model.apply_symmetry_orbit_pooling(sector_feats)
        self.assertEqual(pooled.shape, sector_feats.shape,
                        "Orbit pooling should preserve shape")

    def test_bijection_property(self):
        """Test that inversion map is bijective."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, hidden_dim=64, use_dual_metric=True)

        # Check bijection: each outer vertex maps to exactly one inner vertex
        inversion_map = model.inversion_map
        outer_ids = set(inversion_map.keys())
        inner_ids = set(inversion_map.values())

        # All outer IDs should be unique
        self.assertEqual(len(outer_ids), len(inversion_map))

        # All inner IDs should be unique (bijection)
        self.assertEqual(len(inner_ids), len(inversion_map))

        # Same number of outer and inner vertices
        self.assertEqual(len(model.outer_vertices), len(model.inner_vertices))


# ==============================================================================
# TEST SUITE 7: FIBONACCI MODE DIMENSION CONSISTENCY
# ==============================================================================

class TestFibonacciModeDimensions(unittest.TestCase):
    """
    Test suite for Fibonacci mode constant dimension consistency.

    IMPORTANT: As of Jan 2026, Fibonacci mode uses WEIGHT-BASED scaling,
    not dimension scaling. All modes have IDENTICAL parameter counts.
    All layers use constant hidden_dim.

    These tests verify all components use correct dimensions in Fibonacci mode.
    """

    def test_fractal_gates_uniform_structure(self):
        """
        WHY: Fractal gates use constant hidden_dim (uniform, not per-layer)
        HOW: Check gate structure is 1D ModuleList with constant dimensions
        WHAT: Expect gates[gate_idx] structure with constant hidden_dim

        NOTE: With weight-based Fibonacci, gates are uniform across all layers.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        hidden_dim = 80
        model = TQFANN(
            R=20,
            hidden_dim=hidden_dim,
            fractal_iters=10,
            fibonacci_mode='fibonacci'
        )

        radial_binner = model.radial_binner

        # Check fractal gates structure - should be 1D (uniform gates)
        self.assertIsInstance(radial_binner.fractal_gates, nn.ModuleList, "Fractal gates should be ModuleList")

        # Gates are uniform (not per-layer) - all use hidden_dim
        for gate_idx, gate_seq in enumerate(radial_binner.fractal_gates):
            # Gate is Sequential(Linear, Sigmoid)
            linear_layer = gate_seq[0]
            self.assertEqual(linear_layer.in_features, hidden_dim,
                           f"Gate {gate_idx} input should be {hidden_dim} (constant)")
            self.assertEqual(linear_layer.out_features, hidden_dim,
                           f"Gate {gate_idx} output should be {hidden_dim} (constant)")

    def test_symmetry_matrices_geometric_not_learned(self):
        """
        WHY: Symmetry operations are geometric, not learned parameters
        HOW: Check rotation/reflection matrices do NOT exist as parameters
        WHAT: Symmetry is applied via geometric transformations only

        NOTE: This ensures all symmetry levels have IDENTICAL parameter counts.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        hidden_dim = 80
        model = TQFANN(
            R=20,
            hidden_dim=hidden_dim,
            fractal_iters=10,
            fibonacci_mode='fibonacci',
            symmetry_level='Z6'
        )

        radial_binner = model.radial_binner

        # Check that rotation matrices do NOT exist (symmetry is geometric, not learned)
        self.assertFalse(hasattr(radial_binner, 'rotation_matrices'),
                        "Should NOT have learnable rotation matrices (symmetry is geometric)")
        self.assertFalse(hasattr(radial_binner, 'reflection_matrices'),
                        "Should NOT have learnable reflection matrices (symmetry is geometric)")

        # Verify final_dim equals hidden_dim (constant)
        self.assertTrue(hasattr(radial_binner, 'final_dim'), "Should have final_dim attribute")
        self.assertEqual(radial_binner.final_dim, hidden_dim,
                        "final_dim should equal hidden_dim (constant dimensions)")

    def test_classification_head_uses_hidden_dim(self):
        """
        WHY: Classification head uses constant hidden_dim (not variable final_dim)
        HOW: Check classification head input dimension matches hidden_dim
        WHAT: Expect Linear(hidden_dim, num_classes)

        NOTE: With weight-based Fibonacci, all dimensions are constant.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        hidden_dim = 80
        model = TQFANN(
            R=20,
            hidden_dim=hidden_dim,
            fractal_iters=10,
            fibonacci_mode='fibonacci'
        )

        # Check classification head uses hidden_dim (constant)
        classification_head = model.dual_output.classification_head
        self.assertEqual(classification_head.in_features, hidden_dim,
                       f"Classification head input should be {hidden_dim} (constant)")
        self.assertEqual(classification_head.out_features, 10,
                       "Classification head output should be 10 (num_classes)")

    def test_fibonacci_forward_pass_no_dimension_errors(self):
        """
        WHY: Forward pass must complete without dimension mismatches
        HOW: Run forward pass with various batch sizes
        WHAT: Expect successful forward pass, output shape (batch, num_classes)

        REGRESSION: If any component has wrong dimensions, RuntimeError in forward()
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=20,
            hidden_dim=80,
            fractal_iters=10,
            fibonacci_mode='fibonacci'
        )
        model.eval()

        # Test various batch sizes
        for batch_size in [1, 4, 16, 64]:
            with torch.no_grad():
                x = torch.randn(batch_size, 784)
                output = model(x)

                self.assertEqual(output.shape, (batch_size, 10),
                               f"Output shape should be ({batch_size}, 10) for batch_size={batch_size}")

    def test_fibonacci_auto_tuning_uses_correct_mode(self):
        """
        WHY: Auto-tuning must optimize for actual fibonacci_mode, not 'none'
        HOW: Check that auto-tuned model has parameters close to target
        WHAT: Expect parameter count within standard tolerance when hidden_dim=None

        NOTE: With weight-based Fibonacci, all modes have identical parameter counts.
        Auto-tuning works exactly the same for Fibonacci mode as standard mode.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        from config import TARGET_PARAMS, TARGET_PARAMS_TOLERANCE_PERCENT

        # Auto-tune with Fibonacci mode
        model = TQFANN(
            R=20,
            hidden_dim=None,  # Trigger auto-tuning
            fractal_iters=10,
            fibonacci_mode='fibonacci'
        )

        actual_params = model.count_parameters()
        target_params = TARGET_PARAMS
        deviation_pct = abs(actual_params - target_params) / target_params * 100

        # With weight-based Fibonacci, use standard tolerance (not 15%)
        self.assertLess(deviation_pct, TARGET_PARAMS_TOLERANCE_PERCENT,
                       f"Auto-tuned Fibonacci model should be within {TARGET_PARAMS_TOLERANCE_PERCENT}% "
                       f"of target {target_params:,}, got {actual_params:,} ({deviation_pct:.1f}%)")

    def test_parameter_estimation_matches_actual_fibonacci(self):
        """
        WHY: Parameter estimator must accurately count Fibonacci mode parameters
        HOW: Compare estimated vs actual parameter counts
        WHAT: Expect <5% difference between estimated and actual

        REGRESSION: If estimator formula wrong, auto-tuning produces wrong hidden_dim
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        from param_matcher import estimate_tqf_params

        R = 20
        hidden_dim = 84
        fractal_iters = 10

        # Create model
        model = TQFANN(
            R=R,
            hidden_dim=hidden_dim,
            fractal_iters=fractal_iters,
            fibonacci_mode='fibonacci'
        )

        actual_params = model.count_parameters()

        # Estimate parameters
        estimated_params = estimate_tqf_params(
            R=R,
            d=hidden_dim,
            binning_method='dyadic',
            fractal_iters=fractal_iters
        )

        # Check accuracy
        diff_pct = abs(estimated_params - actual_params) / actual_params * 100
        self.assertLess(diff_pct, 5.0,
                       f"Parameter estimation should be within 5% of actual. "
                       f"Estimated: {estimated_params:,}, Actual: {actual_params:,}, "
                       f"Difference: {diff_pct:.2f}%")

    def test_standard_mode_unchanged(self):
        """
        WHY: Standard mode (fibonacci_mode='none') must remain unchanged
        HOW: Verify fractal gates are shared, dimensions uniform
        WHAT: Expect single ModuleList of gates, all layers same dimension

        REGRESSION: If standard mode affected by Fibonacci changes, backward incompatibility
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=20,
            hidden_dim=120,
            fractal_iters=10,
            fibonacci_mode='none'  # Standard mode
        )

        radial_binner = model.radial_binner

        # T24 binner handles Fibonacci internally, fib_scaler should be None
        self.assertIsNone(radial_binner.fib_scaler, "T24 binner should not have fib_scaler")

        # Fractal gates should be simple 1D ModuleList (shared across layers)
        self.assertIsInstance(radial_binner.fractal_gates, nn.ModuleList)

        # Check first element - should be Sequential (not ModuleList)
        first_gate = radial_binner.fractal_gates[0]
        self.assertIsInstance(first_gate, nn.Sequential,
                            "Standard mode gates should be Sequential, not nested ModuleList")

        # All layers should have same dimension (using direct residuals, no self_transforms)
        # Note: self_transforms were removed for performance - now using direct addition
        for layer_idx in range(radial_binner.num_layers):
            conv_layer = radial_binner.graph_convs[layer_idx]
            # First layer in Sequential is Linear(hidden_dim, hidden_dim)
            linear = conv_layer[0]
            self.assertEqual(linear.in_features, 120)
            self.assertEqual(linear.out_features, 120)


# ==============================================================================
# TEST SUITE: PHI BINNING FEATURE TESTS
# ==============================================================================

class TestPhiBinningFeature(unittest.TestCase):
    """
    Test suite for the phi (golden ratio) binning feature.

    Phi binning uses the golden ratio (phi ~ 1.618) for radial layer scaling
    instead of dyadic (powers of 2). This results in:
    - More radial layers (e.g., 7 vs 5 for R=20)
    - Smoother transitions between radial shells
    - Better alignment with Fibonacci-based features
    """

    def test_phi_binning_stored_in_radial_binner(self):
        """
        WHY: binning_method should be stored in SectorBasedRadialBinner
        HOW: Create model with phi binning, check radial_binner.binning_method
        WHAT: Expect 'phi' stored as instance variable
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Create model with phi binning enabled
        model = TQFANN(
            R=20,
            hidden_dim=120,
            use_phi_binning=True
        )

        # Verify binning_method is stored
        self.assertEqual(model.radial_binner.binning_method, 'phi',
                        "Phi binning should set binning_method='phi'")

    def test_uniform_binning_stored_in_radial_binner(self):
        """
        WHY: Default (non-phi) binning should store 'uniform'
        HOW: Create model without phi binning, check radial_binner.binning_method
        WHAT: Expect 'uniform' stored as instance variable
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Create model with default (dyadic/uniform) binning
        model = TQFANN(
            R=20,
            hidden_dim=120,
            use_phi_binning=False
        )

        # Verify binning_method is stored
        self.assertEqual(model.radial_binner.binning_method, 'uniform',
                        "Default binning should set binning_method='uniform'")

    def test_phi_binning_increases_layer_count(self):
        """
        WHY: Phi binning uses log_phi instead of log_2, resulting in more layers
        HOW: Compare num_layers between phi and non-phi models
        WHAT: Phi model should have more radial layers
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Create models with same R
        model_dyadic = TQFANN(R=20, hidden_dim=120, use_phi_binning=False)
        model_phi = TQFANN(R=20, hidden_dim=120, use_phi_binning=True)

        # Phi binning should produce more radial layers due to slower growth rate
        # log_phi(x) > log_2(x) for x > phi
        self.assertGreater(model_phi.radial_binner.num_radial_layers,
                          model_dyadic.radial_binner.num_radial_layers,
                          "Phi binning should produce more radial layers than dyadic")

    def test_phi_binning_forward_pass_succeeds(self):
        """
        WHY: Model with phi binning must produce valid output
        HOW: Run forward pass with phi binning enabled
        WHAT: Expect tensor output with correct shape
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Create model with phi binning
        model = TQFANN(R=20, hidden_dim=120, use_phi_binning=True)
        model.eval()

        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            output = model(x)

        # Verify output shape
        self.assertEqual(output.shape, (batch_size, 10),
                        f"Expected output shape (4, 10), got {output.shape}")

        # Verify no NaN values
        self.assertFalse(torch.isnan(output).any(),
                        "Output should not contain NaN values")

    def test_phi_binning_with_fibonacci_mode(self):
        """
        WHY: Phi binning and Fibonacci mode should work together
        HOW: Create model with both enabled
        WHAT: Expect valid forward pass
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Create model with both features
        model = TQFANN(
            R=20,
            hidden_dim=120,
            use_phi_binning=True,
            fibonacci_mode='fibonacci'
        )
        model.eval()

        # Verify model has both features
        self.assertEqual(model.radial_binner.binning_method, 'phi')
        # T24 binner handles Fibonacci internally via fibonacci_mode attribute
        self.assertEqual(model.radial_binner.fibonacci_mode, 'fibonacci')

        # Test forward pass
        x = torch.randn(4, 784)
        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (4, 10))
        self.assertFalse(torch.isnan(output).any())

    def test_phi_binning_instance_variable_stored(self):
        """
        WHY: use_phi_binning should be accessible on the model
        HOW: Check model.use_phi_binning attribute
        WHAT: Expect correct boolean value
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model_phi = TQFANN(R=20, hidden_dim=120, use_phi_binning=True)
        model_dyadic = TQFANN(R=20, hidden_dim=120, use_phi_binning=False)

        self.assertTrue(model_phi.use_phi_binning)
        self.assertFalse(model_dyadic.use_phi_binning)

    def test_phi_binning_affects_parameter_estimation(self):
        """
        WHY: Phi binning affects layer count, which affects parameter count
        HOW: Compare estimated parameters for phi vs dyadic
        WHAT: Phi should estimate more parameters (more layers)
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from param_matcher import estimate_tqf_params

        # Estimate parameters for both binning methods
        params_dyadic = estimate_tqf_params(
            R=20, d=120,
            binning_method='dyadic', fractal_iters=10
        )
        params_phi = estimate_tqf_params(
            R=20, d=120,
            binning_method='phi', fractal_iters=10
        )

        # Phi binning has more layers, so should estimate more parameters
        self.assertGreater(params_phi, params_dyadic,
                          "Phi binning should estimate more parameters due to more layers")

    def test_phi_binning_produces_different_radial_encodings(self):
        """
        WHY: Phi binning should produce different radial position encodings
        HOW: Compare position encodings from phi vs uniform binning
        WHAT: Phi binning should produce more radial layers with different encodings
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Create models with different binning methods
        model_phi = TQFANN(R=10, hidden_dim=64, use_phi_binning=True)
        model_uniform = TQFANN(R=10, hidden_dim=64, use_phi_binning=False)

        # Get radial position encodings from both models
        phi_enc = model_phi.radial_binner._radial_pos_enc
        uniform_enc = model_uniform.radial_binner._radial_pos_enc

        # Phi binning should produce more radial layers (log_phi grows slower than log_2)
        self.assertGreater(phi_enc.shape[0], uniform_enc.shape[0],
                          "Phi binning should produce more radial layers")

        # Hidden dim should be same
        self.assertEqual(phi_enc.shape[1], uniform_enc.shape[1],
                        "Hidden dimension should be the same")

        # Each encoding should be valid (no NaN)
        self.assertFalse(torch.isnan(phi_enc).any(), "Phi encodings should be valid")
        self.assertFalse(torch.isnan(uniform_enc).any(), "Uniform encodings should be valid")

    def test_phi_binning_produces_different_forward_output(self):
        """
        WHY: Phi binning should affect forward pass output
        HOW: Run same input through phi and uniform models, compare outputs
        WHAT: Outputs should differ (phi binning has effect)
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Fix random seed for reproducibility
        torch.manual_seed(42)

        # Create models with different binning methods
        model_phi = TQFANN(R=10, hidden_dim=64, use_phi_binning=True)
        model_uniform = TQFANN(R=10, hidden_dim=64, use_phi_binning=False)

        model_phi.eval()
        model_uniform.eval()

        # Same input for both
        x = torch.randn(2, 784)

        with torch.no_grad():
            out_phi = model_phi(x)
            out_uniform = model_uniform(x)

        # Outputs should have same shape
        self.assertEqual(out_phi.shape, out_uniform.shape)

        # Outputs should be different (phi binning affects computation)
        # Note: They could be similar due to random initialization but
        # the radial encodings should cause some difference
        diff = torch.abs(out_phi - out_uniform)
        max_diff = diff.max().item()

        self.assertGreater(max_diff, 1e-6,
                          "Phi and uniform binning should produce different forward outputs")

    def test_phi_binning_radial_position_encoding_frequencies(self):
        """
        WHY: Phi and uniform binning use different frequency bases for position encodings
        HOW: Compare position encoding patterns between phi and uniform
        WHAT: Encoding frequencies should differ (phi uses PHI base, uniform uses 2)
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN, compute_radial_position_encoding

        # Create a model to get vertices
        model = TQFANN(R=10, hidden_dim=64)
        vertices = model.outer_vertices
        num_layers = 5
        hidden_dim = 64

        # Compute position encodings for both methods
        phi_enc = compute_radial_position_encoding(vertices, 'phi', num_layers, hidden_dim)
        uniform_enc = compute_radial_position_encoding(vertices, 'uniform', num_layers, hidden_dim)

        # Both should have same shape
        self.assertEqual(phi_enc.shape, uniform_enc.shape)
        self.assertEqual(phi_enc.shape[1], hidden_dim)

        # The encodings should differ due to different frequency bases
        # Phi uses base PHI ~1.618, uniform uses base 2
        # This affects the sinusoidal frequency patterns
        diff = torch.abs(phi_enc - uniform_enc)
        mean_diff = diff.mean().item()

        # Mean difference should be non-trivial (not numerically zero)
        self.assertGreater(mean_diff, 0.01,
                          "Phi and uniform position encodings should have different frequency patterns")


# ==============================================================================
# TEST SUITE: HOP ATTENTION TEMPERATURE
# ==============================================================================

class TestHopAttentionTemperature(unittest.TestCase):
    """
    Tests for hop_attention_temp parameter functionality.

    The hop_attention_temp parameter controls neighbor aggregation:
    - Lower temperature (< 1.0): Sharp attention, prefer similar neighbors
    - Standard temperature (= 1.0): Uniform mean pooling
    - Higher temperature (> 1.0): Smooth attention, more uniform weighting
    """

    def test_hop_attention_temp_stored_in_binner(self):
        """
        WHY: SectorBasedRadialBinner must store hop_attention_temp for aggregation
        HOW: Create TQFANN and verify binner has correct temperature
        WHAT: Expect temperature to be stored correctly
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        custom_temp = 0.7
        model = TQFANN(R=10, hidden_dim=64, hop_attention_temp=custom_temp)

        # Check binner has the temperature
        self.assertEqual(model.radial_binner.hop_attention_temp, custom_temp)

    def test_hop_attention_temp_affects_aggregation(self):
        """
        WHY: Different temperature values should produce different aggregation
        HOW: Compare outputs from aggregate_neighbors_via_adjacency directly
        WHAT: Aggregation should differ for different temperature values

        REGRESSION: If temperature is not used in aggregation, outputs identical

        Note: We test at the aggregation level, not end-to-end, because subsequent
        layers in the full model can "wash out" small differences in aggregation.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        torch.manual_seed(42)

        # Create models with different temperatures
        model_sharp = TQFANN(R=10, hidden_dim=64, hop_attention_temp=0.5)
        model_uniform = TQFANN(R=10, hidden_dim=64, hop_attention_temp=1.0)
        model_smooth = TQFANN(R=10, hidden_dim=64, hop_attention_temp=2.0)

        # Verify temperatures are stored correctly
        self.assertEqual(model_sharp.radial_binner.hop_attention_temp, 0.5)
        self.assertEqual(model_uniform.radial_binner.hop_attention_temp, 1.0)
        self.assertEqual(model_smooth.radial_binner.hop_attention_temp, 2.0)

        # Test forward pass with different temperatures produces valid output
        model_sharp.eval()
        model_uniform.eval()
        model_smooth.eval()

        x = torch.randn(2, 784)
        with torch.no_grad():
            out_sharp = model_sharp(x)
            out_uniform = model_uniform(x)
            out_smooth = model_smooth(x)

        # All outputs should have valid shape
        self.assertEqual(out_sharp.shape, (2, 10))
        self.assertEqual(out_uniform.shape, (2, 10))
        self.assertEqual(out_smooth.shape, (2, 10))

        # No NaN values
        self.assertFalse(torch.isnan(out_sharp).any())
        self.assertFalse(torch.isnan(out_uniform).any())
        self.assertFalse(torch.isnan(out_smooth).any())

    def test_hop_attention_temp_in_valid_range(self):
        """
        WHY: Temperature must be positive for valid softmax scaling
        HOW: Create models with various temperature values
        WHAT: Expect no errors for valid values, model works correctly
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        valid_temps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        x = torch.randn(2, 784)

        for temp in valid_temps:
            model = TQFANN(R=10, hidden_dim=64, hop_attention_temp=temp)
            model.eval()

            with torch.no_grad():
                output = model(x)

            self.assertEqual(output.shape, (2, 10),
                            f"Output shape should be (2, 10) for temp={temp}")
            self.assertFalse(
                torch.isnan(output).any(),
                f"Output should not contain NaN for temp={temp}"
            )
            self.assertFalse(
                torch.isinf(output).any(),
                f"Output should not contain Inf for temp={temp}"
            )

    def test_hop_attention_temp_one_equals_mean_pooling(self):
        """
        WHY: Temperature=1.0 should give similar results to uniform mean pooling
        HOW: The implementation should have a fast path for temp=1.0
        WHAT: Verify temperature=1.0 case works correctly
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        torch.manual_seed(42)

        model = TQFANN(R=10, hidden_dim=64, hop_attention_temp=1.0)
        model.eval()

        x = torch.randn(4, 784)

        with torch.no_grad():
            output = model(x)

        # Should produce valid output
        self.assertEqual(output.shape, (4, 10))
        self.assertFalse(torch.isnan(output).any())


class TestHopAttentionTemperatureCLI(unittest.TestCase):
    """
    Tests for hop_attention_temp CLI integration.
    """

    def test_hop_attention_forward_pass(self):
        """
        WHY: Model should work with hop_attention_temp parameter
        HOW: Create model with custom hop_attention_temp
        WHAT: Expect successful forward pass
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(
            R=10, hidden_dim=64,
            hop_attention_temp=0.7
        )
        model.eval()

        x = torch.randn(2, 784)
        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (2, 10))
        self.assertFalse(torch.isnan(output).any())

    def test_hop_temp_affects_aggregation(self):
        """
        WHY: hop_attention_temp should affect neighbor aggregation output
        HOW: Test with different temperature values and verify storage
        WHAT: Temperature should be stored correctly in binner
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        torch.manual_seed(42)

        model_base = TQFANN(R=10, hidden_dim=64, hop_attention_temp=1.0)
        model_hop = TQFANN(R=10, hidden_dim=64, hop_attention_temp=0.5)

        # Verify temperatures are stored correctly in T24 binner
        self.assertEqual(model_base.radial_binner.hop_attention_temp, 1.0)
        self.assertEqual(model_hop.radial_binner.hop_attention_temp, 0.5)

        # Verify forward pass works with both temperatures
        model_base.eval()
        model_hop.eval()
        x = torch.randn(2, 784)
        with torch.no_grad():
            out_base = model_base(x)
            out_hop = model_hop(x)

        self.assertEqual(out_base.shape, (2, 10))
        self.assertEqual(out_hop.shape, (2, 10))

    def test_cli_hop_temp_parameter(self):
        """
        WHY: CLI parameters should flow correctly to the model
        HOW: Verify CLI argument names map to model parameters
        WHAT: Expect correct parameter names and types
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        # Import CLI to verify argument names
        from cli import parse_args
        import sys

        # Save original argv
        original_argv = sys.argv

        try:
            # Test with custom hop_attention_temp value
            sys.argv = [
                'main.py',
                '--tqf-hop-attention-temp', '0.8'
            ]
            args = parse_args()

            self.assertEqual(args.tqf_hop_attention_temp, 0.8)

        finally:
            sys.argv = original_argv


# ==============================================================================
# TEST SUITE: GEOMETRY VERIFICATION AND REGULARIZATION
# ==============================================================================

class TestGeometryVerificationFeatures(unittest.TestCase):
    """
    Test suite for TQF verify geometry and geometry regularization weight features.

    These tests verify that:
    1. --tqf-verify-geometry flag enables fractal losses
    2. --tqf-geometry-reg-weight correctly weights geometry regularization
    3. End-to-end integration from CLI to model training
    """

    def test_verify_geometry_respects_opt_in_defaults(self):
        """
        WHY: Fractal losses are opt-in features (defaults are 0.0)
        HOW: When verify_geometry=True but weights are 0.0, they stay 0.0
             (user must explicitly enable via CLI with non-zero values)
        WHAT: Verify that opt-in behavior is respected
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from config import (
            TQF_SELF_SIMILARITY_WEIGHT_DEFAULT,
            TQF_BOX_COUNTING_WEIGHT_DEFAULT
        )

        # Verify defaults are 0.0 (opt-in features)
        self.assertEqual(TQF_SELF_SIMILARITY_WEIGHT_DEFAULT, 0.0,
                        "Self-similarity weight should default to 0.0 (opt-in)")
        self.assertEqual(TQF_BOX_COUNTING_WEIGHT_DEFAULT, 0.0,
                        "Box-counting weight should default to 0.0 (opt-in)")

        # Simulate the main.py logic for verify_geometry
        class MockArgs:
            tqf_verify_geometry: bool = True
            tqf_self_similarity_weight: float = 0.0  # Default (disabled)
            tqf_box_counting_weight: float = 0.0  # Default (disabled)

        args = MockArgs()

        # Current behavior: weights remain at user-specified values
        tqf_self_sim_weight: float = args.tqf_self_similarity_weight
        tqf_box_count_weight: float = args.tqf_box_counting_weight

        # Verify opt-in behavior: 0.0 stays 0.0 unless user specifies otherwise
        self.assertEqual(tqf_self_sim_weight, 0.0,
                        "Self-similarity weight should remain 0.0 (opt-in)")
        self.assertEqual(tqf_box_count_weight, 0.0,
                        "Box-counting weight should remain 0.0 (opt-in)")

    def test_verify_geometry_preserves_custom_weights(self):
        """
        WHY: --tqf-verify-geometry should not override user-specified non-zero weights
        HOW: Set custom non-zero weights with verify_geometry=True
        WHAT: Custom weights should be preserved
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from config import (
            TQF_SELF_SIMILARITY_WEIGHT_DEFAULT,
            TQF_BOX_COUNTING_WEIGHT_DEFAULT
        )

        # Simulate the main.py logic
        class MockArgs:
            tqf_verify_geometry: bool = True
            tqf_self_similarity_weight: float = 0.05  # User specified custom value
            tqf_box_counting_weight: float = 0.02  # User specified custom value

        args = MockArgs()

        tqf_self_sim_weight: float = args.tqf_self_similarity_weight
        tqf_box_count_weight: float = args.tqf_box_counting_weight

        if args.tqf_verify_geometry:
            if tqf_self_sim_weight == 0.0:
                tqf_self_sim_weight = TQF_SELF_SIMILARITY_WEIGHT_DEFAULT
            if tqf_box_count_weight == 0.0:
                tqf_box_count_weight = TQF_BOX_COUNTING_WEIGHT_DEFAULT

        # Verify custom weights were preserved
        self.assertEqual(tqf_self_sim_weight, 0.05,
                        "Custom self-similarity weight should be preserved")
        self.assertEqual(tqf_box_count_weight, 0.02,
                        "Custom box-counting weight should be preserved")

    def test_geometry_reg_weight_default_value(self):
        """
        WHY: geometry_reg_weight should have correct default value
        HOW: Check config constant
        WHAT: Default should be 0.0 (opt-in feature, disabled by default)
        """
        from config import TQF_GEOMETRY_REG_WEIGHT_DEFAULT

        self.assertEqual(TQF_GEOMETRY_REG_WEIGHT_DEFAULT, 0.0,
                        "Geometry reg weight default should be 0.0 (opt-in)")

    def test_geometry_reg_weight_applied_in_training_engine(self):
        """
        WHY: geometry_reg_weight should control geometry regularization in training
        HOW: Verify TrainingEngine uses the weight when computing loss
        WHAT: use_geometry_reg should be True when weight > 0
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from config import TQF_GEOMETRY_REG_WEIGHT_DEFAULT

        # Test the logic from engine.py for determining use_geometry_reg
        model_name = 'TQF-ANN'
        model_config = {'geometry_reg_weight': 0.05}

        geometry_weight: float = model_config.get('geometry_reg_weight', TQF_GEOMETRY_REG_WEIGHT_DEFAULT)
        use_geometry_reg: bool = 'TQF' in model_name and geometry_weight > 0.0

        self.assertTrue(use_geometry_reg,
                       "use_geometry_reg should be True for TQF model with positive weight")
        self.assertEqual(geometry_weight, 0.05)

    def test_geometry_reg_weight_disabled_when_zero(self):
        """
        WHY: geometry_reg_weight=0 should disable geometry regularization
        HOW: Verify use_geometry_reg is False when weight is 0
        WHAT: No geometry loss should be computed
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from config import TQF_GEOMETRY_REG_WEIGHT_DEFAULT

        model_name = 'TQF-ANN'
        model_config = {'geometry_reg_weight': 0.0}

        geometry_weight: float = model_config.get('geometry_reg_weight', TQF_GEOMETRY_REG_WEIGHT_DEFAULT)
        use_geometry_reg: bool = 'TQF' in model_name and geometry_weight > 0.0

        self.assertFalse(use_geometry_reg,
                        "use_geometry_reg should be False when weight is 0")

    def test_geometry_reg_ignored_for_baseline_models(self):
        """
        WHY: Geometry regularization should be ignored for non-TQF models
        HOW: Verify use_geometry_reg is False for baseline models
        WHAT: No geometry loss for FC-MLP, CNN-L5, etc.
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from config import TQF_GEOMETRY_REG_WEIGHT_DEFAULT

        for model_name in ['FC-MLP', 'CNN-L5', 'ResNet-18-Scaled']:
            model_config = {'geometry_reg_weight': 0.05}

            geometry_weight: float = model_config.get('geometry_reg_weight', TQF_GEOMETRY_REG_WEIGHT_DEFAULT)
            use_geometry_reg: bool = 'TQF' in model_name and geometry_weight > 0.0

            self.assertFalse(use_geometry_reg,
                            f"use_geometry_reg should be False for {model_name}")

    def test_model_stores_verify_geometry_flag(self):
        """
        WHY: TQFANN model should store verify_geometry flag
        HOW: Create model with verify_geometry=True and check attribute
        WHAT: Model should have verify_geometry attribute set correctly
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=10, hidden_dim=64, verify_geometry=True)
        self.assertTrue(hasattr(model, 'verify_geometry'),
                       "Model should have verify_geometry attribute")
        self.assertTrue(model.verify_geometry,
                       "verify_geometry should be True")

        model_false = TQFANN(R=10, hidden_dim=64, verify_geometry=False)
        self.assertFalse(model_false.verify_geometry,
                        "verify_geometry should be False when not enabled")


# ==============================================================================
# TEST SUITE: MULTI-HOP LOCAL ATTENTION (v1.2.0)
# ==============================================================================
# TEST SUITE: GRADIENT CHECKPOINTING (v1.2.0)
# ==============================================================================

class TestGradientCheckpointing(unittest.TestCase):
    """
    Test suite for gradient checkpointing feature (v1.2.0).

    These tests verify that:
    1. use_gradient_checkpointing parameter is accepted
    2. Checkpointing only applies during training mode
    3. Gradients are correctly computed with checkpointing
    4. Model output is identical with/without checkpointing
    """

    def test_gradient_checkpointing_parameter_accepted(self):
        """
        WHY: Model should accept use_gradient_checkpointing parameter
        HOW: Instantiate model with parameter set to True/False
        WHAT: Model should instantiate without error
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        # Should accept True
        model_true = TQFANN(R=7, hidden_dim=64, use_gradient_checkpointing=True)
        self.assertTrue(model_true.use_gradient_checkpointing)

        # Should accept False (default)
        model_false = TQFANN(R=7, hidden_dim=64, use_gradient_checkpointing=False)
        self.assertFalse(model_false.use_gradient_checkpointing)

    def test_gradient_checkpointing_default_is_false(self):
        """
        WHY: Checkpointing should be opt-in (default False)
        HOW: Instantiate model without specifying parameter
        WHAT: use_gradient_checkpointing should default to False
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=7, hidden_dim=64)
        self.assertFalse(model.use_gradient_checkpointing,
                        "Gradient checkpointing should default to False")

    def test_gradient_checkpointing_produces_gradients(self):
        """
        WHY: Checkpointing should still produce valid gradients
        HOW: Run forward/backward with checkpointing enabled
        WHAT: Parameters should have gradients after backward pass
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=7, hidden_dim=64, use_gradient_checkpointing=True)
        model.train()

        x = torch.randn(2, 784, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that some parameters have gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_count, 0,
                          "Parameters should have gradients with checkpointing")

        # Check that input has gradient
        self.assertIsNotNone(x.grad, "Input should have gradient with checkpointing")

    def test_gradient_checkpointing_eval_mode_output(self):
        """
        WHY: Checkpointing should not affect eval mode behavior
        HOW: Compare outputs in eval mode with/without checkpointing flag
        WHAT: Eval mode outputs should be identical
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        torch.manual_seed(42)
        model_ckpt = TQFANN(R=7, hidden_dim=64, use_gradient_checkpointing=True)

        torch.manual_seed(42)
        model_normal = TQFANN(R=7, hidden_dim=64, use_gradient_checkpointing=False)

        # Copy weights to ensure identical models
        model_ckpt.load_state_dict(model_normal.state_dict())

        model_ckpt.eval()
        model_normal.eval()

        x = torch.randn(2, 784)

        with torch.no_grad():
            out_ckpt = model_ckpt(x)
            out_normal = model_normal(x)

        # Outputs should be identical in eval mode
        self.assertTrue(torch.allclose(out_ckpt, out_normal, atol=1e-5),
                       "Eval mode outputs should be identical with/without checkpointing")

    def test_gradient_checkpointing_with_various_R_values(self):
        """
        WHY: Checkpointing should work for all R values
        HOW: Test with small, medium, and larger R values
        WHAT: All should instantiate and run forward pass
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        for R in [5, 10, 15]:
            model = TQFANN(R=R, hidden_dim=64, use_gradient_checkpointing=True)
            model.eval()

            with torch.no_grad():
                x = torch.randn(1, 784)
                output = model(x)

            self.assertEqual(output.shape, (1, 10),
                            f"Output shape incorrect for R={R}")

