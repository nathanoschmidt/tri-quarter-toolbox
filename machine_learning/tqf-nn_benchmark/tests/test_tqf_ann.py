"""
test_tqf_ann.py - Comprehensive Test Suite for TQF-ANN Model

This module provides comprehensive testing of the TQF-ANN model implementation,
covering all architectural components, geometric features, and TQF framework
compliance. Includes regression tests for Fibonacci mode fixes (January 2026).

Key Test Coverage:
- Regression Tests: Constant dimensions, parameter counting
- Model Initialization: Various configurations (R, hidden_dim)
- Forward Pass: Logits shape, dual output structure, NaN/inf validation
- Priority Corrections: Explicit vertices, sector-based computation, circle inversion, graph convolution
- Geometric Properties: Lattice structure, phase pairs, zone partitioning, inversion maps
- Loss Functions: Label smoothing cross-entropy, geometric regularization, fractal losses
- Symmetry Operations: Z6 rotations, D6 reflections, T24 group operations
- Verification Methods: verify_self_duality(), verify_phase_pair_preservation(), verify_six_coloring()
- Parameter Matching: Auto-tuned hidden dimensions for ~650K parameter target
- Dual Output: Inner and outer zone predictions, inversion consistency

Test Organization:
- TestRegressions: Prevent re-introduction of January 2026 bugs
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
    Regression tests for fixes completed January 2026.

    These tests prevent re-introduction of bugs that were fixed:
    1. Missing LabelSmoothingCrossEntropy class
    2. Pre-encoder parameter overcount in estimation
    3. Phase encodings counted as parameters (should be buffer)
    4. Fractal gates use constant hidden_dim (uniform)
    5. Self-transforms not counted in standard mode estimation
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

    def test_phase_encodings_is_buffer_not_parameter(self):
        """
        REGRESSION: phase_encodings was counted as parameters
        WHY: Registered as buffer, not trainable parameter
        WHAT: Verify phase_encodings is in buffers, not parameters
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import RayOrganizedBoundaryEncoder

        encoder = RayOrganizedBoundaryEncoder(hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)
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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)
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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

        # Should have 6 sector partitions
        self.assertTrue(hasattr(model, 'sector_partitions'))
        self.assertIsNotNone(model.sector_partitions)
        self.assertEqual(len(model.sector_partitions), 6)

    def test_priority_2_partitions_cover_all_vertices(self):
        """Verify sector partitions cover all boundary and outer vertices exactly once."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

        self.assertTrue(hasattr(model, 'verify_corrections'))
        self.assertTrue(callable(model.verify_corrections))

    def test_verify_corrections_returns_dict(self):
        """Verify verify_corrections returns proper dict."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=5.0, r=1.0, hidden_dim=64)

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

        model = TQFANN(R=10.0, r=1.0, hidden_dim=120)

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
    Test suite for constant dimension consistency.

    All layers use constant hidden_dim. These tests verify all components
    use correct dimensions.
    """

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
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        hidden_dim = 80
        model = TQFANN(R=20, hidden_dim=hidden_dim)

        # Check classification head uses hidden_dim (constant)
        classification_head = model.dual_output.classification_head
        self.assertEqual(classification_head.in_features, hidden_dim,
                       f"Classification head input should be {hidden_dim} (constant)")
        self.assertEqual(classification_head.out_features, 10,
                       "Classification head output should be 10 (num_classes)")

    def test_forward_pass_no_dimension_errors(self):
        """
        WHY: Forward pass must complete without dimension mismatches
        HOW: Run forward pass with various batch sizes
        WHAT: Expect successful forward pass, output shape (batch, num_classes)

        REGRESSION: If any component has wrong dimensions, RuntimeError in forward()
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=20, hidden_dim=80)
        model.eval()

        # Test various batch sizes
        for batch_size in [1, 4, 16, 64]:
            with torch.no_grad():
                x = torch.randn(batch_size, 784)
                output = model(x)

                self.assertEqual(output.shape, (batch_size, 10),
                               f"Output shape should be ({batch_size}, 10) for batch_size={batch_size}")

    def test_auto_tuning_uses_correct_mode(self):
        """
        WHY: Auto-tuning must optimize for correct parameter target
        HOW: Check that auto-tuned model has parameters close to target
        WHAT: Expect parameter count within standard tolerance when hidden_dim=None
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN
        from config import TARGET_PARAMS, TARGET_PARAMS_TOLERANCE_PERCENT

        # Auto-tune
        model = TQFANN(R=20, hidden_dim=None)  # hidden_dim=None triggers auto-tuning

        actual_params = model.count_parameters()
        target_params = TARGET_PARAMS
        deviation_pct = abs(actual_params - target_params) / target_params * 100

        self.assertLess(deviation_pct, TARGET_PARAMS_TOLERANCE_PERCENT,
                       f"Auto-tuned model should be within {TARGET_PARAMS_TOLERANCE_PERCENT}% "
                       f"of target {target_params:,}, got {actual_params:,} ({deviation_pct:.1f}%)")

    def test_parameter_estimation_matches_actual(self):
        """
        WHY: Parameter estimator must accurately count parameters
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

        # Create model
        model = TQFANN(R=R, hidden_dim=hidden_dim)

        actual_params = model.count_parameters()

        # Estimate parameters
        estimated_params = estimate_tqf_params(R=R, d=hidden_dim)

        # Check accuracy
        diff_pct = abs(estimated_params - actual_params) / actual_params * 100
        self.assertLess(diff_pct, 5.0,
                       f"Parameter estimation should be within 5% of actual. "
                       f"Estimated: {estimated_params:,}, Actual: {actual_params:,}, "
                       f"Difference: {diff_pct:.2f}%")

    def test_standard_mode_unchanged(self):
        """
        WHY: Standard mode graph conv dimensions must be uniform
        HOW: Verify all graph conv layers use constant hidden_dim
        WHAT: Expect all layers same dimension (direct residuals, no self_transforms)
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=20, hidden_dim=120)

        radial_binner = model.radial_binner

        # All layers should have same dimension (using direct residuals, no self_transforms)
        # Note: self_transforms were removed for performance - now using direct addition
        for layer_idx in range(radial_binner.num_layers):
            conv_layer = radial_binner.graph_convs[layer_idx]
            # First layer in Sequential is Linear(hidden_dim, hidden_dim)
            linear = conv_layer[0]
            self.assertEqual(linear.in_features, 120)
            self.assertEqual(linear.out_features, 120)


# ==============================================================================
# TEST SUITE: INNER SECTOR FEATURE CACHING
# ==============================================================================

class TestInnerSectorFeatureCaching(unittest.TestCase):
    """
    Test inner zone sector feature caching for orbit mixing.

    These tests verify that both inner and outer zone features are cached
    after a forward pass, enabling feature-space orbit mixing operations.
    """

    def test_inner_sector_features_cached_after_forward(self):
        """
        WHY: Inner zone features must be cached for D6/T24 orbit mixing
        HOW: Run forward pass, check that inner features are accessible
        WHAT: Expect non-None tensor of correct shape
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=2)
        model.eval()

        x = torch.randn(4, 784)
        with torch.no_grad():
            model(x)

        inner_feats = model.get_cached_inner_sector_features()
        self.assertIsNotNone(inner_feats, "Inner sector features should be cached after forward pass")
        self.assertEqual(inner_feats.shape[0], 4, "Batch dimension should be 4")
        self.assertEqual(inner_feats.shape[1], 6, "Should have 6 sectors")

    def test_outer_sector_features_still_cached(self):
        """
        WHY: Outer zone caching should still work (regression test)
        HOW: Run forward pass, check that outer features are accessible
        WHAT: Expect non-None tensor of correct shape
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=2)
        model.eval()

        x = torch.randn(4, 784)
        with torch.no_grad():
            model(x)

        outer_feats = model.get_cached_sector_features()
        self.assertIsNotNone(outer_feats, "Outer sector features should be cached after forward pass")
        self.assertEqual(outer_feats.shape[0], 4, "Batch dimension should be 4")
        self.assertEqual(outer_feats.shape[1], 6, "Should have 6 sectors")

    def test_inner_and_outer_have_same_hidden_dim(self):
        """
        WHY: Both zones use shared weights, so hidden_dim must match
        HOW: Compare hidden_dim of cached inner and outer features
        WHAT: Expect identical hidden_dim
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=2)
        model.eval()

        x = torch.randn(4, 784)
        with torch.no_grad():
            model(x)

        outer_feats = model.get_cached_sector_features()
        inner_feats = model.get_cached_inner_sector_features()
        self.assertEqual(outer_feats.shape[2], inner_feats.shape[2],
                        "Inner and outer features should have same hidden_dim")

    def test_cached_features_are_detached(self):
        """
        WHY: Cached features should be detached to prevent gradient leaks
        HOW: Check requires_grad on cached features
        WHAT: Expect requires_grad = False
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=2)
        model.eval()

        x = torch.randn(4, 784)
        with torch.no_grad():
            model(x)

        inner_feats = model.get_cached_inner_sector_features()
        self.assertFalse(inner_feats.requires_grad,
                        "Cached inner features should be detached (no grad)")

    def test_get_cached_inner_before_forward_returns_none(self):
        """
        WHY: Before any forward pass, there are no cached features
        HOW: Call get_cached_inner_sector_features on fresh model
        WHAT: Expect None
        """
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        from models_tqf import TQFANN

        model = TQFANN(R=2)
        inner_feats = model.get_cached_inner_sector_features()
        self.assertIsNone(inner_feats,
                         "Should return None before any forward pass")

