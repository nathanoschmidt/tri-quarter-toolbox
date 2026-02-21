"""
test_tqf_ann_integration.py - End-to-End TQF-ANN Integration Tests

This module provides end-to-end integration testing of the TQF-ANN model,
validating that all Priority 1-4 TQF framework corrections work together
correctly during model initialization, forward pass, and verification.

Key Test Coverage:
- Explicit Vertex Storage: Model stores ExplicitLatticeVertex objects with all required fields
- Zone Separation: Boundary (6 vertices), outer, and inner zones properly partitioned
- Phase Pair Integration: Phase pairs computed and accessible throughout model
- Circle Inversion: Inversion map correctly maps outer to inner zone vertices
- Graph Convolution Pipeline: Adjacency-based message passing uses true hexagonal neighbors
- Dual Output Structure: Simultaneous inner/outer zone predictions with correct dimensions
- Forward Pass Validation: Complete input-to-output pipeline with dual metrics
- Verification Methods: Self-duality checks, phase pair preservation, six-coloring validation
- Sector Computation: Angular sectors (0-5) align with 60-degree TQF hexagonal symmetry
- Gradient Flow: Backpropagation through entire TQF-ANN architecture

Test Organization:
- TestTQFANNIntegration: Main integration test class (uses R=5 for speed)
- setUp(): Creates small TQF-ANN model for fast testing (~2-3s initialization)
- All tests: Marked as @pytest.mark.slow due to model initialization time

Scientific Rationale:
Integration tests ensure that individual TQF components (lattice, dual metrics,
inversion, graph convolution) compose correctly into a functioning neural network
that maintains TQF framework guarantees end-to-end.

Note:
This module is marked as slow (@pytest.mark.slow) because TQF-ANN initialization
takes ~10-15 seconds for production config. Use --quick flag to skip during
fast test iterations.

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
import warnings

# Optional pytest import for pytest-specific markers
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

# Import shared utilities
from conftest import TORCH_AVAILABLE, PATHS

# Skip entire module if torch not available - prevents import-time slowdown
# Mark as slow since TQFANN initialization takes 10-15 seconds
if PYTEST_AVAILABLE:
    pytestmark = [
        pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required"),
        pytest.mark.slow
    ]

# Only import torch and TQFANN if actually running tests (not during collection)
if TORCH_AVAILABLE:
    import torch
    # Lazy import - only loaded when tests actually run, not during collection
    from models_tqf import TQFANN


class TestTQFANNIntegration(unittest.TestCase):
    """Test end-to-end TQF-ANN integration with all corrections."""

    def setUp(self) -> None:
        """Create a small TQF-ANN for testing."""
        self.model = TQFANN(
            in_features=784,
            hidden_dim=32,
            num_classes=10,
            R=5.0,  # Small for fast tests
            symmetry_level='Z6',
            fractal_iters=2  # Reduced for speed
        )

    def test_model_has_explicit_vertices(self) -> None:
        """Test that model stores ExplicitLatticeVertex objects."""
        self.assertIsNotNone(self.model.vertices)
        self.assertGreater(len(self.model.vertices), 0)

        # Check first vertex has all required fields
        vertex = self.model.vertices[0]
        self.assertIsNotNone(vertex.vertex_id)
        self.assertIsNotNone(vertex.eisenstein)
        self.assertIsNotNone(vertex.cartesian)
        self.assertIsNotNone(vertex.sector)
        self.assertIsNotNone(vertex.zone)
        self.assertIsNotNone(vertex.norm)
        self.assertIsNotNone(vertex.phase)
        self.assertIsNotNone(vertex.phase_pair)  # phase_pair field required

    def test_model_has_zone_separation(self) -> None:
        """Test that vertices are properly separated into zones."""
        boundary = self.model.boundary_vertices
        outer = self.model.outer_vertices
        inner = self.model.inner_vertices

        self.assertEqual(len(boundary), 6, "Boundary should have 6 vertices for r=1")
        self.assertGreater(len(outer), 0, "Outer zone should not be empty")
        self.assertGreater(len(inner), 0, "Inner zone should not be empty")
        self.assertEqual(len(outer), len(inner), "Outer and inner should be equal size (bijection)")

    def test_model_has_inversion_map(self) -> None:
        """Test that model has proper inversion map."""
        self.assertIsNotNone(self.model.inversion_map)
        self.assertEqual(len(self.model.inversion_map), len(self.model.outer_vertices))

        # Check map is injective
        inner_ids = list(self.model.inversion_map.values())
        self.assertEqual(len(inner_ids), len(set(inner_ids)), "Inversion map should be injective")

    def test_model_has_sector_partitions(self) -> None:
        """Test that model has 6 angular sector partitions covering boundary and outer vertices."""
        self.assertIsNotNone(self.model.sector_partitions)
        self.assertEqual(len(self.model.sector_partitions), 6, "Should have exactly 6 sectors")

        # Check all boundary and outer vertices are in some sector (inner vertices are not)
        total_in_sectors = sum(len(s) for s in self.model.sector_partitions)
        expected_count = len(self.model.boundary_vertices) + len(self.model.outer_vertices)
        self.assertEqual(total_in_sectors, expected_count,
                        "Sectors should cover all boundary and outer vertices")

    def test_verify_phase_pair_consistency(self) -> None:
        """Test phase pair consistency verification method."""
        is_consistent = self.model.verify_phase_pair_consistency(verbose=True)
        self.assertTrue(is_consistent, "All vertices should have consistent phase pairs")

    def test_verify_trihexagonal_coloring(self) -> None:
        """Test trihexagonal six-coloring verification method runs without error.

        NOTE: Coloring validity depends on lattice size; it may not hold at small R
        due to boundary effects. Validity is tested at R=6.0 in
        TestTQFANNSixColoring (test_models_tqf_lattice_integration.py).
        """
        # Verify the method runs without error and returns a bool
        result = self.model.verify_trihexagonal_coloring(verbose=True)
        self.assertIsInstance(result, bool)

    def test_verify_inversion_map_bijection(self) -> None:
        """Test inversion map bijection verification method."""
        is_bijective = self.model.verify_inversion_map_bijection(verbose=True)
        self.assertTrue(is_bijective, "Inversion map should be a proper bijection")

    def test_graph_convolution_end_to_end(self) -> None:
        """Test that graph convolution works end-to-end in forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 784)

        # Run forward pass
        with torch.no_grad():
            logits = self.model(x)

        # Check output shape
        self.assertEqual(logits.shape, (batch_size, 10))

        # Check that RadialBinner used graph adjacency
        self.assertIsNotNone(self.model.radial_binner.adjacency_dict)
        self.assertGreater(len(self.model.radial_binner.adjacency_dict), 0)

    def test_geodesic_verification_end_to_end(self) -> None:
        """Test geodesic distance verification in dual output (Step 5)."""
        batch_size = 2
        x = torch.randn(batch_size, 784)

        # Run forward pass with geometry loss
        with torch.no_grad():
            logits, inv_loss, geom_loss = self.model(
                x,
                return_inv_loss=True,
                return_geometry_loss=True
            )

        # Check outputs
        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertGreaterEqual(inv_loss.item(), 0.0, "Inversion loss should be non-negative")
        self.assertGreaterEqual(geom_loss.item(), 0.0, "Geodesic loss should be non-negative")

        # Geodesic loss should be finite
        self.assertTrue(torch.isfinite(geom_loss), "Geodesic loss should be finite")

    def test_dual_output_with_circle_inversion(self) -> None:
        """Test that dual output properly uses geometric circle inversion."""
        batch_size = 2
        x = torch.randn(batch_size, 784)

        # Get model outputs
        with torch.no_grad():
            logits = self.model(x)

        # Check that dual output head has inversion map
        self.assertIsNotNone(self.model.dual_output.inversion_map)
        self.assertEqual(
            len(self.model.dual_output.inversion_map),
            len(self.model.outer_vertices),
            "Dual output should have complete inversion map"
        )

    def test_all_corrections_verified(self) -> None:
        """Test that all priority corrections are verified."""
        corrections = self.model.verify_corrections(verbose=True)

        # Check that key corrections exist and pass
        # Note: The actual keys may have more specific names
        priority_1_keys = [k for k in corrections.keys() if k.startswith('priority_1_')]
        priority_2_keys = [k for k in corrections.keys() if k.startswith('priority_2_')]
        priority_3_keys = [k for k in corrections.keys() if k.startswith('priority_3_')]
        priority_4_keys = [k for k in corrections.keys() if k.startswith('priority_4_')]

        # Check each priority level has at least one check
        self.assertGreater(len(priority_1_keys), 0, "Should have Priority 1 corrections")
        self.assertGreater(len(priority_2_keys), 0, "Should have Priority 2 corrections")
        self.assertGreater(len(priority_3_keys), 0, "Should have Priority 3 corrections")
        self.assertGreater(len(priority_4_keys), 0, "Should have Priority 4 corrections")

        # All corrections should pass
        failed = [k for k, v in corrections.items() if not v]
        self.assertEqual(len(failed), 0, f"Failed corrections: {failed}")

    def test_get_six_color_batches(self) -> None:
        """Test that six-coloring batches are generated correctly."""
        batches = self.model.get_six_color_batches(batch_size=32)

        self.assertEqual(len(batches), 6, "Should have exactly 6 batches")

        # Check all batches are non-empty
        for i, batch in enumerate(batches):
            self.assertGreater(len(batch), 0, f"Batch {i} should not be empty")

        # Check all vertices are in some batch
        all_vertex_ids = set()
        for batch in batches:
            all_vertex_ids.update(batch)

        expected_ids = {v.vertex_id for v in self.model.vertices}
        self.assertEqual(all_vertex_ids, expected_ids, "All vertices should be in exactly one batch")

    def test_forward_pass_works(self) -> None:
        """Test that forward pass executes without errors."""
        batch_size = 4
        x = torch.randn(batch_size, 784)

        # Standard forward
        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, 10))

        # With inversion loss
        logits, inv_loss = self.model(x, return_inv_loss=True)
        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertIsInstance(inv_loss.item(), float)

    def test_verify_tqf_corrections(self) -> None:
        """Test comprehensive TQF corrections verification."""
        results = self.model.verify_corrections(verbose=True)  # Fixed: correct method name

        # Check all verifications pass
        for key, value in results.items():
            if key in ['priority_1_explicit_lattice', 'priority_2_sector_partitions']:
                self.assertTrue(value, f"Verification failed for {key}")


class TestTQFANNSymmetryLevels(unittest.TestCase):
    """Test TQF-ANN with different symmetry levels."""

    def test_symmetry_none(self) -> None:
        """Test TQF-ANN with no symmetry."""
        model = TQFANN(R=4.0, hidden_dim=16, symmetry_level='none', fractal_iters=1)
        x = torch.randn(2, 784)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))

    def test_symmetry_z6(self) -> None:
        """Test TQF-ANN with Z6 rotational symmetry."""
        model = TQFANN(R=4.0, hidden_dim=16, symmetry_level='Z6', fractal_iters=1)
        x = torch.randn(2, 784)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))

        # Verify Z6 symmetry properties
        self.assertEqual(len(model.sector_partitions), 6)

    def test_symmetry_d6(self) -> None:
        """Test TQF-ANN with D6 dihedral symmetry."""
        model = TQFANN(R=4.0, hidden_dim=16, symmetry_level='D6', fractal_iters=1)
        x = torch.randn(2, 784)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))

    def test_symmetry_t24(self) -> None:
        """Test TQF-ANN with T24 full symmetry."""
        model = TQFANN(R=4.0, hidden_dim=16, symmetry_level='T24', fractal_iters=1)
        x = torch.randn(2, 784)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))


class TestTQFANNScalability(unittest.TestCase):
    """Test TQF-ANN scalability with different R values."""

    def test_small_lattice(self) -> None:
        """Test with small R=3."""
        model = TQFANN(R=3.0, hidden_dim=16, fractal_iters=1)
        self.assertGreater(len(model.vertices), 6)  # More than just boundary

    def test_medium_lattice(self) -> None:
        """Test with medium R=8."""
        model = TQFANN(R=8.0, hidden_dim=16, fractal_iters=1)
        self.assertGreater(len(model.vertices), 100)

    def test_vertex_count_scales_quadratically(self) -> None:
        """Test that vertex count scales approximately as R^2."""
        model_small = TQFANN(R=5.0, hidden_dim=16, fractal_iters=1)
        model_large = TQFANN(R=10.0, hidden_dim=16, fractal_iters=1)

        count_small = len(model_small.vertices)
        count_large = len(model_large.vertices)

        # Should be roughly 4x (since R doubled)
        ratio = count_large / count_small
        self.assertGreater(ratio, 3.0, "Vertex count should scale quadratically")
        self.assertLess(ratio, 5.0, "But not more than 5x due to boundary effects")


class TestTQFANNForwardPassCorrections(unittest.TestCase):
    """Test that forward pass uses TQF corrections properly."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(
            R=5.0,
            hidden_dim=16,
            fractal_iters=1,
                    )

    def test_graph_convolution_uses_adjacency(self) -> None:
        """Test that RadialBinner uses actual lattice adjacency."""
        # Check that radial_binner has adjacency_dict
        self.assertIsNotNone(self.model.radial_binner.adjacency_dict)
        self.assertGreater(len(self.model.radial_binner.adjacency_dict), 0)

    def test_circle_inversion_uses_inversion_map(self) -> None:
        """Test that DualOutputHead uses geometric inversion map."""
        dual_output = self.model.dual_output

        # Check inversion map is set
        self.assertIsNotNone(dual_output.inversion_map)
        self.assertTrue(dual_output.use_geometric_inversion)

        # Test inversion with vertex-level features
        batch_size = 2
        num_outer = len(self.model.outer_vertices)
        hidden_dim = self.model.hidden_dim

        outer_feats = torch.randn(batch_size, num_outer, hidden_dim)
        inner_feats = dual_output.apply_circle_inversion_bijection(outer_feats)

        # Shape should be (batch, num_inner, hidden_dim)
        num_inner = len(self.model.inner_vertices)
        self.assertEqual(inner_feats.shape, (batch_size, num_inner, hidden_dim))

        # Inner features should not be all zeros (mapping worked)
        self.assertGreater(inner_feats.abs().mean().item(), 0.01,
                          "Circle inversion should produce non-zero features")

    def test_forward_pass_with_corrections(self) -> None:
        """Test full forward pass with all corrections active."""
        batch_size = 4
        x = torch.randn(batch_size, 784)

        # Forward pass should work
        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, 10))

        # With inversion loss (tests dual output)
        logits, inv_loss = self.model(x, return_inv_loss=True)
        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertIsInstance(inv_loss.item(), float)
        self.assertGreaterEqual(inv_loss.item(), 0.0, "Inversion loss should be non-negative")

    def test_adjacency_matches_lattice_structure(self) -> None:
        """Test that stored adjacency matches actual lattice."""
        from dual_metrics import build_triangular_lattice_zones

        # Rebuild lattice to compare
        vertices, adjacency, _, _, _, _ = build_triangular_lattice_zones(
            R=self.model.R, r_sq=int(self.model.r ** 2)
        )

        # Check full adjacency dict matches rebuilt lattice
        self.assertEqual(
            len(self.model.adjacency_full),
            len(adjacency)
        )

        # Sample check: pick a few vertices and verify neighbors
        sample_vertices = vertices[:10]
        for vertex in sample_vertices:
            vid = vertex.vertex_id

            # Get neighbors from model (full adjacency)
            model_neighbors = set(self.model.adjacency_full.get(vid, []))

            # Get neighbors from ground truth
            true_neighbors = set(adjacency.get(vid, []))

            self.assertEqual(model_neighbors, true_neighbors,
                           f"Adjacency mismatch for vertex {vid}")


class TestT24SymmetryGroup(unittest.TestCase):
    """Test T24 = D6 x Z2 Inversive Hexagonal Dihedral Symmetry Group implementation."""

    def setUp(self) -> None:
        """Create TQF-ANN models with different symmetry levels."""
        self.model_z6 = TQFANN(
            hidden_dim=32, R=5.0, symmetry_level='Z6',
            fractal_iters=2,         )
        self.model_d6 = TQFANN(
            hidden_dim=32, R=5.0, symmetry_level='D6',
            fractal_iters=2,         )
        self.model_t24 = TQFANN(
            hidden_dim=32, R=5.0, symmetry_level='T24',
            fractal_iters=2,         )

    def test_t24_verification_method_exists(self) -> None:
        """Test that T24 verification method is available."""
        self.assertTrue(hasattr(self.model_t24, 'verify_t24_symmetry_group'))
        self.assertTrue(callable(self.model_t24.verify_t24_symmetry_group))

    def test_t24_z6_subgroup_rotations(self) -> None:
        """Test Z6 cyclic subgroup (6 rotations) of T24."""
        is_valid = self.model_t24.verify_t24_symmetry_group(verbose=False)
        self.assertTrue(is_valid, "T24 verification should pass for Z6 subgroup properties")

        # Verify 6 angular sectors
        self.assertEqual(len(self.model_t24.sector_partitions), 6)

        # All sectors should be nonempty
        for sector_idx, sector in enumerate(self.model_t24.sector_partitions):
            self.assertGreater(len(sector), 0, f"Sector {sector_idx} should not be empty")

    def test_t24_d6_subgroup_reflections(self) -> None:
        """Test D6 dihedral subgroup (rotations + reflections) of T24."""
        # Check sector balance (D6 property)
        sector_counts = [len(s) for s in self.model_t24.sector_partitions]
        mean_count = sum(sector_counts) / 6.0

        for count in sector_counts:
            deviation = abs(count - mean_count) / mean_count
            self.assertLess(deviation, 0.6,
                          f"Sector count {count} deviates too much from mean {mean_count}")

    def test_t24_z2_circle_inversion(self) -> None:
        """Test Z2 subgroup (circle inversion) of T24."""
        # Verify inversion map exists and is bijective
        self.assertIsNotNone(self.model_t24.inversion_map)
        self.assertEqual(len(self.model_t24.outer_vertices), len(self.model_t24.inner_vertices))

        # Verify injection (no duplicate inner vertices)
        inner_ids = list(self.model_t24.inversion_map.values())
        self.assertEqual(len(inner_ids), len(set(inner_ids)), "Inversion map should be injective")

    def test_t24_phase_preservation_under_inversion(self) -> None:
        """Test that circle inversion preserves phase pairs (key T24 property)."""
        from dual_metrics import verify_phase_pair_preservation

        preserved_count = 0
        total_checked = 0

        for outer_id, inner_id in list(self.model_t24.inversion_map.items())[:20]:
            outer_vertex = self.model_t24.vertex_dict[outer_id]
            inner_vertex = self.model_t24.vertex_dict[inner_id]

            if verify_phase_pair_preservation(outer_vertex.phase_pair, inner_vertex.phase_pair):
                preserved_count += 1
            total_checked += 1

        # At least 95% should preserve phase pairs (allowing for numerical precision)
        preservation_rate = preserved_count / total_checked
        self.assertGreater(preservation_rate, 0.95,
                          f"Phase preservation rate {preservation_rate:.2%} too low")

    def test_t24_sector_equivariance(self) -> None:
        """Test sector equivariance under T24 operations."""
        # Vertices in the same sector partition should have consistent sector labels
        for sector_idx, sector_vertices in enumerate(self.model_t24.sector_partitions):
            if len(sector_vertices) == 0:
                continue

            sample_vertex = self.model_t24.vertex_dict[sector_vertices[0]]
            expected_sector = sample_vertex.sector

            # Check first 10 vertices in this sector
            for vertex_id in sector_vertices[:10]:
                vertex = self.model_t24.vertex_dict[vertex_id]
                self.assertEqual(vertex.sector, expected_sector,
                               f"Vertex {vertex_id} in partition {sector_idx} has sector {vertex.sector}, expected {expected_sector}")

    def test_t24_full_verification(self) -> None:
        """Test full T24 symmetry group verification."""
        is_valid = self.model_t24.verify_t24_symmetry_group(verbose=True)
        self.assertTrue(is_valid, "Full T24 verification should pass")

    def test_t24_vs_d6_vs_z6_parameter_counts(self) -> None:
        """Test that T24, D6, and Z6 models have identical parameter counts."""
        params_z6 = self.model_z6.count_parameters()
        params_d6 = self.model_d6.count_parameters()
        params_t24 = self.model_t24.count_parameters()

        # All should have EXACTLY the same parameters
        # (symmetry operations are geometric, not learned)
        self.assertEqual(params_z6, params_d6,
                        f"Z6 ({params_z6:,}) and D6 ({params_d6:,}) should have same parameter count")
        self.assertEqual(params_d6, params_t24,
                        f"D6 ({params_d6:,}) and T24 ({params_t24:,}) should have same parameter count")

    def test_t24_forward_pass_with_inversion(self) -> None:
        """Test that T24 model can perform forward pass with circle inversion."""
        x = torch.randn(2, 784)

        with torch.no_grad():
            output = self.model_t24(x)

        self.assertEqual(output.shape, (2, 10), "T24 forward pass should produce correct output shape")

        # Check output is valid (no NaN or Inf)
        self.assertFalse(torch.isnan(output).any(), "T24 output should not contain NaN")
        self.assertFalse(torch.isinf(output).any(), "T24 output should not contain Inf")


class TestSymmetryOperationsIntegration(unittest.TestCase):
    """
    Test integration of D6/T24 symmetry operations with TQF-ANN model.

    WHY: Validates that new symmetry enforcement features work end-to-end:
         - Feature caching
         - T24 augmentation
         - Equivariance loss computation

    HOW: Creates TQF-ANN model, runs forward pass, verifies new methods work.

    WHAT: Tests for Phase 2 (model integration) and Phase 3 (loss functions).
    """

    def setUp(self) -> None:
        """Create small TQF-ANN for testing."""
        self.model = TQFANN(
            in_features=784,
            hidden_dim=32,
            num_classes=10,
            R=5.0,
            symmetry_level='D6',
            fractal_iters=2,
                    )

    def test_feature_caching(self) -> None:
        """
        Test that sector features are cached after forward pass.

        WHY: Equivariance losses require access to cached sector features.
        HOW: Run forward pass, retrieve cached features via get_cached_sector_features().
        WHAT: Verifies _cached_sector_feats is set and has correct shape.
        """
        batch_size = 2
        x = torch.randn(batch_size, 784)

        # Before forward pass, no cached features
        cached_before = self.model.get_cached_sector_features()
        self.assertIsNone(cached_before, "No features should be cached before forward pass")

        # Run forward pass
        with torch.no_grad():
            _ = self.model(x)

        # After forward pass, features should be cached
        cached_after = self.model.get_cached_sector_features()
        self.assertIsNotNone(cached_after, "Features should be cached after forward pass")
        self.assertEqual(cached_after.shape, (batch_size, 6, self.model.hidden_dim),
                        "Cached features should have shape (batch, 6, hidden_dim)")

    def test_cached_features_detached(self) -> None:
        """
        Test that cached features are detached (no gradient tracking).

        WHY: Cached features should not track gradients to avoid memory overhead.
        HOW: Check requires_grad property of cached features.
        WHAT: Verifies cached_sector_feats.requires_grad is False.
        """
        x = torch.randn(2, 784)
        with torch.no_grad():
            _ = self.model(x)

        cached = self.model.get_cached_sector_features()
        self.assertFalse(cached.requires_grad,
                        "Cached features should be detached (no gradient tracking)")

    def test_t24_augmentation_shapes(self) -> None:
        """
        Test that T24 augmentation preserves feature shapes.

        WHY: T24 augmentation applies geometric transformations to features.
        HOW: Apply random T24 operation, verify output shape matches input.
        WHAT: Tests apply_t24_augmentation() method.
        """
        x = torch.randn(2, 784)
        with torch.no_grad():
            _ = self.model(x)

        sector_feats = self.model.get_cached_sector_features()
        augmented = self.model.apply_t24_augmentation(sector_feats)

        self.assertEqual(augmented.shape, sector_feats.shape,
                        "T24 augmentation should preserve feature shape")

    def test_t24_augmentation_non_identity(self) -> None:
        """
        Test that T24 augmentation modifies features (non-identity).

        WHY: Most T24 operations (23 out of 24) should change features.
        HOW: Apply non-identity T24 operations to synthetic sector features
             with guaranteed sector variation, verify they change features.
        WHAT: Ensures augmentation operations actually apply transformations.

        NOTE: Uses synthetic features because model-generated sector features
        may have identical sectors due to shared weights across sectors.
        """
        from symmetry_ops import sample_random_t24_operation

        # Use synthetic features with guaranteed distinct sectors
        sector_feats = torch.randn(2, 6, self.model.hidden_dim)
        num_changed = 0

        for _ in range(10):
            # Sample non-identity operations to avoid flaky test
            op = sample_random_t24_operation()
            while op.rotation_index == 0 and not op.is_reflected and not op.is_inverted:
                op = sample_random_t24_operation()

            augmented = self.model.apply_t24_augmentation(sector_feats, operation=op)

            # Check if features changed (with small tolerance for numerical precision)
            if not torch.allclose(augmented, sector_feats, atol=1e-6):
                num_changed += 1

        # Non-identity D6 operations (rotation/reflection) should change features
        # Inversion is a no-op on sector-aggregated features, but rotation/reflection
        # permute sectors which changes features when sectors are distinct
        self.assertGreater(num_changed, 0,
                          "Non-identity T24 operations should change synthetic sector features")

    def test_z6_equivariance_loss_computable(self) -> None:
        """
        Test that Z6 equivariance loss can be computed.

        WHY: Z6 loss must be computable for training integration.
        HOW: Run forward pass, compute Z6 loss, verify it's non-negative and finite.
        WHAT: Tests compute_z6_rotation_equivariance_loss() integration.
        """
        from symmetry_ops import compute_z6_rotation_equivariance_loss

        batch_size = 2
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            _ = self.model(x)
            sector_feats = self.model.get_cached_sector_features()

            loss = compute_z6_rotation_equivariance_loss(
                self.model, x, sector_feats, num_rotations=2  # Use 2 for speed
            )

        self.assertIsInstance(loss, torch.Tensor, "Z6 loss should return a tensor")
        self.assertGreaterEqual(loss.item(), 0.0, "Z6 loss should be non-negative")
        self.assertTrue(torch.isfinite(loss), "Z6 loss should be finite")

    def test_d6_equivariance_loss_computable(self) -> None:
        """
        Test that D6 equivariance loss can be computed.

        WHY: D6 loss must be computable for training integration.
        HOW: Run forward pass, compute D6 loss, verify it's non-negative and finite.
        WHAT: Tests compute_d6_reflection_equivariance_loss() integration.
        """
        from symmetry_ops import compute_d6_reflection_equivariance_loss

        batch_size = 2
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            _ = self.model(x)
            sector_feats = self.model.get_cached_sector_features()

            loss = compute_d6_reflection_equivariance_loss(
                self.model, x, sector_feats, num_reflections=2  # Use 2 for speed
            )

        self.assertIsInstance(loss, torch.Tensor, "D6 loss should return a tensor")
        self.assertGreaterEqual(loss.item(), 0.0, "D6 loss should be non-negative")
        self.assertTrue(torch.isfinite(loss), "D6 loss should be finite")

    def test_t24_orbit_invariance_loss_computable(self) -> None:
        """
        Test that T24 orbit invariance loss can be computed.

        WHY: T24 loss must be computable for training integration.
        HOW: Run forward pass, compute T24 loss, verify it's non-negative and finite.
        WHAT: Tests compute_t24_orbit_invariance_loss() integration.
        """
        from symmetry_ops import compute_t24_orbit_invariance_loss

        batch_size = 2
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits = self.model(x)
            sector_feats = self.model.get_cached_sector_features()

            inversion_fn = self.model.dual_output.apply_circle_inversion_bijection
            loss = compute_t24_orbit_invariance_loss(
                sector_feats, logits, num_samples=4, inversion_fn=inversion_fn
            )

        self.assertIsInstance(loss, torch.Tensor, "T24 loss should return a tensor")
        self.assertGreaterEqual(loss.item(), 0.0, "T24 loss should be non-negative")
        self.assertTrue(torch.isfinite(loss), "T24 loss should be finite")


def run_all_tests() -> bool:
    """Run all TQF-ANN integration tests."""
    print("\n" + "=" * 80)
    print("TQF-ANN END-TO-END INTEGRATION TEST SUITE")
    print("Testing: Phase Pairs, Six-Coloring, Inversion, Forward Pass, Symmetry Ops")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test suites
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNSymmetryLevels))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNScalability))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNForwardPassCorrections))
    suite.addTests(loader.loadTestsFromTestCase(TestT24SymmetryGroup))
    suite.addTests(loader.loadTestsFromTestCase(TestSymmetryOperationsIntegration))

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
        print("\n[PASS] ALL TQF-ANN INTEGRATION TESTS PASSED")
        print("  The TQF-ANN model correctly uses:")
        print("  1. ExplicitLatticeVertex with phase_pair field")
        print("  2. Zone separation (boundary, outer, inner)")
        print("  3. Circle inversion bijection map")
        print("  4. Angular sector partitions (6 sectors)")
        print("  5. Trihexagonal six-coloring for parallel processing")
        print("  6. Verification methods for all TQF properties")
        print("  7. T24 Inversive Hexagonal Dihedral Symmetry Group (D6 x Z2)")
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        print("  Please review failures before running experiments.")

    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
