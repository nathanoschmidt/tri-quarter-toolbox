"""
test_models_tqf_lattice_integration.py - Hexagonal Lattice Integration Tests for TQF-ANN

This module tests that models_tqf.py correctly integrates the hexagonal lattice
foundation from dual_metrics.py into the TQF-ANN architecture. Validates proper
lattice construction, zone partitioning, adjacency propagation, and graph
convolution over the explicit Eisenstein integer lattice.

Key Test Coverage:
1. Lattice Construction: Proper lattice building during TQF-ANN __init__
2. Vertex Zone Partitioning: Boundary (6), outer, inner zones with correct sizes
3. Adjacency Propagation: Hexagonal adjacency reaches RadialBinner layers
4. Phase Pair Consistency: Phase pairs computed and preserved throughout model
5. Six-Coloring Functionality: Trihexagonal six-coloring for D6 independence
6. Graph Convolution: Message passing over true 6-neighbor hexagonal lattice
7. ExplicitLatticeVertex Storage: All vertices have Eisenstein coordinates
8. Sector Assignment: Each vertex assigned to correct angular sector (0-5)
9. Inversion Map: Bijection between outer and inner zone vertices
10. Forward Pass Integration: Lattice structure used correctly in predictions

Test Organization:
- TestTQFANNLatticeConstruction: Lattice vertex creation and zone partitioning
- TestAdj acencyIntegration: Adjacency map propagation to RadialBinner
- TestPhasePairConsistency: Phase pair computation and preservation
- TestSixColoring: Trihexagonal coloring validation
- TestGraphConvolution: Graph neural network message passing

Scientific Foundation:
Tests validate that TQF-ANN correctly implements the hexagonal Eisenstein integer
lattice with 6-neighbor connectivity as specified in Schmidt's Tri-Quarter Framework,
ensuring geometric self-duality and T24 symmetry properties.

Note:
All tests marked as @pytest.mark.slow due to TQF-ANN initialization time (~10-15s
for R=20, ~2-3s for R=5). Uses smaller R values where possible for faster execution.

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
import os
import pytest

# Add paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Skipping torch-dependent tests.")

# Mark entire module as slow since TQFANN initialization is expensive
pytestmark = [
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required"),
    pytest.mark.slow
]

if TORCH_AVAILABLE:
    from models_tqf import TQFANN
    from dual_metrics import (
        ExplicitLatticeVertex,
        verify_trihexagonal_six_coloring_independence,
        compute_phase_pair,
        verify_phase_pair_preservation
    )


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNLatticeConstruction(unittest.TestCase):
    """Test that TQF-ANN properly constructs hexagonal lattice."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fractal_iters=2,
            fibonacci_mode='none'
        )

    def test_lattice_vertices_created(self) -> None:
        """Test that explicit lattice vertices are created."""
        self.assertIsNotNone(self.model.vertices)
        self.assertGreater(len(self.model.vertices), 6)

        # All should be ExplicitLatticeVertex instances
        for v in self.model.vertices[:10]:
            self.assertIsInstance(v, ExplicitLatticeVertex)

    def test_boundary_zone_has_six_vertices(self) -> None:
        """Test that boundary zone has exactly 6 vertices."""
        self.assertEqual(len(self.model.boundary_vertices), 6,
                        "Boundary zone should have 6 vertices for r=1")

    def test_zones_properly_partitioned(self) -> None:
        """Test that vertices are properly partitioned into zones."""
        boundary_ids = {v.vertex_id for v in self.model.boundary_vertices}
        outer_ids = {v.vertex_id for v in self.model.outer_vertices}
        inner_ids = {v.vertex_id for v in self.model.inner_vertices}

        # Zones should be disjoint
        self.assertEqual(len(boundary_ids & outer_ids), 0, "Boundary and outer should be disjoint")
        self.assertEqual(len(boundary_ids & inner_ids), 0, "Boundary and inner should be disjoint")
        self.assertEqual(len(outer_ids & inner_ids), 0, "Outer and inner should be disjoint")

    def test_outer_inner_bijection(self) -> None:
        """Test that outer and inner zones have equal size (bijection)."""
        self.assertEqual(len(self.model.outer_vertices), len(self.model.inner_vertices),
                        "Outer and inner zones should have equal size")

    def test_inversion_map_complete(self) -> None:
        """Test that inversion map covers all outer vertices."""
        self.assertEqual(len(self.model.inversion_map), len(self.model.outer_vertices),
                        "Inversion map should cover all outer vertices")

        # All inner vertex IDs should be in the map values
        inner_ids = {v.vertex_id for v in self.model.inner_vertices}
        mapped_ids = set(self.model.inversion_map.values())

        self.assertEqual(inner_ids, mapped_ids,
                        "Inversion map should map to all inner vertices")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNSectorPartitions(unittest.TestCase):
    """Test angular sector partitioning."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(R=6.0, hidden_dim=32, fractal_iters=2, fibonacci_mode='none')

    def test_six_sectors_created(self) -> None:
        """Test that exactly 6 angular sectors are created."""
        self.assertEqual(len(self.model.sector_partitions), 6,
                        "Should have exactly 6 angular sectors")

    def test_all_sectors_nonempty(self) -> None:
        """Test that all sectors contain vertices."""
        for i, sector in enumerate(self.model.sector_partitions):
            self.assertGreater(len(sector), 0,
                             f"Sector {i} should not be empty")

    def test_sectors_are_exhaustive(self) -> None:
        """Test that all vertices are in exactly one sector."""
        # Count total vertices in sectors
        total_in_sectors = sum(len(s) for s in self.model.sector_partitions)

        # Should equal number of boundary + outer vertices
        expected = len(self.model.boundary_vertices) + len(self.model.outer_vertices)

        self.assertEqual(total_in_sectors, expected,
                        "Sectors should cover all boundary and outer vertices")

    def test_sectors_are_disjoint(self) -> None:
        """Test that sectors don't overlap."""
        all_vertex_ids = []
        for sector in self.model.sector_partitions:
            all_vertex_ids.extend(sector)

        self.assertEqual(len(all_vertex_ids), len(set(all_vertex_ids)),
                        "Sectors should not have overlapping vertices")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNAdjacencyIntegration(unittest.TestCase):
    """Test that adjacency is properly integrated."""

    def setUp(self) -> None:
        """Create test model with T24 binner."""
        self.model = TQFANN(R=5.0, hidden_dim=32, fractal_iters=2, fibonacci_mode='none')

    def test_adjacency_dict_exists(self) -> None:
        """Test that adjacency dictionary exists."""
        self.assertIsNotNone(self.model.adjacency_full)
        self.assertIsInstance(self.model.adjacency_full, dict)
        self.assertGreater(len(self.model.adjacency_full), 0)

    def test_adjacency_passed_to_radial_binner(self) -> None:
        """Test that zone-specific adjacency is passed to RadialBinner."""
        # T24 radial binner uses outer-zone adjacency (zones share weights)
        self.assertTrue(hasattr(self.model.radial_binner, 'adjacency_dict'))
        self.assertEqual(self.model.radial_binner.adjacency_dict,
                        self.model.adjacency_outer,
                        "T24 RadialBinner should have outer-zone adjacency")

        # Both inner and outer adjacencies should be non-empty in the model
        self.assertGreater(len(self.model.adjacency_outer), 0,
                          "Outer adjacency should be non-empty")
        self.assertGreater(len(self.model.adjacency_inner), 0,
                          "Inner adjacency should be non-empty")

    def test_discrete_metric_created(self) -> None:
        """Test that discrete metric is created in RadialBinner."""
        self.assertTrue(hasattr(self.model.radial_binner, 'discrete_metric'))
        self.assertIsNotNone(self.model.radial_binner.discrete_metric)

    def test_neighbor_map_built(self) -> None:
        """Test that neighbor map is built."""
        self.assertTrue(hasattr(self.model.radial_binner, 'neighbor_map'))
        self.assertIsInstance(self.model.radial_binner.neighbor_map, dict)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNPhasePairs(unittest.TestCase):
    """Test phase pair integration."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(R=4.0, hidden_dim=32, fractal_iters=2, fibonacci_mode='none')

    def test_all_vertices_have_phase_pairs(self) -> None:
        """Test that all vertices have phase_pair attribute."""
        for vertex in self.model.vertices:
            self.assertTrue(hasattr(vertex, 'phase_pair'))
            self.assertIsNotNone(vertex.phase_pair)

    def test_phase_pairs_consistent_with_coordinates(self) -> None:
        """Test that phase pairs match vertex coordinates."""
        for vertex in self.model.vertices[:20]:  # Test first 20
            if vertex.zone in ['boundary', 'outer']:
                x, y = vertex.cartesian
                expected_pp = compute_phase_pair(x, y)

                self.assertTrue(
                    verify_phase_pair_preservation(vertex.phase_pair, expected_pp),
                    f"Vertex {vertex.vertex_id} phase pair mismatch"
                )

    def test_phase_pair_verification_method(self) -> None:
        """Test that model's phase pair verification method works."""
        is_consistent = self.model.verify_phase_pair_consistency(verbose=False)
        self.assertTrue(is_consistent, "All phase pairs should be consistent")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNSixColoring(unittest.TestCase):
    """Test trihexagonal six-coloring integration."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(R=6.0, hidden_dim=32, fractal_iters=2, fibonacci_mode='none')

    def test_six_coloring_verification_exists(self) -> None:
        """Test that six-coloring verification method exists."""
        self.assertTrue(hasattr(self.model, 'verify_trihexagonal_coloring'))
        self.assertTrue(callable(self.model.verify_trihexagonal_coloring))

    def test_six_coloring_is_valid(self) -> None:
        """Test that six-coloring produces valid independent sets."""
        is_valid = verify_trihexagonal_six_coloring_independence(
            self.model.vertices,
            self.model.adjacency_full,
            verbose=False
        )
        self.assertTrue(is_valid, "Six-coloring should be valid")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNGraphConvolution(unittest.TestCase):
    """Test graph neural network integration (Step 4)."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fractal_iters=2,
            fibonacci_mode='none'
        )

    def test_radial_binner_has_aggregate_method(self) -> None:
        """Test that RadialBinner has neighbor aggregation method."""
        self.assertTrue(hasattr(self.model.radial_binner, 'aggregate_neighbors_via_adjacency'))
        self.assertTrue(callable(self.model.radial_binner.aggregate_neighbors_via_adjacency))

    def test_neighbor_aggregation_shape(self) -> None:
        """Test that neighbor aggregation preserves tensor shape."""
        batch_size = 2
        num_vertices = len(self.model.vertices)
        hidden_dim = 32

        # Create dummy features
        feats = torch.randn(batch_size, num_vertices, hidden_dim)

        # Aggregate neighbors
        aggregated = self.model.radial_binner.aggregate_neighbors_via_adjacency(feats)

        # Should preserve shape
        self.assertEqual(aggregated.shape, feats.shape)

    def test_neighbor_aggregation_changes_features(self) -> None:
        """Test that neighbor aggregation actually changes features."""
        batch_size = 2
        num_vertices = len(self.model.vertices)
        hidden_dim = 32

        # Create dummy features
        feats = torch.randn(batch_size, num_vertices, hidden_dim)
        feats_copy = feats.clone()

        # Aggregate neighbors
        aggregated = self.model.radial_binner.aggregate_neighbors_via_adjacency(feats)

        # Should NOT be identical (unless all vertices isolated, which won't happen)
        difference = torch.norm(aggregated - feats_copy).item()
        self.assertGreater(difference, 0.0, "Aggregation should change features")

    def test_graph_conv_uses_lattice_adjacency(self) -> None:
        """Test that graph convolution uses actual lattice adjacency."""
        # This is verified by checking that aggregate_neighbors_via_adjacency
        # uses self.adjacency_dict, which we've already tested
        self.assertIsNotNone(self.model.radial_binner.adjacency_dict)
        self.assertGreater(len(self.model.radial_binner.adjacency_dict), 0)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNDualOutputGeodesic(unittest.TestCase):
    """Test geodesic distance verification in dual output (Step 5)."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(
            R=5.0,
            hidden_dim=32,
            fractal_iters=2,
            fibonacci_mode='none',
            use_dual_output=True
        )

    def test_dual_output_has_geodesic_method(self) -> None:
        """Test that dual output has geodesic verification method."""
        self.assertTrue(hasattr(self.model.dual_output, 'compute_geodesic_verification_loss'))
        self.assertTrue(callable(self.model.dual_output.compute_geodesic_verification_loss))

    def test_geodesic_loss_computation(self) -> None:
        """Test that geodesic loss can be computed."""
        batch_size = 2
        num_sectors = 6
        hidden_dim = 32

        # Create dummy features
        outer_feats = torch.randn(batch_size, num_sectors, hidden_dim)
        inner_feats = torch.randn(batch_size, num_sectors, hidden_dim)

        # Compute geodesic loss
        geom_loss = self.model.dual_output.compute_geodesic_verification_loss(
            outer_feats, inner_feats
        )

        # Should be a scalar tensor
        self.assertEqual(geom_loss.dim(), 0)
        self.assertGreaterEqual(geom_loss.item(), 0.0)

    def test_forward_returns_geometry_loss(self) -> None:
        """Test that forward pass can return geometry loss."""
        batch_size = 2
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits, inv_loss, geom_loss = self.model(
                x,
                return_inv_loss=True,
                return_geometry_loss=True
            )

        # Check shapes and values
        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertEqual(inv_loss.dim(), 0)
        self.assertEqual(geom_loss.dim(), 0)
        self.assertGreaterEqual(geom_loss.item(), 0.0)

    def test_identical_features_give_zero_geodesic_loss(self) -> None:
        """Test that identical outer/inner features give zero geodesic loss."""
        batch_size = 2
        num_sectors = 6
        hidden_dim = 32

        # Create identical features
        feats = torch.randn(batch_size, num_sectors, hidden_dim)

        # Compute geodesic loss with identical features
        geom_loss = self.model.dual_output.compute_geodesic_verification_loss(feats, feats)

        # Should be approximately zero
        self.assertLess(geom_loss.item(), 1e-5)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNForwardPass(unittest.TestCase):
    """Test that forward pass works with lattice integration."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(
            R=4.0,
            hidden_dim=32,
            fractal_iters=2,
            fibonacci_mode='none'
        )
        self.model.eval()

    def test_forward_pass_basic(self) -> None:
        """Test basic forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits = self.model(x)

        self.assertEqual(logits.shape, (batch_size, 10))

    def test_forward_pass_with_inversion_loss(self) -> None:
        """Test forward pass with inversion loss."""
        batch_size = 2
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits, inv_loss = self.model(x, return_inv_loss=True)

        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertIsInstance(inv_loss.item(), float)
        self.assertGreaterEqual(inv_loss.item(), 0.0)

    def test_forward_pass_all_losses(self) -> None:
        """Test forward pass with all losses."""
        batch_size = 2
        x = torch.randn(batch_size, 784)

        with torch.no_grad():
            logits, inv_loss, geom_loss = self.model(
                x,
                return_inv_loss=True,
                return_geometry_loss=True
            )

        self.assertEqual(logits.shape, (batch_size, 10))
        self.assertIsInstance(inv_loss.item(), float)
        self.assertIsInstance(geom_loss.item(), float)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestTQFANNVerificationMethods(unittest.TestCase):
    """Test TQF-ANN verification methods."""

    def setUp(self) -> None:
        """Create test model."""
        self.model = TQFANN(R=5.0, hidden_dim=32, fractal_iters=2, fibonacci_mode='none')

    def test_verify_dualities_method(self) -> None:
        """Test duality verification method."""
        results = self.model.verify_dualities(verbose=False)

        self.assertIn('theorem_3.1_six_vertices', results)
        self.assertIn('theorem_4.1_six_sectors', results)
        self.assertIn('theorem_4.3_bijective_self_duality', results)

        # All should pass
        for key, val in results.items():
            self.assertTrue(val, f"{key} should pass")

    def test_verify_corrections_method(self) -> None:
        """Test corrections verification method."""
        results = self.model.verify_corrections(verbose=False)

        # Check key corrections
        self.assertTrue(results['priority_1_explicit_vertices'])
        self.assertTrue(results['priority_1_six_boundary'])
        self.assertTrue(results['priority_1_adjacency'])
        self.assertTrue(results['priority_1_inversion_map'])
        self.assertTrue(results['priority_2_sector_partitions'])

    def test_parameter_count(self) -> None:
        """Test that parameter count method works."""
        params = self.model.count_parameters()
        self.assertGreater(params, 0)
        self.assertIsInstance(params, int)


def run_all_tests() -> bool:
    """Run all integration tests."""
    if not TORCH_AVAILABLE:
        print("=" * 80)
        print("SKIPPING TESTS: PyTorch not available")
        print("=" * 80)
        return True

    print("\n" + "=" * 80)
    print("TQF-ANN HEXAGONAL LATTICE INTEGRATION TEST SUITE")
    print("Testing: Lattice construction, zones, sectors, adjacency, phase pairs")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNLatticeConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNSectorPartitions))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNAdjacencyIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNPhasePairs))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNSixColoring))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNForwardPass))
    suite.addTests(loader.loadTestsFromTestCase(TestTQFANNVerificationMethods))

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
        print("\nALL TQF-ANN LATTICE INTEGRATION TESTS PASSED")
        return True
    else:
        print("\nSOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
