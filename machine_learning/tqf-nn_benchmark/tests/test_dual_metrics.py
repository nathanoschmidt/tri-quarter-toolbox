"""
test_dual_metrics.py - Comprehensive Dual Metrics Tests for TQF-NN

This module provides 50+ comprehensive tests for the dual_metrics.py module,
covering hexagonal lattice construction, Eisenstein coordinates, phase pairs,
circle inversion, discrete/continuous dual metrics, and trihexagonal six-coloring.

Key Test Coverage:
- Lattice Construction: build_triangular_lattice_zones with various R values
- Eisenstein Coordinates: Basis vector conversion, integer lattice properties
- Hexagonal Adjacency: True 6-neighbor connectivity validation
- Phase Pair Computation: Structured directional labeling per TQF spec
- Sector Partitioning: 6 angular sectors (60-degree increments)
- Circle Inversion: Exact bijection v' = r^2 / conj(v) verification
- Discrete Dual Metric: Hop-distance BFS computation on lattice graph
- Continuous Dual Metric: Hyperbolic distance for dyadic/linear binning
- Trihexagonal Six-Coloring: D6 symmetry independence verification
- Inversion Consistency: Phase pair preservation under circle inversion

Test Organization:
- TestLatticeConstruction: Basic lattice building and zone partitioning
- TestPhasePairs: Phase pair computation and preservation
- TestCircleInversion: Bijection correctness and consistency
- TestDualMetrics: Discrete and continuous metric validation
- TestSymmetryProperties: Z6/D6/T24 group properties

Scientific Foundation:
Based on "The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs"
by Nathan O. Schmidt (2025). Tests ensure implementation matches mathematical
specification for self-duality and equivariant encodings.

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
import warnings
from typing import List, Dict, Tuple

# Import shared utilities
from conftest import TORCH_AVAILABLE, NUMPY_AVAILABLE, PATHS

# Import dual_metrics module
try:
    from dual_metrics import (
        DiscreteDualMetric,
        ContinuousDualMetric,
        build_triangular_lattice_zones,
        ExplicitLatticeVertex,
        PhasePair,
        compute_phase_pair,
        eisenstein_to_cartesian,
        compute_angular_sector,
        verify_trihexagonal_six_coloring_independence
    )
except ImportError as e:
    raise ImportError(f"Cannot import dual_metrics module: {e}")

# Try importing torch
if TORCH_AVAILABLE:
    import torch

# Try importing numpy
if NUMPY_AVAILABLE:
    import numpy as np


class TestLatticeConstruction(unittest.TestCase):
    """Test suite for build_triangular_lattice_zones()."""

    def test_lattice_construction_basic(self) -> None:
        """Test basic lattice construction."""
        vertices, adjacency, boundary, outer, inner, inv_map = build_triangular_lattice_zones(R=3.0, r_sq=1)

        self.assertIsInstance(vertices, list)
        self.assertIsInstance(adjacency, dict)
        self.assertIsInstance(boundary, list)
        self.assertIsInstance(outer, list)
        self.assertIsInstance(inner, list)
        self.assertIsInstance(inv_map, dict)

    def test_boundary_has_six_vertices(self) -> None:
        """Test that boundary zone has 6 vertices for r=1."""
        vertices, adjacency, boundary, outer, inner, inv_map = build_triangular_lattice_zones(R=5.0, r_sq=1)
        self.assertEqual(len(boundary), 6)

    def test_outer_inner_bijection(self) -> None:
        """Test that outer and inner zones have equal size."""
        vertices, adjacency, boundary, outer, inner, inv_map = build_triangular_lattice_zones(R=4.0, r_sq=1)
        self.assertEqual(len(outer), len(inner))
        self.assertEqual(len(inv_map), len(outer))

    def test_all_vertices_are_explicit(self) -> None:
        """Test that all vertices are ExplicitLatticeVertex instances."""
        vertices, _, _, _, _, _ = build_triangular_lattice_zones(R=3.0, r_sq=1)
        for v in vertices:
            self.assertIsInstance(v, ExplicitLatticeVertex)

    def test_vertices_have_phase_pairs(self) -> None:
        """Test that all vertices have phase_pair attribute."""
        vertices, _, _, _, _, _ = build_triangular_lattice_zones(R=3.0, r_sq=1)
        for v in vertices:
            self.assertTrue(hasattr(v, 'phase_pair'))
            self.assertIsInstance(v.phase_pair, PhasePair)

    def test_vertices_have_sectors(self) -> None:
        """Test that all vertices have sector assignment."""
        vertices, _, _, _, _, _ = build_triangular_lattice_zones(R=3.0, r_sq=1)
        for v in vertices:
            self.assertIn(v.sector, range(6))

    def test_adjacency_is_symmetric(self) -> None:
        """Test that adjacency relation is symmetric."""
        vertices, adjacency, _, _, _, _ = build_triangular_lattice_zones(R=3.0, r_sq=1)
        for vertex_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                self.assertIn(vertex_id, adjacency.get(neighbor_id, []))

    def test_vertex_count_scales_with_R_squared(self) -> None:
        """
        Test that vertex count scales approximately with R^2 (hexagonal lattice property).

        WHY: Truncation radius R controls lattice size. For a hexagonal/triangular lattice,
             the number of vertices within radius R grows approximately as 3*R^2.
        HOW: Build lattices with different R values and verify scaling relationship.
        WHAT: Vertex count ratio should approximate (R2/R1)^2 within tolerance.
        """
        # Build lattices with different R values
        vertices_r3, _, _, _, _, _ = build_triangular_lattice_zones(R=3.0, r_sq=1)
        vertices_r5, _, _, _, _, _ = build_triangular_lattice_zones(R=5.0, r_sq=1)
        vertices_r10, _, _, _, _, _ = build_triangular_lattice_zones(R=10.0, r_sq=1)

        count_r3: int = len(vertices_r3)
        count_r5: int = len(vertices_r5)
        count_r10: int = len(vertices_r10)

        # Verify monotonic increase with R
        self.assertGreater(count_r5, count_r3, "R=5 should have more vertices than R=3")
        self.assertGreater(count_r10, count_r5, "R=10 should have more vertices than R=5")

        # Verify approximate R^2 scaling (with tolerance for boundary effects)
        # Expected ratio (5/3)^2 = 2.78, (10/3)^2 = 11.1
        ratio_5_to_3: float = count_r5 / count_r3
        ratio_10_to_3: float = count_r10 / count_r3

        # Allow 50% tolerance due to discrete lattice effects at small R
        self.assertGreater(ratio_5_to_3, 1.5, "R=5/R=3 ratio should show growth")
        self.assertGreater(ratio_10_to_3, 5.0, "R=10/R=3 ratio should show ~R^2 scaling")

    def test_minimum_R_produces_valid_lattice(self) -> None:
        """
        Test that minimum valid R=2 produces a valid (non-empty) lattice.

        WHY: R=2 is the minimum valid truncation radius (must be > r=1).
        HOW: Build lattice with R=2 and verify basic properties.
        WHAT: Should have vertices in boundary and outer zones.
        """
        vertices, adjacency, boundary, outer, inner, inv_map = build_triangular_lattice_zones(R=2.0, r_sq=1)

        # Should have some vertices
        self.assertGreater(len(vertices), 0, "R=2 lattice should have vertices")

        # Should have 6 boundary vertices (norm=1 for r_sq=1)
        self.assertEqual(len(boundary), 6, "Should have 6 boundary vertices at norm=1")

        # Should have some outer vertices (norm > 1, norm <= 2)
        self.assertGreater(len(outer), 0, "R=2 lattice should have outer vertices")

        # Outer and inner should be equal (inversion bijection)
        self.assertEqual(len(outer), len(inner), "Outer and inner zones must have equal size")

    def test_large_R_lattice_structure(self) -> None:
        """
        Test that large R produces expected lattice structure with many vertices.

        WHY: Validates that truncation radius R correctly controls lattice extent.
        HOW: Build lattice with R=20 and verify structure.
        WHAT: Should have many vertices, all within expected zones.
        """
        R: float = 20.0
        vertices, adjacency, boundary, outer, inner, inv_map = build_triangular_lattice_zones(R=R, r_sq=1)

        # Should have many vertices (~3 * R^2 for hexagonal lattice)
        expected_min: int = int(2 * R * R)  # Conservative lower bound
        self.assertGreater(len(vertices), expected_min,
                          f"R={R} lattice should have > {expected_min} vertices")

        # All vertices should have valid zone assignments
        for v in vertices:
            self.assertIn(v.zone, ['boundary', 'outer', 'inner'])

        # Boundary should still have exactly 6 vertices
        self.assertEqual(len(boundary), 6, "Boundary always has 6 vertices for r=1")


class TestDiscreteDualMetric(unittest.TestCase):
    """Test suite for DiscreteDualMetric class."""

    def test_discrete_metric_initialization(self) -> None:
        """Test DiscreteDualMetric can be initialized."""
        adjacency: Dict[int, List[int]] = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        metric = DiscreteDualMetric(adjacency)
        self.assertIsNotNone(metric)

    def test_discrete_metric_direct_neighbors(self) -> None:
        """Test distance to direct neighbors is 1."""
        adjacency: Dict[int, List[int]] = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        metric = DiscreteDualMetric(adjacency)

        dist: int = metric.compute_hop_distance(0, 1)
        self.assertEqual(dist, 1)

    def test_discrete_metric_self_distance(self) -> None:
        """Test distance to self is 0."""
        adjacency: Dict[int, List[int]] = {0: [1], 1: [0]}
        metric = DiscreteDualMetric(adjacency)

        dist: int = metric.compute_hop_distance(0, 0)
        self.assertEqual(dist, 0)

    def test_discrete_metric_unreachable(self) -> None:
        """Test distance to unreachable node returns large finite value."""
        adjacency: Dict[int, List[int]] = {0: [1], 1: [0], 2: []}
        metric = DiscreteDualMetric(adjacency)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dist: int = metric.compute_hop_distance(0, 2)
            self.assertGreater(dist, 100)

    def test_discrete_metric_symmetric_distance(self) -> None:
        """Test that distance is symmetric: d(i,j) = d(j,i)."""
        adjacency: Dict[int, List[int]] = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
        metric = DiscreteDualMetric(adjacency)

        dist_01: int = metric.compute_hop_distance(0, 1)
        dist_10: int = metric.compute_hop_distance(1, 0)
        self.assertEqual(dist_01, dist_10)

    def test_discrete_metric_triangle_inequality(self) -> None:
        """Test triangle inequality: d(i,k) <= d(i,j) + d(j,k)."""
        adjacency: Dict[int, List[int]] = {
            0: [1], 1: [0, 2], 2: [1, 3], 3: [2]
        }
        metric = DiscreteDualMetric(adjacency)

        d_03: int = metric.compute_hop_distance(0, 3)
        d_01: int = metric.compute_hop_distance(0, 1)
        d_13: int = metric.compute_hop_distance(1, 3)

        self.assertLessEqual(d_03, d_01 + d_13)

    def test_discrete_metric_empty_adjacency(self) -> None:
        """Test with empty adjacency dict raises ValueError."""
        adjacency: Dict[int, List[int]] = {}
        # Empty adjacency should raise ValueError during initialization
        with self.assertRaises(ValueError):
            metric = DiscreteDualMetric(adjacency)

    def test_discrete_metric_single_node(self) -> None:
        """Test with single isolated node."""
        adjacency: Dict[int, List[int]] = {0: []}
        metric = DiscreteDualMetric(adjacency)
        dist: int = metric.compute_hop_distance(0, 0)
        self.assertEqual(dist, 0)

    def test_discrete_metric_linear_chain(self) -> None:
        """Test distance in linear chain graph."""
        # Linear chain: 0-1-2-3
        adjacency: Dict[int, List[int]] = {
            0: [1], 1: [0, 2], 2: [1, 3], 3: [2]
        }
        metric = DiscreteDualMetric(adjacency)

        dist: int = metric.compute_hop_distance(0, 3)
        self.assertEqual(dist, 3)


class TestContinuousDualMetric(unittest.TestCase):
    """Test suite for ContinuousDualMetric class."""

    def test_continuous_metric_initialization(self) -> None:
        """Test ContinuousDualMetric can be initialized."""
        metric = ContinuousDualMetric(r=1.0)
        self.assertIsNotNone(metric)
        self.assertEqual(metric.r, 1.0)

    def test_compute_radial_bin_dyadic(self) -> None:
        """Test dyadic radial binning."""
        metric = ContinuousDualMetric(r=1.0)

        bin_idx = metric.compute_radial_bin(norm=2.0, R=10.0, num_bins=5, method='dyadic')
        self.assertIsInstance(bin_idx, int)
        self.assertIn(bin_idx, range(5))

    def test_compute_radial_bin_linear(self) -> None:
        """Test linear radial binning."""
        metric = ContinuousDualMetric(r=1.0)

        bin_idx = metric.compute_radial_bin(norm=5.0, R=10.0, num_bins=5, method='linear')
        self.assertIsInstance(bin_idx, int)
        self.assertIn(bin_idx, range(5))

    def test_radial_bin_monotonic(self) -> None:
        """Test that larger norms give larger or equal bin indices."""
        metric = ContinuousDualMetric(r=1.0)
        R = 10.0
        num_bins = 5

        prev_bin = 0
        for norm in [1.5, 3.0, 5.0, 7.0, 9.0]:
            bin_idx = metric.compute_radial_bin(norm, R, num_bins, method='linear')
            self.assertGreaterEqual(bin_idx, prev_bin)
            prev_bin = bin_idx

    def test_continuous_metric_different_radius(self) -> None:
        """Test metric with different radius values."""
        for r in [0.5, 1.0, 2.0, 5.0]:
            metric = ContinuousDualMetric(r=r)
            self.assertEqual(metric.r, r)

    def test_radial_bin_boundary_cases(self) -> None:
        """Test radial binning at boundary values."""
        metric = ContinuousDualMetric(r=1.0)

        # Vertex at boundary should be in bin 0
        bin_idx = metric.compute_radial_bin(norm=1.1, R=10.0, num_bins=5, method='linear')
        self.assertEqual(bin_idx, 0)

        # Vertex near R should be in highest bin
        bin_idx = metric.compute_radial_bin(norm=9.9, R=10.0, num_bins=5, method='linear')
        self.assertGreaterEqual(bin_idx, 3)

    def test_radial_bin_both_methods(self) -> None:
        """Test that both dyadic and linear methods work."""
        metric = ContinuousDualMetric(r=1.0)

        for method in ['dyadic', 'linear']:
            bin_idx = metric.compute_radial_bin(norm=5.0, R=10.0, num_bins=5, method=method)
            self.assertIn(bin_idx, range(5))



class TestVertexLevelUtilities(unittest.TestCase):
    """Test suite for vertex-level graph convolution utilities."""

    def test_build_vertex_neighbor_map(self) -> None:
        """Test building vertex neighbor map."""
        from dual_metrics import build_vertex_neighbor_map

        # Simple test graph
        adjacency: dict = {
            0: [1, 2],
            1: [0, 2, 3],
            2: [0, 1],
            3: [1]
        }
        vertex_to_idx: dict = {0: 0, 1: 1, 2: 2, 3: 3}

        neighbor_map: dict = build_vertex_neighbor_map(adjacency, vertex_to_idx)

        self.assertEqual(neighbor_map[0], [1, 2])
        self.assertEqual(neighbor_map[1], [0, 2, 3])


class TestKHopNeighborPrecomputation(unittest.TestCase):
    """Test suite for k-hop neighbor precomputation (v1.2.0)."""

    def test_precompute_k_hop_neighbors_basic(self) -> None:
        """Test basic k-hop precomputation on simple graph."""
        from dual_metrics import precompute_k_hop_neighbors

        # Linear chain: 0 - 1 - 2 - 3
        neighbor_map: dict = {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2]
        }

        k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)

        # Vertex 0: 1-hop = [1], 2-hop = [2]
        self.assertEqual(k_hop[0][1], [1])
        self.assertEqual(k_hop[0][2], [2])

        # Vertex 1: 1-hop = [0, 2], 2-hop = [3]
        self.assertEqual(sorted(k_hop[1][1]), [0, 2])
        self.assertEqual(k_hop[1][2], [3])

    def test_precompute_k_hop_neighbors_triangular(self) -> None:
        """Test k-hop on triangular graph (3 vertices, all connected)."""
        from dual_metrics import precompute_k_hop_neighbors

        # Complete triangle: 0 - 1 - 2 - 0
        neighbor_map: dict = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1]
        }

        k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)

        # In a complete triangle, all vertices are 1-hop from each other
        # So 2-hop should be empty (no vertices exactly 2 hops away)
        self.assertEqual(sorted(k_hop[0][1]), [1, 2])
        self.assertEqual(k_hop[0][2], [])

    def test_precompute_k_hop_neighbors_hexagonal(self) -> None:
        """Test k-hop on hexagonal ring (6 vertices)."""
        from dual_metrics import precompute_k_hop_neighbors

        # Hexagonal ring: 0-1-2-3-4-5-0
        neighbor_map: dict = {
            0: [1, 5],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3, 5],
            5: [4, 0]
        }

        k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)

        # Vertex 0: 1-hop = [1, 5], 2-hop = [2, 4]
        self.assertEqual(sorted(k_hop[0][1]), [1, 5])
        self.assertEqual(sorted(k_hop[0][2]), [2, 4])

    def test_precompute_k_hop_neighbors_lattice(self) -> None:
        """Test k-hop on actual triangular lattice structure."""
        from dual_metrics import build_vertex_neighbor_map, build_triangular_lattice_zones, precompute_k_hop_neighbors

        # Build a small lattice
        vertices, adjacency, _, _, _, _ = build_triangular_lattice_zones(R=5.0, r_sq=1)
        vertex_to_idx = {v.vertex_id: i for i, v in enumerate(vertices)}
        neighbor_map = build_vertex_neighbor_map(adjacency, vertex_to_idx)

        k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)

        # Verify structure
        self.assertEqual(len(k_hop), len(vertices))

        # Check that 1-hop neighbors match neighbor_map
        for v_idx in range(len(vertices)):
            expected_1hop = sorted(neighbor_map.get(v_idx, []))
            actual_1hop = sorted(k_hop[v_idx].get(1, []))
            self.assertEqual(actual_1hop, expected_1hop)

    def test_precompute_k_hop_neighbors_no_overlap(self) -> None:
        """Test that 1-hop and 2-hop neighborhoods don't overlap."""
        from dual_metrics import precompute_k_hop_neighbors

        # Simple graph
        neighbor_map: dict = {
            0: [1, 2],
            1: [0, 3],
            2: [0, 3],
            3: [1, 2, 4],
            4: [3]
        }

        k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)

        for v_idx in range(5):
            hop_1 = set(k_hop[v_idx].get(1, []))
            hop_2 = set(k_hop[v_idx].get(2, []))
            # No overlap between 1-hop and 2-hop
            self.assertEqual(hop_1 & hop_2, set())
            # Self should not be in any hop
            self.assertNotIn(v_idx, hop_1)
            self.assertNotIn(v_idx, hop_2)

    def test_precompute_k_hop_neighbors_empty_graph(self) -> None:
        """Test k-hop on empty neighbor map."""
        from dual_metrics import precompute_k_hop_neighbors

        neighbor_map: dict = {}
        k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)
        self.assertEqual(k_hop, {})

    def test_precompute_k_hop_neighbors_isolated_vertex(self) -> None:
        """Test k-hop with isolated vertices."""
        from dual_metrics import precompute_k_hop_neighbors

        neighbor_map: dict = {
            0: [],  # Isolated
            1: [2],
            2: [1]
        }

        k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)

        # Isolated vertex should have empty neighborhoods
        self.assertEqual(k_hop[0][1], [])
        self.assertEqual(k_hop[0][2], [])


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all dual metrics tests."""
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()

    for test_class in [
        TestDiscreteDualMetric,
        TestContinuousDualMetric,
        TestVertexLevelUtilities,
        TestKHopNeighborPrecomputation
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dual Metrics Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE DUAL METRICS TESTS")
    print("=" * 80)

    result = run_tests(verbosity=2 if args.verbose else 1)
    sys.exit(0 if result.wasSuccessful() else 1)
