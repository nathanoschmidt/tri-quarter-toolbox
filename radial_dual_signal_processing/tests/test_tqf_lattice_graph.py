"""
test_tqf_lattice_graph.py - Lattice Graph and Trihexagonal Six-Coloring Tests (Claim C5)

Tests for tqf_lattice_graph.py -- truncated triangular lattice graph and the
trihexagonal six-coloring underpinning claim C5 (conflict-free parallel recovery).

The graph dict exposes: num_vertices, num_edges, coords (N,2), neighbor_idx
(N,6 with sentinel == num_vertices for empty slots), neighbor_cnt (N,),
color_classes (list of 6 index arrays), and proper (bool).

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.0.0
Date: June 24, 2026
"""
import numpy as np
import pytest

import tqf_lattice_graph as g

RADII = [10, 20, 40]


def _colour_array(graph):
    """Reconstruct a per-vertex colour in {0..5} from the color_classes partition."""
    colour = np.full(graph["num_vertices"], -1, dtype=int)
    for c, idx in enumerate(graph["color_classes"]):
        colour[idx] = c
    return colour


def _undirected_edges(graph):
    """Reconstruct the undirected edge set from neighbor_idx (sentinel == N)."""
    n = graph["num_vertices"]
    nbr = graph["neighbor_idx"]
    edges = set()
    for i in range(n):
        for j in nbr[i]:
            j = int(j)
            if 0 <= j < n:
                edges.add((min(i, j), max(i, j)))
    return edges


@pytest.mark.parametrize("radius", RADII)
def test_six_coloring_is_proper_flag(radius):
    assert g.build_lattice_graph(radius)["proper"] is True


@pytest.mark.parametrize("radius", RADII)
def test_six_coloring_proper_against_edges(radius):
    # Independent re-verification: no edge joins two equally-coloured vertices.
    graph = g.build_lattice_graph(radius)
    colour = _colour_array(graph)
    assert colour.min() >= 0  # every vertex coloured
    for i, j in _undirected_edges(graph):
        assert colour[i] != colour[j]


@pytest.mark.parametrize("radius", RADII)
def test_color_classes_partition_all_vertices(radius):
    graph = g.build_lattice_graph(radius)
    assert len(graph["color_classes"]) == 6
    sizes = sum(len(c) for c in graph["color_classes"])
    assert sizes == graph["num_vertices"]
    allidx = np.concatenate(graph["color_classes"])
    assert len(np.unique(allidx)) == graph["num_vertices"]  # disjoint + complete


def test_neighbour_offsets_change_c3():
    # Every nearest-neighbour offset changes (a - b) mod 3, which is exactly why
    # the 3-coloring c3 -- and hence the refined 6-coloring -- is proper.
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    for da, db in offsets:
        assert (da - db) % 3 != 0


@pytest.mark.parametrize("radius", RADII)
def test_max_degree_at_most_six(radius):
    graph = g.build_lattice_graph(radius)
    assert int(graph["neighbor_cnt"].max()) <= 6
    assert int(graph["neighbor_cnt"].min()) >= 1


@pytest.mark.parametrize("radius", RADII)
def test_edge_count_consistent(radius):
    graph = g.build_lattice_graph(radius)
    assert len(_undirected_edges(graph)) == graph["num_edges"]
    assert graph["coords"].shape[0] == graph["num_vertices"]


def test_relaxation_sweep_reduces_mse():
    # The color-ordered diffusion sweep should denoise a smooth field.
    graph = g.build_lattice_graph(30)
    coords = graph["coords"]
    rng = np.random.default_rng(0)
    truth = np.sin(coords[:, 0] * 0.15) * np.cos(coords[:, 1] * 0.15)
    noisy = truth + rng.normal(0.0, 0.5, truth.shape)
    state = noisy.copy()
    for _ in range(20):
        state = g.relaxation_sweep_numpy(
            state, graph["neighbor_idx"], graph["neighbor_cnt"],
            graph["color_classes"], alpha=0.5,
        )
    assert np.mean((state - truth) ** 2) < np.mean((noisy - truth) ** 2)


def test_relaxation_sweep_is_deterministic():
    graph = g.build_lattice_graph(20)
    rng = np.random.default_rng(1)
    field = rng.normal(0.0, 1.0, graph["num_vertices"])

    def run():
        s = field.copy()
        for _ in range(5):
            s = g.relaxation_sweep_numpy(
                s, graph["neighbor_idx"], graph["neighbor_cnt"],
                graph["color_classes"], alpha=0.5,
            )
        return s

    assert np.array_equal(run(), run())
