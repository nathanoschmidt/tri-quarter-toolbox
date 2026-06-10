# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 04: Benchmarking the Standard Clustering Coefficient Computation
#                (EXACT RATIONAL ARITHMETIC)
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
# Last Updated: June 10, 2026
# Version: 1.1.0
#
# Description:
# This script establishes the standard (full recompute) baseline for the
# average local clustering coefficient on the complete truncated radial dual
# triangular lattice graph Lambda_r^R (admissible inversion radius r = 1,
# configurable truncation radius R).
#
# The average local clustering coefficient of Lambda_r^R is an exact rational
# number: each local coefficient is 2*common / (deg*(deg-1)) with integer
# common and deg, and the average is a sum of rationals divided by the vertex
# count. This script therefore computes the coefficient in exact rational
# arithmetic (fractions.Fraction), so the reported value carries no
# floating-point round-off and is bitwise-reproducible. The symmetry-reduced
# Tri-Quarter approach (Simulation 05) computes the identical exact rational,
# which the two scripts verify by exact (==) comparison rather than by a
# floating-point tolerance.
#
# Times are averaged over multiple runs with inner timing repeats and reported
# in milliseconds.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
#   python simulation_04_benchmark_standard_clustering.py R [--runs N]
#                                                           [--timing_repeats M]
# Example:
#   python simulation_04_benchmark_standard_clustering.py 100 --runs 20 --timing_repeats 20
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import time
import argparse
import statistics
from fractions import Fraction

from radial_dual_triangular_lattice_graph import build_complete_lattice_graph


def build_adjacency_sets(G):
    """Return a {vertex: frozenset(neighbors)} adjacency-set view of G.

    Precomputing this view once lets the clustering routine perform triangle
    counting through O(1) average-case set membership tests, keeping the
    baseline's per-vertex constant factor small.
    """
    return {v: frozenset(G.neighbors(v)) for v in G.nodes()}


def local_clustering_exact(adjacency, v):
    """Exact local clustering coefficient of vertex v as a Fraction.

    The value is 2 * (edges among neighbors) / (deg * (deg - 1)). Both the
    numerator and denominator are integers, so the coefficient is an exact
    rational with no floating-point round-off. Vertices of degree below 2
    contribute exactly Fraction(0).
    """
    neigh = adjacency[v]
    deg = len(neigh)
    if deg < 2:
        return Fraction(0)
    neigh_list = tuple(neigh)
    common = 0
    for i in range(deg):
        ni_adj = adjacency[neigh_list[i]]
        for j in range(i + 1, deg):
            if neigh_list[j] in ni_adj:
                common += 1
    return Fraction(2 * common, deg * (deg - 1))


def compute_average_clustering_standard_exact(adjacency):
    """Exact average local clustering coefficient over all vertices.

    Accumulates the per-vertex exact rational coefficients and divides by the
    vertex count, returning a single exact Fraction. Because the arithmetic is
    exact, the result is independent of summation order and is therefore
    bitwise-reproducible across implementations.

    Args:
        adjacency: {vertex: frozenset(neighbors)} adjacency-set view.

    Returns:
        The average local clustering coefficient as an exact Fraction
        (Fraction(0) if the graph is empty).
    """
    num_v = len(adjacency)
    if num_v == 0:
        return Fraction(0)
    total = Fraction(0)
    for v in adjacency:
        total += local_clustering_exact(adjacency, v)
    return total / num_v


def benchmark_standard_clustering(adjacency, runs, timing_repeats):
    """Time the exact standard clustering computation over multiple runs.

    Each run times timing_repeats full-graph exact clustering computations and
    records the mean per-computation wall-clock time in milliseconds. Returns
    the mean and standard deviation across runs.
    """
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(timing_repeats):
            compute_average_clustering_standard_exact(adjacency)
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return statistics.mean(times), std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the standard (full recompute) average local "
            "clustering coefficient on the complete truncated radial dual "
            "triangular lattice graph Lambda_r^R, in exact rational "
            "arithmetic. Times are in milliseconds."
        )
    )
    parser.add_argument("R", type=int, nargs="?", default=10,
                        help="Truncation radius R (default: 10)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of benchmark runs (default: 20)")
    parser.add_argument("--timing_repeats", type=int, default=20,
                        help="Repeats per run for accuracy (default: 20)")
    args = parser.parse_args()

    G, _ = build_complete_lattice_graph(args.R)
    adjacency = build_adjacency_sets(G)
    num_v = len(adjacency)
    print(f"Graph: |V|={num_v}")

    coeff = compute_average_clustering_standard_exact(adjacency)
    print(f"Average clustering coefficient (exact): {coeff} = {float(coeff):.12f}")

    avg, std = benchmark_standard_clustering(
        adjacency, args.runs, args.timing_repeats
    )
    print(f"Standard (exact): {avg:.3f} ms +/- {std:.3f}")
