# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 02: Benchmarking the Standard (Recompute) Approach for Path Mirroring
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
# Last Updated: June 10, 2026
# Version: 1.1.0
#
# Description:
# This script establishes the standard (recompute) baseline for the dual-zone
# shortest-path problem on the truncated radial dual triangular lattice graph
# Lambda_r^R (admissible inversion radius r = 1, configurable truncation
# radius R). The dual-zone problem is: given a source vertex in the outer zone
# subgraph Lambda_{+,r}^R and its inversion twin in the inner zone subgraph
# Lambda_{-,r}^R, obtain single-source shortest-path hop distances (the
# discrete dual metric) in BOTH zones.
#
# The standard approach solves this by running a full single-source
# shortest-path computation independently in each zone. It performs no
# symmetry exploitation and serves as the reference against which the
# Tri-Quarter inversion-based approach (Simulation 03) is measured. Both
# simulations are defined on the identical dual-zone problem, so the timings
# are directly comparable, and Simulation 03 verifies its inner-zone result
# bitwise-equal to the recomputation performed here.
#
# Times are averaged over multiple runs with inner timing repeats and reported
# in milliseconds.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
#   python simulation_02_benchmark_standard_path_mirroring.py R [--runs N]
#                                                              [--timing_repeats M]
# Example:
#   python simulation_02_benchmark_standard_path_mirroring.py 15 --runs 20 --timing_repeats 100
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import time
import random
import argparse
import statistics

import networkx as nx

from radial_dual_triangular_lattice_graph import build_zone_subgraphs


def standard_dual_zone_paths(G_outer, G_inner, start_outer, inversion_map):
    """Solve the dual-zone shortest-path problem by full recomputation.

    Computes single-source shortest-path hop distances (the discrete dual
    metric) in the outer zone from start_outer, and independently in the inner
    zone from the corresponding inversion twin start_inner. Returns both
    distance dictionaries so the result can be verified against the
    inversion-based approach of Simulation 03.

    Args:
        G_outer: Outer zone subgraph Lambda_{+,r}^R.
        G_inner: Inner zone subgraph Lambda_{-,r}^R.
        start_outer: Source vertex in the outer zone.
        inversion_map: Bijection mapping outer vertices to inner twins.

    Returns:
        tuple (outer_dist, inner_dist) of {vertex: hop_distance} dictionaries.
    """
    outer_dist = nx.single_source_shortest_path_length(G_outer, start_outer)
    start_inner = inversion_map.get(start_outer)
    inner_dist = (
        nx.single_source_shortest_path_length(G_inner, start_inner)
        if start_inner is not None
        else {}
    )
    return outer_dist, inner_dist


def benchmark_standard_path_mirroring(G_outer, G_inner, start_outer,
                                      inversion_map, runs, timing_repeats):
    """Time the standard recompute approach over multiple runs.

    Each run times timing_repeats solutions of the dual-zone problem and
    records the mean per-solution wall-clock time in milliseconds. Returns the
    mean and standard deviation across runs.
    """
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(timing_repeats):
            standard_dual_zone_paths(
                G_outer, G_inner, start_outer, inversion_map
            )
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return statistics.mean(times), std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the standard (recompute) baseline for the dual-zone "
            "shortest-path problem on a truncated radial dual triangular "
            "lattice graph Lambda_r^R. Times are in milliseconds (ms)."
        )
    )
    parser.add_argument("R", type=int, nargs="?", default=10,
                        help="Truncation radius R (default: 10)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of benchmark runs (default: 20)")
    parser.add_argument("--timing_repeats", type=int, default=100,
                        help="Repeats per run for accuracy (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for source selection (default: 42)")
    args = parser.parse_args()

    print(f"Building radial dual triangular lattice graph with "
          f"truncation radius R={args.R}...")
    G_outer, G_inner, inversion_map = build_zone_subgraphs(args.R)
    num_outer = len(G_outer.nodes())
    num_inner = len(G_inner.nodes())
    print(f"Graphs built: Outer {num_outer} vertices, "
          f"Inner {num_inner} vertices.")

    # Fixed seed so the standard and Tri-Quarter benchmarks select the same
    # source vertex and are therefore directly comparable and verifiable.
    random.seed(args.seed)
    start_outer = (
        random.choice(sorted(G_outer.nodes())) if num_outer > 0 else None
    )

    print(f"Running {args.runs} benchmarks, each with {args.timing_repeats} "
          f"timing repeats.")
    avg, std = benchmark_standard_path_mirroring(
        G_outer, G_inner, start_outer, inversion_map,
        args.runs, args.timing_repeats
    )
    print(f"Standard Path Mirroring (Recompute): {avg:.3f} ms (+/-{std:.3f})")