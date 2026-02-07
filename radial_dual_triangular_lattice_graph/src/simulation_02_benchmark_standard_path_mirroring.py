# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 02: Benchmarking the Standard Approach for Path Mirroring
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
#
# Description:
# This Python script benchmarks the standard (recompute) approach to path
# mirroring in the truncated radial dual triangular lattice graph Lambda_r^R
# (with admissible inversion radius r = 1 and configurable truncation radius R).
# It computes single-source shortest paths (via the discrete dual metric) from
# a random start vertex in the outer zone subgraph Lambda_{+,r}^R, then
# recomputes equivalent paths in the inner zone subgraph Lambda_{-,r}^R from
# the corresponding inverted start vertex under the circle inversion map iota_r.
# This provides a baseline for comparing against the Tri-Quarter duality-based
# mirroring approach. Times are averaged over multiple runs for reliability and
# reported in ms.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
# Run the script with: python simulation_02_benchmark_standard_path_mirroring.py R
# (e.g., python simulation_02_benchmark_standard_path_mirroring.py 10 for R=10)
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import networkx as nx
import time
import random
import argparse
import statistics

from radial_dual_triangular_lattice_graph import build_zone_subgraphs

# Benchmark the standard path computation in the dual zones (recompute inner
# from scratch). Computes shortest paths from a random outer start vertex,
# then recomputes equivalent paths in the inner zone from the inverted start
# vertex for fairness.
def benchmark_standard_path_mirroring(G_outer, G_inner, start_outer,
                                      inversion_map, runs, timing_repeats):
    times = []  # List to store execution times
    for _ in range(runs):
        t0 = time.perf_counter()  # Start timer
        for _ in range(timing_repeats):  # Repeat for noise reduction
            # Outer paths: compute single-source shortest path lengths
            # in Lambda_{+,r}^R via the discrete dual metric (hop counts)
            nx.single_source_shortest_path_length(G_outer, start_outer)
            # Recompute inner paths: using inverted start for fairness
            # (preserves phase pair constancy along graph rays)
            start_inner = inversion_map.get(start_outer)
            if start_inner:
                nx.single_source_shortest_path_length(G_inner, start_inner)
        # Average time per dual-zone computation in ms
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)
    # Return mean and std dev
    return statistics.mean(times), statistics.stdev(times)

if __name__ == "__main__":
    # Parse arguments for configurable runs
    parser = argparse.ArgumentParser(
        description="Benchmark path mirroring on a truncated Tri-Quarter radial "
                    "dual triangular lattice graph with standard approach "
                    "(recompute inner paths). This provides a baseline for "
                    "duality methods. Times are in milliseconds (ms)."
    )
    parser.add_argument("R", type=int, nargs="?", default=10,
                        help="Truncation radius (default: 10)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of benchmark runs (default: 20)")
    parser.add_argument("--timing_repeats", type=int, default=100,
                        help="Repeats per run for accuracy (default: 100)")

    args = parser.parse_args()

    # Build graphs
    print(f"Building radial dual triangular lattice graph with "
          f"truncation radius R={args.R}...")
    G_outer, G_inner, inversion_map = build_zone_subgraphs(args.R)
    num_outer = len(G_outer.nodes())
    num_inner = len(G_inner.nodes())
    print(f"Graphs built: Outer {num_outer} vertices, Inner {num_inner} vertices.")

    # Random outer start
    start_outer = random.choice(list(G_outer.nodes())) if num_outer > 0 else None

    print(f"Running {args.runs} benchmarks, each with {args.timing_repeats} "
          f"repeats for reliable timing.")

    # Run and display results
    avg, std = benchmark_standard_path_mirroring(G_outer, G_inner, start_outer,
                                                 inversion_map, args.runs,
                                                 args.timing_repeats)
    print(f"Standard Path Mirroring (Recompute): {avg:.2f} ms (+/-{std:.2f})")
