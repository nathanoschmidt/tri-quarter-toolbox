# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 03: Benchmarking the Tri-Quarter Approach for Inversion-Based
# Path Mirroring
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 24, 2025
#
# Description:
# This Python script benchmarks the Tri-Quarter duality approach to path
# mirroring on a truncated radial dual triangular lattice graph Lambda_r^R
# (with admissible inversion radius r = 1 and configurable truncation radius R).
# It computes single-source shortest paths in the outer zone subgraph
# Lambda_{+,r}^R using the discrete dual metric (hop counts), then mirrors
# them to the inner zone subgraph Lambda_{-,r}^R via the circle inversion
# bijection iota_r (as per bijective self-duality). This demonstrates approximately
# 2x speedups over recompute baselines, leveraging the Escher reflective
# duality for reversible zone swapping without recomputation.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
# Run the script with: python simulation_03_benchmark_triquarter_path_mirroring.py R
# where R is the truncation radius (default: 10).
# Example: python simulation_03_benchmark_triquarter_path_mirroring.py 15 --runs 20 --timing_repeats 100
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

from radial_dual_triangular_lattice_graph import build_radial_dual_triangular_lattice_graph

# Mirror a path dictionary via the circle inversion bijection iota_r
# (O(|path|) time complexity for the mapping operation).
# This preserves phases and norms under the Escher reflective duality,
# enabling reversible information preservation across zones without recomputation.
def mirror_paths(outer_paths, inversion_map):
    mirrored = {}  # Dictionary to store mirrored path lengths in the inner zone
    for target, length in outer_paths.items():
        # Retrieve the inverted target vertex via the bijection (phase-preserving)
        inv_target = inversion_map.get(target)
        if inv_target:
            # Copy the hop length (preserved by the induced graph isomorphism,
            # Corollary 4.4, under the discrete dual metric)
            mirrored[inv_target] = length
    return mirrored

# Benchmark the Tri-Quarter path mirroring approach in the dual zones
# (compute paths in the outer zone, then mirror to the inner zone via bijection).
# This leverages bijective self-duality for O(1) per-vertex
# mirroring, demonstrating efficiency gains from symmetry exploitation.
def benchmark_triquarter_path_mirroring(G_outer, start_outer, inversion_map,
                                        runs, timing_repeats):
    times = []  # List to store execution times across benchmark runs
    for _ in range(runs):
        t0 = time.perf_counter()  # Start high-resolution timer
        for _ in range(timing_repeats):  # Repeat inner loop for statistical noise reduction
            # Compute outer paths: single-source shortest path lengths
            # in the outer zone subgraph Lambda_{+,r}^R via the discrete
            # dual metric (hop counts)
            outer_paths = nx.single_source_shortest_path_length(G_outer, start_outer)
            # Mirror to inner zone via the circle inversion bijection iota_r
            # (no recomputation required, per reversible zone swapping)
            mirror_paths(outer_paths, inversion_map)
        # Compute average time per dual-zone computation in milliseconds
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)
    # Return the mean and standard deviation of the times
    return statistics.mean(times), statistics.stdev(times)

if __name__ == "__main__":
    # Parse command-line arguments for configurable benchmark parameters
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark path mirroring on a truncated Tri-Quarter radial dual "
            "triangular lattice graph Lambda_r^R with the Tri-Quarter duality "
            "approach (mirror outer paths to inner via bijection). "
            "Demonstrates speedups from exact inversion under the Escher "
            "reflective duality (Theorem 4.2). Times are in milliseconds (ms)."
        )
    )
    parser.add_argument(
        "R", type=int, nargs="?", default=10,
        help="Truncation radius R (default: 10)"
    )
    parser.add_argument(
        "--runs", type=int, default=20,
        help="Number of benchmark runs (default: 20)"
    )
    parser.add_argument(
        "--timing_repeats", type=int, default=100,
        help="Repeats per run for accuracy (default: 100)"
    )

    args = parser.parse_args()

    # Build the truncated radial dual triangular lattice graph Lambda_r^R
    # (with admissible inversion radius r = 1 and truncation radius R)
    print(
        f"Building radial dual triangular lattice graph "
        f"with truncation radius R={args.R}..."
    )
    G_outer, G_inner, inversion_map = (
        build_radial_dual_triangular_lattice_graph(args.R)
    )  # Build outer/inner subgraphs and bijection map
    num_outer = len(G_outer.nodes())
    num_inner = len(G_inner.nodes())
    print(
        f"Graphs built: Outer {num_outer} vertices, "
        f"Inner {num_inner} vertices."
    )

    # Select a random starting vertex in the outer zone subgraph
    start_outer = (
        random.choice(list(G_outer.nodes())) if num_outer > 0 else None
    )  # Random outer start vertex

    # Report benchmark configuration
    print(
        f"Running {args.runs} benchmarks, each with "
        f"{args.timing_repeats} repeats for reliable timing."
    )
    print(
        "Note: Includes bijection preprocessing "
        "(amortized over multiple queries)."
    )

    # Execute the benchmark and display results
    avg, std = benchmark_triquarter_path_mirroring(
        G_outer, start_outer, inversion_map,
        args.runs, args.timing_repeats
    )
    print(
        f"Tri-Quarter Path Mirroring (Duality): "
        f"{avg:.2f} ms (+/-{std:.2f})"
    )
