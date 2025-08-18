# Tri-Quarter Framework: Simulates and benchmarks the standard (recompute) approach
# to inversion-based path mirroring.
# Version: 1.0
# Authors: Nathan O. Schmidt and Grok
# Created Date: August 17, 2025
# Modified Date: August 17, 2025
# License: MIT (see LICENSE file)

import networkx as nx
import time
import random
import argparse
import statistics

from triangular_lattice_graph import build_radial_dual_triangular_lattice_graph

# Benchmark standard path computation in dual zones (recompute inner from scratch).
# Computes shortest paths from a random outer start, then recomputes equivalent in inner.
def benchmark_standard_path_mirroring(G_outer, G_inner, start_outer, runs, timing_repeats):
    times = []  # List to store execution times
    for _ in range(runs):
        t0 = time.perf_counter()  # Start timer
        for _ in range(timing_repeats):  # Repeat for noise reduction
            # Outer paths
            nx.single_source_shortest_path_length(G_outer, start_outer)
            # Recompute inner paths (random equivalent start in inner)
            start_inner = random.choice(list(G_inner.nodes()))
            nx.single_source_shortest_path_length(G_inner, start_inner)
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)  # Average time per dual-zone computation in ms
    return statistics.mean(times), statistics.stdev(times)  # Return mean and std dev

if __name__ == "__main__":
    # Parse arguments for configurable runs
    parser = argparse.ArgumentParser(
        description="Benchmark path mirroring on a truncated Tri-Quarter radial dual triangular lattice graph "
                    "with standard approach (recompute inner paths). This provides a baseline for duality methods. "
                    "Times are in milliseconds (ms)."
    )
    parser.add_argument("R", type=int, nargs="?", default=10, help="Truncation radius (default: 10)")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs (default: 20)")
    parser.add_argument("--timing_repeats", type=int, default=100, help="Repeats per run for accuracy (default: 100)")

    args = parser.parse_args()

    print(f"Building radial dual lattice graph with truncation radius R={args.R}...")
    G_outer, G_inner, _ = build_radial_dual_triangular_lattice_graph(args.R)  # Build graphs
    num_outer = len(G_outer.nodes())
    num_inner = len(G_inner.nodes())
    print(f"Graphs built: Outer {num_outer} vertices, Inner {num_inner} vertices.")

    start_outer = random.choice(list(G_outer.nodes())) if num_outer > 0 else None  # Random outer start

    print(f"Running {args.runs} benchmarks, each with {args.timing_repeats} repeats for reliable timing.")

    # Run and display results
    avg, std = benchmark_standard_path_mirroring(G_outer, G_inner, start_outer, args.runs, args.timing_repeats)
    print(f"Standard Path Mirroring (Recompute): {avg:.2f} ms (+/-{std:.2f})")
