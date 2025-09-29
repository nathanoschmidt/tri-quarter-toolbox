# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 04: Benchmarking Standard Clustering Coefficient Computation
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
#
# Description:
# This Python script benchmarks the standard (full recompute) approach to
# computing the average local clustering coefficient on the truncated radial
# dual triangular lattice graph Lambda_r^R. It provides a baseline for
# comparing against symmetry-reduced methods that exploit the order-6
# rotational symmetry of Z_6. The script measures execution
# time over multiple runs with inner repeats for statistical reliability.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
# Run the script with: python simulation_04_benchmark_standard_clustering.py R
# where R is the truncation radius (default: 10). Optional flags:
# --runs N (number of benchmark runs, default: 20)
# --timing_repeats M (repeats per run, default: 100)
#
# Example:
# python simulation_04_benchmark_standard_clustering.py 20 --runs 20
# --timing_repeats 100
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

# Import the helper function to build the complete lattice graph
# (includes inner, outer, and boundary zones with twin edges)
from radial_dual_triangular_lattice_graph import build_complete_lattice_graph

def compute_average_clustering_standard(G):
    # Initialize total clustering coefficient sum
    total_clust = 0.0
    # Get the number of vertices in the graph
    num_v = len(G.nodes())
    # Iterate over each node to compute its local clustering coefficient
    for node in G.nodes():
        # Get the list of neighbors for the current node
        neigh = list(G.neighbors(node))
        # Compute the degree of the node
        deg = len(neigh)
        # Skip nodes with degree less than 2 (clustering undefined)
        if deg < 2:
            continue
        # Count the number of edges between neighbors (common neighbors)
        common = sum(
            1
            for i in range(deg)
            for j in range(i + 1, deg)
            if G.has_edge(neigh[i], neigh[j])
        )
        # Compute local clustering coefficient: 2 * edges / possible edges
        clust = (2 * common) / (deg * (deg - 1))
        # Accumulate the local coefficient
        total_clust += clust
    # Return the average clustering coefficient (or 0 if no vertices)
    return total_clust / num_v if num_v > 0 else 0.0

def benchmark_standard_clustering(G, runs, timing_repeats, seed=42):
    random.seed(seed)
    # List to store timing results from each benchmark run
    times = []
    # Perform multiple benchmark runs for statistical reliability
    for _ in range(runs):
        # Start high-resolution timer
        t0 = time.perf_counter()
        # Repeat the clustering computation multiple times per run
        # to average out system noise
        for _ in range(timing_repeats):
            # Compute average clustering (discards result for timing only)
            _ = compute_average_clustering_standard(G)
        # Append average time per repeat in milliseconds
        times.append(
            (time.perf_counter() - t0) * 1000 / timing_repeats
        )
    # Compute mean and standard deviation of the timings
    return statistics.mean(times), statistics.stdev(times)

if __name__ == "__main__":
    # Set up command-line argument parser for configurable benchmarking
    parser = argparse.ArgumentParser(
        description="Benchmark standard clustering on Lambda_r^R."
    )
    # Add argument for truncation radius R (default: 10)
    parser.add_argument(
        "R", type=int, nargs="?", default=10
    )
    # Add argument for number of benchmark runs (default: 20)
    parser.add_argument(
        "--runs", type=int, default=20
    )
    # Add argument for inner repeats per run (default: 100)
    parser.add_argument(
        "--timing_repeats", type=int, default=100
    )
    # Parse the arguments
    args = parser.parse_args()

    # Build the complete truncated radial dual triangular lattice graph
    # (with admissible inversion radius r=1 and truncation radius R)
    G, _ = build_complete_lattice_graph(args.R)
    # Get the number of vertices in the full graph
    num_v = len(G.nodes())
    # Print graph size for reference
    print(f"Graph: |V|={num_v}")

    # Run the benchmark and get mean and std dev timings
    avg, std = benchmark_standard_clustering(
        G, args.runs, args.timing_repeats
    )
    # Print the results in milliseconds with standard deviation
    print(f"Standard: {avg:.2f} ms +/- {std:.2f}")
