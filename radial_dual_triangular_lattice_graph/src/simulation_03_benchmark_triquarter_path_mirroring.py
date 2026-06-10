# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 03: Benchmarking the Tri-Quarter Inversion-Based Approach for
# Path Mirroring
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
# Last Updated: June 10, 2026
# Version: 1.1.0
#
# Description:
# This script benchmarks the Tri-Quarter inversion-based approach to the
# dual-zone shortest-path problem on the truncated radial dual triangular
# lattice graph Lambda_r^R (admissible inversion radius r = 1, configurable
# truncation radius R). The dual-zone problem is identical to the one solved
# by the standard recompute baseline of Simulation 02: obtain single-source
# shortest-path hop distances (the discrete dual metric) in BOTH the outer
# zone subgraph Lambda_{+,r}^R and the inner zone subgraph Lambda_{-,r}^R.
#
# The Tri-Quarter approach solves it by running the shortest-path computation
# once, in the outer zone, and then mapping the result into the inner zone
# through the circle inversion bijection iota_r. The Escher reflective duality
# guarantees that iota_r induces a graph isomorphism between the zones that
# preserves hop distances under the discrete dual metric, so the second
# computation is provably redundant.
#
# This script also verifies the mirrored inner-zone distances against an
# independent recomputation: the verify_mirror_exactness function confirms
# that every mirrored hop distance is exactly equal to the value the standard
# baseline computes. The exactness check establishes that the measured speedup
# comes from eliminating redundant work while preserving the discrete dual
# metric, not from solving a different problem than the baseline.
#
# Times are averaged over multiple runs with inner timing repeats and reported
# in milliseconds.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
#   python simulation_03_benchmark_triquarter_path_mirroring.py R [--runs N]
#                                                                 [--timing_repeats M]
# Example:
#   python simulation_03_benchmark_triquarter_path_mirroring.py 15 --runs 20 --timing_repeats 100
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


def mirror_paths(outer_dist, inversion_map):
    """Map outer-zone hop distances into the inner zone via iota_r.

    For each outer vertex with a known hop distance, the inversion bijection
    iota_r yields the inner-zone twin. The Escher reflective duality makes
    iota_r a distance-preserving graph isomorphism between the zones, so the
    outer hop distance transfers unchanged to the twin. Runs in O(|outer_dist|)
    time and requires no shortest-path computation in the inner zone.

    Args:
        outer_dist: {outer_vertex: hop_distance} from the outer-zone solve.
        inversion_map: Bijection mapping outer vertices to inner twins.

    Returns:
        {inner_vertex: hop_distance} for the inner zone.
    """
    mirrored = {}
    for vertex, distance in outer_dist.items():
        twin = inversion_map.get(vertex)
        if twin is not None:
            mirrored[twin] = distance
    return mirrored


def triquarter_dual_zone_paths(G_outer, start_outer, inversion_map):
    """Solve the dual-zone shortest-path problem by inversion-based mirroring.

    Computes single-source shortest-path hop distances in the outer zone once,
    then obtains the inner-zone distances by mirroring through iota_r instead
    of recomputing. Returns both distance dictionaries.
    """
    outer_dist = nx.single_source_shortest_path_length(G_outer, start_outer)
    inner_dist = mirror_paths(outer_dist, inversion_map)
    return outer_dist, inner_dist


def verify_mirror_exactness(G_inner, start_outer, inversion_map, mirrored):
    """Confirm that mirrored inner-zone distances match an independent solve.

    Independently recomputes single-source shortest-path hop distances in the
    inner zone and checks that every mirrored value is exactly equal. Returns
    True only if the mirrored result is bitwise-identical to the recomputed
    result, demonstrating that the inversion-based shortcut preserves the
    discrete dual metric exactly.
    """
    start_inner = inversion_map.get(start_outer)
    if start_inner is None:
        return len(mirrored) == 0
    recomputed = nx.single_source_shortest_path_length(G_inner, start_inner)
    if set(recomputed.keys()) != set(mirrored.keys()):
        return False
    return all(recomputed[v] == mirrored[v] for v in recomputed)


def benchmark_triquarter_path_mirroring(G_outer, start_outer, inversion_map,
                                        runs, timing_repeats):
    """Time the Tri-Quarter inversion-based approach over multiple runs.

    Each run times timing_repeats solutions of the dual-zone problem and
    records the mean per-solution wall-clock time in milliseconds. Returns the
    mean and standard deviation across runs.
    """
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(timing_repeats):
            triquarter_dual_zone_paths(G_outer, start_outer, inversion_map)
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return statistics.mean(times), std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the Tri-Quarter inversion-based approach to the "
            "dual-zone shortest-path problem on a truncated radial dual "
            "triangular lattice graph Lambda_r^R. Solves the outer zone once "
            "and mirrors into the inner zone via the circle inversion "
            "bijection iota_r, with an exactness check against independent "
            "recomputation. Times are in milliseconds (ms)."
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

    # Fixed seed so this benchmark and the standard baseline (Simulation 02)
    # select the same source vertex, making the comparison and the exactness
    # verification directly meaningful.
    random.seed(args.seed)
    start_outer = (
        random.choice(sorted(G_outer.nodes())) if num_outer > 0 else None
    )

    # Verify exactness once before timing: confirm the inversion-based shortcut
    # reproduces the independently recomputed inner-zone result exactly.
    _, mirrored = triquarter_dual_zone_paths(
        G_outer, start_outer, inversion_map
    )
    exact = verify_mirror_exactness(
        G_inner, start_outer, inversion_map, mirrored
    )
    print(f"Exactness check (mirrored == recomputed): {'PASS' if exact else 'FAIL'}")

    print(f"Running {args.runs} benchmarks, each with {args.timing_repeats} "
          f"timing repeats.")
    avg, std = benchmark_triquarter_path_mirroring(
        G_outer, start_outer, inversion_map, args.runs, args.timing_repeats
    )
    print(f"Tri-Quarter Path Mirroring (Inversion): {avg:.3f} ms (+/-{std:.3f})")