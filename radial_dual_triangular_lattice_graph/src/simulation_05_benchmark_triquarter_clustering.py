# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 05: Benchmarking the Tri-Quarter Symmetry-Reduced Clustering
# Coefficient Computation (EXACT RATIONAL ARITHMETIC)
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
# Last Updated: June 10, 2026
# Version: 1.1.0
#
# Description:
# This script benchmarks the Tri-Quarter symmetry-reduced approach to the
# average local clustering coefficient on the complete truncated radial dual
# triangular lattice graph Lambda_r^R (admissible inversion radius r = 1,
# configurable truncation radius R). It solves the identical problem as the
# standard baseline of Simulation 04 and, like that baseline, computes the
# coefficient in exact rational arithmetic (fractions.Fraction).
#
# The Tri-Quarter approach exploits the order-6 rotational symmetry of the
# lattice. A Z_6-orbit transversal partitions the vertex set into orbits under
# the order-6 rotation action; the local clustering coefficient is computed on
# a single representative per orbit and replicated, weighted by orbit size, to
# the remaining vertices. The order-6 rotation is a graph automorphism of
# Lambda_r^R, so every vertex in an orbit has the identical integer
# neighborhood structure and therefore the identical exact rational
# coefficient. Consequently the orbit-reduced average equals the full-graph
# average not merely to floating-point tolerance but EXACTLY, as the same
# rational number -- which this script verifies by exact (==) comparison
# against Simulation 04 before timing.
#
# Two orbit-transversal constructions are provided and benchmarked:
#   - a pure-Python construction using a visited-set traversal, and
#   - a NumPy-vectorized construction (when NumPy is available).
# Both yield bitwise-identical orbits. Orbit-transversal construction is a
# one-time precomputation amortized over repeated clustering queries; it is
# performed once, outside the timing loop.
#
# Times are averaged over multiple runs with inner timing repeats and reported
# in milliseconds.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
# - NumPy library, optional (install via: pip install numpy)
#
# Usage:
#   python simulation_05_benchmark_triquarter_clustering.py R [--runs N]
#                                                             [--timing_repeats M]
#                                                             [--orbit-method METHOD]
#                                                             [--debug]
# where METHOD is one of: auto (default), python, numpy.
#
# Example:
#   python simulation_05_benchmark_triquarter_clustering.py 100 --runs 20 --timing_repeats 20
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

from radial_dual_triangular_lattice_graph import (
    build_complete_lattice_graph, lattice_rotate
)
from simulation_04_benchmark_standard_clustering import (
    build_adjacency_sets,
    local_clustering_exact,
    compute_average_clustering_standard_exact,
)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def get_symmetry_orbits_python(G, debug=False):
    """Partition the vertex set into Z_6 orbits via a visited-set traversal."""
    visited = set()
    orbits = []
    for node in G.nodes():
        if node in visited:
            continue
        m, n, node_type = node
        orbit = [node]
        for k in range(1, 6):
            rot_m, rot_n = lattice_rotate(m, n, k)
            rot_node = (rot_m, rot_n, node_type)
            if rot_node in G and rot_node not in visited:
                orbit.append(rot_node)
                visited.add(rot_node)
        visited.add(node)
        orbits.append(orbit)
    if debug:
        total = sum(len(set(o)) for o in orbits)
        print(f"Debug (python): {len(orbits)} orbits, "
              f"avg size {total / len(orbits):.2f}, "
              f"coverage {total == len(G)}")
    return orbits


def get_symmetry_orbits_numpy(G, debug=False):
    """Partition the vertex set into Z_6 orbits using NumPy batched rotations."""
    nodes = list(G.nodes())
    index_of = {node: i for i, node in enumerate(nodes)}
    coords = np.array([(m, n) for (m, n, _) in nodes], dtype=np.int64)
    base = np.array([[0, -1], [1, 1]], dtype=np.int64)
    rotations = [np.linalg.matrix_power(base, k) for k in range(6)]
    rotated = [coords @ rot.T for rot in rotations]
    visited = [False] * len(nodes)
    orbits = []
    for i in range(len(nodes)):
        if visited[i]:
            continue
        orbit = []
        for k in range(6):
            rm, rn = int(rotated[k][i, 0]), int(rotated[k][i, 1])
            cand = (rm, rn, nodes[i][2])
            j = index_of.get(cand)
            if j is not None and not visited[j]:
                visited[j] = True
                orbit.append(nodes[j])
        orbits.append(orbit)
    if debug:
        total = sum(len(set(o)) for o in orbits)
        print(f"Debug (numpy): {len(orbits)} orbits, "
              f"avg size {total / len(orbits):.2f}, "
              f"coverage {total == len(G)}")
    return orbits


def get_symmetry_orbits(G, method="auto", debug=False):
    """Dispatch to the requested orbit-transversal construction."""
    if method == "numpy" or (method == "auto" and NUMPY_AVAILABLE):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy requested but not installed.")
        return get_symmetry_orbits_numpy(G, debug=debug)
    return get_symmetry_orbits_python(G, debug=debug)


def compute_average_clustering_triquarter_exact(adjacency, orbits):
    """Exact average local clustering coefficient via orbit replication.

    Computes the exact rational local coefficient on one representative per
    orbit and replicates it across the orbit, weighted by orbit size. Because
    the order-6 rotation is a graph automorphism, every member of an orbit has
    the identical coefficient, so this exactly reproduces the full-graph sum as
    a rational number. The denominator (total weight) equals the vertex count.

    Args:
        adjacency: {vertex: frozenset(neighbors)} adjacency-set view.
        orbits: Z_6-orbit transversal from get_symmetry_orbits.

    Returns:
        The average local clustering coefficient as an exact Fraction
        (Fraction(0) if the graph is empty).
    """
    total = Fraction(0)
    total_weight = 0
    for orbit in orbits:
        rep = orbit[0]
        clust_rep = local_clustering_exact(adjacency, rep)
        orbit_size = len(set(orbit))
        total += clust_rep * orbit_size
        total_weight += orbit_size
    return total / total_weight if total_weight > 0 else Fraction(0)


def verify_orbit_member_equality(adjacency, orbits):
    """Confirm every member of every orbit shares the representative's exact
    coefficient (i.e. the rotation action genuinely preserves local
    clustering). Returns True if the symmetry reduction is lossless."""
    for orbit in orbits:
        rep_val = local_clustering_exact(adjacency, orbit[0])
        for member in orbit:
            if local_clustering_exact(adjacency, member) != rep_val:
                return False
    return True


def benchmark_triquarter_clustering(adjacency, orbits, runs, timing_repeats):
    """Time the exact symmetry-reduced clustering computation over multiple
    runs. The orbit transversal is supplied precomputed (a one-time cost
    amortized over repeated queries)."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(timing_repeats):
            compute_average_clustering_triquarter_exact(adjacency, orbits)
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return statistics.mean(times), std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the Tri-Quarter symmetry-reduced average local "
            "clustering coefficient on the complete truncated radial dual "
            "triangular lattice graph Lambda_r^R in exact rational "
            "arithmetic, using a Z_6-orbit transversal. Times are in "
            "milliseconds."
        )
    )
    parser.add_argument("R", type=int, nargs="?", default=10,
                        help="Truncation radius R (default: 10)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of benchmark runs (default: 20)")
    parser.add_argument("--timing_repeats", type=int, default=20,
                        help="Repeats per run for accuracy (default: 20)")
    parser.add_argument("--orbit-method", choices=["auto", "python", "numpy"],
                        default="auto",
                        help="Orbit-transversal construction (default: auto)")
    parser.add_argument("--debug", action="store_true",
                        help="Print orbit-transversal statistics")
    args = parser.parse_args()

    G, _ = build_complete_lattice_graph(args.R)
    adjacency = build_adjacency_sets(G)
    num_v = len(adjacency)
    print(f"Graph: |V|={num_v}")
    if args.orbit_method in ("auto", "numpy"):
        print(f"NumPy available: {NUMPY_AVAILABLE}")

    t0 = time.perf_counter()
    orbits = get_symmetry_orbits(G, method=args.orbit_method, debug=args.debug)
    orbit_build_ms = (time.perf_counter() - t0) * 1000
    print(f"Orbit transversal: {len(orbits)} orbits "
          f"built in {orbit_build_ms:.2f} ms "
          f"(method={args.orbit_method})")

    # Exactness verification BEFORE timing: the orbit-reduced exact rational
    # must equal the full-graph exact rational bitwise (==), not merely to a
    # floating-point tolerance.
    members_ok = verify_orbit_member_equality(adjacency, orbits)
    ct = compute_average_clustering_triquarter_exact(adjacency, orbits)
    cs = compute_average_clustering_standard_exact(adjacency)
    exact_match = (ct == cs)
    print(f"Orbit member-equality check: {'PASS' if members_ok else 'FAIL'}")
    print(f"Exact match (orbit == standard, as Fraction): "
          f"{'PASS' if exact_match else 'FAIL'}")
    print(f"Float images identical: {float(ct) == float(cs)}")
    print(f"Average clustering coefficient (exact): {ct} = {float(ct):.12f}")

    avg, std = benchmark_triquarter_clustering(
        adjacency, orbits, args.runs, args.timing_repeats
    )
    print(f"Tri-Quarter (exact): {avg:.3f} ms +/- {std:.3f}")
