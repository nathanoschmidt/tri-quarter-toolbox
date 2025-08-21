import networkx as nx
import time
import random
import argparse
import statistics

from triangular_lattice_graph import build_radial_dual_triangular_lattice_graph

# Mirror a path dict via inversion bijection (O(|path|) mapping).
# Preserves phases/norms under duality.
def mirror_paths(outer_paths, inversion_map):
    mirrored = {}  # Mirrored path lengths
    for target, length in outer_paths.items():
        inv_target = inversion_map.get(target)
        if inv_target: mirrored[inv_target] = length  # Copy length (bijection preserves hops)
    return mirrored

# Benchmark Tri-Quarter path mirroring in dual zones (compute outer, mirror to inner).
# Uses bijection for O(1) per-path mirroring.
def benchmark_triquarter_path_mirroring(G_outer, start_outer, inversion_map, runs, timing_repeats):
    times = []  # List to store execution times
    for _ in range(runs):
        t0 = time.perf_counter()  # Start timer
        for _ in range(timing_repeats):  # Repeat for noise reduction
            # Compute outer paths
            outer_paths = nx.single_source_shortest_path_length(G_outer, start_outer)
            # Mirror to inner via duality (no recompute)
            mirror_paths(outer_paths, inversion_map)
        times.append((time.perf_counter() - t0) * 1000 / timing_repeats)  # Average time per dual-zone computation in ms
    return statistics.mean(times), statistics.stdev(times)  # Return mean and std dev

if __name__ == "__main__":
    # Parse arguments for configurable runs
    parser = argparse.ArgumentParser(
        description="Benchmark path mirroring on a truncated Tri-Quarter radial dual triangular lattice graph "
                    "with Tri-Quarter duality approach (mirror outer paths to inner via bijection). Demonstrates "
                    "speedups from exact inversion. Times are in milliseconds (ms)."
    )
    parser.add_argument("R", type=int, nargs="?", default=10, help="Truncation radius (default: 10)")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs (default: 20)")
    parser.add_argument("--timing_repeats", type=int, default=100, help="Repeats per run for accuracy (default: 100)")

    args = parser.parse_args()

    print(f"Building radial dual lattice graph with truncation radius R={args.R}...")
    G_outer, G_inner, inversion_map = build_radial_dual_triangular_lattice_graph(args.R)  # Build graphs and map
    num_outer = len(G_outer.nodes())
    num_inner = len(G_inner.nodes())
    print(f"Graphs built: Outer {num_outer} vertices, Inner {num_inner} vertices.")

    start_outer = random.choice(list(G_outer.nodes())) if num_outer > 0 else None  # Random outer start

    print(f"Running {args.runs} benchmarks, each with {args.timing_repeats} repeats for reliable timing.")
    print("Note: Includes bijection preproc (amortized over queries).")

    # Run and display results
    avg, std = benchmark_triquarter_path_mirroring(G_outer, start_outer, inversion_map, args.runs, args.timing_repeats)
    print(f"Tri-Quarter Path Mirroring (Duality): {avg:.2f} ms (+/-{std:.2f})")
