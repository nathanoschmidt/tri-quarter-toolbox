# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Simulation 06: Benchmarking Conflict-Free Parallel Computation via the
# Trihexagonal Six-Coloring (CPU baseline versus GPU)
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: June 9, 2026
# Last Updated: June 9, 2026
# Version: 1.1.0
#
# Description:
# This script benchmarks a symmetry-aware parallel workload on the complete
# truncated radial dual triangular lattice graph Lambda_r^R (admissible
# inversion radius r = 1, configurable truncation radius R), demonstrating
# that the framework's equivariant trihexagonal six-coloring directly enables
# data-parallel execution on a GPU.
#
# The trihexagonal six-coloring e_6: Lambda_r -> {0,...,5} is the proper
# six-coloring defined as e_6 = 2*c + (s_6 mod 2), where c = (a - b) mod 3 is
# the standard proper three-coloring of the triangular lattice in Eisenstein
# coordinates (a, b), and s_6 is the angular sector index. Because e_6 is a
# proper coloring, each of its six color classes is an independent set: no two
# vertices in the same class are adjacent. Independence is exactly the
# property a parallel scheduler needs, since vertices in one class can be
# updated simultaneously with no read-write conflicts and no locks.
#
# The benchmarked workload is one color-ordered relaxation sweep over the
# lattice: every vertex updates its scalar state to a weighted combination of
# its own state and the mean of its neighbors' states. This is the
# computational kernel shared by graph diffusion, iterative smoothing, label
# propagation, and message-passing layers of graph neural networks. The sweep
# visits the six color classes in turn (a Gauss-Seidel ordering: a class reads
# the updates already written by earlier classes in the same sweep). Because
# the six-coloring is proper, the vertices within a single class are mutually
# non-adjacent, so all updates inside that class are independent and are issued
# as one batched, vectorized operation with no read-write conflicts and no
# locks. The color classes are the unit of parallelism; the ordering across
# classes is sequential by construction.
#
# Two backends are benchmarked on the identical workload:
#   - a CPU baseline that sweeps the six color classes sequentially using
#     NumPy array operations, and
#   - a GPU backend (PyTorch) that sweeps the six color classes using batched
#     tensor operations on the CUDA device.
# Both backends produce numerically identical results; the script verifies
# this agreement before timing. When no CUDA device is available the GPU
# backend transparently falls back to the CPU PyTorch device, and the script
# reports which device was used so the result can be interpreted accordingly.
#
# The neighbor structure is encoded once as a padded adjacency-index tensor,
# so each relaxation sweep is a single gather-and-reduce over that tensor with
# no Python-level per-vertex loop. The color classes are precomputed from e_6;
# this precomputation is a one-time cost amortized over repeated sweeps and is
# excluded from the timed region.
#
# Times are averaged over multiple runs with inner timing repeats and reported
# in milliseconds per sweep.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
# - NumPy library (install via: pip install numpy)
# - PyTorch library (install via: pip install torch); see
#   https://pytorch.org/get-started/locally/ for a CUDA-enabled build matched
#   to the local NVIDIA driver. Without a CUDA build the GPU backend runs on
#   the CPU PyTorch device and the speedup column reflects that fallback.
#
# Usage:
#   python simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py R [--runs N]
#                                                                    [--timing_repeats M]
#                                                                    [--sweeps S]
# Example:
#   python simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py 200 --runs 10 --timing_repeats 20
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import math
import time
import argparse
import statistics

import numpy as np

from radial_dual_triangular_lattice_graph import build_complete_lattice_graph

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Primary lattice ray directions d_t (t in Z_6) as Eisenstein coordinate
# pairs: d_t is the image of (1, 0) under t steps of the order-6 lattice
# rotation, i.e. the six nearest-neighbor directions at angles t * 60 degrees.
SECTOR_RAYS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def angular_sector_index(m, n):
    """Return the angular sector index s_6 in Z_6 of an Eisenstein lattice
    vertex (m, n) (the origin excluded).

    The six sectors are the 60-degree wedges between consecutive primary
    lattice rays SECTOR_RAYS. Wedge membership is decided exactly from the
    integer coordinate pair (m, n) using the sign of the lattice cross product
    cross((a, b), (c, d)) = a * d - b * c, which equals the orientation of the
    two directions up to the positive constant sqrt(3) / 2 and is therefore an
    exact integer (no floating-point phase is computed). A vertex lying on a
    primary ray is assigned to the sector counterclockwise of that ray, so the
    six sectors tile Z_6 without overlap. This rule is exactly equivariant
    under the order-6 lattice rotation---a vertex and its 60-degree image
    differ by exactly one sector---so the resulting sector partition, and the
    s_6 parity of the trihexagonal six-coloring built from it, are independent
    of any floating-point computation.
    """
    for t in range(6):
        dm, dn = SECTOR_RAYS[t]
        em, en = SECTOR_RAYS[(t + 1) % 6]
        # p is counterclockwise of (or on) ray t, and strictly clockwise of
        # ray t+1: this places on-ray vertices in the counterclockwise sector.
        if dm * n - dn * m >= 0 and m * en - n * em > 0:
            return t
    raise ValueError(
        f"angular_sector_index: undefined for origin/invalid vertex ({m}, {n})"
    )


def trihexagonal_six_coloring(node):
    """Return the trihexagonal six-coloring e_6 of a lattice vertex.

    Implements e_6 = 2 * c + (s_6 mod 2), where c = (m - n) mod 3 is the
    proper three-coloring of the triangular lattice in Eisenstein coordinates
    (m, n), and s_6 is the angular sector index. The result lies in {0,...,5}
    and is a proper coloring of the lattice graph: adjacent vertices receive
    distinct colors, so each color class is an independent set.
    """
    m, n, _ = node
    c = (m - n) % 3
    s6_parity = angular_sector_index(m, n) % 2
    return 2 * c + s6_parity


def build_relaxation_arrays(G):
    """Build the index arrays needed for batched neighbor relaxation.

    Produces a stable vertex ordering, a padded neighbor-index matrix, a
    neighbor-count vector, and the six independent color-class index lists
    induced by the trihexagonal six-coloring. The padded neighbor matrix has
    one row per vertex and a fixed number of columns equal to the maximum
    degree; unused entries are padded so that a single gather over the matrix
    yields every vertex's neighbor states at once.

    Args:
        G: The complete truncated radial dual triangular lattice graph.

    Returns:
        dict with keys:
          'num_vertices' : vertex count.
          'neighbor_idx' : int array, shape (num_vertices, max_degree).
          'neighbor_cnt' : int array, shape (num_vertices,).
          'color_classes': list of six int arrays of vertex indices.
          'proper'       : bool, True if the six-coloring is verified proper.
    """
    nodes = list(G.nodes())
    index_of = {node: i for i, node in enumerate(nodes)}
    num_vertices = len(nodes)

    neighbor_lists = [
        [index_of[v] for v in G.neighbors(node)] for node in nodes
    ]
    max_degree = max((len(nl) for nl in neighbor_lists), default=0)

    # Padded neighbor-index matrix. Padding entries point back at the vertex
    # itself; the neighbor-count vector masks them out of the mean reduction.
    neighbor_idx = np.zeros((num_vertices, max_degree), dtype=np.int64)
    neighbor_cnt = np.zeros(num_vertices, dtype=np.int64)
    for i, nl in enumerate(neighbor_lists):
        neighbor_cnt[i] = len(nl)
        for j, nbr in enumerate(nl):
            neighbor_idx[i, j] = nbr
        for j in range(len(nl), max_degree):
            neighbor_idx[i, j] = i

    # Partition vertices into the six independent color classes of e_6.
    color_of = np.array(
        [trihexagonal_six_coloring(node) for node in nodes], dtype=np.int64
    )
    color_classes = [
        np.where(color_of == color)[0] for color in range(6)
    ]

    # Verify that the six-coloring is proper: no edge joins same-colored
    # vertices. Independence of each color class is what licenses
    # conflict-free parallel updates within a class.
    proper = all(
        color_of[index_of[u]] != color_of[index_of[v]]
        for u, v in G.edges()
    )

    return {
        "num_vertices": num_vertices,
        "neighbor_idx": neighbor_idx,
        "neighbor_cnt": neighbor_cnt,
        "color_classes": color_classes,
        "proper": proper,
    }


def relaxation_sweep_cpu(state, arrays, alpha=0.5):
    """Perform one color-ordered relaxation sweep on the CPU with NumPy.

    Each vertex is updated to alpha * (own state) + (1 - alpha) * (mean of
    neighbor states). The six color classes are visited in turn; a class reads
    the updates already written by earlier classes in the same sweep
    (Gauss-Seidel ordering). Within a single class the vertices are mutually
    non-adjacent because the six-coloring is proper, so their updates are
    independent and applied as one batched array operation. The input array is
    not modified; the updated state array is returned.
    """
    neighbor_idx = arrays["neighbor_idx"]
    neighbor_cnt = arrays["neighbor_cnt"]
    state = state.copy()
    for cls in arrays["color_classes"]:
        if cls.size == 0:
            continue
        gathered = state[neighbor_idx[cls]]
        counts = np.maximum(neighbor_cnt[cls], 1).astype(state.dtype)
        neighbor_mean = gathered.sum(axis=1) / counts
        state[cls] = alpha * state[cls] + (1.0 - alpha) * neighbor_mean
    return state


def relaxation_sweep_gpu(state, neighbor_idx, neighbor_cnt,
                         color_classes, alpha=0.5):
    """Perform one color-ordered relaxation sweep with batched PyTorch tensors.

    Equivalent to relaxation_sweep_cpu: the six color classes are visited in
    turn with Gauss-Seidel ordering, and within each class the independent
    updates run as one batched tensor operation on the device holding the
    input tensors (a CUDA device when one is available). The input tensor is
    not modified; the updated state tensor is returned.
    """
    state = state.clone()
    for cls in color_classes:
        if cls.numel() == 0:
            continue
        gathered = state[neighbor_idx[cls]]
        counts = torch.clamp(neighbor_cnt[cls], min=1).to(state.dtype)
        neighbor_mean = gathered.sum(dim=1) / counts
        state[cls] = alpha * state[cls] + (1.0 - alpha) * neighbor_mean
    return state


def run_cpu(arrays, sweeps, runs, timing_repeats):
    """Benchmark the CPU backend; return (result_state, mean_ms, std_ms).

    The reported time is the mean wall-clock time per single relaxation sweep.
    """
    num_vertices = arrays["num_vertices"]
    initial = np.linspace(0.0, 1.0, num_vertices, dtype=np.float64)

    result_state = initial.copy()
    for _ in range(sweeps):
        result_state = relaxation_sweep_cpu(result_state, arrays)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(timing_repeats):
            state = initial.copy()
            for _ in range(sweeps):
                state = relaxation_sweep_cpu(state, arrays)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed / (timing_repeats * sweeps))
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return result_state, statistics.mean(times), std


def run_gpu(arrays, sweeps, runs, timing_repeats):
    """Benchmark the GPU backend; return (result_state, mean_ms, std_ms, device).

    Moves the index tensors and state to the CUDA device when available (CPU
    PyTorch device otherwise) and times batched relaxation sweeps there. The
    reported time is the mean wall-clock time per single relaxation sweep and
    includes CUDA synchronization so the measurement reflects completed device
    work. The result state is returned on the CPU as a NumPy array.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_vertices = arrays["num_vertices"]

    neighbor_idx = torch.from_numpy(arrays["neighbor_idx"]).to(device)
    neighbor_cnt = torch.from_numpy(arrays["neighbor_cnt"]).to(device)
    color_classes = [
        torch.from_numpy(cls).to(device) for cls in arrays["color_classes"]
    ]
    initial = torch.linspace(
        0.0, 1.0, num_vertices, dtype=torch.float64, device=device
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    state = initial.clone()
    for _ in range(sweeps):
        state = relaxation_sweep_gpu(
            state, neighbor_idx, neighbor_cnt, color_classes
        )
    synchronize()
    result_state = state.cpu().numpy()

    times = []
    for _ in range(runs):
        synchronize()
        t0 = time.perf_counter()
        for _ in range(timing_repeats):
            state = initial.clone()
            for _ in range(sweeps):
                state = relaxation_sweep_gpu(
                    state, neighbor_idx, neighbor_cnt, color_classes
                )
        synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed / (timing_repeats * sweeps))
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return result_state, statistics.mean(times), std, device.type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark conflict-free parallel relaxation on the complete "
            "truncated radial dual triangular lattice graph Lambda_r^R, "
            "scheduled by the equivariant trihexagonal six-coloring, on a "
            "CPU baseline versus a GPU backend. Times are in milliseconds "
            "per sweep."
        )
    )
    parser.add_argument("R", type=int, nargs="?", default=100,
                        help="Truncation radius R (default: 100)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of benchmark runs (default: 10)")
    parser.add_argument("--timing_repeats", type=int, default=20,
                        help="Repeats per run for accuracy (default: 20)")
    parser.add_argument("--sweeps", type=int, default=10,
                        help="Relaxation sweeps per timed iteration "
                             "(default: 10)")
    args = parser.parse_args()

    G, _ = build_complete_lattice_graph(args.R)
    arrays = build_relaxation_arrays(G)
    print(f"Graph: |V|={arrays['num_vertices']} |E|={len(G.edges())}")
    print(f"Trihexagonal six-coloring proper: "
          f"{'PASS' if arrays['proper'] else 'FAIL'}")
    class_sizes = [int(cls.size) for cls in arrays["color_classes"]]
    print(f"Six-coloring class sizes: {class_sizes}")

    cpu_state, cpu_avg, cpu_std = run_cpu(
        arrays, args.sweeps, args.runs, args.timing_repeats
    )
    print(f"CPU (NumPy, color-ordered): {cpu_avg:.4f} ms/sweep "
          f"+/- {cpu_std:.4f}")

    if not TORCH_AVAILABLE:
        print("PyTorch not installed; GPU backend skipped. "
              "Install via: pip install torch")
    else:
        gpu_state, gpu_avg, gpu_std, device = run_gpu(
            arrays, args.sweeps, args.runs, args.timing_repeats
        )
        print(f"GPU backend device: {device}")

        # Verify the GPU result matches the CPU result before reporting speed.
        max_abs_diff = float(np.max(np.abs(cpu_state - gpu_state)))
        agree = max_abs_diff < 1e-9
        print(f"CPU/GPU agreement: {'PASS' if agree else 'FAIL'} "
              f"(max abs diff {max_abs_diff:.2e})")
        print(f"GPU (PyTorch, color-ordered): {gpu_avg:.4f} ms/sweep "
              f"+/- {gpu_std:.4f}")
        if gpu_avg > 0:
            print(f"Speedup (CPU / GPU): {cpu_avg / gpu_avg:.2f}x")