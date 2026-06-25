"""
tqf_lattice_graph.py - Truncated Triangular Lattice Graph and Trihexagonal Six-Coloring

The truncated triangular lattice graph and trihexagonal six-coloring used by the
conflict-free parallel signal-recovery benchmark of the Tri-Quarter Framework
(TQF) radial_dual_signal_processing subproject.

This helper constructs the hexagonal sampling lattice on which a signal field
lives: the base triangular lattice L truncated to a Euclidean radius R, with
nearest-neighbour edges (each interior vertex has six neighbours). It also
builds a proper, order-6-equivariant six-coloring of that graph.

The trihexagonal six-coloring partitions the vertices into six independent sets
(no edge lies within a class), so the six classes can be relaxed in turn with
fully data-parallel, lock-free updates. We construct it from the exact integer
coordinates and *verify* it is proper, so the benchmark's parallelism rests on a
checked structural property rather than a heuristic coloring.

The data structures (padded neighbour-index array, neighbour counts, list of
colour-class index arrays) are shared verbatim by the NumPy (CPU) and PyTorch
(GPU) backends in simulation_04 so that both execute the identical workload.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.0.0
Date: June 24, 2026
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# Basis geometry (matches tqf_hex_signal): omega1 = exp(i*pi/3).
_OMEGA1_RE = 0.5
_OMEGA1_IM = math.sqrt(3.0) / 2.0

# The six nearest-neighbour offsets in oblique (a, b) coordinates.
_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
)


def _norm_sq(a: int, b: int) -> int:
    """Integer squared Euclidean length a^2 + a*b + b^2."""
    return a * a + a * b + b * b


def build_lattice_graph(radius: float) -> Dict[str, object]:
    """Build the truncated triangular lattice graph within Euclidean ``radius``.

    Returns a dictionary with:
      ``num_vertices`` : int
      ``coords``       : (N, 2) int array of oblique (a, b) coordinates
      ``neighbor_idx`` : (N, 6) int array of neighbour vertex indices, padded
                          with the sentinel value N (a zero slot) where a vertex
                          has fewer than six in-graph neighbours
      ``neighbor_cnt`` : (N,) int array of actual neighbour counts
      ``num_edges``    : int (undirected edge count)
      ``color_classes``: list of six int arrays (vertex indices per colour)
      ``proper``       : bool, True iff the six-coloring is verified proper

    The padded neighbour scheme lets a relaxation sweep gather neighbour states
    with a single fancy-index over a state vector that has one extra zero entry
    appended at index N, then divide by ``neighbor_cnt`` to form neighbour means.
    """
    r_sq = radius * radius
    # Enumerate vertices: lattice points strictly within the truncation radius.
    bound = int(math.ceil(radius)) + 1
    coords: List[Tuple[int, int]] = []
    index_of: Dict[Tuple[int, int], int] = {}
    for a in range(-bound, bound + 1):
        for b in range(-bound, bound + 1):
            # Euclidean squared length in the plane equals the Eisenstein norm.
            if _norm_sq(a, b) <= r_sq:
                index_of[(a, b)] = len(coords)
                coords.append((a, b))
    n = len(coords)
    coords_arr = np.array(coords, dtype=np.int64)

    neighbor_idx = np.full((n, 6), n, dtype=np.int64)  # sentinel = N (zero slot)
    neighbor_cnt = np.zeros(n, dtype=np.int64)
    edge_count = 0
    for (a, b), i in index_of.items():
        slot = 0
        for da, db in _NEIGHBOR_OFFSETS:
            j = index_of.get((a + da, b + db))
            if j is not None:
                neighbor_idx[i, slot] = j
                slot += 1
                if j > i:
                    edge_count += 1
        neighbor_cnt[i] = slot

    color_classes, proper = six_coloring(coords_arr, index_of)
    return {
        "num_vertices": n,
        "coords": coords_arr,
        "neighbor_idx": neighbor_idx,
        "neighbor_cnt": neighbor_cnt,
        "num_edges": edge_count,
        "color_classes": color_classes,
        "proper": proper,
    }


def six_coloring(coords: np.ndarray,
                 index_of: Dict[Tuple[int, int], int]
                 ) -> Tuple[List[np.ndarray], bool]:
    """Return the trihexagonal six-coloring as six index arrays, plus a proper flag.

    Construction: combine the exact triangular-lattice 3-coloring residue
    c3 = (a - b) mod 3 with the parity c2 = (a + b) mod 2 into the six-colour
    label  colour = 2 * c3 + c2.  Because adjacent lattice vertices never share
    c3, any refinement of the 3-coloring (here by c2) is automatically a proper
    coloring; the result is also equivariant under the order-6 rotation. The
    function verifies properness against the actual edge set before returning.
    """
    a = coords[:, 0]
    b = coords[:, 1]
    c3 = np.mod(a - b, 3)
    c2 = np.mod(a + b, 2)
    colour = (2 * c3 + c2).astype(np.int64)

    # Note: the 3-coloring c3 alone is already proper (every nearest-neighbour
    # offset changes (a - b) mod 3), so three colour classes would already be
    # conflict-free. The refinement to six classes by the parity c2 is chosen for
    # continuity with the lattice paper -- it is the order-6-equivariant
    # trihexagonal coloring tied to T_24, not a requirement of conflict-freedom.

    # Verify properness: no edge connects two equally-coloured vertices.
    proper = True
    for (av, bv), i in index_of.items():
        for da, db in _NEIGHBOR_OFFSETS:
            j = index_of.get((av + da, bv + db))
            if j is not None and colour[i] == colour[j]:
                proper = False
                break
        if not proper:
            break

    classes = [np.where(colour == c)[0].astype(np.int64) for c in range(6)]
    return classes, proper


def relaxation_sweep_numpy(state: np.ndarray, neighbor_idx: np.ndarray,
                           neighbor_cnt: np.ndarray,
                           color_classes: List[np.ndarray],
                           alpha: float) -> np.ndarray:
    """One color-ordered relaxation (graph-diffusion / MRF smoothing) sweep, CPU.

    For each of the six (mutually non-adjacent) colour classes in turn, every
    vertex updates to  alpha * (own state) + (1 - alpha) * (mean of neighbours).
    Updates within a class are independent, so they are issued as one vectorized
    batch -- the colour class is the unit of parallelism. This is the shared
    computational kernel of graph diffusion, iterative smoothing, and the
    message-passing layers of graph neural networks, here used as a lattice-
    signal denoiser.
    """
    n = state.shape[0]
    for cls in color_classes:
        if cls.size == 0:
            continue
        cnt = neighbor_cnt[cls]
        state_ext = np.concatenate([state, np.zeros(1, dtype=state.dtype)])
        nbr = neighbor_idx[cls]                      # (k, 6), sentinel -> N
        neighbor_sum = state_ext[nbr].sum(axis=1)    # zero slot adds nothing
        # Guard the degenerate isolated-vertex case (cnt == 0, possible only at
        # tiny radii): such a vertex has no neighbour mean, so it is left
        # unchanged rather than dividing by zero.
        denom = np.where(cnt > 0, cnt, 1)
        updated = alpha * state[cls] + (1.0 - alpha) * (neighbor_sum / denom)
        state[cls] = np.where(cnt > 0, updated, state[cls])
    return state


if __name__ == "__main__":
    # Smoke test: build a small graph, confirm the coloring is proper and balanced.
    g = build_lattice_graph(20)
    sizes = [int(c.size) for c in g["color_classes"]]
    print(f"R=20: |V|={g['num_vertices']} |E|={g['num_edges']} "
          f"six-coloring proper={g['proper']} class sizes={sizes}")
