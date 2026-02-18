"""
dual_metrics.py - Dual Metric System for TQF Radial Dual Triangular Lattice Graph

This module implements the mathematical foundations for Schmidt's Tri-Quarter Framework (TQF)
radial dual triangular lattice graph, providing explicit geometric representations, symmetry
operations, and dual metrics for neural network architectures.

Key Features:
- Explicit Eisenstein integer lattice vertex generation with 6-neighbor hexagonal adjacency
- Phase pair computation for structured directional labeling per TQF specification
- Continuous dual metric for dyadic/linear radial binning strategies
- Discrete dual metric for hop-distance computation on lattice graphs
- Circle inversion mapping between inner and outer zones for self-duality
- Trihexagonal six-coloring verification for D6 symmetry group independence

Scientific Foundation:
Based on "The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal Dihedral
Symmetry Group T24" by Nathan O. Schmidt (2025).

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Date: February 2026
"""

# MIT License
#
# Copyright (c) 2026 Nathan O. Schmidt, Cold Hammer Research & Development LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from dataclasses import dataclass


# ==============================================================================
# COORDINATE TRANSFORMATIONS
# ==============================================================================

def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """
    Convert Cartesian (x, y) to polar (norm, phase).

    Args:
        x: Real coordinate
        y: Imaginary coordinate

    Returns:
        (norm, phase): Euclidean norm and phase in radians [0, 2*pi)
    """
    norm: float = math.sqrt(x * x + y * y)
    phase: float = math.atan2(y, x)
    if phase < 0:
        phase += 2.0 * math.pi
    return (norm, phase)


def compute_angular_sector(phase: float) -> int:
    """
    Compute angular sector index (0-5) from phase for Z6 rotational symmetry.

    Sectors divide the 2*pi circle into 6 equal parts (60 degrees each):
    S_0: [0, pi/3), S_1: [pi/3, 2*pi/3), S_2: [2*pi/3, pi)
    S_3: [pi, 4*pi/3), S_4: [4*pi/3, 5*pi/3), S_5: [5*pi/3, 2*pi)

    Args:
        phase: Angular phase in radians [0, 2*pi)

    Returns:
        sector: Integer sector index in [0, 5]
    """
    sector: int = int((phase * 3.0) / math.pi) % 6
    return sector


def eisenstein_to_cartesian(m: int, n: int) -> Tuple[float, float]:
    """
    Convert Eisenstein integer (m, n) to Cartesian (x, y) coordinates.

    The Eisenstein integers form a hexagonal lattice in the complex plane with basis
    vectors {1, omega} where omega = exp(2*pi*i/3) = -1/2 + i*sqrt(3)/2.

    Formula: z = m + n*omega = m + n*(-1/2 + i*sqrt(3)/2)
    Cartesian: (x, y) = (m - n/2, n*sqrt(3)/2)

    This representation naturally encodes the hexagonal symmetry of the triangular lattice.

    Args:
        m: First Eisenstein coordinate
        n: Second Eisenstein coordinate

    Returns:
        (x, y): Cartesian coordinates in the complex plane
    """
    x: float = float(m) - float(n) * 0.5
    y: float = float(n) * (math.sqrt(3.0) / 2.0)
    return (x, y)


# ==============================================================================
# PHASE PAIR ASSIGNMENTS
# ==============================================================================

@dataclass
class PhasePair:
    """
    Structured orientation phase pair for directional labeling per TQF specification.

    Per Schmidt's Table 1 and Table 2, each vertex v in the lattice is assigned a phase
    pair phi(v) = (<v_R>, <v_I>)_phi that encodes the directional quadrant position.

    The phase pair provides a discrete directional label that is preserved under circle
    inversion, enabling exact bijective mappings between inner and outer zones.

    Attributes:
        real_phase: Phase of real component <v_R> in {'+', '-', '.'}
        imag_phase: Phase of imaginary component <v_I> in {'+', '-', '.'}
        quadrant: Quadrant index (I, II, III, IV) = (0, 1, 2, 3)
    """
    real_phase: str
    imag_phase: str
    quadrant: int

    def __str__(self) -> str:
        return f"({self.real_phase}, {self.imag_phase})_phi"


def compute_phase_pair(x: float, y: float, epsilon: float = 1e-9) -> PhasePair:
    """
    Compute structured orientation phase pair from Cartesian coordinates.

    Implements Table 1 (Quadrant Phases) and Table 2 (Axis Cases) from Schmidt's TQF paper.
    The phase pair encodes directional orientation and is preserved under circle inversion.

    Quadrant mapping:
    - Quadrant I   (x > 0, y > 0): (+, +)
    - Quadrant II  (x < 0, y > 0): (-, +)
    - Quadrant III (x < 0, y < 0): (-, -)
    - Quadrant IV  (x > 0, y < 0): (+, -)

    Axis cases (using '.' for neutral):
    - Positive real axis (y = 0, x > 0): (+, .)
    - Positive imag axis (x = 0, y > 0): (., +)
    - Negative real axis (y = 0, x < 0): (-, .)
    - Negative imag axis (x = 0, y < 0): (., -)

    Args:
        x: Real (horizontal) Cartesian coordinate
        y: Imaginary (vertical) Cartesian coordinate
        epsilon: Tolerance for detecting axis alignment

    Returns:
        PhasePair: Structured orientation phase pair with quadrant index
    """
    real_phase: str = '.'
    imag_phase: str = '.'
    quadrant: int = 0

    x_near_zero: bool = abs(x) < epsilon
    y_near_zero: bool = abs(y) < epsilon

    # Determine phase pair based on quadrant or axis position
    if not x_near_zero and not y_near_zero:
        # Off-axis: standard quadrant cases
        if x > 0 and y > 0:
            real_phase, imag_phase, quadrant = '+', '+', 0
        elif x < 0 and y > 0:
            real_phase, imag_phase, quadrant = '-', '+', 1
        elif x < 0 and y < 0:
            real_phase, imag_phase, quadrant = '-', '-', 2
        elif x > 0 and y < 0:
            real_phase, imag_phase, quadrant = '+', '-', 3
    elif y_near_zero:
        # On horizontal axis
        if x > 0:
            real_phase, imag_phase, quadrant = '+', '.', 0
        elif x < 0:
            real_phase, imag_phase, quadrant = '-', '.', 2
    elif x_near_zero:
        # On vertical axis
        if y > 0:
            real_phase, imag_phase, quadrant = '.', '+', 1
        elif y < 0:
            real_phase, imag_phase, quadrant = '.', '-', 3

    return PhasePair(
        real_phase=real_phase,
        imag_phase=imag_phase,
        quadrant=quadrant
    )


def verify_phase_pair_preservation(
    v1_phase_pair: PhasePair,
    v2_phase_pair: PhasePair
) -> bool:
    """
    Verify that two vertices have consistent phase pairs.

    Used to verify phase pair preservation under geometric transformations such as
    circle inversion. This is a key property of the TQF dual mapping.

    Args:
        v1_phase_pair: Phase pair of first vertex
        v2_phase_pair: Phase pair of second vertex

    Returns:
        True if phase pairs match exactly, False otherwise
    """
    return (v1_phase_pair.real_phase == v2_phase_pair.real_phase and
            v1_phase_pair.imag_phase == v2_phase_pair.imag_phase and
            v1_phase_pair.quadrant == v2_phase_pair.quadrant)


# ==============================================================================
# EXPLICIT LATTICE VERTEX REPRESENTATION
# ==============================================================================

@dataclass
class ExplicitLatticeVertex:
    """
    Explicit representation of a triangular lattice vertex with all geometric properties.

    This data structure maintains complete geometric information throughout the neural
    network forward pass, enabling exact symmetry operations and circle inversion mappings.
    Maintaining explicit coordinates (rather than learned embeddings) ensures bijective
    dualities and verifiable self-consistency.

    Attributes:
        vertex_id: Unique integer identifier
        eisenstein: (m, n) coordinates in Eisenstein integer basis
        cartesian: (x, y) coordinates in Cartesian plane
        sector: Angular sector index (0-5 for Z6 symmetry)
        zone: Zone classification ('boundary', 'outer', or 'inner')
        norm: Euclidean norm ||v||
        phase: Angular phase in radians [0, 2*pi)
        phase_pair: Structured orientation phase pair for directional labeling
    """
    vertex_id: int
    eisenstein: Tuple[int, int]
    cartesian: Tuple[float, float]
    sector: int
    zone: str
    norm: float
    phase: float
    phase_pair: PhasePair


# ==============================================================================
# HEXAGONAL LATTICE CONSTRUCTION
# ==============================================================================

def build_triangular_lattice_zones(
    R: float,
    r_sq: int = 1
) -> Tuple[List[ExplicitLatticeVertex], Dict[int, List[int]], List[ExplicitLatticeVertex],
           List[ExplicitLatticeVertex], List[ExplicitLatticeVertex], Dict[int, int]]:
    """
    Build explicit triangular lattice with Eisenstein coordinates and 6-neighbor adjacency.

    This function constructs the actual hexagonal lattice graph as described in Schmidt's
    paper, partitioned into three zones:
    - Boundary zone (Lambda_T): vertices with norm^2 = r_sq (typically 6 vertices for r=1)
    - Outer zone (Lambda_+): vertices with r_sq < norm^2 <= R^2
    - Inner zone (Lambda_-): mirror images via circle inversion of outer zone vertices

    Each vertex has up to 6 neighbors in the triangular lattice (hexagonal coordination).
    The construction uses Eisenstein integers to naturally encode the hexagonal symmetry.

    Scientific rationale:
    The triangular lattice provides the optimal packing and natural D6 symmetry group
    structure. Eisenstein integers ensure exact integer arithmetic without floating-point
    approximation errors in neighbor detection.

    Args:
        R: Truncation radius (maximum norm)
        r_sq: Boundary zone radius squared (default: 1 for r=1, giving 6 boundary vertices)

    Returns:
        vertices: List of all ExplicitLatticeVertex objects
        adjacency: Dict mapping vertex_id -> list of neighbor vertex_ids (6 neighbors each)
        boundary_vertices: List of 6 boundary zone vertices
        outer_vertices: List of outer zone vertices
        inner_vertices: List of inner zone vertices (created via inversion)
        inversion_map: Dict mapping outer_vertex_id -> inner_vertex_id for dual outputs
    """
    vertices: List[ExplicitLatticeVertex] = []
    adjacency: Dict[int, List[int]] = {}
    vertex_id: int = 0

    # Track (m, n) -> vertex_id for adjacency construction
    eisenstein_to_id: Dict[Tuple[int, int], int] = {}

    # Generate all lattice vertices within radius R using Eisenstein integers
    max_coord: int = int(math.ceil(R)) + 1

    for m in range(-max_coord, max_coord + 1):
        for n in range(-max_coord, max_coord + 1):
            # Convert to Cartesian for norm computation
            x, y = eisenstein_to_cartesian(m, n)
            norm_sq: float = x * x + y * y

            # Only include vertices within truncation radius (excluding origin)
            if norm_sq <= R * R and norm_sq > 0:
                norm, phase = cartesian_to_polar(x, y)
                sector: int = compute_angular_sector(phase)
                phase_pair: PhasePair = compute_phase_pair(x, y)

                # Classify zone based on norm squared
                if abs(norm_sq - r_sq) < 1e-9:
                    zone: str = 'boundary'
                elif norm_sq > r_sq:
                    zone: str = 'outer'
                else:
                    zone: str = 'inner'

                vertex: ExplicitLatticeVertex = ExplicitLatticeVertex(
                    vertex_id=vertex_id,
                    eisenstein=(m, n),
                    cartesian=(x, y),
                    sector=sector,
                    zone=zone,
                    norm=norm,
                    phase=phase,
                    phase_pair=phase_pair
                )
                vertices.append(vertex)
                eisenstein_to_id[(m, n)] = vertex_id
                vertex_id += 1

    # Build 6-neighbor hexagonal adjacency using Eisenstein integer offsets
    # For Eisenstein integers z = m + n*omega where omega = exp(2*pi*i/3),
    # the 6 unit elements (neighbors at distance 1) are: 1, -1, omega, -omega, 1+omega, -(1+omega)
    # In (m, n) coordinates: (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)
    # With Cartesian conversion x = m - n/2, y = n*sqrt(3)/2, all these have |z| = 1
    neighbor_offsets: List[Tuple[int, int]] = [
        (1, 0),   # +1: x=1, y=0
        (-1, 0),  # -1: x=-1, y=0
        (0, 1),   # +omega: x=-0.5, y=sqrt(3)/2
        (0, -1),  # -omega: x=0.5, y=-sqrt(3)/2
        (1, 1),   # 1+omega: x=0.5, y=sqrt(3)/2
        (-1, -1)  # -(1+omega): x=-0.5, y=-sqrt(3)/2
    ]

    for vertex in vertices:
        if vertex.zone in ['boundary', 'outer']:  # Only outer and boundary vertices have adjacency
            m, n = vertex.eisenstein
            neighbors: List[int] = []
            for dm, dn in neighbor_offsets:
                neighbor_coord: Tuple[int, int] = (m + dm, n + dn)
                if neighbor_coord in eisenstein_to_id:
                    neighbor_id: int = eisenstein_to_id[neighbor_coord]
                    neighbor_vertex: ExplicitLatticeVertex = vertices[neighbor_id]
                    # Only add neighbors in same zone or boundary
                    if neighbor_vertex.zone in ['boundary', 'outer']:
                        neighbors.append(neighbor_id)
            adjacency[vertex.vertex_id] = neighbors

    # Partition vertices by zone
    boundary_vertices: List[ExplicitLatticeVertex] = [v for v in vertices if v.zone == 'boundary']
    outer_vertices: List[ExplicitLatticeVertex] = [v for v in vertices if v.zone == 'outer']

    # Create inner zone vertices via circle inversion (inversion radius r=1)
    inner_vertices: List[ExplicitLatticeVertex] = []
    inversion_map: Dict[int, int] = {}
    r: float = math.sqrt(r_sq)

    for outer_vertex in outer_vertices:
        x, y = outer_vertex.cartesian
        norm_sq: float = x * x + y * y

        # Circle inversion formula: v' = (r^2 / ||v||^2) * v
        scale: float = (r * r) / norm_sq
        x_inv: float = scale * x
        y_inv: float = scale * y

        norm_inv, phase_inv = cartesian_to_polar(x_inv, y_inv)
        sector_inv: int = compute_angular_sector(phase_inv)
        phase_pair_inv: PhasePair = compute_phase_pair(x_inv, y_inv)

        inner_vertex: ExplicitLatticeVertex = ExplicitLatticeVertex(
            vertex_id=vertex_id,
            eisenstein=(-1, -1),  # Inner vertices don't have integer Eisenstein coords
            cartesian=(x_inv, y_inv),
            sector=sector_inv,
            zone='inner',
            norm=norm_inv,
            phase=phase_inv,
            phase_pair=phase_pair_inv
        )
        inner_vertices.append(inner_vertex)
        inversion_map[outer_vertex.vertex_id] = vertex_id
        vertex_id += 1

    # Add inner vertices to main list
    vertices.extend(inner_vertices)

    # Build inner zone adjacency by mirroring the outer zone adjacency structure
    # This ensures the inner zone has proper graph connectivity for graph convolutions.
    # The mirroring preserves the lattice topology: if outer_v1 <-> outer_v2, then inner_v1 <-> inner_v2

    # Create reverse inversion map: inner_id -> outer_id
    reverse_inversion_map: Dict[int, int] = {v: k for k, v in inversion_map.items()}

    # Build inner vertex adjacency by mirroring outer adjacency
    for inner_vertex in inner_vertices:
        inner_id: int = inner_vertex.vertex_id
        outer_id: int = reverse_inversion_map[inner_id]

        # Get the outer vertex's neighbors
        outer_neighbors: List[int] = adjacency.get(outer_id, [])
        inner_neighbors: List[int] = []

        for outer_neighbor_id in outer_neighbors:
            # Find the corresponding inner neighbor
            outer_neighbor_vertex: ExplicitLatticeVertex = vertices[outer_neighbor_id]

            if outer_neighbor_vertex.zone == 'boundary':
                # Boundary vertices are shared between zones - add directly
                inner_neighbors.append(outer_neighbor_id)
            elif outer_neighbor_vertex.zone == 'outer':
                # Map to corresponding inner vertex via inversion
                if outer_neighbor_id in inversion_map:
                    inner_neighbor_id: int = inversion_map[outer_neighbor_id]
                    inner_neighbors.append(inner_neighbor_id)

        adjacency[inner_id] = inner_neighbors

    # Also update boundary vertices to include their inner zone neighbors
    # This creates bidirectional connections between boundary and inner zones
    for boundary_vertex in boundary_vertices:
        boundary_id: int = boundary_vertex.vertex_id
        current_neighbors: List[int] = adjacency.get(boundary_id, [])

        # Find inner vertices that should connect to this boundary vertex
        # (i.e., inner vertices whose corresponding outer vertices neighbor this boundary vertex)
        for outer_id, inner_id in inversion_map.items():
            outer_neighbors: List[int] = adjacency.get(outer_id, [])
            if boundary_id in outer_neighbors and inner_id not in current_neighbors:
                current_neighbors.append(inner_id)

        adjacency[boundary_id] = current_neighbors

    return vertices, adjacency, boundary_vertices, outer_vertices, inner_vertices, inversion_map


def verify_trihexagonal_six_coloring_independence(
    vertices: List[ExplicitLatticeVertex],
    adjacency: Dict[int, List[int]],
    verbose: bool = False
) -> Tuple[bool, Dict[int, Set[int]]]:
    """
    Verify that the trihexagonal six-coloring produces valid independent sets.

    A valid six-coloring means no two adjacent vertices share the same color.
    The six-coloring combines triangular lattice 3-coloring with sector parity.

    For D6 symmetry group independence, the six color classes should form independent
    sets (no edges within each color class), enabling parallel processing without
    interference. This is a key property of the trihexagonal tiling used in TQF-ANN.

    Args:
        vertices: List of explicit lattice vertices
        adjacency: Adjacency dict for the lattice graph
        verbose: Whether to print detailed verification results

    Returns:
        (is_valid, color_classes): Tuple of validity bool and color class dictionary
    """
    # Partition vertices by six-color
    color_sets: Dict[int, Set[int]] = {i: set() for i in range(6)}

    for vertex in vertices:
        if vertex.zone in ['boundary', 'outer']:
            m, n = vertex.eisenstein
            # Skip inner vertices (they have dummy Eisenstein coords)
            if m == -1 and n == -1:
                continue
            color: int = compute_trihexagonal_six_coloring(m, n, vertex.sector)
            color_sets[color].add(vertex.vertex_id)

    # Check independence for each color class
    violation_count: int = 0
    for color in range(6):
        for vertex_id in color_sets[color]:
            neighbors: List[int] = adjacency.get(vertex_id, [])
            for neighbor_id in neighbors:
                if neighbor_id in color_sets[color]:
                    violation_count += 1
                    if verbose:
                        # Find neighbor vertex for sector info
                        neighbor_vertex: Optional[ExplicitLatticeVertex] = None
                        for v in vertices:
                            if v.vertex_id == neighbor_id:
                                neighbor_vertex = v
                                break
                        if neighbor_vertex:
                            print(f"  Violation: Vertex {vertex_id} (color {color}) "
                                  f"adjacent to vertex {neighbor_id} (color {color})")

    is_valid: bool = (violation_count == 0)

    if verbose:
        if is_valid:
            print(f"  Six-coloring valid: No adjacent vertices share the same color")
        else:
            print(f"  Six-coloring invalid: {violation_count} violations found")

    return is_valid, color_sets


def build_vertex_neighbor_map(
    adjacency: Dict[int, List[int]],
    vertex_to_idx: Dict[int, int]
) -> Dict[int, List[int]]:
    """
    Build efficient mapping from vertex index to neighbor indices.

    This is a utility function for models that need to index into feature tensors
    using vertex indices rather than vertex IDs. It translates the adjacency structure
    from vertex IDs to tensor indices.

    Args:
        adjacency: Dict mapping vertex_id -> list of neighbor vertex_ids
        vertex_to_idx: Dict mapping vertex_id -> index in feature tensor

    Returns:
        neighbor_map: Dict mapping vertex_idx -> list of neighbor tensor indices
    """
    neighbor_map: Dict[int, List[int]] = {}

    for vertex_id, vertex_idx in vertex_to_idx.items():
        if vertex_id in adjacency:
            neighbor_ids: List[int] = adjacency[vertex_id]
            neighbor_indices: List[int] = [
                vertex_to_idx[nid] for nid in neighbor_ids
                if nid in vertex_to_idx
            ]
            neighbor_map[vertex_idx] = neighbor_indices
        else:
            neighbor_map[vertex_idx] = []

    return neighbor_map


def precompute_k_hop_neighbors(
    neighbor_map: Dict[int, List[int]],
    max_k: int = 2
) -> Dict[int, Dict[int, List[int]]]:
    """
    Precompute k-hop neighborhoods for O(1) lookup during forward passes.

    For local attention mechanisms, we often need to know which vertices are
    within k hops of a given vertex. Computing this on-the-fly is O(k * degree).
    By precomputing at initialization, we achieve O(1) lookup during forward.

    The triangular lattice has coordination number 6, so:
    - 1-hop: up to 6 neighbors
    - 2-hop: up to 18 neighbors (6 + 12 second neighbors)
    - k-hop: grows linearly with k due to planar structure

    Args:
        neighbor_map: Dict mapping vertex_idx -> list of 1-hop neighbor indices
                      (output of build_vertex_neighbor_map)
        max_k: Maximum number of hops to precompute (default 2)

    Returns:
        k_hop_neighbors: Dict[vertex_idx, Dict[k, List[neighbor_indices]]]
                         where k_hop_neighbors[v][k] gives all vertices
                         exactly k hops from v (not including v itself)

    Example:
        >>> neighbor_map = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
        >>> k_hop = precompute_k_hop_neighbors(neighbor_map, max_k=2)
        >>> k_hop[0][1]  # 1-hop neighbors of vertex 0
        [1, 2]
        >>> k_hop[0][2]  # 2-hop neighbors of vertex 0 (excluding 0 and 1-hop)
        [3]
    """
    num_vertices: int = len(neighbor_map)
    k_hop_neighbors: Dict[int, Dict[int, List[int]]] = {}

    for vertex_idx in range(num_vertices):
        k_hop_neighbors[vertex_idx] = {}

        # Track all vertices seen so far (to avoid revisiting)
        seen: set = {vertex_idx}
        current_frontier: set = {vertex_idx}

        for k in range(1, max_k + 1):
            # Find all vertices at exactly k hops
            next_frontier: set = set()

            for v in current_frontier:
                if v in neighbor_map:
                    for neighbor in neighbor_map[v]:
                        if neighbor not in seen:
                            next_frontier.add(neighbor)

            # Store the k-hop neighbors (sorted for determinism)
            k_hop_neighbors[vertex_idx][k] = sorted(next_frontier)

            # Update seen set and frontier for next iteration
            seen.update(next_frontier)
            current_frontier = next_frontier

    return k_hop_neighbors


def compute_triangular_lattice_three_coloring(m: int, n: int) -> int:
    """
    Compute proper 3-coloring of triangular lattice vertex at Eisenstein (m, n).

    The triangular lattice admits a natural 3-coloring where no two adjacent vertices
    share the same color. This uses the formula: c(m,n) = (m - n) mod 3.

    Scientific rationale: The 3-coloring is a fundamental property of the triangular
    lattice structure and is preserved under translations. It forms the basis for the
    trihexagonal 6-coloring used in TQF symmetry operations.

    Args:
        m: First Eisenstein coordinate
        n: Second Eisenstein coordinate

    Returns:
        color: Integer color in {0, 1, 2}
    """
    color: int = (m - n) % 3
    return color


def compute_trihexagonal_six_coloring(
    m: int,
    n: int,
    sector: int
) -> int:
    """
    Compute trihexagonal six-coloring for equivariant independent set decomposition.

    Per Schmidt's TQF paper Section 4, the trihexagonal six-coloring combines:
    1. Three-coloring c: Lambda_r -> {0, 1, 2} (triangular lattice coloring)
    2. Angular sector parity s_6 mod 2 (hexagonal symmetry)

    Formula: e_6(v) = 2 * c(v) + (s_6(v) mod 2)

    This produces 6 independent sets that can be processed in parallel without
    interference, enabling efficient D6-equivariant computation.

    Args:
        m: First Eisenstein coordinate
        n: Second Eisenstein coordinate
        sector: Angular sector index in [0, 5]

    Returns:
        six_color: Integer color in {0, 1, 2, 3, 4, 5}
    """
    three_color: int = compute_triangular_lattice_three_coloring(m, n)
    sector_parity: int = sector % 2
    six_color: int = 2 * three_color + sector_parity
    return six_color


# ==============================================================================
# CONTINUOUS DUAL METRIC (for radial binning)
# ==============================================================================

class ContinuousDualMetric:
    """
    Continuous dual metric for radial binning in TQF-ANN architecture.

    Implements dyadic (logarithmic) and linear binning strategies for partitioning the
    outer zone into radial layers. Dyadic binning provides finer resolution near the
    boundary and coarser resolution far from the origin, aligning with hierarchical
    feature extraction in deep networks.

    The binning respects the dual structure by treating inner and outer zones symmetrically.
    """

    def __init__(self, r: float = 1.0):
        """
        Initialize continuous dual metric.

        Args:
            r: Inversion radius (default: 1.0 for standard TQF configuration)
        """
        self.r: float = r

    def compute_radial_bin(
        self,
        norm: float,
        R: float,
        num_bins: int,
        method: str = 'dyadic'
    ) -> int:
        """
        Compute radial bin index for a given norm.

        The radial bin determines which hidden layer a vertex belongs to in the TQF-ANN
        architecture. Dyadic binning provides logarithmic spacing (more bins near boundary),
        while linear binning provides uniform spacing.

        Args:
            norm: Euclidean norm of vertex
            R: Truncation radius
            num_bins: Number of radial bins (hidden layers)
            method: 'dyadic' for log2-based binning, 'linear' for uniform spacing

        Returns:
            bin_index: Integer bin index in [0, num_bins-1]

        Raises:
            ValueError: If method is not 'dyadic' or 'linear'
        """
        # Boundary vertices (norm <= r) go to bin 0
        if norm <= self.r:
            return 0

        if method == 'dyadic':
            # Dyadic binning: log2-based for hierarchical resolution
            # Provides finer bins near boundary, coarser bins near truncation radius
            normalized: float = (norm - self.r) / (R - self.r)
            bin_idx: int = int(math.log2(normalized * (2 ** num_bins - 1) + 1))
            return min(bin_idx, num_bins - 1)
        elif method == 'linear':
            # Linear binning: uniform spacing across radial range
            normalized: float = (norm - self.r) / (R - self.r)
            bin_idx: int = int(normalized * num_bins)
            return min(bin_idx, num_bins - 1)
        else:
            raise ValueError(f"Invalid binning method: {method}. Use 'dyadic' or 'linear'")


# ==============================================================================
# DISCRETE DUAL METRIC (for graph operations)
# ==============================================================================

class DiscreteDualMetric:
    """
    Discrete dual metric for hop-distance computation on lattice graphs.

    Implements BFS-based shortest path distances with self-duality verification. The discrete
    metric measures distance in terms of graph hops (edges) rather than Euclidean distance,
    which is essential for graph convolution operations in TQF-ANN.

    Note: For truncated graphs (finite radius R), disconnected zones may occur, resulting
    in infinite (or very large) distances. This is expected behavior and does not indicate
    an error - it reflects the zone-specific nature of the lattice graph structure.
    """

    def __init__(self, adjacency: Dict[int, List[int]], validate: bool = True):
        """
        Initialize discrete dual metric from adjacency structure.

        Args:
            adjacency: Dict mapping vertex_id -> list of neighbor vertex_ids
            validate: Whether to validate adjacency structure (checks for dangling references)
        """
        if validate:
            self._validate_adjacency(adjacency)
        self.adjacency: Dict[int, List[int]] = adjacency

    def _validate_adjacency(self, adjacency: Dict[int, List[int]]) -> None:
        """
        Validate adjacency list structure for consistency.

        Checks that all referenced neighbor vertices exist in the adjacency keys.
        Warnings are issued for dangling references but do not prevent construction.
        """
        if not adjacency:
            raise ValueError("Adjacency list cannot be empty")

        all_vertices: Set[int] = set(adjacency.keys())
        for vertex, neighbors in adjacency.items():
            for neighbor in neighbors:
                if neighbor not in all_vertices:
                    warnings.warn(
                        f"Vertex {vertex} has neighbor {neighbor} not in adjacency keys"
                    )

    def compute_hop_distance(self, source: int, target: int) -> int:
        """
        Compute shortest path length (hop count) between two vertices using BFS.

        Returns a large finite value for unreachable pairs instead of -1 or infinity,
        which ensures numerical stability in downstream computations (e.g., attention weights).

        Args:
            source: Source vertex ID
            target: Target vertex ID

        Returns:
            Shortest path length in hops, or max(1000, graph_size) if unreachable
        """
        if source == target:
            return 0

        # Check if both vertices exist in adjacency
        if source not in self.adjacency or target not in self.adjacency:
            return max(len(self.adjacency), 1000)

        # BFS for unweighted shortest path
        queue: deque = deque([(source, 0)])
        visited: Set[int] = {source}

        while queue:
            current, dist = queue.popleft()

            for neighbor in self.adjacency.get(current, []):
                if neighbor == target:
                    return dist + 1

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Unreachable: return large finite distance for numerical stability
        return max(len(self.adjacency), 1000)

    def compute_distance_matrix(self, vertex_ids: List[int]) -> np.ndarray:
        """
        Compute pairwise hop distance matrix for given vertices.

        Useful for graph convolution kernels and attention mechanisms that depend on
        distance-based weights.

        Args:
            vertex_ids: List of vertex IDs to compute distances for

        Returns:
            Distance matrix of shape (n, n) where n = len(vertex_ids)
        """
        n: int = len(vertex_ids)
        dist_matrix: np.ndarray = np.zeros((n, n), dtype=np.int32)

        for i, src in enumerate(vertex_ids):
            for j, tgt in enumerate(vertex_ids):
                dist_matrix[i, j] = self.compute_hop_distance(src, tgt)

        return dist_matrix
