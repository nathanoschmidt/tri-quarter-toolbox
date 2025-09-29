# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Utility Helper: Building Radial Dual Triangular Lattice Graphs
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
#
# Description:
# This Python module provides functions to construct the radial dual triangular
# lattice graph Lambda_r and its truncated version Lambda_r^R.
# The build_zone_subgraphs function generates the separate outer and inner zone subgraphs
# (Lambda_{+,r}^R and Lambda_{-,r}^R) for inversion-based operations like path mirroring
# (Simulations 02 and 03). The build_complete_lattice_graph function composes them
# into the full graph, adding boundary vertices and twin edges for global computations
# like clustering (Simulations 04 and 05). Nodes use 3-tuples (m, n, type) for consistency,
# with attributes for positions, phases, and squared norms to support phase pair
# assignments and angular sector partitioning. The lattice_rotate function implements
# Z_6 rotations on Eisenstein integers for orbit computation in symmetry-reduced algorithms.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
# For zone subgraphs (e.g., path mirroring):
#   G_outer, G_inner, inversion_map = build_zone_subgraphs(R=10, r_sq=1)
# For full graph (e.g., clustering):
#   G, inversion_map = build_complete_lattice_graph(R=10, r_sq=1)
# For orbits:
#   orbits = get_symmetry_orbits(G)  # Requires lattice_rotate
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import networkx as nx
import math
import cmath  # For complex rotations (unused in core, but kept for potential extensions)

def build_zone_subgraphs(R, r_sq=1):
    """
    Build separate outer and inner zone subgraphs for the truncated radial dual
    triangular lattice graph Lambda_r^R.
    This supports isolated zone operations like inversion-based
    path mirroring without boundary or cross-zone edges.

    Args:
        R (float): Truncation radius (R >> r for balanced approximation).
        r_sq (int, optional): Squared inversion radius (admissible N = r^2
                              representable as m^2 + m n + n^2 for m,n in Z,
                              not both zero; default 1 for unit hexagon boundary).

    Returns:
        tuple: (G_outer, G_inner, inversion_map)
            - G_outer: Outer zone subgraph Lambda_{+,r}^R (nx.Graph).
            - G_inner: Inner zone subgraph Lambda_{-,r}^R (nx.Graph, induced via iota_r).
            - inversion_map: Dict mapping outer nodes to inner twins (bijection).
    """
    G_outer = nx.Graph()
    G_inner = nx.Graph()
    inversion_map = {}

    # Neighbor deltas for degree-6 triangular lattice connectivity
    # (aligned with order-6 rotational symmetry of D_6)
    deltas = [(1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)]

    outer_nodes = []
    # Expand range to ensure coverage within truncation radius R
    # (symmetric around origin, excluding punctured origin per X = C \\ {0})
    max_range = int(math.ceil(2 * R))
    for m in range(-max_range, max_range + 1):
        for n in range(-max_range, max_range + 1):
            if m == 0 and n == 0: continue  # Exclude origin
            # Integer squared Euclidean norm (exact for Eisenstein integers)
            norm_sq = m*m + m*n + n*n
            if norm_sq <= r_sq or math.sqrt(norm_sq) > R: continue
            # Unified complex-Cartesian coordinates (Equation 2.1)
            x, y = m + n*0.5, n*(math.sqrt(3)/2)
            node = (m, n, 'outer')  # 3-tuple for consistency across zones
            G_outer.add_node(
                node, pos=(x,y), phase=math.atan2(y, x), norm_sq=norm_sq
            )
            outer_nodes.append(node)

    # Connect outer edges: nearest-neighbor at Euclidean distance 1
    # (standard triangular lattice spacing, preserving combinatorial duality)
    for u in outer_nodes:
        m, n, _ = u  # Unpack lattice coordinates and type
        for dm, dn in deltas:
            v = (m+dm, n+dn, 'outer')
            if v in G_outer:
                G_outer.add_edge(u, v)

    # Invert outer to inner via circle inversion iota_r
    # (preserves phases for directional consistency)
    for node in outer_nodes:
        m, n, _ = node
        pos = G_outer.nodes[node]['pos']
        norm_sq = G_outer.nodes[node]['norm_sq']
        x_inv = r_sq * pos[0] / norm_sq
        y_inv = r_sq * pos[1] / norm_sq
        inv_node = (m, n, 'inner')  # Twin node in inner zone
        G_inner.add_node(
            inv_node, pos=(x_inv, y_inv),
            phase=math.atan2(y_inv, x_inv),
            norm_sq=r_sq**2 / norm_sq
        )
        inversion_map[node] = inv_node

    # Connect inner edges by mirroring outer (induce isomorphism via reflective duality)
    # (preserves adjacency topologically)
    for u, v in list(G_outer.edges()):
        inv_u = (u[0], u[1], 'inner')
        inv_v = (v[0], v[1], 'inner')
        G_inner.add_edge(inv_u, inv_v)

    return G_outer, G_inner, inversion_map

def build_complete_lattice_graph(R, r_sq=1):
    """
    Build the complete truncated radial dual triangular lattice graph Lambda_r^R,
    composing zone subgraphs with boundary vertices and twin edges.
    This supports global computations like clustering coefficients over the full
    structure, including the combinatorial dual boundary separator.

    Args:
        R (float): Truncation radius.
        r_sq (int, optional): Squared inversion radius (default 1).

    Returns:
        tuple: (G, inversion_map)
            - G: Full nx.Graph with inner/outer/boundary zones and twins.
            - inversion_map: Bijection outer <-> inner (boundary fixed).
    """
    G_outer, G_inner, inversion_map = build_zone_subgraphs(R, r_sq)

    # Compose outer and inner subgraphs (mutually disjoint, no cross edges yet)
    G = nx.compose(G_outer, G_inner)

    # Neighbor deltas (reused for boundary and twins)
    deltas = [(1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)]

    boundary_nodes = []
    # Expand range to capture boundary within R
    max_range = int(math.ceil(R)) + 10
    for m in range(-max_range, max_range + 1):
        for n in range(-max_range, max_range + 1):
            if m == 0 and n == 0: continue
            norm_sq = m*m + m*n + n*n
            if norm_sq == r_sq:  # Exact boundary zone V_{T,r}
                x = m + n*0.5
                y = n*(math.sqrt(3)/2)
                phase = math.atan2(y, x)
                node = (m, n, 'boundary')
                G.add_node(
                    node, pos=(x,y), phase=phase, norm_sq=norm_sq
                )
                boundary_nodes.append(node)

    # Connect boundary cycle: nearest neighbors on V_{T,r} (e.g., hexagon for r=1)
    for i, u in enumerate(boundary_nodes):
        m_u, n_u, _ = u  # Unpack
        # Candidate neighbors on boundary
        v_cand = [
            (m_u+1, n_u, 'boundary'), (m_u, n_u+1, 'boundary'),
            (m_u-1, n_u, 'boundary'), (m_u, n_u-1, 'boundary'),
            (m_u+1, n_u-1, 'boundary'), (m_u-1, n_u+1, 'boundary')
        ]
        for v in v_cand:
            if v in G.nodes and math.isclose(
                G.nodes[v].get('norm_sq', 0), r_sq
            ):
                G.add_edge(u, v)

    # Add twin edges: for each outer-boundary edge {outer, b}, add {iota_r(outer), b}
    # (implements Escher reflective duality across boundary separator)
    for b in boundary_nodes:
        m_b, n_b, _ = b
        for dm, dn in deltas:
            m_o, n_o = m_b + dm, n_b + dn
            outer_node = (m_o, n_o, 'outer')
            if outer_node in G.nodes:
                pos_b = G.nodes[b]['pos']
                pos_o = G.nodes[outer_node]['pos']
                # Check standard lattice spacing ~1
                if math.isclose(
                    math.hypot(pos_o[0]-pos_b[0], pos_o[1]-pos_b[1]),
                    1.0, abs_tol=1e-6
                ):
                    G.add_edge(outer_node, b)
                    # Twin: inner counterpart to boundary
                    twin_outer = (m_o, n_o, 'inner')
                    if twin_outer in G.nodes:
                        G.add_edge(twin_outer, b)

    return G, inversion_map

def lattice_rotate(m, n, k):
    """
    Apply Z_6 rotation (order-6 cyclic group action) to Eisenstein integer
    coordinates (m, n) by k steps of 60 degrees, as per the rotational symmetry
    of the base triangular lattice L. This supports orbit computation
    for symmetry-reduced algorithms like clustering.

    Args:
        m, n (int): Lattice coordinates.
        k (int): Rotation steps (mod 6).

    Returns:
        tuple: Rotated (m', n').
    """
    k = k % 6  # Normalize to [0,5]
    for _ in range(k):
        temp = m
        m = -n
        n = temp + n

    return m, n
