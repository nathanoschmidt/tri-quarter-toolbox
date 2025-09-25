# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Utility: Building the Truncated Radial Dual Triangular Lattice Graph
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 24, 2025
#
# Description:
# This Python utility function constructs a truncated radial dual triangular
# lattice graph Lambda_r^R (with admissible inversion radius r and truncation
# radius R), generating the outer zone subgraph Lambda_{+,r}^R as a standard
# triangular lattice, inverting its vertices via the circle inversion map
# iota_r to form the inner zone subgraph Lambda_{-,r}^R while preserving
# phase pairs and graph ray directions through bijective self-duality, and
# symmetrically connecting edges to maintain the induced graph isomorphism
# between zones. Node attributes store positions, phases, and squared norms
# for rapid directional labeling via O(1) sign-based phase pair assignments
# or modular angular sector indexing.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
# Import as: from radial_dual_triangular_lattice_graph import
#            build_radial_dual_triangular_lattice_graph
# Then call: G_outer, G_inner, inversion_map =
#            build_radial_dual_triangular_lattice_graph(R=10, r_sq=1)
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import networkx as nx
import math

# Function to build a truncated radial dual triangular lattice graph.
# Generates outer lattice vertices (norm > sqrt(r_sq), truncated by R),
# inverts them for inner zone via circle inversion (preserving phase pairs
# and graph ray directions per bijective self-duality), and connects edges
# symmetrically to maintain the induced graph isomorphism between zones.
# Positions, phases, and squared norms stored as node attributes for rapid
# directional labeling and zone assignment via O(1) sign-based phase pair
# checks or modular angular sector indexing.
def build_radial_dual_triangular_lattice_graph(R, r_sq=1):
    # Outer zone subgraph Lambda_{+,r} (standard triangular lattice connections)
    G_outer = nx.Graph()
    # Inner zone subgraph Lambda_{-,r} (inverted via circle inversion iota_r)
    G_inner = nx.Graph()
    # Bijection: outer node -> inner node (supports reversible zone swapping)
    inversion_map = {}

    # Basis for triangular lattice: neighbor offsets aligned with
    # order-6 rotational symmetry of D_6
    deltas = [(1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1)]
    # Primitive direction vectors for adjacency

    # Generate outer vertices (norm > sqrt(r_sq), truncated by R) using
    # Eisenstein integer basis
    # (omega_0 = 1, omega_1 = exp(i pi / 3)); excludes origin to align with
    # punctured complex plane X
    outer_nodes = []
    max_range = int(math.ceil(2 * R))
    for m in range(-max_range, max_range + 1):
        for n in range(-max_range, max_range + 1):
            if m == 0 and n == 0: continue
            # Integer squared Euclidean norm (exact, no floating-point)
            norm_sq = m*m + m*n + n*n
            if norm_sq <= r_sq or math.sqrt(norm_sq) > R: continue
            x, y = m + n*0.5, n*(math.sqrt(3)/2)
            node = (m, n)
            G_outer.add_node(
                node,
                pos=(x,y),
                phase=math.atan2(y, x),
                norm_sq=norm_sq
            )
            outer_nodes.append(node)

    # Connect outer edges: nearest-neighbor connections at Euclidean distance 1
    # (standard lattice spacing)
    for u in outer_nodes:
        m, n = u
        for dm, dn in deltas:
            v = (m+dm, n+dn)
            if v in G_outer: G_outer.add_edge(u, v)

    # Invert outer to inner (bijection via iota_r: r^2 * v / ||v||^2) to realize
    # Escher reflective duality
    # Preserves phase (directional consistency via phase pair assignments) and
    # induces graph isomorphism
    for node in outer_nodes:
        m, n = node
        pos = G_outer.nodes[node]['pos']
        norm_sq = G_outer.nodes[node]['norm_sq']
        x_inv = r_sq * pos[0] / norm_sq
        y_inv = r_sq * pos[1] / norm_sq
        # Tag inner nodes with lattice coords (avoids float precision issues)
        inv_node = (m, n, 'inner')
        G_inner.add_node(
            inv_node,
            pos=(x_inv, y_inv),
            phase=math.atan2(y_inv, x_inv),
            norm_sq=r_sq**2 / norm_sq
        )
        inversion_map[node] = inv_node

    # Connect inner edges by mirroring outer (preserve adjacency via bijection
    # from reflective duality)
    # Ensures |E_{-,r}| = |E_{+,r}| and topological structure for dual metrics
    # and equivariant encodings
    for u, v in G_outer.edges():
        inv_u = (u[0], u[1], 'inner')
        inv_v = (v[0], v[1], 'inner')
        G_inner.add_edge(inv_u, inv_v)

    # Return outer, inner, and bijection map
    return G_outer, G_inner, inversion_map
