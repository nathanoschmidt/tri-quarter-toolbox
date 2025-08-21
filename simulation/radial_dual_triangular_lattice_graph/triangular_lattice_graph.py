import networkx as nx
import math

# Function to build a truncated radial dual triangular lattice graph.
# Generates outer lattice vertices, inverts them for inner zone (via circle inversion),
# and connects edges symmetrically. Positions stored for phases/norms.
def build_radial_dual_triangular_lattice_graph(R, r_sq=1):
    G_outer = nx.Graph()  # Outer zone graph
    G_inner = nx.Graph()  # Inner zone graph (inverted)
    inversion_map = {}  # Bijection: outer node -> inner node

    # Basis for triangular lattice
    deltas = [(1,0),(0,1),(-1,0),(0,-1),(1,-1),(-1,1)]  # Neighbor offsets

    # Generate outer vertices (norm > sqrt(r_sq), truncated by R)
    outer_nodes = []
    max_range = int(math.ceil(2 * R))
    for m in range(-max_range, max_range + 1):
        for n in range(-max_range, max_range + 1):
            if m == 0 and n == 0: continue
            norm_sq = m*m + m*n + n*n
            if norm_sq <= r_sq or math.sqrt(norm_sq) > R: continue
            x, y = m + n*0.5, n*(math.sqrt(3)/2)
            node = (m, n)
            G_outer.add_node(node, pos=(x,y), phase=math.atan2(y, x), norm_sq=norm_sq)
            outer_nodes.append(node)

    # Connect outer edges
    for u in outer_nodes:
        m, n = u
        for dm, dn in deltas:
            v = (m+dm, n+dn)
            if v in G_outer: G_outer.add_edge(u, v)

    # Invert outer to inner (bijection via i_r)
    for node in outer_nodes:
        m, n = node
        pos = G_outer.nodes[node]['pos']
        norm_sq = G_outer.nodes[node]['norm_sq']
        x_inv = r_sq * pos[0] / norm_sq
        y_inv = r_sq * pos[1] / norm_sq
        inv_node = (m, n, 'inner')  # Use exact lattice coords with tag (avoids float precision issues)
        G_inner.add_node(inv_node, pos=(x_inv, y_inv), phase=math.atan2(y_inv, x_inv), norm_sq=r_sq**2 / norm_sq)
        inversion_map[node] = inv_node

    # Connect inner edges by mirroring outer (preserve adjacency via bijection)
    for u, v in G_outer.edges():
        inv_u = (u[0], u[1], 'inner')
        inv_v = (v[0], v[1], 'inner')
        G_inner.add_edge(inv_u, inv_v)

    return G_outer, G_inner, inversion_map  # Return outer, inner, and bijection map
