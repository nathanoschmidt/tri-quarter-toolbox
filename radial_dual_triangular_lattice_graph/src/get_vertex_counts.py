# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Helper Tool: Counting Vertices in Zones and Angular Sectors
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
# Last Updated: June 10, 2026
# Version: 1.1.0
#
# Description:
# This Python script computes vertex counts in the radial dual triangular
# lattice graph Lambda_r across zones (inner, boundary, outer) and angular
# sectors S_t (t in Z_6), for a given admissible inversion radius r (yielding
# integer r_sq = r^2 with lattice points) and truncation radius R. It leverages
# the norm trichotomy for zone partitioning and modular angular sector indexing
# for order-6 rotational symmetry. The script validates r_sq via Eisenstein integer
# representations and outputs counts for direct use in simulations or analysis,
# ensuring balanced distributions under D_6 symmetry for equivariant encodings.
#
# Requirements:
# - Python 3.x
# - NetworkX library (install via: pip install networkx)
#
# Usage:
# Run the script with: python get_vertex_counts.py r R
# Examples:
# python get_vertex_counts.py 1 4
# python get_vertex_counts.py 2.64575 10  (for r approx sqrt(7))
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import math
import argparse
from radial_dual_triangular_lattice_graph import build_zone_subgraphs

def compute_eisenstein_representations(max_nsq):
    """
    Compute the number of Eisenstein integer representations of integers as
    sums of the form m^2 + m*n + n^2 in the triangular lattice L, up to the
    maximum squared norm max_nsq. This supports validation of admissible
    inversion radii r where r^2 = N has positive representations.
    """
    representations = {}  # Dictionary to store the count of representations
                          # for each squared norm nsq
    max_m = int(math.ceil(math.sqrt(max_nsq))) + 10  # Safe range for m, n to
                                                     # cover all possible
                                                     # nsq <= max_nsq
    for m in range(-max_m, max_m + 1):  # Loop over possible m values
        for n in range(-max_m, max_m + 1):  # Loop over possible n values
            if m == 0 and n == 0:
                continue  # Skip the origin to align with punctured complex
                          # plane X
            nsq = m * m + m * n + n * n  # Compute the squared norm in
                                         # triangular lattice L
            if nsq > max_nsq:
                continue  # Skip if beyond the maximum
            if nsq not in representations:
                representations[nsq] = 0  # Initialize count if not present
            representations[nsq] += 1  # Increment the representation count
    return representations


# Primary lattice ray directions d_t (t in Z_6) as Eisenstein coordinate
# pairs: d_t is the image of (1, 0) under t steps of the order-6 lattice
# rotation, i.e. the six nearest-neighbor directions at angles t * 60 degrees.
SECTOR_RAYS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def angular_sector_index(m, n):
    """
    Determine the angular sector index t in Z_6 (0 to 5) of an Eisenstein
    lattice vertex (m, n), the origin excluded, using exact integer arithmetic.

    The six sectors are the 60-degree wedges between consecutive primary
    lattice rays SECTOR_RAYS. Wedge membership is decided from the integer
    coordinate pair (m, n) via the sign of the lattice cross product
    cross((a, b), (c, d)) = a * d - b * c, which is the orientation of the two
    directions up to the positive constant sqrt(3) / 2 and is therefore an
    exact integer (no floating-point phase is used). A vertex on a primary ray
    is assigned to the sector counterclockwise of that ray, so the six sectors
    tile Z_6 without overlap and the rule is exactly equivariant under the
    order-6 rotational symmetry of D_6.
    """
    for t in range(6):
        dm, dn = SECTOR_RAYS[t]
        em, en = SECTOR_RAYS[(t + 1) % 6]
        if dm * n - dn * m >= 0 and m * en - n * em > 0:
            return t
    raise ValueError(
        f"angular_sector_index: undefined for origin/invalid vertex ({m}, {n})"
    )


def on_primary_ray(m, n):
    """
    Return the index t in Z_6 of the primary lattice ray that vertex (m, n)
    lies on, or None if (m, n) is strictly interior to a sector. A vertex lies
    on ray d_t iff it is a positive integer multiple of d_t, tested exactly via
    a zero lattice cross product with d_t together with a positive dot product.
    """
    for t in range(6):
        dm, dn = SECTOR_RAYS[t]
        if dm * n - dn * m == 0 and (m * dm + n * dn) > 0:
            return t
    return None


def generate_boundary_zone_vertices(truncation_radius, r_sq):
    """
    Generate vertices in the boundary zone V_{T,r} at exact norm sqrt(r_sq)
    within the truncation radius R, using the base triangular lattice L.
    Stores (m, n) coordinate pairs for phase pair assignments and exact
    integer angular sector partitioning.
    """
    boundary_zone_vertices = []  # List to store boundary vertices (m, n)
    max_m = int(math.ceil(truncation_radius)) + 10  # Safe range for m, n
    for m in range(-max_m, max_m + 1):  # Loop over possible m values
        for n in range(-max_m, max_m + 1):  # Loop over possible n values
            if m == 0 and n == 0:
                continue  # Skip the origin to align with punctured complex
                          # plane X
            nsq = m * m + m * n + n * n  # Compute squared norm
            norm = math.sqrt(nsq)  # Compute actual norm
            if norm > truncation_radius:
                continue  # Skip if outside truncation radius
            if nsq == r_sq:  # Check if on the boundary zone V_{T,r}
                boundary_zone_vertices.append((m, n))  # Add coordinate pair
    return boundary_zone_vertices


def main():
    # Parse command-line arguments for inversion radius r and truncation radius R
    parser = argparse.ArgumentParser(
        description="Calculate vertex counts in Tri-Quarter radial dual "
                    "triangular lattice graph Lambda_r across zones and "
                    "angular sectors."
    )
    parser.add_argument(
        "r", type=float,
        help="Inversion radius r (must yield integer r_sq with lattice points)"
    )
    parser.add_argument(
        "R", type=float, help="Truncation radius R"
    )
    args = parser.parse_args()

    # Propose r_sq and compute max_nsq for Eisenstein representations
    r_sq_proposed = round(args.r * args.r)
    max_nsq = int(args.R * args.R) + 10
    representations = compute_eisenstein_representations(max_nsq)

    # Validate proposed r_sq as admissible (positive representations)
    if r_sq_proposed in representations and representations[r_sq_proposed] > 0:
        r_sq = r_sq_proposed
        effective_r = math.sqrt(r_sq)
        print(f"Valid r_sq = {r_sq}, effective r = {effective_r:.6f}")
    else:
        print(f"Invalid r = {args.r:.6f} (r_sq = {r_sq_proposed}, no lattice "
              f"points at this exact norm).")
        # Find next lower valid N
        lower_n = r_sq_proposed - 1
        while lower_n >= 1 and (
            lower_n not in representations or representations[lower_n] == 0
        ):
            lower_n -= 1
        # Find next higher valid N
        higher_n = r_sq_proposed + 1
        while higher_n <= max_nsq and (
            higher_n not in representations or representations[higher_n] == 0
        ):
            higher_n += 1
        if lower_n >= 1:
            print(
                f"Next lower valid r = sqrt({lower_n}) approx "
                f"{math.sqrt(lower_n):.6f}"
            )
        if higher_n <= max_nsq:
            print(
                f"Next higher valid r = sqrt({higher_n}) approx "
                f"{math.sqrt(higher_n):.6f}"
            )
        return

    # Generate truncated radial dual triangular lattice graph using imported function
    G_outer, G_inner, inversion_map = build_zone_subgraphs(args.R, r_sq)

    # Count vertices in outer and inner zones (excluding boundary)
    count_outer = len(G_outer.nodes)
    count_inner = len(G_inner.nodes)

    # Generate and count boundary zone vertices V_{T,r}
    boundary_zone_vertices = generate_boundary_zone_vertices(args.R, r_sq)
    count_boundary = len(boundary_zone_vertices)

    # Calculate total vertices in truncated Lambda_r^R
    total = count_outer + count_inner + count_boundary

    # Initialize sector counts for outer, inner, and boundary zones
    sector_counts_outer = [0] * 6
    for node, _ in G_outer.nodes(data=True):
        sector = angular_sector_index(node[0], node[1])
        sector_counts_outer[sector] += 1

    sector_counts_inner = [0] * 6
    for node, _ in G_inner.nodes(data=True):
        sector = angular_sector_index(node[0], node[1])
        sector_counts_inner[sector] += 1

    sector_counts_boundary = [0] * 6
    for m_v, n_v in boundary_zone_vertices:
        sector = angular_sector_index(m_v, n_v)
        sector_counts_boundary[sector] += 1

    # Print zone counts
    print(f"Outer zone vertices: {count_outer}")
    print(f"Inner zone vertices: {count_inner}")
    print(f"Boundary zone vertices: {count_boundary}")
    print(f"Total vertices: {total}")
    print(
        "Vertices per angular sector (outer + boundary + inner = total):"
    )
    for k in range(6):
        total_sector = (
            sector_counts_outer[k]
            + sector_counts_inner[k]
            + sector_counts_boundary[k]
        )
        print(
            f"S_{k}: {sector_counts_outer[k]} + "
            f"{sector_counts_boundary[k]} + "
            f"{sector_counts_inner[k]} = {total_sector}"
        )

    # Calculate and print average vertex count per angular sector
    print("\nAverage vertex count per angular sector:")
    print(f"Outer: {count_outer / 6:.2f}")
    print(f"Inner: {count_inner / 6:.2f}")
    print(f"Boundary: {count_boundary / 6:.2f}")
    print(f"Total: {total / 6:.2f}")

    # Count vertices on angular sector borders (primary rays at t * 60 degrees),
    # detected exactly from the integer coordinate pair (m, n) via on_primary_ray.
    ray_labels = [
        "East (0 deg)", "North-East (60 deg)", "North-West (120 deg)",
        "West (180 deg)", "South-West (240 deg)", "South-East (300 deg)"
    ]
    ray_counts_outer = [0] * 6
    for node, _ in G_outer.nodes(data=True):
        k = on_primary_ray(node[0], node[1])
        if k is not None:
            ray_counts_outer[k] += 1

    ray_counts_inner = [0] * 6
    for node, _ in G_inner.nodes(data=True):
        k = on_primary_ray(node[0], node[1])
        if k is not None:
            ray_counts_inner[k] += 1

    ray_counts_boundary = [0] * 6
    for m_v, n_v in boundary_zone_vertices:
        k = on_primary_ray(m_v, n_v)
        if k is not None:
            ray_counts_boundary[k] += 1

    print("\nVertices on angular sector borders (primary rays):")
    for k in range(6):
        print(
            f"{ray_labels[k]}: outer {ray_counts_outer[k]}, "
            f"boundary {ray_counts_boundary[k]}, "
            f"inner {ray_counts_inner[k]}"
        )


if __name__ == "__main__":
    main()
