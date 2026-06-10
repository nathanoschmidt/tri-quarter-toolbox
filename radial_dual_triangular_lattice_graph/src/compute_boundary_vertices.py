# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Helper Tool: Computing Boundary Vertices for Admissible Inversion Radii
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 28, 2025
# Last Updated: June 9, 2026
# Version: 1.1.0
#
# Description:
# This Python script computes the explicit boundary vertices in the boundary
# zone V_{T,r} for a given admissible inversion radius r where r^2 = N is an
# integer representable as m^2 + m n + n^2 for integers m, n not both zero
# (as per the definition of admissible inversion radii). It finds integer
# solutions (m, n) to this equation, computes their Cartesian positions,
# phases, and angular sectors S_t for t in Z_6, and groups them by sector
# to illustrate uniform equidistribution under the order-6 rotational
# symmetry of the dihedral group D_6. This aligns with examples like the
# boundary vertices for r = sqrt(7) (N=7) that yield exactly two vertices
# per angular sector, forming symmetric orbits as shown in the table of
# admissible inversion radii examples.
#
# Requirements:
# - Python 3.x (standard libraries: math, argparse)
#
# Usage:
# Run the script with: python compute_boundary_vertices.py [N]
# (e.g., python compute_boundary_vertices.py 7 for r = sqrt(7))
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import math
import argparse

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


def find_representations(N):
    """
    Find integer solutions (m, n) to m^2 + m*n + n^2 = N in the base triangular
    lattice L and assign each to its angular sector index t in Z_6 using exact
    integer arithmetic (no floating-point phase). This supports verification of
    boundary zone vertices V_{T,r} for admissible inversion radii r where
    r^2 = N has positive representations, ensuring symmetric distribution across
    angular sectors S_t under D_6 rotational symmetry.
    """
    reps = []  # List of (sector, (m, n))
    # Safe range for m, n to cover all possible solutions without overflow for small N
    max_m = int(math.ceil(math.sqrt(N))) + 10
    for m in range(-max_m, max_m + 1):
        for n in range(-max_m, max_m + 1):
            # Skip origin to align with punctured complex plane X = C \ {0}
            if m == 0 and n == 0: continue
            # Compute squared Euclidean norm (integer, exact for Eisenstein integers)
            norm_sq = m*m + m*n + n*n
            if norm_sq == N:  # Check if on the boundary circle of radius r
                # Exact integer angular sector index t in Z_6 for mod 6
                # partitioning and order-6 rotational invariance
                sector = angular_sector_index(m, n)
                reps.append((sector, (m, n)))
    # Sort by sector index then coordinates for ordered output that highlights
    # equidistribution across angular sectors S_t
    reps.sort(key=lambda item: (item[0], item[1]))
    return reps

def main():
    # Parse command-line arguments for flexibility in specifying N = r^2
    parser = argparse.ArgumentParser(
        description="Compute boundary vertices for admissible r^2 = N in the "
                    "base triangular lattice L, grouped by angular sector to "
                    "demonstrate symmetric distribution under D_6."
    )
    parser.add_argument(
        "N", type=int, nargs="?", default=7,
        help="Integer N = r^2 (default: 7 for r = sqrt(7) example with 12 "
             "boundary vertices)"
    )
    args = parser.parse_args()

    # Find all representations for the given N
    representations = find_representations(args.N)

    # Print results grouped by sector, showing uniform equidistribution
    # (e.g., exactly k vertices per sector for |V_{T,r}| = 6k)
    print(f"Boundary vertices for N={args.N} (r=sqrt({args.N})):")
    for rep in representations:
        sector, (m, n) = rep
        print(f"Sector {sector}: (m,n)=({m},{n})")

if __name__ == "__main__":
    main()
