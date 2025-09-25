# =============================================================================
# The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact
# Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal
# Dihedral Symmetry Group T_24
#
# Helper Tool: Computing Lambda_r Truncation Errors
#
# Author: Nathan O. Schmidt
# Affiliation: Cold Hammer Research & Development LLC, Eagle, Idaho, USA
# Email: nate.o.schmidt@coldhammer.net
# Date: September 24, 2025
#
# Description:
# This Python script computes truncation error percentages for various
# truncation radii R with inversion radius r=1 for the radial dual triangular
# lattice graph Lambda_r to quantify the unresolved area near the origin in
# the inner zone Lambda_{-,1} as a fraction of the total viewed area.
# The unresolved area is pi (r^2 / R)^2 = pi / R^2 (for r=1), and the total
# viewed area is approximated as pi R^2. The results are output in a table-like
# format.
#
# Requirements:
# - Python 3.x (uses standard math module; no external dependencies)
#
# Usage:
# Run the script with: python compute_truncation_errors.py
#
# Source code is freely available at:
# https://github.com/nathanoschmidt/tri-quarter-toolbox/
# (MIT License; see repository LICENSE for details)
#
# =============================================================================

import math

# Parameters for the radial dual triangular lattice graph Lambda_r
r = 1.0  # Admissible inversion radius (r=1 yields symmetric boundary
         # set V_{T,1} with |V_{T,1}|=6 vertices forming a unit hexagon,
         # aligned with primary rays at phases t pi / 3 for t in Z_6)
Rs = [4, 10, 20, 50]  # List of truncation radii R to compute errors for
                      # (ensures R >> r for balanced finite approximations
                      # of the infinite Lambda_r with gaps scaling as O(1/R^2))

# Compute truncation errors for each R
# (unresolved area near punctured origin in inner zone Lambda_{-,r} as fraction
# of total viewed area; aligns with "looking scope" approximation)
print("Truncation Error Percentages for Various R (with r=1):")
print("R | Unresolved Area (pi r^4 / R^2) | Total Viewed Area (~ pi R^2) | "
      "Percentage (%)")
for R in Rs:
    # Unresolved area: pi (r^2 / R)^2 near origin in Lambda_{-,r}
    # (vanishing as R -> infinity, enabling scalable finite simulations)
    unresolved_area = math.pi * (r**4 / R**2)
    # Total viewed area approximation: pi R^2 (disk area up to truncation R)
    total_area_approx = math.pi * R**2
    # Error percentage: (unresolved / total) * 100
    percentage = (unresolved_area / total_area_approx) * 100
    print(f"{R} | {unresolved_area:.4f} | {total_area_approx:.2f} | "
          f"{percentage:.4f}%")
