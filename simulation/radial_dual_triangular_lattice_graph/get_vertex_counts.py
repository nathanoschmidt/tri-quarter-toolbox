import math
import argparse
from triangular_lattice_graph import build_radial_dual_triangular_lattice_graph

def compute_representations(max_nsq):
    """Compute the number of representations of integers as sums of three squares in the triangular lattice."""
    representations = {}  # Dictionary to store the count of representations for each nsq
    max_m = int(math.ceil(math.sqrt(max_nsq))) + 10  # Safe range for m, n to cover all possible nsq <= max_nsq
    for m in range(-max_m, max_m + 1):  # Loop over possible m values
        for n in range(-max_m, max_m + 1):  # Loop over possible n values
            if m == 0 and n == 0:
                continue  # Skip the origin
            nsq = m * m + m * n + n * n  # Compute the squared norm in triangular lattice
            if nsq > max_nsq:
                continue  # Skip if beyond the maximum
            if nsq not in representations:
                representations[nsq] = 0  # Initialize count if not present
            representations[nsq] += 1  # Increment the representation count
    return representations

def compute_sector(phase):
    """Determine the angular sector (0-5) for a given phase angle."""
    if phase < 0:
        phase += 2 * math.pi  # Normalize phase to [0, 2*pi)
    return math.floor(6 * phase / (2 * math.pi)) % 6  # Compute sector index

def generate_boundary_vertices(R, r_sq):
    """Generate vertices on the boundary at radius sqrt(r_sq) within truncation R."""
    boundary_vertices = []  # List to store boundary vertices (m, n, phase)
    max_m = int(math.ceil(R)) + 10  # Safe range for m, n
    for m in range(-max_m, max_m + 1):  # Loop over possible m values
        for n in range(-max_m, max_m + 1):  # Loop over possible n values
            if m == 0 and n == 0:
                continue  # Skip the origin
            nsq = m * m + m * n + n * n  # Compute squared norm
            norm = math.sqrt(nsq)  # Compute actual norm
            if norm > R:
                continue  # Skip if outside truncation radius
            if nsq == r_sq:  # Check if on the boundary
                x = m + n * 0.5  # Compute Cartesian x-coordinate
                y = n * (math.sqrt(3) / 2)  # Compute Cartesian y-coordinate
                phase = math.atan2(y, x)  # Compute phase angle
                boundary_vertices.append((m, n, phase))  # Add to list
    return boundary_vertices

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate vertex counts in Tri-Quarter radial dual triangular lattice graph.")
    parser.add_argument("r", type=float, help="Inversion radius r (must yield integer r_sq with lattice points)")
    parser.add_argument("R", type=float, help="Truncation radius R")
    args = parser.parse_args()

    # Propose r_sq and compute max_nsq for representations
    r_sq_proposed = round(args.r * args.r)
    max_nsq = int(args.R * args.R) + 10
    representations = compute_representations(max_nsq)

    # Check if proposed r_sq is valid
    if r_sq_proposed in representations and representations[r_sq_proposed] > 0:
        r_sq = r_sq_proposed
        effective_r = math.sqrt(r_sq)
        print(f"Valid r_sq = {r_sq}, effective r = {effective_r:.6f}")
    else:
        print(f"Invalid r = {args.r:.6f} (r_sq = {r_sq_proposed}, no lattice points at this exact norm).")
        # Find next lower valid N
        lower_n = r_sq_proposed - 1
        while lower_n >= 1 and (lower_n not in representations or representations[lower_n] == 0):
            lower_n -= 1
        # Find next higher valid N
        higher_n = r_sq_proposed + 1
        while higher_n <= max_nsq and (higher_n not in representations or representations[higher_n] == 0):
            higher_n += 1
        if lower_n >= 1:
            print(f"Next lower valid r = sqrt({lower_n}) ~ {math.sqrt(lower_n):.6f}")
        if higher_n <= max_nsq:
            print(f"Next higher valid r = sqrt({higher_n}) ~ {math.sqrt(higher_n):.6f}")
        return

    # Generate graphs using the imported function
    G_outer, G_inner, inversion_map = build_radial_dual_triangular_lattice_graph(args.R, r_sq)

    # Count vertices in outer and inner zones
    count_outer = len(G_outer.nodes)
    count_inner = len(G_inner.nodes)

    # Generate and count boundary vertices
    boundary_vertices = generate_boundary_vertices(args.R, r_sq)
    count_boundary = len(boundary_vertices)

    # Calculate total vertices
    total = count_outer + count_inner + count_boundary

    # Initialize sector counts for outer, inner, and boundary
    sector_counts_outer = [0] * 6
    for _, data in G_outer.nodes(data=True):
        sector = compute_sector(data['phase'])
        sector_counts_outer[sector] += 1

    sector_counts_inner = [0] * 6
    for _, data in G_inner.nodes(data=True):
        sector = compute_sector(data['phase'])
        sector_counts_inner[sector] += 1

    sector_counts_boundary = [0] * 6
    for _, _, phase in boundary_vertices:
        sector = compute_sector(phase)
        sector_counts_boundary[sector] += 1

    # Print counts
    print(f"Outer zone vertices: {count_outer}")
    print(f"Inner zone vertices: {count_inner}")
    print(f"Boundary zone vertices: {count_boundary}")
    print(f"Total vertices: {total}")
    print("Vertices per angular sector (outer + boundary + inner = total):")
    for k in range(6):
        total_sector = sector_counts_outer[k] + sector_counts_inner[k] + sector_counts_boundary[k]
        print(f"S_{k}: {sector_counts_outer[k]} + {sector_counts_boundary[k]} + {sector_counts_inner[k]} = {total_sector}")

    # Calculate and print averages
    print("\nAverage vertex count per angular sector:")
    print(f"Outer: {count_outer / 6:.2f}")
    print(f"Inner: {count_inner / 6:.2f}")
    print(f"Boundary: {count_boundary / 6:.2f}")
    print(f"Total: {total / 6:.2f}")

if __name__ == "__main__":
    main()
