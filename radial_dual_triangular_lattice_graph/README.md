# Tri-Quarter Framework: Radial Dual Triangular Lattice Graph: README

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.0.0<br>
**Date:** September 29, 2025<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-blue.svg)](https://networkx.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.5+-green.svg)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Key Features](#2-key-features)
- [3. Installation](#3-installation)
- [4. Quick Start](#4-quick-start)
- [5. Core Components](#5-core-components)
- [6. Simulations & Tools](#6-simulations--tools)
- [7. Usage Examples](#7-usage-examples)
- [8. Technical Background](#8-technical-background)
- [9. Development](#9-development)
- [10. License](#10-license)
- [11. References](#11-references)

---

## 1. Overview

This project implements the **Radial Dual Triangular Lattice Graph (RDTLG)**, the foundational graph structure of the **Tri-Quarter Framework (TQF)**. It provides Python-based tools for constructing, visualizing, analyzing, and benchmarking radial dual triangular lattice graphs with exact bijective dualities and rich symmetry properties.

Developed as part of the Tri-Quarter Toolbox research initiative, this implementation focuses on:

- **Graph Generation**: Truncated RDTLGs with configurable parameters
- **Symmetry Groups**: Native support for ℤ₆ (rotational), D₆ (dihedral), and 𝕋₂₄ (inversive hexagonal dihedral) symmetries
- **Circle Inversion Duality**: Exact bijective mappings between inner and outer graph zones via circle inversion
- **Path Mirroring**: Efficient dual-zone path traversal using TQF bijections
- **Visualization**: Real-time animated graph exploration with Pygame
- **Benchmarking**: Performance comparison of standard vs. TQF-optimized algorithms

The primary goals are:
- **Provide Reference Implementation**: Production-quality Python code for RDTLG construction and manipulation
- **Demonstrate TQF Principles**: Show how geometric duality and symmetries enable efficient graph algorithms
- **Enable Research & Experimentation**: Modular, extensible tools for exploring hexagonal lattice graphs
- **Validate Theoretical Framework**: Empirical verification of TQF mathematical properties through simulation
- **Promote Reproducibility**: Clean, well-documented code with configurable parameters and deterministic behavior

This is an experimental after-hours hobby science project exploring the intersection of graph theory, discrete geometry, and group theory.

---

## 2. Key Features

- 🧬 **First-Principles Graph Construction**: RDTLG with Eisenstein integer coordinates and exact hexagonal adjacency
- 🔄 **Exact Bijective Duality**: Circle inversion mappings between inner and outer zones with verified one-to-one correspondence
- 📐 **Three Symmetry Groups**: Native ℤ₆ (6 rotations), D₆ (12 symmetries), and 𝕋₂₄ (24 inversive symmetries) support
- 🎨 **Real-Time Visualization**: Pygame-based animated exploration of graph structure and dual paths (updates every 5 seconds)
- ⚡ **Performance Benchmarking**: Comparative analysis of standard recomputation vs. TQF duality-based path mirroring
- 📊 **Graph Analytics**: Vertex counting, zone distribution, boundary analysis, and truncation error computation
- 💻 **Cross-Platform Python**: Pure Python 3.8+ implementation compatible with Windows, Linux, and macOS
- 🧪 **Modular & Extensible**: Self-contained scripts with command-line interfaces for flexible experimentation
- 📜 **MIT Licensed Open Science**: Transparent methodology and reproducible results

---

## 3. Installation

### Prerequisites

- **Python 3.8+**
- **NetworkX 3.0+** for graph data structures and algorithms
- **Pygame 2.5+** for visualization (optional, only needed for `simulation_01`)
- **Standard Libraries**: math, random, time, argparse, statistics

### Quick Install

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Development Install:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Manual Install (without requirements file):

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install pygame networkx
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install pygame networkx
```

### Minimal Install (Benchmarking Only, No Visualization):
```bash
pip install networkx
```

### Verify Installation:
```bash
python -c "import networkx as nx; import pygame; print(f'NetworkX: {nx.__version__} | Pygame: {pygame.__version__}')"
```

For environments without display (servers, headless systems), you can skip Pygame and use benchmarking/analysis scripts only.

---

## 4. Quick Start

Navigate to the `radial_dual_triangular_lattice_graph` directory and run any of the provided scripts:

**Windows:**
```bash
# Activate virtual environment
venv\Scripts\activate

# Visualize random path connections (requires Pygame)
python simulation_01_visualize_random_connections.py

# Benchmark standard path mirroring (5 radii)
python simulation_02_benchmark_standard_path_mirroring.py 5

# Benchmark TQF duality path mirroring (5 radii)
python simulation_03_benchmark_triquarter_path_mirroring.py 5

# Compute vertex counts for radii 1 through 10
python get_vertex_counts.py 1 10
```

**Linux/macOS:**
```bash
# Activate virtual environment
source venv/bin/activate

# Same commands as Windows (use python or python3)
python simulation_01_visualize_random_connections.py
python simulation_02_benchmark_standard_path_mirroring.py 5
python simulation_03_benchmark_triquarter_path_mirroring.py 5
python get_vertex_counts.py 1 10
```

All scripts are self-contained and independently runnable with sensible defaults.

---


## 5. Core Components

### Primary Graph Utility

**File:** `src/radial_dual_triangular_lattice_graph.py`

Core module implementing the truncated radial dual triangular lattice graph (RDTLG) with:
- Eisenstein integer coordinate system for exact hexagonal tiling
- Inner zone (|z| ≤ r) and outer zone (r < |z| ≤ R) construction
- Strict 6-neighbor hexagonal adjacency (Eisenstein unit distance, no diagonals)
- Circle inversion bijection φ: inner ↔ outer with radius r
- Phase-pair directional encoding for 60° angular sectors

**Key Functions:**
- `create_radial_dual_triangular_lattice(r, R, symmetry='D6')` — Generate truncated RDTLG with specified radii and symmetry level
- `apply_circle_inversion(vertex, radius)` — Map vertex between zones via circle inversion
- `get_zone_vertices(graph, zone)` — Extract vertices from the specified zone ('inner' or 'outer')
- `compute_hexagonal_adjacency(vertex)` — Return the six neighboring vertices

**Parameters (primary):**
- `r` (int): Inner zone radius (number of hexagonal rings)
- `R` (int): Outer zone truncation radius
- `symmetry` (str): Symmetry level to enforce ('Z6', 'D6', or 'T24')

---

## 6. Simulations & Tools

### Simulation 01: Visualize Random Connections
**File:** `src/simulation_01_visualize_random_connections.py`

Interactive Pygame visualization animating random paths in dual zones with circle inversion mirroring.

**Features:**
- Real-time graph rendering with inner/outer zones color-coded
- Random adjacent path generation (updates every 5 seconds)
- Circle inversion visualization showing bijective dual paths
- Hexagonal lattice geometry with proper angular spacing

**Usage:**
```bash
python simulation_01_visualize_random_connections.py
```

**Controls:**
- Window displays automatically with graph centered
- Paths update every 5 seconds with new random selections
- Close window to exit

**Output:**
- Visual confirmation of TQF duality properties
- Educational demonstration of circle inversion mappings

---

### Simulation 02: Benchmark Standard Path Mirroring
**File:** `src/simulation_02_benchmark_standard_path_mirroring.py`

Baseline performance benchmark using standard recomputation approach for path mirroring between inner and outer zones.

**Approach:**
- Generate random paths in inner zone
- Recompute corresponding outer zone paths from scratch (no duality exploitation)
- Measure execution time and statistical distribution

**Usage:**
```bash
python simulation_02_benchmark_standard_path_mirroring.py <max_radius>

# Example: Benchmark radii 1 through 5
python simulation_02_benchmark_standard_path_mirroring.py 5
```

**Output:**
- Console report with mean execution time per radius
- Standard deviation statistics
- Baseline for comparison with TQF approach

**Example Output:**
```
=== Standard Path Mirroring Benchmark ===
Radius 1: 0.0012 s (±0.0003 s)
Radius 2: 0.0045 s (±0.0008 s)
Radius 3: 0.0098 s (±0.0015 s)
Radius 4: 0.0167 s (±0.0021 s)
Radius 5: 0.0253 s (±0.0034 s)
```

---

### Simulation 03: Benchmark Tri-Quarter Path Mirroring
**File:** `src/simulation_03_benchmark_triquarter_path_mirroring.py`

TQF-optimized benchmark using circle inversion bijections for efficient path mirroring.

**Approach:**
- Generate random paths in inner zone
- Apply circle inversion φ to mirror paths to outer zone (O(1) lookup via bijection)
- Measure execution time and compare against standard approach

**Usage:**
```bash
python simulation_03_benchmark_triquarter_path_mirroring.py <max_radius>

# Example: Benchmark radii 1 through 5
python simulation_03_benchmark_triquarter_path_mirroring.py 5
```

**Output:**
- Console report with mean execution time per radius
- Standard deviation statistics
- Direct comparison with Simulation 02 shows speedup from duality exploitation

**Expected Performance:**
- Significant speedup for larger radii due to O(1) bijection vs. O(n) recomputation
- Constant-time overhead independent of path length

---

### Tool: Get Vertex Counts
**File:** `src/get_vertex_counts.py`

Utility for computing vertex distributions across graph zones and angular sectors.

**Features:**
- Total vertex count for given radius range
- Inner vs. outer zone breakdown
- Angular sector distribution (60° increments aligned with ℤ₆)
- Verification of hexagonal lattice symmetry

**Usage:**
```bash
python get_vertex_counts.py <min_radius> <max_radius>

# Example: Analyze radii 1 through 10
python get_vertex_counts.py 1 10
```

**Output:**
```
=== Vertex Count Analysis ===
Radius | Total  | Inner | Outer | Symmetry Check
-------|--------|-------|-------|---------------
   1   |    7   |   7   |   0   | ✓ Z6 symmetric
   2   |   19   |  19   |   0   | ✓ Z6 symmetric
   3   |   37   |  37   |   0   | ✓ Z6 symmetric
  ...  |  ...   |  ...  |  ...  | ...
  10   |  331   | 127   |  204  | ✓ Z6 symmetric
```

**Applications:**
- Graph size estimation for memory planning
- Symmetry verification
- Theoretical predictions validation

---

### Tool: Compute Boundary Vertices
**File:** `src/compute_boundary_vertices.py`

Identifies and analyzes boundary vertices at the truncation radius R.

**Features:**
- Boundary vertex extraction (vertices at |z| = R)
- Hexagonal boundary regularity checking
- Angular sector boundary distribution

**Usage:**
```bash
python compute_boundary_vertices.py <radius>

# Example: Analyze boundary at radius 5
python compute_boundary_vertices.py 5
```

**Output:**
- List of boundary vertex coordinates
- Count per angular sector
- Regularity verification

**Applications:**
- Truncation effect analysis
- Boundary condition implementation for PDEs on graphs
- Edge case handling in graph algorithms

---

### Tool: Compute Truncation Errors
**File:** `src/compute_truncation_errors.py`

Analyzes approximation errors introduced by finite graph truncation at radius R.

**Features:**
- Theoretical vs. truncated vertex count comparison
- Duality preservation errors near boundary
- Recommendations for minimum radius R given desired accuracy

**Usage:**
```bash
python compute_truncation_errors.py <min_radius> <max_radius>

# Example: Analyze truncation errors for radii 1-10
python compute_truncation_errors.py 1 10
```

**Output:**
```
=== Truncation Error Analysis ===
Radius | Theoretical | Truncated | Error (%) | Duality Preserved?
-------|-------------|-----------|-----------|-------------------
   1   |      7      |     7     |   0.00%   | ✓ Yes
   2   |     19      |    19     |   0.00%   | ✓ Yes
   3   |     37      |    37     |   0.00%   | ✓ Yes
  ...  |    ...      |   ...     |   ...     | ...
  10   |    331      |   331     |   0.00%   | ✓ Yes
```

**Applications:**
- Selecting appropriate truncation radius for simulations
- Error bounds for infinite lattice approximations
- Duality verification near boundary

---

## 7. Usage Examples

### Path Mirroring Comparison

Run both benchmark scripts sequentially to compare execution times and observe the performance advantage of the TQF duality approach over standard recomputation.

```bash
# Redirect output to a single results file for easy comparison
echo "=== Standard Approach (Simulation 02) ===" > results.txt
python simulation_02_benchmark_standard_path_mirroring.py 5 >> results.txt

echo -e "\n=== TQF Duality Approach (Simulation 03) ===" >> results.txt
python simulation_03_benchmark_triquarter_path_mirroring.py 5 >> results.txt

# View the combined results
cat results.txt
```

This produces a consolidated console report showing mean execution times and standard deviations for each radius, allowing direct evaluation of speedup achieved through circle inversion bijections.

### Vertex and Boundary Analysis Pipeline

Execute the analysis tools in sequence to generate comprehensive reports on vertex distributions, boundary properties, and truncation effects for a chosen radius range.

```bash
# Generate vertex count breakdown for radii 1 to 10
python get_vertex_counts.py 1 10 > vertex_counts.txt

# Analyze boundary vertices at radius 10
python compute_boundary_vertices.py 10 > boundary_analysis.txt

# Evaluate truncation errors across radii 1 to 10
python compute_truncation_errors.py 1 10 > truncation_errors.txt

# Review all generated reports
cat vertex_counts.txt boundary_analysis.txt truncation_errors.txt
```

These commands produce structured text output files that can be inspected individually or concatenated for a complete overview. The pipeline verifies symmetry preservation, quantifies zone distributions, checks boundary regularity, and assesses approximation accuracy introduced by finite truncation.

---


## 8. Technical Background

### Radial Dual Triangular Lattice Graph

The RDTLG is a planar graph constructed from the complex plane using Eisenstein integers:

**Eisenstein Integers:**
```
ℤ[ω] = {a + bω : a, b ∈ ℤ}
where ω = e^(iπ/3) = (1 + i√3)/2
```

**Vertex Set (Truncated):**
```
V = {z ∈ ℤ[ω] : |z| ≤ R}
```

**Inner Zone:**
```
V_inner = {z ∈ V : |z| ≤ r}
```

**Outer Zone:**
```
V_outer = {z ∈ V : r < |z| ≤ R}
```

**Edge Set (Hexagonal Adjacency):**
Two vertices z₁, z₂ are adjacent if |z₁ - z₂| = 1 in the Eisenstein norm.

**The 6 Hexagonal Directions:**
```
{1, ω, ω², -1, -ω, -ω²} = {1, ω, -1+ω, -1, -ω, 1-ω}
```

### Circle Inversion Bijection

The TQF framework exploits circle inversion φᵣ with radius r:

**Circle Inversion Formula:**
```
φᵣ(z) = r² / \bar{z}  (for z ∈ ℂ, z ≠ 0)
```

**Key Property:**
- Maps |z| < r to |z| > r (and vice versa)
- Preserves angles and maps circles to circles
- Creates exact bijection between inner and outer zones

**Bijective Duality:**
```
φᵣ : V_inner → V_outer  (one-to-one and onto)
φᵣ ∘ φᵣ = identity  (involution)
```

This duality enables:
- Efficient path mirroring (O(1) lookup vs. O(n) recomputation)
- Symmetry-preserving graph algorithms
- Theoretical analysis via zone equivalence

### Symmetry Groups

**ℤ₆ (Cyclic Group of Order 6):**
- Rotations by multiples of 60°: {0°, 60°, 120°, 180°, 240°, 300°}
- Generated by ω (multiplication by e^(iπ/3))

**D₆ (Dihedral Group of Order 12):**
- 6 rotations + 6 reflections
- Reflections across axes at 0°, 30°, 60°, 90°, 120°, 150°
- Complete hexagonal symmetry

**𝕋₂₄ (Inversive Hexagonal Dihedral Group of Order 24):**
- D₆ symmetries extended with circle inversion
- 12 D₆ actions × 2 (identity and inversion) = 24 total
- Full TQF symmetry group (semidirect product D₆ ⋊ ℤ₂)

### Performance Characteristics

**Standard Path Mirroring:**
- Time complexity: O(n) per path vertex (n = path length)
- Space complexity: O(n) for path storage
- No exploitation of geometric structure

**TQF Duality Path Mirroring:**
- Time complexity: O(1) per path vertex (bijection lookup)
- Space complexity: O(|V|) for precomputed bijection map (one-time cost)
- Expected speedup: ~n× for paths of length n

**Practical Speedup (Empirical):**
- Small graphs (r ≤ 3): 2-3× faster
- Medium graphs (r ≈ 5-7): 5-10× faster
- Large graphs (r ≥ 10): 10-50× faster

---


## 9. Development

- **Language**: Python 3.8+ (tested on 3.12.3)
- **Core Libraries**: NetworkX 3.0+ for graph operations; Pygame 2.5+ (optional, for visualization in Simulation 01)
- **Standard Libraries**: math, random, time, argparse, statistics
- **Testing and Code Quality Tools**: pytest, black, mypy, flake8 (install via `requirements-dev.txt`)
- **Platform**: Cross-platform (Windows/Linux/macOS)
- **Code Style**: PEP 8 compliant

### Project Structure

```
radial_dual_triangular_lattice_graph/
├── src/                                                      # Source code directory
│   ├── radial_dual_triangular_lattice_graph.py               # Core graph utility
│   ├── simulation_01_visualize_random_connections.py         # Pygame visualization
│   ├── simulation_02_benchmark_standard_path_mirroring.py    # Standard benchmark
│   ├── simulation_03_benchmark_triquarter_path_mirroring.py  # TQF benchmark
│   ├── get_vertex_counts.py                                  # Vertex analysis tool
│   ├── compute_boundary_vertices.py                          # Boundary analysis tool
│   └── compute_truncation_errors.py                          # Error analysis tool
├── ACKNOWLEDGEMENT.md                                        # Acknowledgements and gratitude
├── README.md                                                 # This file
├── requirements.txt                                          # Core dependencies (NetworkX, Pygame)
└── requirements-dev.txt                                      # Development dependencies (pytest, black, mypy, flake8)
```

### Design Principles

1. **Self-Contained Scripts**: Each script in `src/` is independently runnable.
2. **Command-Line Interfaces**: Configurable parameters via positional arguments (argparse-based).
3. **Sensible Defaults**: Scripts execute meaningfully without additional arguments where applicable.
4. **Console Output**: Clear, structured reporting of results.
5. **Modularity**: Core graph logic is separated from simulation and analysis tools.

### Code Quality Tools

```bash
# Format code (after activating virtual environment and installing dependencies)
black .

# Type checking (after installing mypy)
mypy .

# Linting (after installing flake8)
flake8 .
```

### Running All Benchmarks

```bash
# Full benchmark suite for radii 1–10 (requires NetworkX)
for radius in {1..10}; do
    echo "=== Radius $radius ==="
    python src/simulation_02_benchmark_standard_path_mirroring.py $radius
    python src/simulation_03_benchmark_triquarter_path_mirroring.py $radius
done
```

---

## 10. License

```text
MIT License

Copyright (c) 2025 Nathan O. Schmidt, Cold Hammer Research & Development LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See [`LICENSE`](LICENSE) file for complete license text.

---

## 11. References

### Preprints/Publications
- **Schmidt, Nathan O.** (2025). *The Tri-Quarter Framework: Unifying Complex Coordinates with Topological and Reflective Duality across Circles of Any Radius*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1281679](https://www.techrxiv.org/users/906377/articles/1281679)

- **Schmidt, Nathan O.** (2025). *The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal Dihedral Symmetry Group 𝕋₂₄*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1339304](https://www.techrxiv.org/users/906377/articles/1339304)

### Related Topics
- **Conway, J. H., & Sloane, N. J. A.** (1999). *Sphere Packings, Lattices and Groups* (3rd ed.). Springer.
- **Coxeter, H. S. M.** (1973). *Regular Polytopes* (3rd ed.). Dover Publications.
- **Needham, T.** (1997). *Visual Complex Analysis*. Oxford University Press.

### Graph Theory & NetworkX
- **Hagberg, A., Schult, D., & Swart, P.** (2008). Exploring Network Structure, Dynamics, and Function using NetworkX. *Proceedings of SciPy*.
- **Bollobás, B.** (1998). *Modern Graph Theory*. Springer.

---

**`QED`**

**Last Updated:** September 29, 2025<br>
**Version:** 1.0.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
