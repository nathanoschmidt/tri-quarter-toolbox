# Tri-Quarter Framework: Radial Dual Triangular Lattice Graph: README

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.1.0<br>
**Last Updated:** June 10, 2026<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-blue.svg)](https://networkx.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.5+-green.svg)](https://www.pygame.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
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

- **Graph Generation**: Truncated RDTLGs (zone subgraphs and the complete lattice) with configurable truncation radius R and admissible inversion radius r
- **Symmetry Groups**: Native support for ℤ₆ (rotational), D₆ (dihedral), and 𝕋₂₄ (inversive hexagonal dihedral) symmetries
- **Circle Inversion Duality**: Exact bijective mappings between inner and outer graph zones via circle inversion ιᵣ
- **Path Mirroring**: Efficient dual-zone shortest-path traversal using TQF inversion bijections
- **Symmetry-Reduced Clustering**: Exact-rational average local clustering via ℤ₆-orbit reduction
- **Conflict-Free Parallelism**: Equivariant trihexagonal six-coloring enabling data-parallel relaxation sweeps on CPU and GPU
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

- 🧬 **First-Principles Graph Construction**: RDTLG with Eisenstein integer coordinates and exact degree-6 triangular-lattice adjacency
- 🔄 **Exact Bijective Duality**: Circle inversion mappings between inner and outer zones with verified one-to-one correspondence
- 📐 **Three Symmetry Groups**: Native ℤ₆ (6 rotations), D₆ (12 symmetries), and 𝕋₂₄ (24 inversive symmetries) support
- 🎨 **Real-Time Visualization**: Pygame-based animated exploration of graph structure and dual paths (updates every 5 seconds)
- ⚡ **Performance Benchmarking**: Comparative analysis of standard recomputation vs. TQF duality-based path mirroring and symmetry-reduced clustering
- 🎯 **Exact-Rational Arithmetic**: Clustering coefficients computed as exact `fractions.Fraction` values, bitwise-reproducible and verified by `==` (not float tolerance)
- 🌈 **Equivariant Six-Coloring**: Proper trihexagonal six-coloring scheduling conflict-free, lock-free parallel relaxation sweeps (CPU NumPy vs. GPU PyTorch)
- 📊 **Graph Analytics**: Vertex counting, zone/angular-sector distribution, boundary analysis, and truncation error computation
- 💻 **Cross-Platform Python**: Python 3.8+ implementation compatible with Windows, Linux, and macOS
- 🧪 **Modular & Extensible**: Self-contained scripts with command-line interfaces for flexible experimentation
- 📜 **MIT Licensed Open Science**: Transparent methodology and reproducible results

---

## 3. Installation

### Prerequisites

- **Python 3.8+**
- **NetworkX 3.0+** for graph data structures and algorithms (core module + Simulations 02–06 + `get_vertex_counts.py`)
- **NumPy 1.24+** required for `simulation_06`; optional vectorized acceleration for `simulation_05`
- **Pygame 2.5+** for visualization (optional, only needed for `simulation_01`)
- **PyTorch 2.0+** optional, only for the GPU backend of `simulation_06` (CPU NumPy baseline runs without it)
- **Standard Libraries**: math, cmath, random, time, argparse, statistics, fractions

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
pip install networkx numpy pygame
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install networkx numpy pygame
```

### Optional GPU Backend (Simulation 06):
```bash
# Install a PyTorch build matched to your platform / CUDA driver.
# See https://pytorch.org/get-started/locally/ for the correct command.
pip install torch
```

### Minimal Install (Benchmarking Only, No Visualization):
```bash
# Covers the core module, path-mirroring (02/03) and clustering (04/05) benchmarks
pip install networkx numpy
```

### Verify Installation:
```bash
python -c "import networkx as nx; import numpy as np; print(f'NetworkX: {nx.__version__} | NumPy: {np.__version__}')"
```

For environments without a display (servers, headless systems), you can skip Pygame and use the benchmarking/analysis scripts only. The GPU backend of Simulation 06 falls back to the CPU PyTorch device when no CUDA device is present, and the CPU/NumPy baseline runs even when PyTorch is not installed.

---

## 4. Quick Start

Navigate to the `radial_dual_triangular_lattice_graph` directory and run any of the provided scripts (each script adds its own `src/` directory to the import path, so the `src/` prefix shown below works from this base directory):

**Windows:**
```bash
# Activate virtual environment
venv\Scripts\activate

# Visualize random connections with circle-inversion mirroring (requires Pygame)
python src/simulation_01_visualize_random_connections.py

# Benchmark standard (recompute) dual-zone path mirroring at truncation radius R=15
python src/simulation_02_benchmark_standard_path_mirroring.py 15

# Benchmark TQF inversion-based path mirroring at truncation radius R=15
python src/simulation_03_benchmark_triquarter_path_mirroring.py 15

# Benchmark standard vs. TQF symmetry-reduced clustering at R=100
python src/simulation_04_benchmark_standard_clustering.py 100
python src/simulation_05_benchmark_triquarter_clustering.py 100

# Benchmark trihexagonal six-coloring parallel relaxation (CPU vs. GPU) at R=200
python src/simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py 200

# Count vertices for inversion radius r=1, truncation radius R=4
python src/get_vertex_counts.py 1 4
```

**Linux/macOS:**
```bash
# Activate virtual environment
source venv/bin/activate

# Same commands as Windows (use python or python3)
python src/simulation_01_visualize_random_connections.py
python src/simulation_02_benchmark_standard_path_mirroring.py 15
python src/simulation_03_benchmark_triquarter_path_mirroring.py 15
python src/simulation_04_benchmark_standard_clustering.py 100
python src/simulation_05_benchmark_triquarter_clustering.py 100
python src/simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py 200
python src/get_vertex_counts.py 1 4
```

All scripts are self-contained and independently runnable with sensible defaults (each accepts `-h`/`--help` for its full argument list).

---


## 5. Core Components

### Primary Graph Utility

**File:** `src/radial_dual_triangular_lattice_graph.py`

Core module implementing the truncated radial dual triangular lattice graph (RDTLG) Λᵣ and its truncation Λᵣᴿ, with:
- Eisenstein integer coordinate system `(m, n)` for exact triangular tiling; nodes are 3-tuples `(m, n, type)` where `type ∈ {'outer', 'inner', 'boundary'}`
- Outer zone (r < |z| ≤ R) and inner zone (0 < |z| < r) subgraph construction, plus boundary zone V_{T,r} at |z| = r
- Strict degree-6 triangular-lattice adjacency via the neighbor deltas `(1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)` (Euclidean unit spacing)
- Circle inversion bijection ιᵣ: outer ↔ inner, encoded as an `inversion_map` dict
- Node attributes `pos` (Cartesian position) and `norm_sq` (squared Eisenstein norm) for angular-sector partitioning and inversion

**Key Functions:**
- `build_zone_subgraphs(R, r_sq=1)` — Build the separate outer (`Λ₊,ᵣᴿ`) and inner (`Λ₋,ᵣᴿ`) zone subgraphs plus the outer→inner `inversion_map`, for isolated zone operations like path mirroring (Simulations 02–03). Returns `(G_outer, G_inner, inversion_map)`.
- `build_complete_lattice_graph(R, r_sq=1)` — Compose the zone subgraphs into the full graph, adding boundary vertices and twin edges across the boundary separator, for global computations like clustering (Simulations 04–05). Returns `(G, inversion_map)`.
- `lattice_rotate(m, n, k)` — Apply the order-6 (ℤ₆) lattice rotation to Eisenstein coordinates by `k` steps of 60°, for orbit computation in symmetry-reduced algorithms. Returns `(m', n')`.

**Parameters (primary):**
- `R` (float): Truncation radius (`R ≫ r` for a balanced finite approximation)
- `r_sq` (int): Squared inversion radius `r² = N`, admissible when representable as `m² + mn + n²` for integers `m, n` not both zero (default `1` → unit-hexagon boundary)

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

Baseline performance benchmark establishing the standard (full recompute) approach to the dual-zone shortest-path problem: given a source in the outer zone subgraph and its inversion twin in the inner zone, obtain single-source shortest-path hop distances (the discrete dual metric) in **both** zones.

**Approach:**
- Run a full single-source shortest-path computation independently in each zone (no symmetry exploitation)
- Use a fixed random seed so the source vertex matches Simulation 03 for a directly comparable, verifiable benchmark
- Time the solution over multiple runs with inner timing repeats; report mean and standard deviation in milliseconds

**Usage:**
```bash
python simulation_02_benchmark_standard_path_mirroring.py R [--runs N] [--timing_repeats M] [--seed S]

# Example: Benchmark at truncation radius R=15
python simulation_02_benchmark_standard_path_mirroring.py 15 --runs 20 --timing_repeats 100
```

**Arguments:**
- `R` (int, default 10): Truncation radius
- `--runs` (default 20): Number of benchmark runs
- `--timing_repeats` (default 100): Repeats per run for timing accuracy
- `--seed` (default 42): Random seed for source-vertex selection

**Output:**
```
Building radial dual triangular lattice graph with truncation radius R=15...
Graphs built: Outer <N> vertices, Inner <M> vertices.
Running 20 benchmarks, each with 100 timing repeats.
Standard Path Mirroring (Recompute): <avg> ms (+/-<std>)
```

This is the baseline against which the TQF inversion-based approach (Simulation 03) is measured.

---

### Simulation 03: Benchmark Tri-Quarter Path Mirroring
**File:** `src/simulation_03_benchmark_triquarter_path_mirroring.py`

TQF-optimized benchmark that solves the **same** dual-zone shortest-path problem as Simulation 02, but exploits the circle inversion bijection ιᵣ to avoid the second computation.

**Approach:**
- Compute single-source shortest-path hop distances once, in the outer zone
- Map the result into the inner zone through the inversion bijection ιᵣ (a distance-preserving graph isomorphism under the Escher reflective duality), making the inner-zone recomputation provably redundant
- Verify the mirrored inner-zone distances are bitwise-equal to an independent recomputation (`verify_mirror_exactness`) before timing
- Time over multiple runs; report mean and standard deviation in milliseconds

**Usage:**
```bash
python simulation_03_benchmark_triquarter_path_mirroring.py R [--runs N] [--timing_repeats M] [--seed S]

# Example: Benchmark at truncation radius R=15
python simulation_03_benchmark_triquarter_path_mirroring.py 15 --runs 20 --timing_repeats 100
```

**Arguments:** Same as Simulation 02 (`R`, `--runs`, `--timing_repeats`, `--seed`); the default seed (42) matches Simulation 02 so the two benchmarks select the same source vertex.

**Output:**
```
Building radial dual triangular lattice graph with truncation radius R=15...
Graphs built: Outer <N> vertices, Inner <M> vertices.
Exactness check (mirrored == recomputed): PASS
Running 20 benchmarks, each with 100 timing repeats.
Tri-Quarter Path Mirroring (Inversion): <avg> ms (+/-<std>)
```

Run alongside Simulation 02 (same `R` and seed) to compare timings; the speedup comes from eliminating the redundant inner-zone solve while preserving the discrete dual metric exactly.

---

### Simulation 04: Benchmark Standard Clustering
**File:** `src/simulation_04_benchmark_standard_clustering.py`

Standard (full recompute) baseline for the **average local clustering coefficient** on the complete truncated lattice graph Λᵣᴿ, computed in **exact rational arithmetic** (`fractions.Fraction`).

**Approach:**
- Build the complete lattice graph and a `{vertex: frozenset(neighbors)}` adjacency view
- Compute each local coefficient as the exact rational `2·common / (deg·(deg−1))` and average over all vertices — no floating-point round-off, bitwise-reproducible
- Time the full-graph computation over multiple runs; report mean and standard deviation in milliseconds

**Usage:**
```bash
python simulation_04_benchmark_standard_clustering.py R [--runs N] [--timing_repeats M]

# Example: Benchmark at truncation radius R=100
python simulation_04_benchmark_standard_clustering.py 100 --runs 20 --timing_repeats 20
```

**Arguments:**
- `R` (int, default 10): Truncation radius
- `--runs` (default 20): Number of benchmark runs
- `--timing_repeats` (default 20): Repeats per run for timing accuracy

**Output:**
```
Graph: |V|=<num_vertices>
Average clustering coefficient (exact): <p>/<q> = <float value>
Standard (exact): <avg> ms +/- <std>
```

This is the reference against which the Tri-Quarter symmetry-reduced approach (Simulation 05) is verified by exact (`==`) comparison.

---

### Simulation 05: Benchmark Tri-Quarter Clustering
**File:** `src/simulation_05_benchmark_triquarter_clustering.py`

TQF symmetry-reduced benchmark for the **same** exact-rational average local clustering coefficient as Simulation 04, exploiting the order-6 rotational symmetry of the lattice.

**Approach:**
- Partition the vertex set into ℤ₆ orbits under the order-6 rotation (a graph automorphism), via either a pure-Python visited-set traversal or a NumPy-vectorized construction (both yield bitwise-identical orbits)
- Compute the local coefficient on one representative per orbit and replicate it across the orbit, weighted by orbit size
- Verify orbit member-equality and confirm the orbit-reduced rational equals the full-graph rational **exactly** (`==`) before timing
- The orbit transversal is a one-time precomputation, performed outside the timing loop

**Usage:**
```bash
python simulation_05_benchmark_triquarter_clustering.py R [--runs N] [--timing_repeats M] [--orbit-method METHOD] [--debug]

# Example: Benchmark at truncation radius R=100
python simulation_05_benchmark_triquarter_clustering.py 100 --runs 20 --timing_repeats 20
```

**Arguments:**
- `R` (int, default 10): Truncation radius
- `--runs` (default 20), `--timing_repeats` (default 20): Timing parameters
- `--orbit-method` (`auto` | `python` | `numpy`, default `auto`): Orbit-transversal construction (`auto` uses NumPy when available)
- `--debug`: Print orbit-transversal statistics

**Output:**
```
Graph: |V|=<num_vertices>
NumPy available: True
Orbit transversal: <K> orbits built in <t> ms (method=auto)
Orbit member-equality check: PASS
Exact match (orbit == standard, as Fraction): PASS
Float images identical: True
Average clustering coefficient (exact): <p>/<q> = <float value>
Tri-Quarter (exact): <avg> ms +/- <std>
```

> **Note:** NumPy is optional here — with `--orbit-method python` (or if NumPy is absent) the pure-Python orbit construction is used and produces identical orbits.

---

### Simulation 06: Benchmark Trihexagonal Six-Coloring (CPU vs. GPU)
**File:** `src/simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py`

Benchmarks a symmetry-aware parallel workload showing that the framework's equivariant **trihexagonal six-coloring** directly enables conflict-free, lock-free data-parallel execution on a GPU.

**Approach:**
- Color the lattice with the proper six-coloring `e₆ = 2·c + (s₆ mod 2)`, where `c = (m − n) mod 3` is the triangular-lattice three-coloring and `s₆` is the angular sector index; each of the six color classes is an independent set
- Run one color-ordered (Gauss-Seidel) relaxation sweep — every vertex updates to `α·(own) + (1−α)·(neighbor mean)` — issuing each color class as a single batched, vectorized operation with no read-write conflicts or locks
- Benchmark two backends on the identical workload: a CPU baseline (NumPy) and a GPU backend (PyTorch on CUDA, with transparent CPU fallback)
- Verify the two backends produce numerically identical results before reporting timing
- Neighbor structure is encoded once as a padded adjacency-index tensor (gather-and-reduce, no per-vertex Python loop); the coloring is precomputed outside the timed region

**Usage:**
```bash
python simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py R [--runs N] [--timing_repeats M] [--sweeps S]

# Example: Benchmark at truncation radius R=200
python simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py 200 --runs 10 --timing_repeats 20
```

**Arguments:**
- `R` (int, default 100): Truncation radius
- `--runs` (default 10), `--timing_repeats` (default 20): Timing parameters
- `--sweeps` (default 10): Relaxation sweeps per timed iteration

**Requirements:** NumPy (required); PyTorch (optional — enables the GPU backend; without it the GPU backend is skipped and only the CPU/NumPy baseline runs). Times are reported in milliseconds **per sweep**.

**Output:**
```
Graph: |V|=<num_vertices> |E|=<num_edges>
Trihexagonal six-coloring proper: PASS
Six-coloring class sizes: [<six counts>]
CPU (NumPy, color-ordered): <avg> ms/sweep +/- <std>
GPU backend device: <cuda|cpu>
CPU/GPU agreement: PASS (max abs diff <value>)
GPU (PyTorch, color-ordered): <avg> ms/sweep +/- <std>
Speedup (CPU / GPU): <ratio>x
```

---

### Tool: Get Vertex Counts
**File:** `src/get_vertex_counts.py`

Utility for computing vertex counts across graph zones (outer, inner, boundary) and the six angular sectors S_t (t ∈ ℤ₆).

**Features:**
- Outer / inner / boundary zone vertex counts and total
- Per-sector distribution and average vertices per sector (60° increments aligned with ℤ₆)
- Validation of the proposed inversion radius via Eisenstein representations (`r² = m² + mn + n²`), suggesting the next admissible radius if invalid
- Count of vertices lying on the angular-sector borders (primary rays)

**Usage:**
```bash
python get_vertex_counts.py r R

# Example: inversion radius r=1, truncation radius R=4
python get_vertex_counts.py 1 4

# Example: r ≈ sqrt(7), truncation radius R=10
python get_vertex_counts.py 2.64575 10
```

**Arguments:**
- `r` (float): Inversion radius — must yield an integer `r² = N` with lattice points (the script validates this and proposes nearby admissible radii otherwise)
- `R` (float): Truncation radius

**Output:**
```
Valid r_sq = 1, effective r = 1.000000
Outer zone vertices: <count>
Inner zone vertices: <count>
Boundary zone vertices: <count>
Total vertices: <count>
Vertices per angular sector (outer + boundary + inner = total):
S_0: <o> + <b> + <i> = <total>
...
Average vertex count per angular sector:
...
Vertices on angular sector borders (primary rays):
...
```

**Applications:**
- Graph size estimation for memory planning
- Symmetry / equidistribution verification under D₆
- Validation of admissible inversion radii

---

### Tool: Compute Boundary Vertices
**File:** `src/compute_boundary_vertices.py`

Computes the explicit boundary-zone vertices V_{T,r} for an admissible inversion radius, where `r² = N` is representable as `m² + mn + n²`. Finds all integer solutions `(m, n)`, assigns each to its angular sector via exact integer arithmetic, and groups them by sector to illustrate uniform equidistribution under D₆.

**Features:**
- Exact integer solutions `(m, n)` to `m² + mn + n² = N`
- Angular sector assignment (exact, no floating-point phase)
- Output sorted by sector to highlight symmetric orbits (e.g. N=7 yields two vertices per sector)
- Requires only the standard library (`math`, `argparse`) — no NetworkX needed

**Usage:**
```bash
python compute_boundary_vertices.py [N]

# Example: N=7 → r=sqrt(7), 12 boundary vertices (default N=7)
python compute_boundary_vertices.py 7
```

**Arguments:**
- `N` (int, optional, default 7): The integer `N = r²`

**Output:**
```
Boundary vertices for N=7 (r=sqrt(7)):
Sector 0: (m,n)=(...,...)
Sector 0: (m,n)=(...,...)
Sector 1: (m,n)=(...,...)
...
```

**Applications:**
- Verifying boundary-vertex equidistribution under D₆
- Constructing the boundary separator for the complete lattice graph
- Selecting admissible inversion radii

---

### Tool: Compute Truncation Errors
**File:** `src/compute_truncation_errors.py`

Quantifies the truncation error for a fixed inversion radius `r = 1`: the unresolved area near the punctured origin in the inner zone Λ₋,₁ as a fraction of the total viewed area. The unresolved area is `π(r²/R)² = π/R²` and the total viewed area is approximated as `πR²`, so the error percentage scales as O(1/R⁴) and vanishes as R → ∞.

**Features:**
- Error percentages for a fixed set of truncation radii `R ∈ {4, 10, 20, 50}` (hardcoded; `r = 1`)
- Pure standard-library script (`math` only) — no command-line arguments and no external dependencies

**Usage:**
```bash
python compute_truncation_errors.py
```

> **Note:** This script takes no arguments; the truncation radii and `r = 1` are fixed in the source. Edit the `Rs` list near the top of the file to evaluate other radii.

**Output:**
```
Truncation Error Percentages for Various R (with r=1):
R | Unresolved Area (pi r^4 / R^2) | Total Viewed Area (~ pi R^2) | Percentage (%)
4 | <...> | <...> | <...>%
10 | <...> | <...> | <...>%
20 | <...> | <...> | <...>%
50 | <...> | <...> | <...>%
```

**Applications:**
- Selecting an appropriate truncation radius R for simulations
- Error bounds for finite approximations of the infinite lattice Λᵣ

---

## 7. Usage Examples

### Path Mirroring Comparison

Run both benchmark scripts at the **same** truncation radius and seed to compare execution times and observe the advantage of the TQF inversion-based approach over standard recomputation.

```bash
# Redirect output to a single results file for easy comparison
echo "=== Standard Approach (Simulation 02) ===" > results.txt
python src/simulation_02_benchmark_standard_path_mirroring.py 15 >> results.txt

echo "=== TQF Inversion Approach (Simulation 03) ===" >> results.txt
python src/simulation_03_benchmark_triquarter_path_mirroring.py 15 >> results.txt

# View the combined results
cat results.txt
```

Both scripts use the same default seed (42), so they select the same source vertex and solve the identical dual-zone problem — allowing a direct evaluation of the speedup achieved by mirroring through the circle inversion bijection ιᵣ instead of recomputing the inner zone.

### Clustering Comparison

Compare the standard and symmetry-reduced clustering benchmarks; both compute the **identical exact rational** coefficient (verified by `==`), so the difference is purely runtime.

```bash
echo "=== Standard Clustering (Simulation 04) ===" > clustering.txt
python src/simulation_04_benchmark_standard_clustering.py 100 >> clustering.txt

echo "=== Tri-Quarter Clustering (Simulation 05) ===" >> clustering.txt
python src/simulation_05_benchmark_triquarter_clustering.py 100 >> clustering.txt

cat clustering.txt
```

### Vertex and Boundary Analysis Pipeline

Execute the analysis tools to generate reports on vertex distributions, boundary properties, and truncation effects.

```bash
# Vertex count breakdown for inversion radius r=1, truncation radius R=10
python src/get_vertex_counts.py 1 10 > vertex_counts.txt

# Boundary vertices for N = r^2 = 7 (r = sqrt(7))
python src/compute_boundary_vertices.py 7 > boundary_analysis.txt

# Truncation error percentages (fixed R set, r=1; takes no arguments)
python src/compute_truncation_errors.py > truncation_errors.txt

# Review all generated reports
cat vertex_counts.txt boundary_analysis.txt truncation_errors.txt
```

These commands produce structured text output that can be inspected individually or concatenated for a complete overview. The pipeline quantifies zone/sector distributions, demonstrates boundary equidistribution under D₆, and assesses the approximation error introduced by finite truncation.

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

**Dual-Zone Path Mirroring (Simulations 02 vs. 03):**
- *Standard:* runs a full single-source shortest-path (BFS) computation independently in **both** zones.
- *TQF inversion:* runs the BFS **once** in the outer zone, then transfers the result to the inner zone in O(|outer_dist|) time via the precomputed inversion map, eliminating the redundant second BFS.
- The mirrored inner-zone distances are verified bitwise-equal to an independent recomputation before timing, so the speedup reflects eliminated redundant work — not a different (approximate) result.

**Symmetry-Reduced Clustering (Simulations 04 vs. 05):**
- *Standard:* computes the exact-rational local coefficient for every vertex.
- *TQF orbit reduction:* computes one representative per ℤ₆ orbit (orbit size up to 6) and replicates by orbit weight, reducing per-query work toward a ~1/6 fraction for the dominant interior orbits, after a one-time orbit-transversal precomputation.
- Because the order-6 rotation is a graph automorphism, the orbit-reduced average equals the full-graph average **exactly** as a rational number (verified by `==`, not float tolerance).

**Trihexagonal Six-Coloring Parallelism (Simulation 06):**
- The proper six-coloring partitions the vertices into six independent sets, so each color class updates with no read-write conflicts and no locks — the property that licenses batched, vectorized CPU (NumPy) and GPU (PyTorch) execution.
- Both backends are verified to produce numerically identical results before the speedup is reported.

> Actual timings depend on R, hardware, and (for Simulation 06) CUDA availability. Run the benchmarks locally to obtain numbers for your environment.

---


## 9. Development

- **Language**: Python 3.8+ (tested on 3.12.3)
- **Core Libraries**: NetworkX 3.0+ for graph operations; NumPy 1.24+ (required for Simulation 06, optional acceleration for Simulation 05); Pygame 2.5+ (optional, for visualization in Simulation 01); PyTorch 2.0+ (optional, for the GPU backend of Simulation 06)
- **Standard Libraries**: math, cmath, random, time, argparse, statistics, fractions
- **Testing and Code Quality Tools**: pytest, black, mypy, flake8 (install via `requirements-dev.txt`)
- **Platform**: Cross-platform (Windows/Linux/macOS)
- **Code Style**: PEP 8 compliant

### Project Structure

```
radial_dual_triangular_lattice_graph/
├── src/                                                          # Source code directory
│   ├── radial_dual_triangular_lattice_graph.py                   # Core graph utility (zone subgraphs, complete lattice, rotation)
│   ├── simulation_01_visualize_random_connections.py             # Pygame visualization
│   ├── simulation_02_benchmark_standard_path_mirroring.py        # Standard path-mirroring benchmark
│   ├── simulation_03_benchmark_triquarter_path_mirroring.py      # TQF inversion path-mirroring benchmark
│   ├── simulation_04_benchmark_standard_clustering.py            # Standard clustering benchmark (exact rational)
│   ├── simulation_05_benchmark_triquarter_clustering.py          # TQF Z6-orbit clustering benchmark (exact rational)
│   ├── simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py   # Six-coloring parallel relaxation (CPU vs. GPU)
│   ├── get_vertex_counts.py                                      # Vertex/zone/sector count tool
│   ├── compute_boundary_vertices.py                              # Boundary-vertex tool
│   └── compute_truncation_errors.py                              # Truncation-error tool
├── ACKNOWLEDGEMENT.md                                            # Acknowledgements and gratitude
├── README.md                                                     # This file
├── requirements.txt                                              # Core dependencies (NetworkX, NumPy, Pygame; optional Torch)
└── requirements-dev.txt                                          # Development dependencies (pytest, black, mypy, flake8)
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
# Path-mirroring suite across several truncation radii (requires NetworkX)
for R in 5 10 15 20; do
    echo "=== Truncation radius R=$R ==="
    python src/simulation_02_benchmark_standard_path_mirroring.py $R
    python src/simulation_03_benchmark_triquarter_path_mirroring.py $R
done

# Clustering suite (requires NetworkX; NumPy optional for Simulation 05)
for R in 25 50 100; do
    echo "=== Truncation radius R=$R ==="
    python src/simulation_04_benchmark_standard_clustering.py $R
    python src/simulation_05_benchmark_triquarter_clustering.py $R
done

# Trihexagonal six-coloring parallel relaxation (requires NumPy; PyTorch for GPU)
python src/simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py 200
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

- **Schmidt, Nathan O.** (2026). *The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal Dihedral Symmetry Group 𝕋₂₄*. Zenodo.
[https://zenodo.org/records/20636058](https://zenodo.org/records/20636058)

### Related Topics
- **Conway, J. H., & Sloane, N. J. A.** (1999). *Sphere Packings, Lattices and Groups* (3rd ed.). Springer.
- **Coxeter, H. S. M.** (1973). *Regular Polytopes* (3rd ed.). Dover Publications.
- **Needham, T.** (1997). *Visual Complex Analysis*. Oxford University Press.

### Graph Theory & NetworkX
- **Hagberg, A., Schult, D., & Swart, P.** (2008). Exploring Network Structure, Dynamics, and Function using NetworkX. *Proceedings of SciPy*.
- **Bollobás, B.** (1998). *Modern Graph Theory*. Springer.

---

**`QED`**

**Last Updated:** June 4, 2026<br>
**Version:** 1.1.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
