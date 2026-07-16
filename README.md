# Tri-Quarter Toolbox: README

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Last Dated:** July 15, 2026<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Getting Started](#3-getting-started)
- [4. Projects & Tools](#4-projects--tools)
- [5. The Tri-Quarter Framework](#5-the-tri-quarter-framework)
- [6. License](#6-license)
- [7. Contributing](#7-contributing)

---

## 1. Overview

The **Tri-Quarter Toolbox** is a public open-source software repository containing scientific research tools, implementations, and experiments regarding Nathan O. Schmidt's **Tri-Quarter Framework (TQF)**—a mathematical and computational framework that upgrades the complex numbers for working with unified 2D coordinates, inversive geometry, Eisenstein integers, **radial dual triangular lattice graph (RDTLG)**, hexagonal symmetries, signal processing, equivariant encodings, and geometric deep learning.

The TQF is based on prior (2007-2017) and current (2025-present) academic research in advanced computer science and mathematics. This TQF toolbox is developed by Cold Hammer Research & Development LLC as an after-hours hobby science project driven by rigorous creativity, a first principles mindset, scientific methodology, and a passion for exploring the intersection of computational/mathematical fields such as data structures, algorithms, parallel computing, graph theory, discrete mathematics, group theory, cryptography, machine learning, and more.

The repository is organized into subdirectories containing various projects and tools, each with their own documentation. All code aims to be cross-platform compatible (Windows/Linux/macOS) with solid practices. 

**Core Philosophy:**
- 🏀 **Richard Feynman's Quote**: "*Science is the belief in the ignorance of experts*"
- ⛓️‍💥 **Freedom and Technical Problem Solving**: Ask questions, challenge groupthink/censorship, and seek answers via the Scientific Method
- 🧬 **First Principles Design**: Mathematical rigor meets practical implementation
- 🔄 **Symmetry-Aware Computing**: Exploit geometric structure for efficient elegant solutions
- 📊 **Reproducibility**: Fixed seeds, documented environments, and transparent methods
- 🔬 **Open Science**: MIT-licensed code and public research artifacts
- 💡 **Continuous Learning**: Experimental projects exploring new ideas
- 🧪 **Automated Testing**: Each time code is deployed to production without automated testing, a child gets a mullet (and yes, some of these tools have contributed to mullets—it's work in progress 🤣)

Dev and scientist for life. 💾🗲🚀

---

## 2. Repository Structure

```
tri-quarter-toolbox/
├── README.md                              # This file
├── LICENSE                                # MIT license
├── case_study_bpsk/                       # BPSK signal processing case study
├── machine_learning/                      # Machine learning tools/projects
│   └── tqf-nn_benchmark/                  # TQF neural network benchmark tools
├── radial_dual_signal_processing/         # Exact/parallel hex-lattice signal processing
├── radial_dual_triangular_lattice_graph/  # Core RDTLG implementation & tools
└── theorem_animation/                     # Foundational theorem visualization
```

Each subdirectory contains its own README with detailed information about that specific project or tool.

---

## 3. Getting Started

### Prerequisites

Most tools and projects in this repository require:
- **Python 3.8+** 
- **Virtual Environment** (strongly recommended)
- **Platform**: Windows, Linux, or macOS

Common dependencies across projects include:
- NumPy, SciPy, Matplotlib (signal processing, visualization)
- NetworkX, Pygame (graph algorithms, lattice visualization)
- PyTorch + torchvision (machine learning benchmarks, CUDA 12.6 recommended)

### Installation Pattern (all projects)

1. Navigate to the desired project directory
2. Create and activate a virtual environment
3. Install dependencies from the project’s `requirements.txt` (or `requirements-dev.txt` for development)

**Example (Windows):**
```bash
cd case_study_bpsk
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Example (Linux/macOS):**
```bash
cd radial_dual_triangular_lattice_graph
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the machine learning benchmark (GPU acceleration):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

See each project’s own README for exact installation steps, optional dependencies (e.g., Pygame, CUDA), and verification commands.

### Quick Navigation

- **Educational Visualization** → [`theorem_animation/`](./theorem_animation/)
- **Core Lattice & Graph Algorithms** → [`radial_dual_triangular_lattice_graph/`](./radial_dual_triangular_lattice_graph/)
- **Signal Processing / Communications (1D BPSK)** → [`case_study_bpsk/`](./case_study_bpsk/)
- **Signal Processing / Communications (2D Hex Constellations)** → [`radial_dual_signal_processing/`](./radial_dual_signal_processing/)
- **Geometric Deep Learning / Neural Networks** → [`machine_learning/tqf-nn_benchmark/`](./machine_learning/tqf-nn_benchmark/)

---

## 4. Projects & Tools

### Tri-Quarter Theorem Animation
**Location:** [`theorem_animation/`](./theorem_animation/)<br>
**Associated Preprint:** [TechRxiv 1281679](https://www.techrxiv.org/users/906377/articles/1281679)

Interactive Matplotlib animation visualizing the core TQF unit-circle theorem: rotating point on the unit circle, real/imaginary vector decomposition, phase angle progression, and labeled zones/boundaries.

**Key Features:**
- Real-time vector projection display
- Play/pause and light/dark theme toggle
- Exportable to GIF/MP4

**Quick Start:**
```bash
cd theorem_animation
# ... activate venv ...
python tri_quarter.py
```

### Radial Dual Triangular Lattice Graph (RDTLG)
**Location:** [`radial_dual_triangular_lattice_graph/`](./radial_dual_triangular_lattice_graph/)  
**Associated Preprint:** [Zenodo 20636058](https://zenodo.org/records/20636058)

Reference implementation of the truncated RDTLG — the foundational geometric structure of TQF. Uses Eisenstein integers, exact circle-inversion bijection, and full ℤ₆/D₆/𝕋₂₄ symmetry support.

**Key Features:**
- Graph construction and analytics (zone subgraphs and the complete lattice graph)
- Circle inversion duality for inner ↔ outer zone mapping
- Path mirroring benchmarks (standard recompute vs. TQF inversion-based)
- Symmetry-reduced clustering benchmarks (standard vs. ℤ₆-orbit reduction, in exact rational arithmetic)
- Conflict-free parallel relaxation via the equivariant trihexagonal six-coloring (CPU NumPy vs. GPU PyTorch)
- Real-time Pygame visualization
- Vertex count, boundary, and truncation error tools

**Quick Start:**
```bash
cd radial_dual_triangular_lattice_graph
# ... activate venv ...
python src/simulation_01_visualize_random_connections.py
python src/simulation_03_benchmark_triquarter_path_mirroring.py 15
python src/simulation_05_benchmark_triquarter_clustering.py 100
python src/simulation_06_benchmark_trihexagonal_sixcoloring_gpu.py 200
```

### BPSK Signal Processing Case Study
**Location:** [`case_study_bpsk/`](./case_study_bpsk/)  
**Associated Preprint:** [TechRxiv 1311875](https://www.techrxiv.org/users/906377/articles/1311875)

Implements and compares three BPSK demodulation strategies in AWGN channels:
- TQF phase-pair directional encoding (primary contribution)
- Majority voting ensemble (classical baseline)
- Gaussian-tuned soft decision decoding (adaptive baseline)

**Key Features:**
- Full BPSK modulation/demodulation pipeline
- BER vs. SNR sweeps (0–12 dB)
- Reproducible simulations with fixed seeds
- Pure Python + NumPy/SciPy

**Quick Start:**
```bash
cd case_study_bpsk
# ... activate venv ...
python simulation_01_tri-quarter_framework_ber.py
```

### Radial Dual Signal Processing (Hexagonal Lattice)
**Location:** [`radial_dual_signal_processing/`](./radial_dual_signal_processing/)  
**Associated Paper:** *Tri-Quarter Framework Method: Exact, Symmetry-Reduced, and Parallel Signal Processing on the Radial Dual Triangular Lattice* (in preparation)

Carries the TQF from the 1D BPSK case study into 2D hexagonal signal constellations, where the framework's order-6 symmetry does real work. Every claim is backed by a script that prints a copy-pasteable results table and an automated test that pins the underlying invariant. Validates eleven falsifiable claims (C1–C4, C6–C12) across eight studies (Episodes I–III), kept strictly separate so no claim borrows credit from another.

**Key Features:**
- Closed-form Eisenstein/A₂ O(1) fast-path decoder that is bitwise-identical to exhaustive ML decoding, with constant-time decode, throughput speedup vs M, and an exact `ℤ[√3]` bit-reproducible nearest-point predicate (C1, C2, C12)
- Hexagonal vs. square-QAM packing gain at matched M and energy, with paired McNemar + Holm significance and exact Clopper–Pearson intervals across AWGN, impulsive, and Rayleigh channels (C3)
- Symmetry-reduced exact constellation metrics via ℤ₆-orbit reduction (exact 6× evaluation reduction, verified by `==`) plus a D₆-canonical symmetry-reduced design search (C4, C10)
- Decoder equivariance under order-6 rotation, a differential hexagonal (DPSK-analogue) scheme, and a combined rotation+inversion ℤ₆×ℤ₂ differential codec (C6, C8)
- Inversion-closed "radial dual" constellation with a fundamental-domain folded ML decoder, its honestly-priced AWGN geometry crossover, and an inversion-pair rate-1/2 block code (C7, C9, C11)
- Standalone, self-documenting studies (each writes its CSV tables and a `simNN_provenance.json` sidecar) and a 186-test automated suite

**Quick Start:**
```bash
cd radial_dual_signal_processing
# ... activate venv ...
python src/simulation_01_exact_demod_and_throughput.py
python src/simulation_02_hex_vs_square_packing_gain.py
python src/simulation_05_radial_dual_structure_and_folded_decoder.py
```

### TQF Neural Network (TQF-NN) Benchmark Tools
**Location:** [`machine_learning/tqf-nn_benchmark/`](./machine_learning/tqf-nn_benchmark/)  
**Focus:** Rotated MNIST symmetry-aware benchmarking

Implements a first-principles TQF-NN architecture based on truncated radial dual triangular lattice graphs with explicit ℤ₆, D₆, and 𝕋₂₄ symmetry exploitation. Compares against parameter-matched (~650k params) baselines: FC-MLP, CNN-L5, scaled ResNet-18.

**Key Features:**
- Native symmetry enforcement (equivariance/invariance losses)
- Z₆-aligned rotated MNIST (60° increments)
- Orbit mixing inference
- PyTorch + CUDA support, rich CLI

**Quick Start:**
```bash
cd machine_learning/tqf-nn_benchmark
# ... activate venv + install torch (cu126 recommended) ...
python src/main.py
```

---

## 5. The Tri-Quarter Framework

### Summary

The TQF is a mathematical and computational framework that unifies complex, Cartesian, and polar coordinate systems on the complex plane ℂ. It centers on truncated radial dual triangular lattice graphs (RDTLGs) constructed over Eisenstein integers (ℤ[ω]), while establishing fundamental topological and reflective properties across circles of arbitrary radius.

The foundational contribution, as presented in the unifying preprint, is the **Tri-Quarter Topological Duality Theorem**, which equips any circle T_r (of radius r > 0) with a novel topological property. This theorem positions T_r as an active boundary zone with intrinsic directional characteristics, enabling consistent separation of the inner zone X_{−,r} (|z| < r) and outer zone X_{+,,r} (|z| > r) through a phase pair map that encodes additional directional information and unifies the treatment of real and imaginary components across coordinate systems.

Complementing this is the **Escher Tri-Quarter Reflective Duality Theorem**, which proves reflective duality across T_r via circle inversion. This map preserves phase pairs while bijectively swapping the inner and outer zones, providing an exact topological and reflective correspondence.

Building on these duality principles, TQF provides:

- **Exact bijective duality** between inner and outer zones via circle inversion
- **Native support for three symmetry groups**:
  - **ℤ₆** — rotational symmetry (order 6, 60° increments)
  - **D₆** — dihedral symmetry (order 12, rotations + reflections)
  - **𝕋₂₄** — inversive hexagonal dihedral symmetry (order 24, D₆ extended by circle inversion)
- **Phase-pair directional encoding** aligned with 60° angular sectors, serving as the unifying mechanism for radial separation and orientation
- **Path mirroring** and symmetry-aware algorithms exploiting geometric structure

TQF enables novel approaches in:
- Geometric deep learning and equivariant neural architectures
- Signal processing (e.g., BPSK demodulation with hexagonal symmetry)
- Graph algorithms on dual-zone lattices
- Visual and algebraic understanding of complex-plane geometry, including streamlined directional mappings and computational efficiency gains (e.g., reduced conditional checks in quadrant-based transformations)

These properties originate from the topological and reflective dualities proven across circles of any radius, forming the theoretical foundation upon which subsequent TQF extensions—such as lattice graph implementations and symmetry-aware neural networks—are constructed.


### References: Preprints/Publications
- **Schmidt, Nathan O.** (2025). *The Tri-Quarter Framework: Unifying Complex Coordinates with Topological and Reflective Duality across Circles of Any Radius*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1281679](https://www.techrxiv.org/users/906377/articles/1281679)

- **Schmidt, Nathan O.** (2025). *Tri-Quarter Framework Case Study: BPSK Signal Processing*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1311875](https://www.techrxiv.org/users/906377/articles/1311875)

- **Schmidt, Nathan O.** (2026). *The Tri-Quarter Framework: Radial Dual Triangular Lattice Graphs with Exact Bijective Dualities and Equivariant Encodings via the Inversive Hexagonal Dihedral Symmetry Group 𝕋₂₄*. Zenodo.
[https://zenodo.org/records/20636058](https://zenodo.org/records/20636058)

- **Schmidt, Nathan O.** (2026). *Tri-Quarter Framework Method: Exact, Symmetry-Reduced, and Parallel Signal Processing on the Radial Dual Triangular Lattice* (in preparation).

### Project-Specific Documentation
See individual project READMEs and documentation for additional info. 

---

## 6. License

```text
MIT License

Copyright (c) 2026 Nathan O. Schmidt, Cold Hammer Research & Development LLC

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

## 7. Contributing

This is currently a personal research project, but feedback and suggestions are welcome:

- **Issues**: Report bugs or suggest features via GitHub Issues
- **Discussions**: Reach out via email for collaboration ideas or questions
- **Citations**: If you use this work, please cite the relevant publications

All contributions must adhere to the MIT License and maintain the reproducibility standards established in the existing codebase.

---

**`QED`**

For tool-specific questions, please consult the relevant tool's documentation first.

**Last Updated:** July 15, 2026<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
