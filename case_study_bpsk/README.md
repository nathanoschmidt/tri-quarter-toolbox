# Tri-Quarter Framework Case Study: BPSK Signal Processing Simulation Scripts: README

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.0.0<br>
**Date:** September 24, 2025<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-blue.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Key Features](#2-key-features)
- [3. Installation](#3-installation)
- [4. Quick Start](#4-quick-start)
- [5. Simulations](#5-simulations)
- [6. Usage Examples](#6-usage-examples)
- [7. Development](#7-development)
- [8. License](#8-license)
- [9. References](#9-references)

---

## 1. Overview

This case study demonstrates the application of the **Tri-Quarter Framework (TQF)** to **Binary Phase-Shift Keying (BPSK)** modulation systems, exploring novel approaches to improve **Bit Error Rate (BER)** performance in digital communication systems.

Developed as part of the Tri-Quarter Toolbox research initiative, this project implements and compares three distinct signal processing strategies for BPSK demodulation and error correction:

1. **Tri-Quarter Framework-based BER optimization** (TQF approach)
2. **Majority voting ensemble method** (baseline comparison)
3. **Gaussian-tuned soft decision decoding** (adaptive approach)

The primary goals are:
- **Apply TQF Principles to Communication Systems**: Leverage the radial dual triangular lattice graph structure and hexagonal symmetries for signal processing in BPSK systems.
- **Benchmark BER Performance**: Compare TQF-based methods against traditional and adaptive approaches across various SNR (Signal-to-Noise Ratio) conditions.
- **Demonstrate Cross-Domain Applicability**: Show how geometric deep learning principles from TQF can extend to classical digital communications.
- **Provide Reproducible Simulations**: Deliver clean, well-documented Python code for researchers and practitioners to replicate and extend.

This is an experimental after-hours hobby science project exploring the intersection of geometric frameworks, group theory, and digital signal processing.

---

## 2. Key Features

- 📡 **BPSK Modulation & Demodulation**: Complete implementation of binary phase-shift keying with AWGN (Additive White Gaussian Noise) channel modeling.
- 🧬 **TQF-Based Signal Processing**: Novel application of radial dual triangular lattice graphs and hexagonal symmetries to communication systems.
- 📊 **Comprehensive BER Analysis**: SNR sweep simulations with statistical analysis and performance visualization.
- ⚖️ **Comparative Benchmarking**: Three distinct approaches implemented with identical test conditions for fair comparison.
- 🔬 **Reproducible Science**: Fixed random seeds, documented simulation parameters, and transparent methodology.
- 💻 **Cross-Platform Python**: Pure Python implementation compatible with Windows, Linux, and macOS.
- 📈 **Visualization Ready**: Outputs designed for plotting BER curves and performance metrics.

---

## 3. Installation

### Prerequisites

- **Python 3.8+**
- **NumPy 1.24+** for numerical computing
- **SciPy 1.10+** for signal processing
- **Matplotlib 3.5+** for visualization

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
pip install numpy scipy matplotlib
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib
```

### Verify Installation:
```bash
python -c "import numpy; import scipy; print(f'NumPy: {numpy.__version__} | SciPy: {scipy.__version__}')"
```

---

## 4. Quick Start

Navigate to the `case_study_bpsk` directory and run any of the three simulation scripts:

**Windows:**
```bash
# Activate virtual environment
venv\Scripts\activate

# Run TQF-based simulation
python simulation_01_tri-quarter_framework_ber.py

# Run majority voting simulation
python simulation_02_majority_voting_ber.py

# Run Gaussian-tuned soft decision simulation
python simulation_03_gaussian-tuned_soft_decision_ber.py
```

**Linux/macOS:**
```bash
# Activate virtual environment
source venv/bin/activate

# Run TQF-based simulation
python simulation_01_tri-quarter_framework_ber.py

# Run majority voting simulation
python simulation_02_majority_voting_ber.py

# Run Gaussian-tuned soft decision simulation
python simulation_03_gaussian-tuned_soft_decision_ber.py
```

Each simulation will output BER statistics across a range of SNR values (typically 0 dB to 12 dB).

---

## 5. Simulations

### Simulation 01: Tri-Quarter Framework BER
**File:** `simulation_01_tri-quarter_framework_ber.py`

This simulation implements the TQF-based phase-pair directional encoding approach to BPSK demodulation with robust symbol detection.

**Key Parameters:**
- SNR range: 0–12 dB (configurable)
- Bits per SNR point: 10⁴ to 10⁶ (adjustable for precision)
- Modulation: BPSK (±1 constellation)
- Channel: AWGN (Additive White Gaussian Noise)

**Expected Output:**
- BER vs. SNR data points
- Statistical confidence intervals (optional)
- Comparison against theoretical BPSK BER curve

---

### Simulation 02: Majority Voting BER
**File:** `simulation_02_majority_voting_ber.py`

Baseline ensemble method using majority voting across multiple independent BPSK demodulators.

**Approach:**
- N parallel BPSK receivers with independent noise realizations
- Majority voting decision rule (hard decision combining)
- Standard approach for diversity combining in communications

**Purpose:**
- Provides a classical benchmark for comparison
- Demonstrates the improvement (or lack thereof) of TQF-based methods over traditional ensemble techniques

---

### Simulation 03: Gaussian-Tuned Soft Decision BER
**File:** `simulation_03_gaussian-tuned_soft_decision_ber.py`

Adaptive soft decision decoder with Gaussian-optimized thresholds.

**Approach:**
- Soft decision decoding with likelihood ratios
- Adaptive threshold tuning based on estimated noise statistics
- Represents state-of-the-art baseline without TQF

**Purpose:**
- Benchmarks TQF against modern adaptive approaches
- Tests whether geometric symmetry exploitation offers advantages over statistical optimization

---

## 6. Usage Examples

### Basic BER Simulation

```bash
# Run default simulation with standard parameters
python simulation_01_tri-quarter_framework_ber.py
```

### Comparing All Three Methods

```bash
# Run all simulations sequentially
python simulation_01_tri-quarter_framework_ber.py > results_tqf.txt
python simulation_02_majority_voting_ber.py > results_majority.txt
python simulation_03_gaussian-tuned_soft_decision_ber.py > results_gaussian.txt

# Compare results (manual analysis or use plotting script)
```

---

## 7. Development

- **Language**: Python 3.8+
- **Key Libraries**: NumPy, SciPy, Matplotlib (optional)
- **Code Quality**: black, mypy, flake8 (install via `requirements-dev.txt`)
- **Platform**: Cross-platform (Windows/Linux/macOS)

### Code Structure

```
case_study_bpsk/
├── src/                                                   # Source code directory
│   ├── simulation_01_tri-quarter_framework_ber.py         # TQF-based approach
│   ├── simulation_02_majority_voting_ber.py               # Majority voting baseline
│   └── simulation_03_gaussian-tuned_soft_decision_ber.py  # Adaptive soft decision
├── ACKNOWLEDGEMENT.md                      # Some gratitude
├── README.md                               # This file
├── requirements.txt                        # Core dependencies
└── requirements-dev.txt                    # Development dependencies
```

### Code Quality Tools

```bash
# Format code
black .

# Type checking
mypy .

# Linting
flake8 .
```

---

## 8. License

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

## 9. References

### Preprints/Publications
- **Schmidt, Nathan O.** (2025). *The Tri-Quarter Framework: Unifying Complex Coordinates with Topological and Reflective Duality across Circles of Any Radius*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1281679](https://www.techrxiv.org/users/906377/articles/1281679)

- **Schmidt, Nathan O.** (2025). *Tri-Quarter Framework Case Study: BPSK Signal Processing*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1311875](https://www.techrxiv.org/users/906377/articles/1311875)

### Related References
- Proakis, J. G., & Salehi, M. (2008). *Digital Communications* (5th ed.). McGraw-Hill.
- Sklar, B. (2001). *Digital Communications: Fundamentals and Applications* (2nd ed.). Prentice Hall.

---

**`QED`**

**Last Updated:** September 24, 2025<br>
**Version:** 1.0.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
