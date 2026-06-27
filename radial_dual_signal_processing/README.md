# Tri-Quarter Framework Method: Exact, Symmetry-Reduced, and Parallel Signal Processing on the Radial Dual Triangular Lattice: README

Reproducible experiments for the paper *"Tri-Quarter Framework Method: Exact,
Symmetry-Reduced, and Parallel Signal Processing on the Radial Dual Triangular
Lattice."* This `radial_dual_signal_processing` subproject applies the Tri-Quarter
Framework (TQF) hexagonal-lattice machinery — Eisenstein/A₂ basis, the order-6
rotation, sector/shell/colour encodings, circle inversion, and the trihexagonal
six-colouring — to digital communications. Every claim in the paper is backed by
a script here that prints a clean, copy-pasteable results table, and by an
automated test that pins the underlying invariant.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Core Components](#5-core-components)
6. [Simulations & Tools](#6-simulations--tools)
7. [Reproducing the Paper](#7-reproducing-the-paper)
8. [Development](#8-development)
9. [License](#9-license)
10. [References](#10-references)

---

## 1. Overview

This subproject carries the TQF from the 1D BPSK case study into 2D hexagonal
signal constellations, where the framework's order-6 symmetry does real work. It
validates eight precise, falsifiable claims (C1–C8), kept strictly separate so no
claim borrows credit from another:

- **C1 — Exact demodulation.** A closed-form Eisenstein/A₂ nearest-point decoder
  is *bitwise-identical* to exhaustive maximum-likelihood (ML) decoding.
- **C2 — Constant-time decode on the fast path.** The decoder is O(1) per symbol
  on its interior fast path; exterior symbols take an O(M) ML fallback, so the
  worst case is O(M) and the amortized cost is contingent on fast-path coverage
  (reported vs SNR). Throughput speedup over ML grows monotonically with M above
  a small crossover.
- **C3 — Packing gain.** At matched order M and matched average energy, the
  hexagonal constellation beats square QAM on **SER**, with per-point significance
  established by a paired **McNemar + Holm** test (common random numbers, α=0.05).
  Under **AWGN** this is a resolved coding gain that grows from M=16 and then
  **saturates near ~0.4 dB** at M≥64 (the ~0.6 dB packing-density figure is the
  asymptotic bound, *not reached* at these M). Under **Rayleigh** fading the
  dB-gain-at-target-SER is **CI-limited** — the point estimates are positive but the
  dB interval does not resolve at 100k trials — so significance is led by the
  per-point McNemar test, which resolves for M≥64 but not M=16. Under heavy
  **impulsive** noise the impulse error floor masks the gain, so that channel shows
  robustness *parity*, not a packing gain. SER is the headline (labeling-independent);
  BER is secondary (the hex labeling is Gray-*like*).
- **C4 — Symmetry-reduced exact evaluation.** Order-6 orbit reduction computes an
  exact constellation metric on one sector and replicates it, reproducing the full
  value as the *identical* integer/rational quantity (verified by `==`), with an
  exact **6× reduction in distance evaluations**. On the C7 radial-dual object a
  separate, label-domain combinatorial enumerator additionally folds by the full
  C₆×Z₂ group (a storage/label reduction, **not** multiplied with the exact 6×
  Euclidean reduction — inversion is conformal, not isometric).
- **C5 — Conflict-free parallel recovery.** The trihexagonal six-colouring
  schedules a lock-free parallel lattice-signal denoiser; on a CUDA GPU (RTX 4060)
  the speedup over a single-threaded CPU baseline widens with lattice size —
  median **~13.8× at ~290k vertices**, crossing unity at ~73k and sub-unity below
  it — a systems result that bundles the colouring's parallelism with GPU hardware.
  (The six-colouring is **proper** but **not** rotation-equivariant; only the
  underlying triangular-lattice 3-colouring is order-6-equivariant. The schedule
  needs only properness, so this distinction does not affect C5.)
- **C6 — Equivariant / rotation-robust decode.** The decoder commutes with the
  order-6 rotation (verified exactly on a real hex constellation); a differential
  hexagonal scheme is invariant to any static phase ambiguity that is a multiple
  of π/3 (the hexagonal analogue of DPSK).
- **C7 — Radial-dual constellation.** There is a filled hexagonal constellation
  (shells {3,4,9,12,16,36,48}, M=42) that is simultaneously shell-complete
  (exact C₆), phase-pair-uniform (identical sector occupancy per shell), and
  inversion-paired (closed under the exact circle inversion ι_r about r²=12, which
  maps shell N to its **integer-dual** shell 144/N: 3↔48, 4↔36, 9↔16, 12 self-dual).
  It is invariant under the full **order-12 C₆×Z₂** group — the rotation×inversion
  subgroup of the centrosymmetric hexagonal point group D₆ₕ (stated as such, not
  over-claimed as the full 24). The phase-pair + inversion **folded decoder** is
  bitwise-identical to exhaustive ML while storing only the fundamental domain:
  the label table folds **6×** by rotation and **10.5×** by rotation+inversion
  (reported separately; the self-dual boundary shell limits the latter below 12×).
- **C8 — Combined rotation+inversion differential codec.** A differential codec
  carrying a (sector∈ℤ₆, inversion-bit∈ℤ₂) pair as component-wise differences is
  invariant under all **12** static C₆×Z₂ actions — extending the C6 DPSK analogue
  to absorb a static amplitude-inversion ambiguity. The inversion bit is a discrete
  label state, never a Euclidean operation.

**Honesty discipline.** Because the closed-form decoder *is* ML-equivalent, it
does not change BER/SER versus ML — the wins are in **cost** (C1/C2/C4/C5/C7) and in
**constellation geometry** (C3/C6/C7/C8), never in beating ML. The exact 6× rotation
(Euclidean) reduction and the inversion (label-only) fold are kept distinct and are
never multiplied into a combined figure.

---

## 2. Key Features

- **Exact integer/rational arithmetic** for all geometry (sector, shell, colour,
  inversion, orbit reduction) — no floating-point tolerance in the exactness claims.
- **Apples-to-apples methodology**: matched M, matched average energy, common
  random numbers (identical noise/fading realizations across the compared schemes),
  a true-Gray square-QAM baseline, and exact Clopper–Pearson confidence intervals.
- **Closed-form O(1) fast-path decoder** with a provably sufficient 3×3 candidate
  window and an exhaustive-ML fallback that guarantees exactness for every symbol.
- **Three channel models**: complex AWGN, 2D impulsive, and flat Rayleigh fading
  (perfect CSI).
- **GPU-ready** trihexagonal six-colouring denoiser with a verified-proper colouring
  and CPU/GPU agreement to machine precision (~1e-16); the GPU run
  self-certifies via a `RAN_ON_CUDA` verdict and a `sim04_provenance.json` sidecar.
  (The colouring is proper but not rotation-equivariant; only the underlying
  3-colouring is — a checked artifact, `sim04_coloring_equivariance.csv`.)
- **Phase-pair + inversion exactness layer**: a documented `phase_pair_sector`
  integer primitive, an exact label-space circle-inversion duality (involution +
  sector preservation, verified with zero violations), a radial-dual constellation
  builder, and a folded ML decoder that is bitwise-identical to exhaustive ML while
  storing only the fundamental domain.
- **Self-documenting runs**: every study writes a `simNN_provenance.json`
  (versions + hardware) for the paper's Methods table.
- **Automated test suite** (129 tests) pinning the invariants behind every claim.
- **One-command reproduction** (`run_all.ps1` on Windows, `run_all.sh` on
  Linux/macOS) and a CSV-driven figure generator.
- **Cross-platform**: Windows | Linux | macOS.

---

## 3. Installation

### Prerequisites

- Python 3.10+
- `pip` (and, optionally, a virtual environment)

### Quick Install

```bash
# from the subproject root
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux/macOS:    source .venv/bin/activate
pip install -r requirements.txt
```

### Development Install

```bash
pip install -r requirements-dev.txt
```

### Manual Install (without requirements file)

```bash
pip install numpy scipy matplotlib
```

### Optional GPU Backend (Simulation 04)

```bash
# Install a PyTorch build matched to your platform / CUDA driver.
# See https://pytorch.org/get-started/locally/ for the correct command.
# Without CUDA, Simulation 04 still runs CPU-only and labels its GPU column n/a.
```

On PEP 668 ("externally-managed-environment") systems, add `--break-system-packages`
to the `pip install` commands.

### Verify Installation

```bash
python -m pytest tests/ -q
```

---

## 4. Quick Start

**Windows:**

```bash
# Activate virtual environment
.venv\Scripts\activate

# Exact decode + throughput (C1, C2)
python src\simulation_01_hex_demod_correctness_and_latency.py

# Hex vs square packing gain (C3)
python src\simulation_02_hex_vs_square_ber_packing_gain.py

# Symmetry-reduced exact metric (C4)
python src\simulation_03_symmetry_reduced_metric_exact.py

# Six-colouring parallel denoiser, CPU vs GPU (C5)
python src\simulation_04_sixcoloring_denoise_gpu.py

# Phase-rotation robustness, differential hex (C6) + C6×Z2 differential (C8)
python src\simulation_05_phase_rotation_robustness.py

# Phase-pair + inversion-folded decoder (C1/C2 storage; C7 object)
python src\simulation_06_phasepair_inversion_folded_decoder.py

# Radial-dual constellation structure (C7)
python src\simulation_07_radial_dual_constellation.py
```

**Linux/macOS:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Same commands as Windows (use python or python3, with forward slashes)
python3 src/simulation_01_hex_demod_correctness_and_latency.py
python3 src/simulation_02_hex_vs_square_ber_packing_gain.py --rayleigh_ebn0_extra 26 28 30 32 34 36 38 40 42 44
python3 src/simulation_03_symmetry_reduced_metric_exact.py
python3 src/simulation_04_sixcoloring_denoise_gpu.py
python3 src/simulation_05_phase_rotation_robustness.py
python3 src/simulation_06_phasepair_inversion_folded_decoder.py
python3 src/simulation_07_radial_dual_constellation.py
```

To regenerate **everything** (all seven studies plus figures) in one step:

```bash
./run_all.sh
```

---

## 5. Core Components

### Core Library: `src/tqf_hex_signal.py`

The shared signal-processing library: the Eisenstein/A₂ basis and exact
integer/rational primitives (the documented `phase_pair_sector` integer primitive
— with `sector_index` kept as a backward-compatible alias — shell, colour residue,
circle inversion, order-6 orbits); hexagonal (filled and 6-fold-symmetric disk),
square-QAM, and **radial-dual** (`build_radial_dual_constellation`) constellation
builders, each normalized to unit average energy; the closed-form O(1) fast-path
decoder, an exhaustive-ML decoder, and the phase-pair + inversion **folded** ML
decoder (`make_folded_decode_context` / `decode_hex_folded`); the exact label-space
inversion duality (`invert_label`, `invert_sector_shell`, `dual_shell_norm`,
`verify_inversion_commutativity` — the lattice paper's Prop. 4.15, cited and
empirically verified, not re-derived); the three channel models; the differential
hexagonal encoder/decoder and the combined rotation+inversion (C₆×Z₂)
`differential_encode_t24` / `differential_decode_t24`; and exact Clopper–Pearson
intervals.

### Lattice Graph: `src/tqf_lattice_graph.py`

The truncated triangular lattice graph and the trihexagonal six-colouring used by
the parallel denoiser (Simulation 04). The colouring is `2·((a−b) mod 3) + ((a+b)
mod 2)` and is verified proper against the edge set on construction. It is
**proper but not rotation-equivariant**: only the underlying triangular-lattice
3-colouring `(a−b) mod 3` (exposed as `three_coloring`) is order-6-equivariant
(rotation permutes its three classes); the parity refinement that yields six
classes breaks equivariance. The conflict-free schedule needs only properness. A
colour-ordered relaxation sweep (`relaxation_sweep_numpy`) provides the CPU
baseline that the GPU kernel mirrors.

---

## 6. Simulations & Tools

### Simulation 01: Exact Decode + Throughput (C1, C2)

```bash
python3 src/simulation_01_hex_demod_correctness_and_latency.py \
    --M 16 64 256 1024 --trials 100000 --latency_batch 200000 \
    --timing_repeats 9 --fastpath_ebn0 0 4 8 12
```

Verifies the closed-form decoder equals ML over a dense grid and a Monte-Carlo
stream (target: zero mismatches), then reports amortized per-symbol throughput vs
M, the speedup over ML, and the O(1) fast-path coverage vs Eᵦ/N₀. Writes
`sim01_latency.csv` (with the operating-`ebn0_db` label) and
`sim01_coverage_vs_snr.csv`.

### Simulation 02: Hex vs Square Packing Gain (C3)

```bash
python3 src/simulation_02_hex_vs_square_ber_packing_gain.py \
    --M 16 64 256 --channels awgn impulsive rayleigh \
    --ebn0 0 2 4 6 8 10 12 14 16 18 20 22 24 \
    --rayleigh_ebn0_extra 26 28 30 32 34 36 38 40 42 44 \
    --trials 100000 --targets 1e-2 1e-3
```

SER and BER vs Eᵦ/N₀ for hex (TQF = ML) vs square QAM, with Clopper–Pearson bands
and a per-point SER-significance flag. The Rayleigh grid extends to ~44 dB
(`--rayleigh_ebn0_extra`) so its slow 1/SNR roll-off reaches the low-SER targets
at M ≥ 64 — appended only for Rayleigh, leaving the AWGN/impulsive streams
bit-identical. Writes a durable `sim02_gain_summary.csv` with the dB packing gain,
a Clopper–Pearson-derived confidence band and a `ci_resolved` flag per
(channel, M, target), and the impulsive error floor ≈ p·(1−1/M) with the implied p.
The impulsive channel is reported as robustness *parity at the floor* (any hex<sq
flag belongs to the pre-floor waterfall). SER leads; the hex Gray-*like* BER
penalty is disclosed, never plotted as a win.

### Simulation 03: Symmetry-Reduced Exact Metric (C4)

```bash
python3 src/simulation_03_symmetry_reduced_metric_exact.py
```

Computes the pairwise squared-distance enumerator (→ d_min, kissing number, mean
squared distance) on one sector representative per Z₆ orbit and verifies it equals
the full enumerator bitwise (`==`), with an exact 6× operation-count reduction
(`sim03_symmetry.csv`). It then writes a separate `sim03_inversion_reduction.csv`
for the C7 radial-dual object: the **Euclidean** squared-distance enumerator folds
by rotation **exactly 6×** (and inversion is shown *not* to be an isometry, so it
cannot fold a metric — the firewall), while a discrete inversion-invariant
**label** enumerator folds by the full C₆×Z₂ group (rotation 6×, combined 10.5×).
The two reductions are reported separately and never multiplied.

### Simulation 04: Six-Colouring Parallel Denoiser, CPU vs GPU (C5)

```bash
# Example: scaling curve to ~290k vertices
python3 src/simulation_04_sixcoloring_denoise_gpu.py \
    50 100 150 200 250 300 --sweeps 50 --sessions 5 --timing_repeats 5
```

Runs a conflict-free, colour-scheduled graph-diffusion denoiser on the lattice and
benchmarks CPU (single-threaded NumPy) vs GPU in one pass. Prints a decisive
`RAN_ON_CUDA = True/False` verdict, stamps `device` + `ran_on_cuda` onto every CSV
row, and writes `sim04_provenance.json` (torch/CUDA build, GPU name, compute
capability, memory) so the artifact self-certifies where it ran; a runtime
`assert` confirms the timed tensors are GPU-resident. CPU/GPU agreement (machine
precision) is the exactness check, verified properness the conflict-free check.
Requires CUDA + torch for the headline speedup; without it the GPU column is a
torch-on-CPU reference backend and `RAN_ON_CUDA = False`. It also writes a CPU-only
`sim04_coloring_equivariance.csv` recording the checked fact that the 3-colouring
is rotation-equivariant while the (proper) six-colouring is not.

### Simulation 05: Phase-Rotation Robustness (C6) + C₆×Z₂ Differential (C8)

```bash
python3 src/simulation_05_phase_rotation_robustness.py --trials 100000 --ebn0 10
```

Verifies decoder equivariance exactly on a real hex constellation (persisted as
`sim05_equivariance_check.csv`), then compares coherent vs differential hex SER
across a phase-offset sweep crossing 60° (`sim05_phase.csv`). It additionally
verifies the **C8** combined rotation+inversion (C₆×Z₂) differential codec —
recovering both the senary sector data and the inversion bit under all 12 static
actions with zero violations (`sim05_t24_check.csv`). The C8 block uses an
independent RNG (`--t24_seed`), so the two pre-existing CSVs are byte-identical.

### Simulation 06: Phase-Pair + Inversion-Folded Decoder (C1/C2 storage; C7)

```bash
python3 src/simulation_06_phasepair_inversion_folded_decoder.py \
    --r_sq 12 --max_norm_sq 60 --trials 200000
```

Builds the C7 radial-dual constellation and verifies the folded ML decoder is
bitwise-identical to exhaustive ML on a dense grid and a Monte-Carlo stream
(`sim06_folded_summary.csv`). Reports the storage ablation — full per-point label
table vs phase-pair (rotation) fold (**6×**) vs phase-pair + inversion fold
(**10.5×**) — as two separate factors (`sim06_folded_ablation.csv`), the fast-path
coverage vs SNR (`sim06_coverage_vs_snr.csv`, intentionally moderate because the
shell-complete object has radial gaps), and the exact involution/commutativity of
the label-space inversion. Every Euclidean decision uses true distances; inversion
folds storage only (the firewall).

### Simulation 07: Radial-Dual Constellation Structure (C7)

```bash
python3 src/simulation_07_radial_dual_constellation.py --r_sq 12 --max_norm_sq 60
```

Verifies the C7 object's exact structure: the integer-dual shell pairs
(3↔48, 4↔36, 9↔16, 12 self-dual; `sim07_shell_pairs.csv`), phase-pair uniformity
(every complete shell has k points per sector, all six equal; `sim07_occupancy.csv`),
and inversion pairing (a same-sector involution fixing the boundary shell pointwise;
`sim07_structure.csv`). The object is invariant under the order-12 C₆×Z₂ group —
the rotation×inversion subgroup of D₆ₕ, stated without over-claiming the full 24.

### Tool: Figure Generator

```bash
python3 src/make_figures.py --format pdf        # reads results/, writes figures/
python3 src/make_figures.py --format png        # quick preview
```

Renders one figure per study from the result CSVs (degrading gracefully when a
CSV is missing, and to a CPU-only panel when Simulation 04 had no GPU).

---

## 7. Reproducing the Paper

```bash
# Regenerate all CSV tables (printed to console too) and all figures:
./run_all.sh

# Or step through individually, then build figures:
python3 src/simulation_01_hex_demod_correctness_and_latency.py
python3 src/simulation_02_hex_vs_square_ber_packing_gain.py --rayleigh_ebn0_extra 26 28 30 32 34 36 38 40 42 44
python3 src/simulation_03_symmetry_reduced_metric_exact.py
python3 src/simulation_04_sixcoloring_denoise_gpu.py 50 100 150 200 250 300 --sweeps 50 --sessions 5
python3 src/simulation_05_phase_rotation_robustness.py
python3 src/simulation_06_phasepair_inversion_folded_decoder.py
python3 src/simulation_07_radial_dual_constellation.py
python3 src/make_figures.py --format pdf
```

Each study saves its results to `results/` (CSV tables + a `simNN_provenance.json`)
and prints a console summary; `run_all.ps1` also tees everything to
`results/run_all_console.log`. The C3 headline dB gains live in
`results/sim02_gain_summary.csv`. The one number that requires your hardware is
**Simulation 04's GPU speedup**: it counts only when the printed
`RAN_ON_CUDA = True` verdict (and `sim04_provenance.json`) confirm a real CUDA run
— a torch-on-CPU fallback is a reference backend, not the headline. Report the
*measured* GPU speedup from your run rather than any fixed figure.

---

## 8. Development

- **Language**: Python 3.10+
- **Key Libraries**: NumPy, SciPy, Matplotlib (PyTorch optional, for Simulation 04)
- **Code Quality**: black, mypy, flake8 (install via `requirements-dev.txt`)
- **Platform**: Cross-platform (Windows/Linux/macOS)

### Testing

The `tests/` suite pins the invariant behind each claim (exact decode == ML, unit
energy, proper six-colouring, exact orbit reduction, decoder equivariance, the
differential round-trip, the C3 packing-gain helpers — dB gain with CI band and the
impulsive floor — and the run-provenance metadata), plus the Mark 2 additions: the
phase-pair primitive and aliases, the exact inversion involution/commutativity, the
C7 radial-dual structure, the folded decoder's bitwise-ML equivalence and storage
folds, the C₆×Z₂ (C8) differential invariance, and the corrected colouring
equivariance. It runs in a few seconds and needs no GPU (PyTorch is optional). For
the full testing documentation — coverage breakdown, how to interpret results,
troubleshooting, and a `pytest` command reference — see
[`tests/TESTS_README.md`](tests/TESTS_README.md).

```bash
# Run the full suite (129 tests)
python -m pytest tests/ -q

# With coverage
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_lattice_graph -q
```

### Code Structure

```
radial_dual_signal_processing/
├── src/                                                  # Source code directory
│   ├── tqf_hex_signal.py                                  # Core signal library
│   ├── tqf_lattice_graph.py                               # Lattice graph + six-colouring
│   ├── simulation_01_hex_demod_correctness_and_latency.py # C1, C2
│   ├── simulation_02_hex_vs_square_ber_packing_gain.py    # C3
│   ├── simulation_03_symmetry_reduced_metric_exact.py     # C4
│   ├── simulation_04_sixcoloring_denoise_gpu.py           # C5
│   ├── simulation_05_phase_rotation_robustness.py         # C6
│   ├── simulation_06_phasepair_inversion_folded_decoder.py # C1/C2 storage, C7
│   ├── simulation_07_radial_dual_constellation.py         # C7 structure
│   └── make_figures.py                                    # Figure generator
├── tests/                                                # Automated test suite
│   ├── conftest.py                                        # Puts src/ on the path
│   ├── test_tqf_hex_signal.py                             # C1 + core primitives
│   ├── test_tqf_lattice_graph.py                          # C5 graph + colouring
│   ├── test_symmetry_and_equivariance.py                  # C4 + C6
│   ├── test_packing_gain.py                               # C3 gain helpers (sim02)
│   ├── test_phasepair_inversion.py                        # C7 + C8 + folded decoder
│   └── TESTS_README.md                                    # Testing documentation
├── run_all.ps1                             # One-command full reproduction (Windows)
├── run_all.sh                              # One-command full reproduction (Linux/macOS)
├── ACKNOWLEDGEMENT.md                      # Some gratitude
├── README.md                               # This file
├── requirements.txt                        # Core dependencies
├── requirements-dev.txt                    # Development dependencies
└── LICENSE                                 # MIT license
```

(Running the experiments also creates `results/` and `figures/`.)

### Code Quality Tools

```bash
# Format code
black src tests

# Type checking
mypy src

# Linting
flake8 src tests
```

---

## 9. License

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

## 10. References

### Preprints/Publications

- **Schmidt, Nathan O.** (2025). *The Tri-Quarter Framework: Unifying Complex Coordinates with Topological and Reflective Duality across Circles of Any Radius*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1281679](https://www.techrxiv.org/users/906377/articles/1281679)

- **Schmidt, Nathan O.** (2025). *Tri-Quarter Framework Case Study: BPSK Signal Processing*. TechRxiv.
[https://www.techrxiv.org/users/906377/articles/1311875](https://www.techrxiv.org/users/906377/articles/1311875)

- **Schmidt, Nathan O.** (2026). *Tri-Quarter Framework: Radial Dual Triangular Lattice Graph*. TechRxiv (companion subproject).

### Related References

- Conway, J. H., & Sloane, N. J. A. (1999). *Sphere Packings, Lattices and Groups* (3rd ed.). Springer. (Hexagonal/A₂ packing gain.)
- Proakis, J. G., & Salehi, M. (2008). *Digital Communications* (5th ed.). McGraw-Hill.
- Sklar, B. (2001). *Digital Communications: Fundamentals and Applications* (2nd ed.). Prentice Hall.

---

**`QED`**

**Last Updated:** June 23, 2026<br>
**Version:** 1.0.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
