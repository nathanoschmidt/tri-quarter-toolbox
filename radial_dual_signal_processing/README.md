# Tri-Quarter Framework Method: Exact, Symmetry-Reduced, and Parallel Signal Processing on the Radial Dual Triangular Lattice: README

Reproducible experiments for the paper *"Tri-Quarter Framework Method: Exact,
Symmetry-Reduced, and Parallel Signal Processing on the Radial Dual Triangular
Lattice."* This `radial_dual_signal_processing` subproject applies the Tri-Quarter
Framework (TQF) hexagonal-lattice machinery — Eisenstein/A₂ basis, the order-6
rotation, sector/shell/color encodings, circle inversion, and exact `Z[sqrt(3)]`
nearest-point predicates — to digital communications. Every claim in the paper is
backed by a script here that prints a clean, copy-pasteable results table, and by
an automated test that pins the underlying invariant.

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
validates eleven precise, falsifiable claims (**C1–C4, C6–C12**), kept strictly
separate so no claim borrows credit from another. The eight reproducible studies
that back them are organized into three Episodes — **Episode I** (Studies 1–2),
**Episode II** (Studies 3–4), and **Episode III** (Studies 5–8):

- **C1 — Exact demodulation.** A closed-form Eisenstein/A₂ nearest-point decoder
  is *bitwise-identical* to exhaustive maximum-likelihood (ML) decoding.
- **C2 — Constant-time decode on the fast path.** The decoder is O(1) per symbol
  on its interior fast path; exterior symbols take an O(M) ML fallback, so the
  worst case is O(M) and the amortized cost is contingent on fast-path coverage
  (reported vs SNR). Throughput is in the same O(1) class as a square-QAM slicer.
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
- **C4 — Symmetry-reduced exact evaluation.** Orbit reduction computes an exact
  constellation metric (the squared-distance enumerator) on one representative per
  orbit and replicates it, reproducing the full value as the *identical*
  integer/rational quantity (verified by `==`), with an exact **6× reduction in
  distance evaluations** under Z6 (and the full dihedral D6 fold, also exact). A
  separate, label-domain combinatorial enumerator additionally folds by the full
  Z₆×Z₂ group (a storage/label reduction, **not** multiplied with the exact
  Euclidean reduction — inversion is conformal, not isometric).
- **C6 — Equivariant / rotation-robust decode.** The decoder commutes with the
  order-6 rotation (verified exactly on a real hex constellation); a differential
  hexagonal scheme is invariant to any static phase ambiguity that is a multiple
  of π/3 (the hexagonal analogue of DPSK).
- **C7 — Radial dual constellation.** There is a filled hexagonal constellation
  (shells {3,4,9,12,16,36,48}, M=42) that is simultaneously shell-complete
  (exact Z₆), phase-pair-uniform (identical sector occupancy per shell), and
  inversion-paired (closed under the exact circle inversion ι_r about r²=12, which
  maps shell N to its **integer-dual** shell 144/N: 3↔48, 4↔36, 9↔16, 12 self-dual).
  It is invariant under the full **order-12 Z₆×Z₂** group — the rotation×inversion
  subgroup of the centrosymmetric hexagonal point group D₆ₕ (stated as such, not
  over-claimed as the full 24). The phase-pair + inversion **folded decoder** is
  bitwise-identical to exhaustive ML while storing only the fundamental domain:
  the label table folds **6×** by rotation and **10.5×** by rotation+inversion
  (reported separately; the self-dual boundary shell limits the latter below 12×).
- **C8 — Combined rotation+inversion differential codec.** A differential codec
  carrying a (sector∈ℤ₆, inversion-bit∈ℤ₂) pair as component-wise differences is
  invariant under all **12** static Z₆×Z₂ actions — extending the C6 DPSK analogue
  to absorb a static amplitude-inversion ambiguity. The inversion bit is a discrete
  label state, never a Euclidean operation.
- **C9 — Radial dual geometry crossover.** The shell-sparse inversion-paired
  constellation has both the smaller nearest-neighbor multiplicity **and** the
  smaller minimum distance, so a genuine SER crossover exists: it wins at low
  Es/N₀ and the matched filled constellation wins at high Es/N₀ (a single sign
  flip, located empirically by a common-random-number paired AWGN sweep).
- **C10 — Symmetry-reduced design search.** An exhaustive constellation design
  search that canonicalizes candidate point sets under the dihedral group D6
  evaluates strictly fewer candidates than the unreduced search while finding the
  identical exact optimum.
- **C11 — Inversion-pair block code.** Transmitting the inversion pair
  (x, ι_r(x)) and decoding both legs with an exact integer consistency cross-check
  gives a rate-1/2 block code; because inversion is **not** an isometry, the joint
  (4D) distance spectrum is genuinely thinned relative to the isometric repetition
  and rotated-repetition baselines.
- **C12 — Exact, bit-reproducible decision.** The A2 nearest-point decision is
  computed exactly in `Z[sqrt(3)]`, so the decoded symbol is a provable,
  platform-independent function of the received-sample bits rather than a
  floating-point verdict that can flip under a different math library, FMA
  contraction, or vectorization order. A Shewchuk-style adaptive filter keeps the
  common case fast and reports the (small, measured) exact-escalation fraction.

**Honesty discipline.** Because the closed-form decoder *is* ML-equivalent, it
does not change BER/SER versus ML — the wins are in **cost** (C1/C2/C4/C7/C12) and in
**constellation geometry** (C3/C6/C7/C8/C9/C10/C11), never in beating ML. The exact
6× rotation (Euclidean) reduction and the inversion (label-only) fold are kept
distinct and are never multiplied into a combined figure. And where a structural
choice *costs* error-rate performance, that cost is measured and **pre-registered**
rather than hidden: Simulation 06 prices the radial dual constellation against a
matched filled baseline in AWGN and reports the measured crossover and high-SNR dB
price honestly either way.

> **Note on C5.** Earlier releases included a claim **C5** (a trihexagonal
> six-coloring GPU denoiser). It has been **retired** from the study set. The
> supporting module `src/tqf_lattice_graph.py` is retained on disk as a standalone
> lattice-geometry utility but is no longer wired into a numbered study.

---

## 2. Key Features

- **Exact integer/rational arithmetic** for all geometry (sector, shell, color,
  inversion, orbit reduction) — no floating-point tolerance in the exactness claims.
- **Apples-to-apples methodology**: matched M, matched average energy, common
  random numbers (identical noise/fading realizations across the compared schemes),
  a true-Gray square-QAM baseline, and exact Clopper–Pearson confidence intervals.
- **Closed-form O(1) fast-path decoder** with a provably sufficient 3×3 candidate
  window and an exhaustive-ML fallback that guarantees exactness for every symbol.
- **Three channel models**: complex AWGN, 2D impulsive, and flat Rayleigh fading
  (perfect CSI).
- **Exact `Z[sqrt(3)]` nearest-point predicate** (`tqf_exact_predicate.py`): a
  bit-reproducible decision with a Shewchuk-style adaptive floating-point filter,
  so decode decisions are a provable function of the received-sample bits.
- **Exact radial dual admissibility audit** (`tqf_admissibility.py`): family
  enumeration closed under circle inversion `N → r⁴/N`, and exact **Burnside**
  fold factors for the Z6 / D6 geometric groups and the Z₆×Z₂ / D₆×Z₂ label groups.
- **Phase-pair + inversion exactness layer**: a documented `phase_pair_sector`
  integer primitive, an exact label-space circle-inversion duality (involution +
  sector preservation, verified with zero violations), a radial dual constellation
  builder, and a folded ML decoder that is bitwise-identical to exhaustive ML while
  storing only the fundamental domain.
- **Self-documenting runs**: every study writes a `simNN_provenance.json`
  (versions + hardware) for the paper's Methods table.
- **Automated test suite** (186 tests) pinning the invariants behind every claim.
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

# Exact decode + throughput + exact-predicate escalation (C1, C2, C12)
python src\simulation_01_exact_demod_and_throughput.py

# Hex vs square packing gain (C3)
python src\simulation_02_hex_vs_square_packing_gain.py

# Symmetry-reduced exact metric (C4)
python src\simulation_03_symmetry_reduced_metric.py

# Phase-rotation robustness (C6) + Z6×Z2 differential codec (C8)
python src\simulation_04_phase_rotation_and_differential.py

# Radial dual structure + inversion-folded decoder (C7)
python src\simulation_05_radial_dual_structure_and_folded_decoder.py

# Radial dual AWGN geometry price + crossover (C7 honesty, C9)
python src\simulation_06_radial_dual_geometry_price.py

# Symmetry-reduced constellation design search (C10)
python src\simulation_07_design_search_symmetry.py

# Inversion-pair dual transmission block code (C11)
python src\simulation_08_dual_pair_transmission.py
```

**Linux/macOS:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Same commands as Windows (use python or python3, with forward slashes)
python3 src/simulation_01_exact_demod_and_throughput.py
python3 src/simulation_02_hex_vs_square_packing_gain.py
python3 src/simulation_03_symmetry_reduced_metric.py
python3 src/simulation_04_phase_rotation_and_differential.py
python3 src/simulation_05_radial_dual_structure_and_folded_decoder.py
python3 src/simulation_06_radial_dual_geometry_price.py
python3 src/simulation_07_design_search_symmetry.py
python3 src/simulation_08_dual_pair_transmission.py
```

Each simulation runs standalone with sensible defaults and prints a copy-pasteable
results table; most also accept a `--results_dir` to redirect their CSV / JSON
sidecar output.

---

## 5. Core Components

### Core Library: `src/tqf_hex_signal.py`

The shared signal-processing library: the Eisenstein/A₂ basis and exact
integer/rational primitives (the documented `phase_pair_sector` integer primitive
— with `sector_index` kept as a backward-compatible alias — shell, color residue,
circle inversion, order-6 orbits); hexagonal (filled and 6-fold-symmetric disk),
square-QAM, and **radial dual** (`build_radial_dual_constellation`) constellation
builders, each normalized to unit average energy; the closed-form O(1) fast-path
decoder, an exhaustive-ML decoder, and the phase-pair + inversion **folded** ML
decoder (`make_folded_decode_context` / `decode_hex_folded`); the exact label-space
inversion duality (`invert_label`, `invert_sector_shell`, `dual_shell_norm`,
`verify_inversion_commutativity` — the lattice paper's Prop. 4.15, cited and
empirically verified, not re-derived); the three channel models; the differential
hexagonal encoder/decoder and the combined rotation+inversion (Z₆×Z₂)
`differential_encode_t24` / `differential_decode_t24`; and exact Clopper–Pearson
intervals.

### Exact Predicate: `src/tqf_exact_predicate.py`

The exact `Z[sqrt(3)]` nearest-point predicate behind claim C12. The A2 decoder's
nearest-point decision reduces to the sign of `α + β·sqrt(3)` (α, β exact
rationals), computed exactly so the decision never depends on floating-point
rounding. A Shewchuk-style adaptive filter (`filtered_nearest_in_window`) returns
the certified float sign in the common case and escalates to the exact-rational
path (`exact_nearest_in_window`) only for samples on a Voronoi bisector; the two
paths agree bit-for-bit, and the escalation fraction is small and measured.

### Admissibility: `src/tqf_admissibility.py`

The exact radial dual family enumeration and Burnside fold-factor audit consumed
by Studies 3, 5, 7, and 8. It answers, in integer arithmetic, which radial dual
constellations exist (shell sets closed under circle inversion `N → r⁴/N`) and by
how much symmetry folds the work (`burnside_geometric_fold` for Z6 / D6,
`burnside_label_fold` for Z₆×Z₂ / D₆×Z₂). Run as a script it prints the family
table and the fold-factor table and writes `mark4_admissibility.json`.

### Lattice Graph (retired utility): `src/tqf_lattice_graph.py`

The truncated triangular lattice graph and the trihexagonal six-coloring. This is
a **standalone lattice-geometry utility** — it backed the retired claim C5 and is
no longer wired into any numbered study, but is kept on disk for the conflict-free
parallel construction it encodes. The coloring is `2·((a−b) mod 3) + ((a+b) mod 2)`
and is verified proper against the edge set on construction; it is **proper but
not rotation-equivariant** (only the underlying 3-coloring `(a−b) mod 3`, exposed
as `three_coloring`, is order-6-equivariant).

---

## 6. Simulations & Tools

### Simulation 01: Exact Decode + Throughput (C1, C2, C12)

```bash
python3 src/simulation_01_exact_demod_and_throughput.py
```

Verifies the closed-form decoder and the exact-predicate referee both equal
exhaustive ML over a dense grid and a noisy stream (target: zero mismatches), then
reports median per-symbol decode time for the hex fast-path decoder, the square-QAM
slicer, and exhaustive ML across M ∈ {16, 64, 256, 1024} (headline: the hex:slicer
cost-class parity, C2). It also reports the O(1) fast-path coverage vs Eᵦ/N₀ and the
**exact-predicate escalation fraction** (C12) — the small, measured share of symbols
whose float filter cannot certify the nearest-point sign and must escalate to the
exact `Z[sqrt(3)]` path. Writes `sim01_latency.csv`, `sim01_coverage_vs_snr.csv`,
and `sim01_provenance.json`.

### Simulation 02: Hex vs Square Packing Gain (C3)

```bash
python3 src/simulation_02_hex_vs_square_packing_gain.py
```

SER and BER vs Eᵦ/N₀ for hex (TQF = ML) vs square QAM at matched order and unit
average energy, over AWGN / Rayleigh / impulsive with common random numbers,
Clopper–Pearson bands, and per-point significance by a paired **McNemar + Holm**
test. Writes the per-(channel, M) tables (`sim02_ber_*.csv`), a durable
`sim02_gain_summary.csv` (dB packing gain with a CI band and a `ci_resolved` flag,
and the impulsive floor ≈ p·(1−1/M)), and `sim02_significance_summary.csv`. SER
leads; the hex Gray-*like* BER penalty is disclosed, never plotted as a win.

### Simulation 03: Symmetry-Reduced Exact Metric (C4)

```bash
python3 src/simulation_03_symmetry_reduced_metric.py
```

Computes the exact pairwise squared-distance enumerator (→ d_min, kissing number)
from one representative per orbit and verifies it equals the full enumerator
bitwise (`==`) for **both** the rotation group Z6 and the full dihedral group D6,
with an exact 6× operation-count reduction (`sim03_symmetry.csv`). It cross-checks
each fold against the exact **Burnside** factor (`sim03_fold_audit.csv`) and writes
`sim03_provenance.json`. Inversion is shown *not* to be an isometry (the firewall),
so the Euclidean 6× fold and the label-domain Z₆×Z₂ fold are kept strictly separate.

### Simulation 04: Phase-Rotation Robustness (C6) + Z₆×Z₂ Differential (C8)

```bash
python3 src/simulation_04_phase_rotation_and_differential.py
```

Verifies decoder equivariance exactly on a real hex constellation
(`sim04_equivariance_check.csv`), then compares coherent vs differential hex SER
across a phase-offset sweep crossing 60° (`sim04_phase.csv`): the differential
scheme returns to the noise floor at 0° **and** 60° while coherent decoding suffers
a catastrophic sector slip near 60°. It additionally verifies the **C8** combined
rotation+inversion (Z₆×Z₂) differential codec, recovering both the senary sector
data and the inversion bit under all 12 static actions with zero violations
(`sim04_t24_check.csv`).

### Simulation 05: Radial Dual Structure + Inversion-Folded Decoder (C7)

```bash
python3 src/simulation_05_radial_dual_structure_and_folded_decoder.py
```

Builds the C7 radial dual constellation(s) and verifies the folded ML decoder is
bitwise-identical to exhaustive ML while storing only the fundamental domain. It
reports the structure (shell-complete, phase-pair-uniform, inversion-paired, the
self-dual boundary shell) and the storage ablation — full label table vs rotation
fold (**6×**) vs rotation+inversion fold (**10.5×**) — as two separate factors,
cross-checked against the exact Burnside label folds. Writes `sim05_structure.csv`,
`sim05_fold_audit.csv`, `sim05_throughput.csv`, and `sim05_provenance.json`.

### Simulation 06: Radial Dual AWGN Geometry Price + Crossover (C7 honesty, C9)

```bash
python3 src/simulation_06_radial_dual_geometry_price.py
```

Prices the radial dual constellation honestly against a matched **filled** baseline
at equal order and equal average energy, over a common-random-number paired AWGN
Es/N₀ sweep with Clopper–Pearson bands and paired **McNemar + Holm** significance.
From the exact (d_min, N_nn) pair it predicts the crossover **direction** (C9:
radial dual better at low Es/N₀, filled better at high Es/N₀) and locates the sign
flip empirically, then reports the high-SNR dB price. Writes `sim06_nn_model.csv`,
`sim06_crossover_awgn.csv`, `sim06_price_summary.csv`, and `sim06_provenance.json`.

### Simulation 07: Symmetry-Reduced Design Search (C10)

```bash
python3 src/simulation_07_design_search_symmetry.py
```

Runs an exhaustive constellation design search over a point pool, canonicalizing
each candidate subset under the 12 D6 isometries. It reports the orbit count and the
exact D6 reduction factor, confirms the reduced search finds the identical optimum
as the unreduced search, and times both (`sim07_search_summary.csv`,
`sim07_timing.csv`). A separate `sim07_inversion_firewall.csv` records the fraction
of subsets whose inversion image differs from the subset itself — inversion changes
the objective, so it can never be used to fold the search. Writes
`sim07_provenance.json`.

### Simulation 08: Inversion-Pair Dual Transmission (C11)

```bash
python3 src/simulation_08_dual_pair_transmission.py
```

Transmits the inversion pair (x, ι_r(x)) and decodes both legs with an exact
integer consistency cross-check (a rate-1/2 block code). It computes the exact 4D
product-distance spectrum and shows it is thinned relative to the isometric
repetition and rotated-repetition baselines (the isometry theorem holds exactly),
then sweeps AWGN / Rayleigh / impulsive channels comparing the pair scheme to
repetition. Writes `sim08_product_distance.csv`, `sim08_awgn.csv` /
`sim08_rayleigh.csv` / `sim08_impulsive.csv`, `sim08_summary.csv`, and
`sim08_provenance.json`.

---

## 7. Reproducing the Paper

```bash
# Step through all eight studies (each prints its table and writes its CSV/JSON):
python3 src/simulation_01_exact_demod_and_throughput.py
python3 src/simulation_02_hex_vs_square_packing_gain.py
python3 src/simulation_03_symmetry_reduced_metric.py
python3 src/simulation_04_phase_rotation_and_differential.py
python3 src/simulation_05_radial_dual_structure_and_folded_decoder.py
python3 src/simulation_06_radial_dual_geometry_price.py
python3 src/simulation_07_design_search_symmetry.py
python3 src/simulation_08_dual_pair_transmission.py
```

Each study prints a console summary and writes its CSV tables plus a
`simNN_provenance.json` (versions + hardware) for the paper's Methods table, into
the current directory by default (or a `--results_dir` you supply). The C3 headline
dB gains live in `sim02_gain_summary.csv`; the C9 crossover and high-SNR price live
in `sim06_price_summary.csv`.

For a captured reference run of all eight studies — the real console output and
result tables on one machine with fixed seeds, plus a claim-by-claim summary — see
[`EXAMPLE_RESULTS.md`](EXAMPLE_RESULTS.md) (raw sidecars in `example_results/`).

---

## 8. Development

- **Language**: Python 3.10+
- **Key Libraries**: NumPy, SciPy
- **Code Quality**: black, mypy, flake8 (install via `requirements-dev.txt`)
- **Platform**: Cross-platform (Windows/Linux/macOS)

### Testing

The `tests/` suite pins the invariant behind each claim: exact decode == ML, unit
energy, exact orbit reduction (Z6 and D6), decoder equivariance and the differential
round-trip, the C3 packing-gain helpers (dB gain with CI band and the impulsive
floor), the phase-pair primitive and inversion involution/commutativity, the C7
radial dual structure and the folded decoder's bitwise-ML equivalence and storage
folds, the Z₆×Z₂ (C8) differential invariance, the Study 6 geometry-price crossover
(C9), the D6-canonical design search (C10), the inversion-pair block code (C11), the
exact `Z[sqrt(3)]` nearest-point predicate (C12), and the radial dual admissibility /
Burnside fold audit. It runs in ~90 seconds and needs no GPU. For the full testing
documentation — coverage breakdown, how to interpret results, troubleshooting, and a
`pytest` command reference — see [`tests/TESTS_README.md`](tests/TESTS_README.md).

```bash
# Run the full suite (186 tests)
python -m pytest tests/ -q

# With coverage
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_exact_predicate --cov=tqf_admissibility -q
```

### Code Structure

```
radial_dual_signal_processing/
├── src/                                                       # Source code directory
│   ├── tqf_hex_signal.py                                       # Core signal library
│   ├── tqf_exact_predicate.py                                  # Exact Z[sqrt(3)] predicate (C12)
│   ├── tqf_admissibility.py                                    # Family enumeration + Burnside folds
│   ├── tqf_lattice_graph.py                                    # Retired six-coloring utility (no study)
│   ├── simulation_01_exact_demod_and_throughput.py             # C1, C2, C12
│   ├── simulation_02_hex_vs_square_packing_gain.py             # C3
│   ├── simulation_03_symmetry_reduced_metric.py                # C4
│   ├── simulation_04_phase_rotation_and_differential.py        # C6, C8
│   ├── simulation_05_radial_dual_structure_and_folded_decoder.py # C7
│   ├── simulation_06_radial_dual_geometry_price.py             # C7 honesty, C9
│   ├── simulation_07_design_search_symmetry.py                 # C10
│   └── simulation_08_dual_pair_transmission.py                 # C11
├── tests/                                                     # Automated test suite
│   ├── conftest.py                                             # Puts src/ on the path
│   ├── test_tqf_hex_signal.py                                  # C1 + core primitives
│   ├── test_tqf_exact_predicate.py                             # C12 exact predicate
│   ├── test_tqf_admissibility.py                               # Family enumeration + Burnside folds
│   ├── test_symmetry_and_equivariance.py                       # C4 + C6
│   ├── test_packing_gain.py                                    # C3 gain helpers (sim02)
│   ├── test_phasepair_inversion.py                             # C7 + C8 + folded decoder
│   ├── test_constellation_geometry.py                          # C3/C7/C9 geometry + price
│   ├── test_noise_and_decoding.py                              # C1/C3/C6 noise mapping + folded decoder
│   ├── test_design_search_and_dual_pair.py                     # C10 + C11
│   └── TESTS_README.md                                         # Testing documentation
├── example_results/                        # Captured reference run (console logs + CSV/JSON)
├── ACKNOWLEDGEMENT.md                      # Some gratitude
├── CHANGELOG.md                            # Release history
├── EXAMPLE_RESULTS.md                      # Real output of all eight studies + summaries
├── HAPPY_BIRTHDAY.md                       # A birthday note
├── QA.md                                   # Q&A / anticipated referee questions
├── README.md                               # This file
├── requirements.txt                        # Core dependencies
├── requirements-dev.txt                    # Development dependencies
└── LICENSE                                 # MIT license
```

(Running the experiments also writes each study's CSV tables and
`simNN_provenance.json` sidecars into the working directory or a `--results_dir`.)

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

**Last Updated:** July 15, 2026<br>
**Version:** 1.3.1<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
