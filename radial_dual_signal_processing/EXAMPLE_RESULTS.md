# Tri-Quarter Framework Method — Radial Dual Signal Processing: Example Results

This document records the **real, verbatim output** of one complete run of all eight
studies (Episodes I–III, backing claims **C1–C4, C6–C12**) on a single machine with
fixed seeds. It is provided so a reader can see exactly what the simulations produce
without running them, and so each falsifiable claim can be checked against a concrete
number.

Every figure below was copied from the captured console logs and CSV sidecars in
[`example_results/`](example_results/). All results are **deterministic under the
documented seeds and reproduce bit-for-bit on re-run — with one exception: wall-clock
timing** (nanoseconds-per-symbol, wall-speedup), which is machine- and load-specific
and will differ on other hardware. Timing is reported as an *illustrative systems
observation*, never as part of an exactness claim.

---

## Table of Contents

1. [Environment & Reproducibility](#1-environment--reproducibility)
2. [Methodology & Honesty Discipline](#2-methodology--honesty-discipline)
3. [Claim → Result Summary](#3-claim--result-summary)
4. [Episode I — Studies 1–2](#4-episode-i--studies-12)
5. [Episode II — Studies 3–4](#5-episode-ii--studies-34)
6. [Episode III — Studies 5–8](#6-episode-iii--studies-58)
7. [Limitations & Caveats](#7-limitations--caveats)
8. [How to Regenerate](#8-how-to-regenerate)

---

## 1. Environment & Reproducibility

| Item | Value |
|---|---|
| Date of run | 2026-07-15 |
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.14 |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| `tqf_hex_signal` version | 1.3.1 |
| Git commit (base) | `a4fac34` (working tree at v1.3.1) |
| Master seed | `42` (every study; Study 4 also uses an independent `t24_seed=808`) |
| Hardware note | single CPU core; **no GPU required** |

Environment metadata is emitted by each study into its own
`example_results/simNN_provenance.json`. Every study was launched at its **built-in
defaults** (100k Monte-Carlo trials for the stochastic studies; full Eb/N0 grids
including the Rayleigh extension to 44 dB). The exact commands appear in
[§8](#8-how-to-regenerate). All eight studies exited with status `0`.

---

## 2. Methodology & Honesty Discipline

These preconditions are enforced (and independently tested) so the comparisons are
fair and the exactness claims have no floating-point wiggle room:

- **Matched order M and matched average energy** — every constellation is normalized
  to unit average energy, so no scheme wins by transmitting harder.
- **Common random numbers (CRN)** — the compared schemes see the *identical*
  noise/fading realizations.
- **Exact Clopper–Pearson confidence intervals** on measured error rates, and a
  **paired McNemar + Holm–Bonferroni** test for per-point significance (α = 0.05).
- **Exact integer/rational arithmetic** for all geometry (sector, shell, orbit fold,
  inversion), so the exactness claims are checked with `==`, not a tolerance.

**The cost-vs-geometry firewall (the one idea to keep in mind).** The closed-form and
folded decoders *are* maximum-likelihood (ML), so they **cannot and do not beat ML on
error rate** — every win is either in **cost** (fewer operations / less storage) or in
**constellation geometry** (packing / robustness). Rotation is an isometry and may
fold a Euclidean distance computation (the exact **6×**); circle inversion is conformal
but **not** an isometry and may fold only a discrete **label/storage** computation (the
**10.5×** at M=42). These two folds are always reported separately and **never
multiplied**.

---

## 3. Claim → Result Summary

| Claim | Headline result (this run) | Source |
|---|---|---|
| **C1** Exact demod = ML | Dense-grid mismatches (hex + exact predicate vs ML) = **0**; hex O(1) decoder == ML at **every** SER point in Study 2 | S1, S2 |
| **C2** O(1) cost class | Hex ≈ 127–298 ns/symbol (flat in M's cost class) vs ML O(M); ML/hex speedup **0.7× → 29.6×** as M: 16 → 1024; fast-path coverage **95.6% → 100%** rising with SNR | S1 |
| **C3** Packing gain | AWGN resolved gain **+0.29 / +0.40 / +0.50 dB** @ SER 1e-2 for M = 16 / 64 / 256; Rayleigh CI-limited; impulsive floor parity | S2 |
| **C4** Symmetry-reduced exact metric | Orbit-folded enumerator **== full** (`exact_match_c6 = exact_match_d6 = 1`) for all constellations; exact Burnside **Z6 fold = 6** (42 evals → 7) | S3 |
| **C6** Equivariant decode | Decoder equivariance on a 138-point hex constellation: **PASS, 0 violations**; differential scheme returns to noise floor at 60° (coherent SER 0.9998 vs differential **8.2e-4**) | S4 |
| **C7** Radial dual constellation | M=42 object is shell-complete, `phase_pair_uniform = 1`, `inversion_paired = 1`, one self-dual boundary shell; folded decoder bitwise-ML; label fold **21/2 = 10.5×** | S5 |
| **C8** Z6×Z2 differential codec | Recovers (sector, inversion-bit) under **all 12** static actions: **PASS, 0 violations** | S4 |
| **C9** Geometry-price crossover | Radial dual has smaller `d_min` **and** smaller `N_nn`; predicted crossover **direction confirmed** for all M; measured crossover ≈ **12–13 dB**; high-SNR price **3.41–5.96 dB** | S6 |
| **C10** Symmetry-reduced design search | Exact **D6 reduction 11.915×** (593,775 → 49,833 evals); reduced search finds the **identical** optimum (`optima_objective_match = 1`) | S7 |
| **C11** Inversion-pair block code | Pair 4D distance spectrum **thinned**: NN multiplicity pair = 12 vs repetition = 96 (**8.0×**); isometric-leg check True; rate-1/2 detect/correct across channels | S8 |
| **C12** Exact bit-reproducible decision | Grid mismatches **0**; exact-`Z[√3]` referee agrees bit-for-bit; escalation fraction **0.0%** at these SNRs (float filter certified every sample; exact path guarantees the measure-zero bisector cases) | S1 |

---

## 4. Episode I — Studies 1–2

### Study 1 — Exact Decode + Throughput + Exact Predicate (C1, C2, C12)

**Command:** `python src/simulation_01_exact_demod_and_throughput.py`
**Artifacts:** [`sim01_console.txt`](example_results/sim01_console.txt), [`sim01_latency.csv`](example_results/sim01_latency.csv), [`sim01_coverage_vs_snr.csv`](example_results/sim01_coverage_vs_snr.csv), [`sim01_provenance.json`](example_results/sim01_provenance.json)

```text
Study 1 complete.
  dense-grid mismatches (hex+exact vs ML): 0
  M=  16  hex=132.9 ns  slicer=6.3 ns  hex/slicer=21.24  ML/hex=0.7  fast=100.00%
  M=  64  hex=126.6 ns  slicer=6.6 ns  hex/slicer=19.28  ML/hex=4.8  fast=99.95%
  M= 256  hex=151.9 ns  slicer=6.2 ns  hex/slicer=24.58  ML/hex=14.7  fast=98.99%
  M=1024  hex=297.8 ns  slicer=6.0 ns  hex/slicer=49.35  ML/hex=29.6  fast=97.94%
```

Fast-path coverage rises monotonically with SNR (from [`sim01_coverage_vs_snr.csv`](example_results/sim01_coverage_vs_snr.csv)):

| M | 6 dB | 9 dB | 12 dB | 15 dB | escalation fraction |
|---|---|---|---|---|---|
| 16 | 0.9556 | 0.9931 | 0.9998 | 1.0000 | 0.0 |
| 64 | 0.9280 | 0.9655 | 0.9918 | 0.9995 | 0.0 |
| 256 | 0.9294 | 0.9531 | 0.9724 | 0.9898 | 0.0 |
| 1024 | 0.9360 | 0.9544 | 0.9680 | 0.9792 | 0.0 |

**Summary.** The closed-form Eisenstein/A₂ decoder (and its exact `Z[√3]` referee) is
**bitwise-identical to exhaustive ML** over the dense grid — *zero* mismatches (C1,
C12). Per-symbol cost is essentially flat in M's cost class (~127–298 ns) while ML
grows O(M), so the ML/hex speedup climbs to **29.6× at M=1024** (C2). The exact-predicate
**escalation fraction is 0.0%** at these SNRs: the adaptive float filter certified the
nearest-point sign for every sample, and the exact integer path stands as the proven
guarantee for the measure-zero Voronoi-bisector cases (C12).

**Caveat.** Worst-case decode is O(M): boundary symbols whose nearest point falls
outside the finite constellation take the exact O(M) ML fallback (hence coverage < 100%
at low SNR / large M). The clean story is "O(1) fast path with an exact O(M) safety
net," not "O(1) always." Timing is machine-specific.

### Study 2 — Hexagonal vs Square-QAM Packing Gain (C3)

**Command:** `python src/simulation_02_hex_vs_square_packing_gain.py`
**Artifacts:** [`sim02_console.txt`](example_results/sim02_console.txt) (full 9-panel dump), [`sim02_gain_summary.csv`](example_results/sim02_gain_summary.csv), [`sim02_significance_summary.csv`](example_results/sim02_significance_summary.csv), per-panel [`sim02_ber_*.csv`](example_results/), [`sim02_provenance.json`](example_results/sim02_provenance.json)

Representative panel (AWGN, M=64) — the full console has all 9 channel×M panels:

```text
--- channel=awgn  M=64  (hex O(1) decoder == ML: True) ---
 Eb/N0     hexSER      sqSER   SERcmp     hexBER      sqBER
   0.0  7.646e-01  7.682e-01      tie  2.867e-01  1.999e-01
   6.0  4.210e-01  4.361e-01   hex<sq  1.476e-01  8.343e-02
  10.0  1.334e-01  1.509e-01   hex<sq  4.689e-02  2.623e-02
  12.0  4.624e-02  5.694e-02   hex<sq  1.613e-02  9.613e-03
  14.0  8.630e-03  1.294e-02   hex<sq  3.002e-03  2.167e-03
  16.0  6.200e-04  1.360e-03   hex<sq  2.167e-04  2.267e-04
   [paired McNemar + Holm @ alpha=0.05: hex<sq at 9/11 Eb/N0 points, hex>sq at 0, tie at 2; min McNemar p=2.03e-88]
   packing gain @ SER=1e-02: +0.40 dB (hex 13.82 vs square 14.23 dB)  [CI +0.27..+0.54 dB, resolved=True]
```

Headline dB gains at SER = 1e-2 (from [`sim02_gain_summary.csv`](example_results/sim02_gain_summary.csv)):

| Channel | M=16 | M=64 | M=256 |
|---|---|---|---|
| **AWGN** | +0.29 dB (resolved) | +0.40 dB (resolved) | +0.50 dB (resolved) |
| **Rayleigh** | +0.08 dB (CI not resolved) | +0.28 dB (CI not resolved) | +0.31 dB (CI not resolved) |
| **Impulsive** | floor parity | floor parity | floor parity |

**Summary.** At matched M and matched average energy with common random numbers, the
hexagonal constellation beats square QAM on **SER**. Under **AWGN** the coding gain
grows with M and **saturates near ~0.4–0.5 dB** at M ≥ 64 (the textbook ~0.6 dB
packing figure is the *asymptotic* bound, not reached at these finite M) — each AWGN
gain is CI-**resolved** and per-point significant (min McNemar p as low as 3.8e-152 at
M=256). SER is the headline (labeling-independent); the hex Gray-*like* BER penalty is
disclosed in the tables, never plotted as a win.

**Caveats (pre-registered honesty).** Under **Rayleigh** the point estimates are
positive but the **dB CI does not resolve at 100k trials** — significance there is
carried by the per-point McNemar test (which resolves for M ≥ 64). Under **impulsive**
noise an error floor ≈ p·(1 − 1/M) ≈ 0.09–0.13 masks the gain, so the honest verdict is
robustness **parity at the floor**, not a hex win. All hex O(1) decoders matched
exhaustive ML at every point (`hex O(1) decoder == ML: True`).

---

## 5. Episode II — Studies 3–4

### Study 3 — Symmetry-Reduced Exact Metric (C4)

**Command:** `python src/simulation_03_symmetry_reduced_metric.py`
**Artifacts:** [`sim03_console.txt`](example_results/sim03_console.txt), [`sim03_symmetry.csv`](example_results/sim03_symmetry.csv), [`sim03_fold_audit.csv`](example_results/sim03_fold_audit.csv), [`sim03_provenance.json`](example_results/sim03_provenance.json)

```text
Study 3 complete.
             disk-42  M= 42  orbitsZ6=  7 orbitsD6=  6  foldZ6=     6 foldD6=      7  exact(Z6,D6)=(1,1)
            disk-126  M=126  orbitsZ6= 21 orbitsD6= 15  foldZ6=     6 foldD6=   42/5  exact(Z6,D6)=(1,1)
            disk-312  M=312  orbitsZ6= 52 orbitsD6= 33  foldZ6=     6 foldD6= 104/11  exact(Z6,D6)=(1,1)
      radial-dual-42  M= 42  orbitsZ6=  7 orbitsD6=  7  foldZ6=     6 foldD6=      6  exact(Z6,D6)=(1,1)
```

**Summary.** The exact pairwise squared-distance enumerator (which yields d_min and the
kissing number) is computed from **one representative per orbit** and replicated, and
it reproduces the full enumerator as the **identical integer multiset** — verified with
`==`, giving `exact(Z6,D6) = (1,1)` for every constellation. The rotation (Z6) fold is
an **exact 6×** reduction in distance evaluations (e.g. 42 evals → 7 orbit evals), and
the full dihedral D6 fold is also exact (an exact rational, e.g. 104/11 for disk-312),
each cross-checked against its exact **Burnside** factor.

**Caveat.** The wall-clock timing shows the folded computation is *not* faster here
(`wall_speedup_c6 ≈ 0.81–0.96×`) — orbit bookkeeping costs more than it saves at these
small M. C4 is an exactness/operation-count claim (verified by `==` and Burnside), **not**
a wall-time claim.

### Study 4 — Phase-Rotation Robustness (C6) + Z6×Z2 Differential Codec (C8)

**Command:** `python src/simulation_04_phase_rotation_and_differential.py`
**Artifacts:** [`sim04_console.txt`](example_results/sim04_console.txt), [`sim04_phase.csv`](example_results/sim04_phase.csv), [`sim04_equivariance_check.csv`](example_results/sim04_equivariance_check.csv), [`sim04_t24_check.csv`](example_results/sim04_t24_check.csv), [`sim04_provenance.json`](example_results/sim04_provenance.json)

```text
[C6 verify] grounding the equivariance on the actual hex lattice (not only the 6-PSK demo):
   decoder equivariance: rotating all 138 points of a 6-fold-symmetric hex (disk)
     constellation by +60 deg permutes the decoded sector index by +1 -> PASS (0 violations)
   differential round-trip: exact data recovery at every k*60 deg offset (zero noise) -> PASS

dtheta(deg)   coherent SER   differential SER
          0      3.100e-04          6.200e-04
         30      4.998e-01          4.999e-01
         60      9.998e-01          8.200e-04
         65      1.000e+00          2.280e-03

[C8 verify] combined rotation + inversion (Z6 x Z2) differential codec:
   Invariance over all 12 static (rotation,inversion) actions on a 20000-symbol
   stream -> PASS (0 violations)
```

**Summary.** The decoder **commutes exactly** with the order-6 rotation — rotating all
138 points of a hex constellation by 60° permutes the decoded sector index by +1 with
**0 violations** (C6). The differential hexagonal scheme (the "hex DPSK") is invariant
to any static phase offset that is a multiple of 60°: at 60° coherent decoding suffers a
catastrophic sector slip (SER 0.9998) while the differential scheme returns to the noise
floor (SER **8.2e-4**). The combined **Z6×Z2** codec recovers *both* the senary sector
and the inversion bit under **all 12** static actions with **0 violations** (C8).

**Caveat.** At 0° the differential scheme pays the usual ~2× DPSK detection penalty
(6.2e-4 vs 3.1e-4) from error propagation in successive differences — expected, and
disclosed.

---

## 6. Episode III — Studies 5–8

### Study 5 — Radial Dual Structure + Inversion-Folded Decoder (C7)

**Command:** `python src/simulation_05_radial_dual_structure_and_folded_decoder.py`
**Artifacts:** [`sim05_console.txt`](example_results/sim05_console.txt), [`sim05_structure.csv`](example_results/sim05_structure.csv), [`sim05_fold_audit.csv`](example_results/sim05_fold_audit.csv), [`sim05_throughput.csv`](example_results/sim05_throughput.csv), [`sim05_provenance.json`](example_results/sim05_provenance.json)

```text
Study 5 complete.
  M= 42  shells=7 stored(rot->inv)=7->4  labelfold Z6xZ2= 21/2 (self_dual=1)  folded/ML=1.10x
  M= 48  shells=8 stored(rot->inv)=8->4  labelfold Z6xZ2=   12 (self_dual=0)  folded/ML=0.76x
  M= 54  shells=9 stored(rot->inv)=9->5  labelfold Z6xZ2= 54/5 (self_dual=1)  folded/ML=0.49x
  M= 60  shells=10 stored(rot->inv)=10->5  labelfold Z6xZ2=   12 (self_dual=0)  folded/ML=0.71x
```

**Summary.** Each radial dual object is shell-complete, `phase_pair_uniform = 1`, and
`inversion_paired = 1` (from [`sim05_structure.csv`](example_results/sim05_structure.csv)),
and the phase-pair + inversion **folded decoder stores only the fundamental domain**
while remaining **bitwise-identical to exhaustive ML**. The storage ablation reports two
**separate** factors: rotation folds the label table **6×**, and rotation+inversion folds
it by the full Z6×Z2 group. For M=42 the combined fold is **21/2 = 10.5×** rather than a
clean 12× — because the single **self-dual boundary shell** cannot be folded by the
inversion half of the group. Objects with no self-dual shell (M=48, 60) reach the clean
**12×**. Each is cross-checked against the exact Burnside label fold, which also confirms
`reflections_add_nothing = 1`.

**Caveat.** `folded/ML` throughput ranges 0.49×–1.10×: folding saves *storage*, not
necessarily wall time. The 10.5× is a **label/storage** fold and is never multiplied
with the Euclidean 6× fold of Study 3.

### Study 6 — Radial Dual AWGN Geometry Price + Crossover (C7 honesty, C9)

**Command:** `python src/simulation_06_radial_dual_geometry_price.py`
**Artifacts:** [`sim06_console.txt`](example_results/sim06_console.txt), [`sim06_nn_model.csv`](example_results/sim06_nn_model.csv), [`sim06_crossover_awgn.csv`](example_results/sim06_crossover_awgn.csv), [`sim06_price_summary.csv`](example_results/sim06_price_summary.csv), [`sim06_provenance.json`](example_results/sim06_provenance.json)

```text
Study 6 complete.
  M= 42  rd(dmin=0.234,NN=2.14) fl(dmin=0.413,NN=4.76)  pred_dir=1 meas_xover=12  dir_confirmed=1  price=3.41dB
  M= 48  rd(dmin=0.169,NN=2.38) fl(dmin=0.385,NN=4.88)  pred_dir=1 meas_xover=13  dir_confirmed=1  price=5.03dB
  M= 54  rd(dmin=0.182,NN=2.67) fl(dmin=0.367,NN=5.00)  pred_dir=1 meas_xover=12  dir_confirmed=1  price=5.05dB
  M= 60  rd(dmin=0.138,NN=2.10) fl(dmin=0.349,NN=5.10)  pred_dir=1 meas_xover=13  dir_confirmed=1  price=5.96dB
```

**Summary.** This is the project's pre-registered honest-cost study. The radial dual
object has **both** a smaller minimum distance **and** a smaller nearest-neighbor
multiplicity than a matched **filled** baseline (equal order, equal average energy). The
union-bound proxy therefore predicts a genuine **SER crossover**: the sparse radial dual
wins at low Es/N0 (fewer near neighbors) and the filled constellation wins at high Es/N0
(larger d_min). The predicted **direction is confirmed** for every M
(`dir_confirmed = 1`), the sign flip is located empirically at **~12–13 dB**, and the
honest high-SNR price is **3.41–5.96 dB** (C9; the geometry that makes C7 elegant is not
the geometry that minimizes AWGN SER at high SNR).

**Caveat.** The union bound is too loose to pin the crossover's *dB location* — it
predicts only the **direction**; the dB value is measured empirically from the
CRN-paired AWGN sweep.

### Study 7 — Symmetry-Reduced Design Search (C10)

**Command:** `python src/simulation_07_design_search_symmetry.py`
**Artifacts:** [`sim07_console.txt`](example_results/sim07_console.txt), [`sim07_search_summary.csv`](example_results/sim07_search_summary.csv), [`sim07_timing.csv`](example_results/sim07_timing.csv), [`sim07_inversion_firewall.csv`](example_results/sim07_inversion_firewall.csv), [`sim07_provenance.json`](example_results/sim07_provenance.json)

```text
Study 7 complete.
  pool = 30 points; search = C(30,6) = 593775
  D6 orbits = 49833  exact reduction = 9425/791 = 11.915x  (optimum match=True)
  eval-count reduction = 11.915x; wall speedup = 0.055x (0.9s -> 15.7s)
  FIREWALL: inversion changed the objective on 3589/5000 sampled subsets (71.8%) => inversion cannot fold a Euclidean design search
  optimum: d2min=7 mult=6 K=6
```

**Summary.** Brute-force constellation design over all C(30,6) = **593,775** candidate
subsets reduces to **49,833 D6-orbit representatives** — an exact **11.915× reduction in
evaluations** — and the reduced search finds the **identical exact optimum**
(`optimum match = True`). The inversion firewall is confirmed empirically: circle
inversion changes the design objective on **71.8%** of sampled subsets, so (unlike the
D6 isometries) it **cannot** be used to fold a Euclidean design search.

**Caveat (honest).** The **wall speedup is 0.055× — i.e. the reduced search was slower
in wall time** (0.9 s → 15.7 s) because canonicalizing each subset to its D6-lex-minimal
image costs more than the evaluations it saves at this problem size. C10 is an
**evaluation-count** reduction claim, exact and verified; it is not a wall-time claim,
and the run reports the slowdown plainly rather than hiding it.

### Study 8 — Inversion-Pair Dual Transmission Block Code (C11)

**Command:** `python src/simulation_08_dual_pair_transmission.py`
**Artifacts:** [`sim08_console.txt`](example_results/sim08_console.txt), [`sim08_product_distance.csv`](example_results/sim08_product_distance.csv), [`sim08_awgn.csv`](example_results/sim08_awgn.csv) / [`sim08_rayleigh.csv`](example_results/sim08_rayleigh.csv) / [`sim08_impulsive.csv`](example_results/sim08_impulsive.csv), [`sim08_summary.csv`](example_results/sim08_summary.csv), [`sim08_provenance.json`](example_results/sim08_provenance.json)

```text
Study 8 complete.
  codebook M=54 (r_sq=12), self-dual (repetition) msgs=6
  exact d2min_pair (energy) = 0.0659  [pair d2min == repetition d2min: True; NN multiplicity pair=12 vs rep=96 (8.0x)]
  isometric-leg check (rotated repetition == repetition): True
  impulsive @10dB: detected 89.0% of pairs, undetected SER 3.14e-02, joint SER 3.62e-01
  criterion @20dB awgn: integer 2.46e-03 vs margin 0.00e+00 (int/margin nan)
  rayleigh @15dB: joint SER 2.49e-01, repetition SER 4.05e-01, joint/rep 0.62
```

**Summary.** Transmitting the inversion pair (x, ι_r(x)) and decoding both legs with an
**exact integer consistency cross-check** gives a rate-1/2 block code. Because inversion
is **not** an isometry, the joint 4D distance spectrum is genuinely **thinned**: the pair
has the same minimum distance as plain repetition but a nearest-neighbor multiplicity of
**12 vs 96 — an 8.0× reduction** in near neighbors (the isometry theorem is confirmed:
`isometric-leg check = True`, so rotated repetition equals repetition and buys nothing).
The built-in error *detection* is visible across channels — e.g. under Rayleigh @15 dB
the joint SER (0.249) is **0.62×** the repetition SER (0.405).

**Caveat.** This is a structural/geometry result about the distance spectrum plus a
detection mechanism; it is not a claim of a coding gain over ML on a single symbol.

---

## 7. Limitations & Caveats

- **Decoders are ML.** No study beats ML on error rate — by construction. All wins are
  in **cost** or **geometry** (see the firewall, [§2](#2-methodology--honesty-discipline)).
- **Timing is illustrative.** All nanosecond and wall-speedup figures are single-machine,
  single-core, and load-dependent; only the exactness, fold-factor, and coverage figures
  are hardware-independent.
- **Rayleigh dB gain is CI-limited** at 100k trials (Study 2): positive point estimates,
  unresolved dB intervals; significance is carried by the per-point McNemar test.
- **Impulsive shows robustness parity at the floor** (Study 2): the impulse error floor
  ≈ p·(1 − 1/M) masks the packing gain.
- **Union bound predicts direction only** (Study 6): the C9 crossover *direction* is
  proven from the exact (d_min, N_nn) pair; the dB *location* is measured empirically.
- **Design-search reduction is eval-count, not wall time** (Study 7): the exact 11.915×
  fewer evaluations came with a wall-time *slowdown* at this problem size.
- **Exactness is integer/rational** for the structural bookkeeping; the Euclidean
  *decisions* still use true distances (that is what makes the decoders ML).
- **Worst-case decode is O(M)** (Study 1): interior symbols use the O(1) fast path;
  boundary symbols use an exact O(M) ML fallback.

---

## 8. How to Regenerate

From the subproject root, with the virtual environment active, run each study (each
writes its CSV tables and a `simNN_provenance.json` into the current directory):

```bash
python src/simulation_01_exact_demod_and_throughput.py
python src/simulation_02_hex_vs_square_packing_gain.py
python src/simulation_03_symmetry_reduced_metric.py
python src/simulation_04_phase_rotation_and_differential.py
python src/simulation_05_radial_dual_structure_and_folded_decoder.py
python src/simulation_06_radial_dual_geometry_price.py
python src/simulation_07_design_search_symmetry.py
python src/simulation_08_dual_pair_transmission.py
```

The raw sidecars captured for this document live in [`example_results/`](example_results/)
(one `simNN_console.txt` per study plus all CSV/JSON files). Under the documented seed
(`42`), **every figure except wall-clock timing reproduces bit-for-bit.** The studies
run standalone and need no GPU. For the invariant behind each claim, see
[`tests/TESTS_README.md`](tests/TESTS_README.md) (186 tests, ~90 s); for the claim
narrative and honesty discipline, see [`README.md`](README.md) and [`QA.md`](QA.md).

---

**`QED`**

**Last Updated:** July 15, 2026<br>
**Version:** 1.3.1<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
