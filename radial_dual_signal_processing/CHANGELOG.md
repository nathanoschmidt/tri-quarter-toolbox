# Tri-Quarter Framework Method — Radial Dual Signal Processing: Changelog

**We're hammering hexagons into the radio waves — it's a work in progress, and this time the tests came along for the ride, so no child gets a mullet!**

All notable changes to this `radial_dual_signal_processing` subproject are
documented in this file. Every entry maps to one of the eight falsifiable claims
(C1–C8) described in the [`README.md`](README.md). The exact 6× rotation (Euclidean)
reduction and the inversion (label-only) fold are always kept distinct here, exactly
as they are in the code — they are never multiplied into a single headline figure.

---

## [1.2.0] - 2026-07-04 — *"Mark 3"* — Happy 250th Birthday USA!

A methodology- and honesty-hardening pass. No new claims (still C1–C8); instead, a
new **pre-registered geometry-price study** that measures what the radial-dual
constellation *costs* in raw AWGN, an exact **per-information-bit Eb/N0** correction,
an **Es/N0 axis relabel**, **apples-to-apples throughput** timing, and an exact
**constellation-geometry / fairness block** that predicts the C3 gains from first
principles — plus a reorganized, larger test suite.

### Added

**New simulation**
- **Simulation 08** — radial-dual AWGN geometry price
  (`simulation_08_radial_dual_geometry_price.py`). A **pre-registered
  null-to-negative result** backing C7's honesty: it measures the AWGN SER *cost* of
  the C7 radial-dual constellation against a matched filled hex-42 baseline at equal
  order (M=42) and equal average energy — common random numbers, paired **McNemar +
  Holm**, Clopper–Pearson bands — with **both** practical decoders asserted
  bitwise-identical to ML at every point (the C1 tie-in). The nearest-neighbor
  approximation (the same model Simulation 03 uses for the C3 gains) predicts a price
  of **~+3.33 dB @ SER 1e-2**; the pre-registered bracket **2.8–4.0 dB** is centered on
  that prediction, and the measured price is reported against it either way. Writes
  `sim08_ser.csv`, `sim08_gap_summary.csv`, and `sim08_provenance.json`.
  (C7 honesty; C1 tie-in.)

**New library surface in `src/tqf_hex_signal.py`**
- `build_filled_constellation_any(m)` — the *m* lowest-energy A₂ lattice points for
  **any** m ≥ 2 (no power-of-two requirement, no bit labeling), used as the matched
  hex-42 baseline for Simulation 08. For power-of-two *m* its point set and exact
  scale agree with `build_filled_constellation`.
- `_noise_sigma_sq` now accepts **fractional** bits/symbol, so a senary (log₂6-bit)
  constellation is given a per-information-bit Eb/N0 exactly; `bits_per_symbol = 0`
  parameterizes Es/N0 directly.
- `decode_hex_folded`'s membership test is vectorized (`np.isin` against a
  precomputed stored-shell array, exposed as `stored_arr`); the decisions and the
  fast-path mask are bitwise-identical to the previous per-symbol implementation.

**New readouts on existing simulations**
- **Simulation 03** now also writes an exact constellation-geometry / fairness block
  (`--skip_geometry_block` to disable): exact d_min², nearest-neighbor multiplicity
  K̄, PAPR, and Gray-map Hamming distance for hex and square QAM at M ∈ {16, 64, 256}
  (self-checked with `==` against a pre-registered table), plus a
  nearest-neighbor-approximation SER prediction that **quantitatively explains** the
  measured C3 gains: the finite-M d_min² gain is 0.46–0.81 dB (above the 0.6 dB
  asymptote for M ≥ 64), and the larger hex multiplicity pulls the net measured gain
  down to 0.29–0.50 dB. Writes `sim03_constellation_geometry.csv` and
  `sim03_nn_prediction.csv`.
- **Simulation 04** gains an optional torch-CPU middle baseline (`--also_torch_cpu`):
  it times the identical torch backend on `device='cpu'` between the single-threaded
  NumPy baseline and the GPU (additive columns; off by default, so the pre-existing
  schema and CSVs are unchanged).

**Tests**
- Reorganized and expanded. The version-named test grouping is gone; its coverage now
  lives in two topic files — `tests/test_constellation_geometry.py` (the any-size
  builder, the Simulation 03 exact hex/square geometry and NN gain prediction, and the
  Simulation 08 radial-dual/hex-42 exact geometry, CRN pairing, decoder tie-in, and NN
  price) and `tests/test_noise_and_decoding.py` (the fractional-bits Eb/N0 → noise
  mapping, the vectorized folded decoder vs a reference loop and vs exhaustive ML, and
  a phase-offset decoding sanity check).
- Suite total: **129 → 173 tests** across 7 files. Still ~5 seconds, still no GPU
  required.

**Reproduction tooling**
- `run_all.ps1` and `run_all.sh` now drive all **eight** studies (01–08).

### Changed
- **Exact per-information-bit Eb/N0 (Simulation 05).** The senary constellation
  carries exactly log₂6 ≈ 2.585 information bits/symbol. The `--ebn0` flag is now
  mapped through the true fractional bit count rather than an integer-rounded 3 bits,
  so Eb/N0 is per information bit exactly (for this constellation
  Es/N0 = Eb/N0 + 10·log₁₀(log₂6) ≈ Eb/N0 + 4.12 dB). SER values in `sim05_phase.csv`
  shift slightly; the qualitative story (collapse at 60°, ~2× differential penalty at
  0°, return to floor at 60°) and the exact zero-noise equivariance / round-trip
  checks are unchanged.
- **Es/N0 axis relabel (Simulation 06).** The radial-dual constellation carries no bit
  labeling, so the SNR knob parameterizes **Es/N0** directly, not Eb/N0. The flags are
  now `--esn0` / `--esn0_grid` (`--ebn0` / `--ebn0_grid` retained as aliases) and the
  coverage CSV column is `esn0_db`. The coverage values are **byte-identical** — a
  name-only correction.
- **Apples-to-apples throughput (Simulations 01, 06).** Simulation 01 interleaves the
  TQF/ML timing repeats (A/B) on one batch and reports median [min, max] dispersion,
  and drops an earlier note about an upward drift in the TQF cost that its own data
  contradicted (M=1024 was the fastest row). The folded decoder's membership test is
  vectorized so its throughput comparison against ML is apples-to-apples (both fully
  vectorized NumPy). Every exactness, fold, and coverage figure is unchanged.
- **Comment hygiene.** Library and simulation comments/docstrings were standardized to
  American English and stripped of version-history narration — this changelog is the
  single source of truth for what changed between releases.

---

## [1.1.0] - 2026-06-27 — *"Mark 2"*

The radial-dual half of the project arrives: the constellation that is closed under
circle inversion, the decoder that stores only its fundamental domain, and the
differential codec that shrugs off a static amplitude inversion.

### Added

**New claims**
- **C7 — Radial-dual constellation.** A filled hexagonal constellation
  (shells {3,4,9,12,16,36,48}, M=42) that is simultaneously shell-complete (exact
  C₆), phase-pair-uniform (identical sector occupancy per shell), and inversion-paired
  under the exact circle inversion ι_r about r²=12 (mapping shell N to its
  integer-dual shell 144/N: 3↔48, 4↔36, 9↔16, 12 self-dual). Invariant under the
  full order-12 **C₆×Z₂** group — stated as such, not over-claimed as the full 24.
- **C8 — Combined rotation+inversion differential codec.** A differential codec
  carrying a (sector ∈ ℤ₆, inversion-bit ∈ ℤ₂) pair as component-wise differences,
  invariant under all 12 static C₆×Z₂ actions. Extends the C6 DPSK analogue to
  absorb a static amplitude-inversion ambiguity. The inversion bit is a discrete
  label state, never a Euclidean operation.

**New simulations**
- **Simulation 06** — phase-pair + inversion-folded decoder
  (`simulation_06_phasepair_inversion_folded_decoder.py`). Builds the C7 object and
  verifies the folded ML decoder is bitwise-identical to exhaustive ML while storing
  only the fundamental domain; reports the storage ablation (6× rotation, 10.5×
  rotation+inversion — separately), fast-path coverage vs SNR, and the exact
  involution/commutativity of the label-space inversion. (C1/C2 storage; C7 object.)
- **Simulation 07** — radial-dual constellation structure
  (`simulation_07_radial_dual_constellation.py`). Verifies the integer-dual shell
  pairs, phase-pair uniformity, and inversion pairing of the C7 object. (C7.)

**New library surface in `src/tqf_hex_signal.py`**
- The documented `phase_pair_sector` integer primitive (see *Changed* for the rename).
- The exact label-space inversion duality: `invert_label`, `invert_sector_shell`,
  `dual_shell_norm`, and `verify_inversion_commutativity` (the lattice paper's
  Prop. 4.15 — cited and empirically verified, not re-derived).
- The radial-dual constellation builder `build_radial_dual_constellation`.
- The phase-pair + inversion **folded** ML decoder: `make_folded_decode_context`
  and `decode_hex_folded`.
- The combined rotation+inversion (C₆×Z₂) differential codec:
  `differential_encode_t24` / `differential_decode_t24`.

**New readouts on existing simulations**
- Simulation 03 now also writes `sim03_inversion_reduction.csv` for the C7 object:
  the **Euclidean** enumerator folds by rotation exactly 6×, while a discrete
  inversion-invariant **label** enumerator folds by the full C₆×Z₂ group (rotation
  6×, combined 10.5×). The two reductions are reported separately and never multiplied.
- Simulation 05 now also verifies the **C8** C₆×Z₂ differential codec
  (`sim05_t24_check.csv`), recovering both the senary sector data and the inversion
  bit under all 12 static actions with zero violations. The C8 block uses an
  independent RNG (`--t24_seed`) so the two pre-existing CSVs stay byte-identical.

**Tests**
- New `tests/test_phasepair_inversion.py` (31 tests): the phase-pair primitive and
  aliases, the exact inversion involution/commutativity, the C7 radial-dual
  structure, the folded decoder's bitwise-ML equivalence and storage folds, the
  C₆×Z₂ (C8) differential invariance, and the corrected coloring equivariance.
- Suite total: **98 → 129 tests**. Still ~5 seconds, still no GPU required.

### Changed
- **Corrected the coloring-equivariance claim** (the honesty fix). The trihexagonal
  six-coloring is **proper** but **not** rotation-equivariant; only the underlying
  triangular-lattice 3-coloring `(a−b) mod 3` (now exposed as `three_coloring` in
  `tqf_lattice_graph.py`) is order-6-equivariant. Simulation 04 records this as a
  checked artifact (`sim04_coloring_equivariance.csv`). C5's conflict-free schedule
  needs only properness, so this distinction does not weaken the claim — but the
  earlier "order-6-equivariant six-coloring" wording was an over-claim and is gone.
- Renamed the sector primitive `sector_index` → **`phase_pair_sector`** (and
  `sector_index_array` → the corresponding array form) to reflect that it encodes the
  full phase-pair label, not merely a sector index. `sector_index` is kept as a
  backward-compatible alias so nothing downstream breaks.
- Expanded `README.md` and `tests/TESTS_README.md` to cover C7/C8, simulations
  06–07, the phase-pair + inversion exactness layer, and the 129-test suite.

---

## [1.0.0] - 2026-06-23 — *"Mark 1"*

### Added
- **Initial release!** The Tri-Quarter Framework carried out of the 1D BPSK case
  study and into 2D hexagonal signal constellations, where the order-6 symmetry
  does real work.
- **Core libraries.** `src/tqf_hex_signal.py` — the Eisenstein/A₂ basis and exact
  integer/rational primitives (sector, shell, color residue, circle inversion,
  order-6 orbits); hexagonal and square-QAM constellation builders normalized to unit
  average energy; the closed-form O(1) fast-path decoder plus an exhaustive-ML
  decoder; three channel models (AWGN, 2D impulsive, flat Rayleigh); the
  differential hexagonal codec; and exact Clopper–Pearson intervals.
  `src/tqf_lattice_graph.py` — the truncated triangular lattice graph, the
  trihexagonal six-coloring (verified proper against the edge set), and the
  color-ordered relaxation sweep.
- **Simulations 01–05** backing claims **C1–C6**:
  - **C1/C2** — exact decode equals ML over a dense grid and a Monte-Carlo stream;
    O(1) fast-path coverage and throughput speedup vs M.
  - **C3** — hex vs square QAM packing gain at matched M and matched average energy,
    with Clopper–Pearson bands and per-point significance by a paired **McNemar +
    Holm** test (common random numbers). AWGN coding gain saturates ~0.4 dB at M≥64;
    Rayleigh is CI-limited; impulsive is robustness parity at the error floor.
  - **C4** — order-6 orbit reduction computing an exact constellation metric on one
    sector and replicating it, with an exact 6× reduction in distance evaluations.
  - **C5** — the six-coloring scheduling a lock-free parallel lattice-signal
    denoiser, benchmarked CPU vs GPU with a `RAN_ON_CUDA` self-certification.
  - **C6** — decoder equivariance under the order-6 rotation, plus a differential
    hexagonal scheme invariant to any static phase ambiguity that is a multiple of
    π/3 (the hexagonal analogue of DPSK).
- **Automated test suite** — 98 tests across 4 files pinning the invariant behind
  every claim, plus `tests/TESTS_README.md`.
- **Reproduction tooling** — one-command reproduction (`run_all.ps1` on Windows,
  `run_all.sh` on Linux/macOS), a CSV-driven figure generator (`make_figures.py`),
  and a per-study `simNN_provenance.json` (versions + hardware) for the paper's
  Methods table.

---

**`QED`**

**Last Updated:** July 2, 2026<br>
**Version:** 1.2.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
