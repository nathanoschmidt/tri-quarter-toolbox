# Tri-Quarter Framework Method: Radial Dual Signal Processing: Automated Testing Framework: TESTS_README

**Each time code is deployed to production without test automation, a child gets a mullet**

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.3.0<br>
**Date:** July 8, 2026<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-8caae6.svg)](https://scipy.org/)
[![pytest](https://img.shields.io/badge/pytest-compatible-green.svg)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/core--library%20coverage-83%25-green.svg)](https://coverage.readthedocs.io/)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Testing Philosophy](#2-testing-philosophy)
- [3. File Structure](#3-file-structure)
- [4. Running Tests](#4-running-tests)
- [5. Test Coverage](#5-test-coverage)
- [6. Interpreting Test Results](#6-interpreting-test-results)
- [7. Best Practices Implemented](#7-best-practices-implemented)
- [8. Adding New Tests](#8-adding-new-tests)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Reference: pytest Commands](#10-reference-pytest-commands)
- [11. Recap Summary](#11-recap-summary)

---

## 1. Overview

This automated testing framework pins the mathematical invariant behind each of
the subproject's falsifiable claims (C1-C4, C6-C12), so that the Mark 3
experiments and the paper rest on continuously-verified foundations rather than
one-off runs.

Key Features:
- **186 total test cases** across 9 test files
- **~90 second** full test suite execution time (dominated by two exhaustive
  100k-sample exact-predicate sweeps and the Monte-Carlo folded-decode checks)
- **No GPU required** (CPU-only; PyTorch not needed for the suite)
- Shared path configuration (`conftest.py`) so tests import `src/` modules directly
- Exact integer/rational verification (no tolerance) for the exactness claims
- Adversarial brute-force cross-checks for the nearest-point decoder
- Mathematical property verification (rotation equivariance, energy normalization)
- Statistical-tooling checks (exact Clopper-Pearson intervals, exact McNemar +
  Holm-Bonferroni family-wise correction)
- C3 packing-gain helpers: log-linear SER interpolation, CI-bounded dB gain, impulsive floor
- Phase-pair + inversion exactness: involution, sector-preserving commutativity, folded ML
- Exact constellation geometry: min-distance enumerator and the hexagonal packing
  advantage over square QAM at matched energy (Study 3)
- Radial-dual geometry price (C7/C9): the exact `(d_min, N_nn)` crossover direction
  and the CRN-paired AWGN low-/high-SNR sign flip (Study 6)
- Fractional-bits Eb/N0 -> noise mapping (log2(6) senary) and the vectorized
  folded-decoder membership vs a reference loop
- **Exact nearest-point predicate (C12):** the `Z[sqrt(3)]` sign law and the
  float-filtered path agreeing bit-for-bit with the exact-rational referee
- **Radial-dual admissibility (Burnside):** family enumeration closed under circle
  inversion and exact fold factors for C6 / D6 / C6xZ2
- **Design search (C10)** and **dual-pair transmission (C11):** the D6-canonical
  candidate reduction and the exact inversion-pair consistency cross-check
- Reproducibility metadata checks (provenance dict shape; `ran_on_cuda` honesty; JSON sidecar)
- Claim-mapped coverage: every test traces to a specific claim

All tests are compatible with Python 3.10+ and run under `pytest`.

> **Mark 3 note.** The Mark 3 study set is Studies 1-8, backing claims C1-C4 and
> C6-C12. Claim **C5** (the trihexagonal six-coloring GPU denoiser) has been
> retired from the study set; `src/tqf_lattice_graph.py` is retained on disk as a
> standalone lattice-geometry utility but is no longer exercised by a dedicated
> test module.

---

## 2. Testing Philosophy

The testing framework is built on five pillars:

**A. Exactness Before Performance**
- The headline claims (C1, C4, C12) are *exactness* claims, so the tests assert
  bitwise/integer equality (`==`, `array_equal`, exact `Fraction`/`Counter`),
  never a floating-point tolerance.
- The closed-form decoder is checked against exhaustive ML on a dense grid and a
  noisy stream; the A2 nearest-point routine and its exact `Z[sqrt(3)]` predicate
  are checked against a wide brute-force search, including adversarial
  near-bisector inputs.

**B. Scientific Rigor**
- Mathematical properties are verified directly: order-6 rotation equivariance of
  the sector label, the order-6 group property, unit average energy of every
  constellation, the exact 6x orbit-reduction operation count, and the exact
  Burnside fold factors.
- Reproducibility is enforced via fixed NumPy `default_rng` seeds.

**C. Apples-to-Apples Preconditions**
- The fairness preconditions for the C3 comparison are tested independently of the
  comparison itself: both hex and square constellations are verified to carry unit
  average energy, the Eb/N0 -> noise-variance mapping is checked against the
  closed-form relation, and the hexagonal minimum-distance advantage is asserted
  directly.

**D. Robustness**
- Edge cases: the origin is excluded from sector tests; boundary/deep-hole points
  are explicitly exercised for the decoder; the exact fold routine rejects a
  non-group-closed point set rather than rounding.
- Platform independence: Windows, Linux, macOS (pure NumPy/SciPy, no GPU).

**E. Maintainability**
- Clear, self-documenting test names (`test_<thing>_<property>`).
- A single `conftest.py` handles the `src/` import path (no per-file boilerplate).
- Tests exercise the *shipped* code (the library and the claim-bearing simulation
  helpers), not re-implementations, so a regression in the real code is caught.

---

## 3. File Structure

```
tests/
|-- conftest.py                           # Puts src/ on the import path (no fixtures needed)
|-- test_tqf_hex_signal.py                # Core library: C1 + primitives + channels + differential
|-- test_tqf_exact_predicate.py           # Exact Z[sqrt(3)] nearest-point predicate (C12)
|-- test_tqf_admissibility.py             # Radial-dual family enumeration + Burnside fold factors
|-- test_symmetry_and_equivariance.py     # Exact orbit reduction (C4) + decoder equivariance (C6)
|-- test_packing_gain.py                  # C3 packing-gain helpers: dB gain + CI band, impulsive floor, McNemar + Holm
|-- test_phasepair_inversion.py           # Phase-pair primitive, exact inversion duality, folded decoder, C7 structure, C6xZ2 (C8) codec
|-- test_constellation_geometry.py        # Any-size builder + exact Study 3 geometry + Study 6 radial-dual price (C3/C7/C9)
|-- test_noise_and_decoding.py            # Fractional-bits Eb/N0 mapping + vectorized folded decoder + phase-offset (C1/C3/C6)
|-- test_design_search_and_dual_pair.py   # D6-canonical design search (C10) + inversion-pair block code (C11)
`-- TESTS_README.md                       # This file
```

### Core Infrastructure File

**conftest.py**

Purpose: shared test configuration.

Contains:
- Automatic path configuration: inserts the subproject's `src/` directory onto
  `sys.path` so test modules can `import tqf_hex_signal` / `import tqf_admissibility`
  / `import tqf_exact_predicate` (and the simulation helpers) directly, exactly as
  the simulations do when run from `src/`.

Benefits:
- Eliminates per-file path-setup boilerplate.
- Single source of truth for how tests locate the code under test.

This subproject deliberately keeps the test infrastructure minimal: there is no
`run_tests.py`, no `pytest.ini`, and no custom markers/fixtures, because the suite
is small, fast, and dependency-light. The tests are plain `pytest` functions
(many parameterized), runnable with a single `pytest` invocation.

---

## 4. Running Tests

All commands are run from the **subproject root** (`radial_dual_signal_processing/`),
i.e. the directory that contains `src/` and `tests/`.

### A. Quick Start (Recommended)

Install the test dependency (once) and run the full suite:

```bash
python -m pip install -r requirements-dev.txt    # provides pytest, pytest-cov
python -m pytest tests/ -q
```

On PEP 668 ("externally-managed-environment") systems, add
`--break-system-packages` to the `pip install` command. On Windows, `python` is
usually correct; the launcher form `py -m pytest tests/ -q` also works.

### B. Common Variants

Verbose (one line per test):

```bash
python -m pytest tests/ -v
```

Run a single file:

```bash
python -m pytest tests/test_tqf_hex_signal.py
```

Run a single test (or a parameterized case):

```bash
python -m pytest tests/test_symmetry_and_equivariance.py::test_orbit_reduced_enumerator_equals_full
```

Select by keyword:

```bash
python -m pytest tests/ -k equivariance
python -m pytest tests/ -k "decode and ml"
```

Stop on first failure / show prints / re-run last failures:

```bash
python -m pytest tests/ -x
python -m pytest tests/ -s
python -m pytest tests/ --lf
```

### C. Coverage

Core-library coverage (the tested `src/` modules):

```bash
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_exact_predicate --cov=tqf_admissibility --cov-report=term-missing
```

HTML report:

```bash
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_exact_predicate --cov=tqf_admissibility --cov-report=html
# then open htmlcov/index.html
```

---

## 5. Test Coverage

### Overall

- **Total test cases:** 186
- **Test files:** 9 (+ `conftest.py`)
- **Full suite execution time:** ~90 seconds (dominated by the two 100k-sample
  exact-predicate sweeps and the Monte-Carlo folded-decode checks)
- **GPU required:** No
- **Core-library code coverage:** ~83%

### Test Suite Breakdown by File

| Test File | Test Cases | Status | Focus Area |
|-----------|------------|--------|------------|
| test_tqf_hex_signal.py | 46 | All passing | C1 exact decode vs ML, A2 nearest-point vs brute force, unit-energy normalization, sector/rotation primitives, channel + Eb/N0 mapping, Clopper-Pearson intervals, differential hex, and provenance/reproducibility metadata |
| test_tqf_exact_predicate.py | 6 | All passing | C12 exact `Z[sqrt(3)]` sign law, exact decision vs a 100k-sample brute-force argmin, the float-filtered path agreeing bit-for-bit with the exact referee, and an engineered near-bisector escalation |
| test_tqf_admissibility.py | 19 | All passing | Exact lattice primitives (Eisenstein norm, order-6 rotation, reflection involution), Loeschian shell enumeration, radial-dual family enumeration closed under `N -> r^4/N`, and exact Burnside fold factors for C6 / D6 / C6xZ2 (incl. the non-group-closed rejection) |
| test_symmetry_and_equivariance.py | 19 | All passing | C4 exact orbit-reduced enumerator (`==` Counter) for C6 and D6, exact 6x op-count, unit `d_min`; C6 decoder equivariance on a real hex constellation + differential round-trip |
| test_packing_gain.py | 19 | All passing | C3 packing-gain helpers (sim02): log-linear SER interpolation, CI-bounded dB gain + `ci_resolved` flag, impulsive floor `~ p*(1-1/M)` and recovered p; exact McNemar p-value + Holm-Bonferroni correction, and `run_channel` emitting the paired significance columns |
| test_phasepair_inversion.py | 29 | All passing | Phase-pair primitive + aliases + rotation increment; exact label-space inversion (involution + sector-preserving commutativity, Prop. 4.15); C7 radial-dual structure (phase-pair-uniform, inversion-paired, integer-dual shell pairs, boundary fixed); folded decoder bitwise-ML on grid + Monte-Carlo and the 6x / 10.5x storage folds; inversion-is-not-isometric firewall; the exact 6x Euclidean fold on the radial-dual object; C6xZ2 (C8) differential invariance over all 12 actions |
| test_constellation_geometry.py | 13 | All passing | The any-size filled builder vs the power-of-two builder and the hex-42 baseline; Study 3 exact min-distance enumerator + the hexagonal packing advantage over square QAM at matched energy; Study 6 radial-dual geometry price: the exact `(d_min, N_nn)` crossover direction and the CRN-paired AWGN low-/high-SNR sign flip |
| test_noise_and_decoding.py | 29 | All passing | The fractional-bits Eb/N0 -> sigma^2 mapping (log2(6) senary) with the integer-bits regression and the label-free Es/N0 clamp; the vectorized folded-decoder membership vs a reference loop and vs exhaustive ML under both folds, plus the derived stored-shell array; a phase-offset sanity check (coherent slips at 60 deg, differential returns to floor) |
| test_design_search_and_dual_pair.py | 6 | All passing | C10 exact objective + D6-canonical candidate reduction that preserves the exact optimum + canonical invariance under the group action; C11 inversion-pair codebook (inversion-paired, self-dual boundary), the exact 4D product-distance spectrum thinned vs the isometric repetition baseline, and the exact integer consistency cross-check (noiseless recovery + corruption detection) |

### Claim Coverage (what each claim's invariant is pinned by)

| Claim | Invariant pinned | Representative tests |
|-------|------------------|----------------------|
| **C1** | Closed-form decode is bitwise-identical to exhaustive ML; the 3x3 window equals a wide brute-force nearest-point search (incl. deep holes); folding preserves bitwise-ML | `test_decode_fast_equals_ml_on_noisy_stream`, `test_nearest_lattice_point_matches_bruteforce`, `test_folded_decoder_matches_ml_dense_grid`, `test_folded_decode_equals_ml` |
| **C2** | Fast-path coverage rises with SNR (the precondition for the O(1) fast path); folded label-table storage shrinks (6x / 10.5x) | `test_fast_path_fraction_increases_with_snr`, `test_folded_decoder_storage_reduction_factors` |
| **C3** | Matched unit average energy and correct Eb/N0 -> sigma^2 mapping; the hexagonal minimum-distance advantage; the dB packing-gain readout with a CI-bounded gain and impulsive floor | `test_hex_packs_tighter_than_square_at_matched_energy`, `test_noise_sigma_mapping`, `test_gain_ci_band_brackets_the_point_estimate`, `test_impulsive_floor_matches_formula_and_recovers_p` |
| **C4** | Orbit-reduced squared-distance enumerator equals the full enumerator exactly for C6 and D6; exact 6x operation-count reduction; `d_min^2 = 1` | `test_orbit_reduced_enumerator_equals_full`, `test_d6_orbit_reduced_enumerator_equals_full`, `test_exact_six_times_operation_reduction`, `test_disk_min_distance_squared_is_one` |
| **C6** | Decoder commutes with the order-6 rotation (sector index permutes by +1); differential scheme invariant to any k*60-degree offset | `test_decoder_equivariance_on_real_hex_constellation`, `test_rotation_increments_phase_pair_sector_by_one`, `test_sim04_offset_sweep_sanity` |
| **C7** | Radial-dual constellation is shell-complete, phase-pair-uniform, and inversion-paired about r^2=12 with exact integer-dual shell pairs; inversion is a same-sector involution fixing the boundary; the folded decoder is bitwise-ML | `test_radial_dual_constellation_basic_shape`, `test_radial_dual_every_shell_has_one_point_per_sector`, `test_radial_dual_inversion_is_same_sector_involution_fixing_boundary`, `test_inversion_is_not_a_euclidean_isometry` |
| **C8** | Combined rotation+inversion (C6xZ2) differential codec recovers the (sector, inversion-bit) data under all 12 static actions; the inversion bit is a pure label state | `test_t24_codec_invariant_under_all_twelve_actions`, `test_t24_inversion_bit_is_pure_label_state` |
| **C9** | Radial-dual has both the smaller `N_nn` and the smaller `d_min`, so a crossover exists; the CRN-paired AWGN sweep shows radial-dual winning at low Es/N0 and losing at high Es/N0 | `test_radial_dual_smaller_dmin_and_nn_predicts_crossover`, `test_paired_sweep_shows_low_high_snr_sign_flip` |
| **C10** | The D6-canonical design search evaluates strictly fewer candidates than the unreduced search yet finds the identical exact optimum; canonicalization is invariant under the group action | `test_d6_canonicalization_reduces_and_preserves_optimum`, `test_d6_canonical_is_invariant_under_the_group_action` |
| **C11** | The inversion-pair codebook is inversion-paired with a self-dual boundary; the exact 4D product-distance spectrum is thinned vs the isometric repetition baseline; the integer consistency cross-check recovers noiselessly and detects corruption | `test_pair_codebook_is_inversion_paired_and_has_a_boundary`, `test_exact_product_distance_thins_the_spectrum_vs_isometric_baselines`, `test_consistency_check_recovers_message_noiselessly_and_flags_corruption` |
| **C12** | The `Z[sqrt(3)]` sign law is exact; the exact decision matches a brute-force float argmin except at genuine near-bisector ties; the float-filtered path agrees bit-for-bit with the exact referee and escalates on the bisector | `test_sign_basic`, `test_exact_matches_float_argmin`, `test_filtered_equals_exact`, `test_engineered_bisector_escalates` |
| **Repro.** | Provenance dict carries the Methods-table fields; `ran_on_cuda` can only be True with a CUDA device; the JSON sidecar round-trips | `test_collect_provenance_has_methods_table_fields`, `test_collect_provenance_ran_on_cuda_requires_a_cuda_device`, `test_emit_provenance_writes_json_roundtrip` |

### Module-Specific Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| tqf_exact_predicate.py | 95% | The exact `Z[sqrt(3)]` sign law, the exact-rational window decision, and the float-filtered path (including escalation) are exercised end-to-end. |
| tqf_admissibility.py | 83% | Lattice primitives, Loeschian shells, radial-dual family enumeration, and the exact Burnside geometric/label folds are exercised. Uncovered lines are mainly the `__main__` table printer and JSON emitter. |
| tqf_hex_signal.py | 82% | Primitives, constellations (including the any-size builder and square QAM), the closed-form / ML / folded decoders, the fractional-bits Eb/N0 mapping, AWGN, Clopper-Pearson, McNemar/Holm, differential coding, and the provenance helpers are exercised. Uncovered lines are mainly the impulsive/Rayleigh channels, some bit-labeling utilities, the torch/CUDA branch of `collect_provenance` (no GPU in CI), and the `__main__` self-test -- all exercised end-to-end by the simulations. |
| **TOTAL (tested core)** | **83%** | The simulation CLIs are validated by end-to-end runs, not by unit tests; the suite focuses on the reusable libraries and the claim-bearing simulation helpers, which are imported and tested directly. `tqf_lattice_graph.py` is a retired standalone utility (no dedicated test). |

---

## 6. Interpreting Test Results

### A. Understanding pytest Output

When you run `python -m pytest tests/ -v`, you'll see output like:

```
======================================================== test session starts ========================================================
platform win32 -- Python 3.13.x, pytest-8.x, pluggy-1.x
collected 186 items

tests/test_tqf_hex_signal.py::test_eisenstein_norm_equals_squared_euclidean PASSED                                          [  1%]
tests/test_tqf_admissibility.py::test_rotate60_has_order_six PASSED                                                          [ 40%]
...
tests/test_design_search_and_dual_pair.py::test_consistency_check_recovers_message_noiselessly_and_flags_corruption PASSED  [100%]

==================================================== 186 passed in ~90s ====================================================
```

**Key Elements:**
- **Platform info:** OS, Python version, pytest version
- **collected N items:** total tests discovered (parameterized cases count individually)
- **Test status:** PASSED, FAILED, SKIPPED, ERROR
- **Progress percentage:** completion progress
- **Summary:** final count and execution time

### B. Test Status Meanings

| Status | Symbol | Meaning | Action Required |
|--------|--------|---------|-----------------|
| PASSED | `.` or `PASSED` | Test executed successfully | None |
| FAILED | `F` or `FAILED` | Assertion failed or unexpected error | Investigate and fix |
| SKIPPED | `s` or `SKIPPED` | Test intentionally skipped | None (none are expected in this suite) |
| ERROR | `E` | Error during setup/collection | Fix test infrastructure (often an import/path issue) |

This suite has **no conditional skips** and needs no GPU: a healthy run is simply
`186 passed`. A `SKIPPED` or `ERROR` here is unexpected and worth investigating
(usually a missing dependency or a `src/` path problem -- see Troubleshooting).

### C. Reading a Failure Message

Example (an intentionally broken exactness assertion):

```python
FAILED tests/test_symmetry_and_equivariance.py::test_orbit_reduced_enumerator_equals_full[37]
______________________ test_orbit_reduced_enumerator_equals_full[37] ______________________

    def test_orbit_reduced_enumerator_equals_full(T):
        ...
>       assert folded == full
E       assert Counter({...}) == Counter({...})

tests/test_symmetry_and_equivariance.py:39: AssertionError
```

Because the exactness tests assert exact `==` (integer `Counter`, `Fraction`, or
`array_equal`), any failure means the code no longer reproduces the exact
quantity -- a real regression, not a tolerance drift.

### D. Coverage Reports

```bash
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_exact_predicate --cov=tqf_admissibility --cov-report=term-missing
```

Output (abridged):

```
Name                         Stmts   Miss  Cover   Missing
----------------------------------------------------------
src\tqf_admissibility.py       209     35    83%   ...
src\tqf_exact_predicate.py      96      5    95%   ...
src\tqf_hex_signal.py          614    112    82%   ...
----------------------------------------------------------
TOTAL                          919    152    83%
```

### E. Common Patterns to Watch For

**Pattern 1: All tests in one file fail at collection**
- Almost always a `src/` import/path problem -- confirm `conftest.py` is present in
  `tests/` and that you ran pytest from the subproject root.

**Pattern 2: Only the exactness tests fail**
- A change altered the decoder, the orbit reduction, the exact predicate, or the
  lattice geometry. Re-derive the affected primitive; these tests are the canary.

**Pattern 3: Flaky/random failures**
- This suite is deterministic (fixed seeds). A flaky failure usually means a seed
  was removed, or a tolerance was loosened where exact equality was expected.

### F. Debugging Failed Tests

```bash
python -m pytest tests/test_tqf_hex_signal.py::test_nearest_lattice_point_matches_bruteforce -vv -s
python -m pytest tests/ --pdb
git diff HEAD~1 -- src tests
```

### G. Expected Test Results (Health Check)

[OK] **Healthy test suite:**
```
186 passed in ~90s
```

[WARN] **Needs attention:**
```
182 passed, 4 failed in ~90s
```
- A regression in one area. Read the failing assertion; map it to its claim via
  Section 5.

[X] **Critical issues:**
```
collected 0 items / errors during collection
```
- Import/path breakage (often a missing `conftest.py` or running from the wrong
  directory). Fix before proceeding.

---

## 7. Best Practices Implemented

- Self-documenting test names and module docstrings stating which claim each file pins.
- Exact assertions (`==`, `np.array_equal`, `Fraction`, `Counter`) for exactness
  claims; tolerances only where a statistic is genuinely random.
- Reproducibility via fixed `np.random.default_rng(seed)` in every stochastic test.
- Adversarial inputs (lattice deep holes, near-bisector samples) for the decoder
  and the exact predicate.
- Tests run against the shipped modules and simulation helpers, not re-implementations.

### Test Organization

- One file per concern: core library, exact predicate, admissibility, symmetry,
  packing gain, phase-pair/inversion, constellation geometry, noise/decoding,
  design search + dual pair.
- Parameterization (`@pytest.mark.parametrize`) over M, orbit sizes, and radii.
- A single `conftest.py` for path setup; no hidden global state.
- Heavy simulation sweeps are kept fast via small trial counts (and `monkeypatch`
  of module-level sweep constants where needed).

---

## 8. Adding New Tests

### Basic Template

```python
import numpy as np
import pytest

import tqf_hex_signal as t   # src/ is already on the path via conftest.py


def test_new_property_holds():
    """One-line statement of WHAT invariant this pins, and WHY it matters."""
    rng = np.random.default_rng(0)                  # fixed seed -> reproducible
    con = t.build_filled_constellation(64)          # arrange
    result = np.mean(np.abs(con.points_unit) ** 2)  # act
    assert result == pytest.approx(1.0, abs=1e-12)  # assert (exactness where possible)


@pytest.mark.parametrize("m", [16, 64, 256])
def test_new_property_over_several_M(m):
    con = t.build_filled_constellation(m)
    assert con.size == m
```

### Guidelines

1. Put the file in `tests/` as `test_<area>.py`; `conftest.py` handles imports.
2. Use a descriptive name: `test_<thing>_<property>_<condition>`.
3. State the claim the test supports in the docstring.
4. Prefer exact assertions for exactness claims; reserve `pytest.approx`/tolerances
   for genuinely random quantities.
5. Seed every stochastic test with `np.random.default_rng(<int>)`.
6. Parameterize over regimes (M, radius, orbit size) instead of copy-pasting.
7. Keep tests independent, isolated, and fast; `monkeypatch` heavy module-level
   sweep constants (e.g. `TRIALS`) instead of running the full production sweep.

---

## 9. Troubleshooting

### A. Import / Collection Errors

**Problem:** `ModuleNotFoundError: No module named 'tqf_hex_signal'`

**Solution:**
- Run pytest from the **subproject root** (the directory containing `src/` and `tests/`).
- Confirm `tests/conftest.py` is present -- it adds `src/` to `sys.path`.
- Verify manually: `python -c "import sys; sys.path.insert(0,'src'); import tqf_hex_signal; print('ok')"`.

### B. pytest Not Found

**Problem:** `pytest: command not found` or `No module named pytest`

**Solution:**
- Install it: `python -m pip install pytest` (or `-r requirements-dev.txt`).
- Always invoke as `python -m pytest` so it uses the interpreter you installed into.

### C. Missing Dependencies

**Problem:** `ImportError` for numpy/scipy

**Solution:**
- Install core deps: `python -m pip install -r requirements.txt`.
- PyTorch is **not** required for the test suite.

### D. Coverage Tool Missing

**Problem:** `unrecognized arguments: --cov`

**Solution:**
- Install coverage plugin: `python -m pip install pytest-cov` (included in `requirements-dev.txt`).

### E. Unexpected Failures After a Change

**Problem:** Exactness tests fail after editing the decoder / orbit reduction / predicate

**Solution:**
- Read the failing assertion and map it to its claim via Section 5.
- Re-run just that test with `-vv -s` to see the differing values.
- These tests are intentionally strict (`==`): the fix is to restore the exact
  behavior, not to loosen the assertion.

---

## 10. Reference: pytest Commands

### Basic Execution

```bash
python -m pytest tests/                                  # Run all tests
python -m pytest tests/test_tqf_hex_signal.py            # Run one file
python -m pytest tests/test_tqf_admissibility.py::test_rotate60_has_order_six  # One test
python -m pytest tests/ -k "decode or equivariance"      # Keyword match
```

### Output Control

```bash
python -m pytest tests/ -v          # Verbose (one line per test)
python -m pytest tests/ -vv         # Very verbose (full assertion detail)
python -m pytest tests/ -q          # Quiet
python -m pytest tests/ -s          # Show print()/stdout
python -m pytest tests/ --tb=short  # Short tracebacks
```

### Execution Control

```bash
python -m pytest tests/ -x           # Stop on first failure
python -m pytest tests/ --maxfail=3  # Stop after 3 failures
python -m pytest tests/ --lf         # Re-run last failed
python -m pytest tests/ --ff         # Failed first, then the rest
python -m pytest tests/ --pdb        # Drop into debugger on failure
```

### Coverage

```bash
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_exact_predicate --cov=tqf_admissibility
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_exact_predicate --cov=tqf_admissibility --cov-report=term-missing
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_exact_predicate --cov=tqf_admissibility --cov-report=html
```

### Parallel Execution (optional, requires pytest-xdist)

```bash
python -m pip install pytest-xdist
python -m pytest tests/ -n auto
```

---

## 11. Recap Summary

This automated testing framework validates the `radial_dual_signal_processing`
subproject through **186 test cases** organized across **9 test modules**, pinning
the exact invariant behind each Mark 3 claim (C1-C4, C6-C12). It emphasizes
exactness (integer/rational/`Counter` `==` for the exactness claims),
reproducibility (fixed seeds), and minimal, dependency-light infrastructure
(a single `conftest.py`).

### Key Statistics

- **Total test cases:** 186
- **Test modules:** 9 (+ `conftest.py`)
- **Skipped tests:** 0
- **Execution time:** ~90 seconds (full suite)
- **Core-library code coverage:** ~83% (`tqf_hex_signal.py` 82%, `tqf_exact_predicate.py` 95%, `tqf_admissibility.py` 83%)
- **GPU required:** No

### Key Strengths

- **Exactness-first:** the headline claims (C1, C4, C12) are checked with bitwise/integer equality.
- **Tests the shipped code:** library and simulation helpers are imported and exercised directly.
- **Adversarial correctness checks:** the decoder and the exact predicate are validated against brute force on deep-hole and near-bisector inputs.
- **Fast and portable:** ~90 seconds, CPU-only, Windows/Linux/macOS.
- **Claim-mapped:** every test traces to a specific claim (see Section 5).

### Testing Categories

1. **Primitive/unit tests:** geometry, constellations (incl. the any-size builder and square QAM), channels, the fractional-bits Eb/N0 mapping, statistics, exact lattice primitives, packing-gain helpers, phase-pair primitive, inversion duality, provenance.
2. **Property tests:** rotation equivariance, group order, exact orbit reduction, exact Burnside fold factors, energy normalization, involution/commutativity, radial-dual structure and geometry price, C6xZ2 invariance, phase-offset robustness, D6-canonical invariance.
3. **Correctness-vs-reference tests:** closed-form and folded decode vs exhaustive ML; the exact predicate vs a 100k-sample brute-force argmin; vectorized folded membership vs a reference loop; storage-fold factors; the inversion-pair consistency cross-check.

### Claim Coverage

- **C1 (exact decode):** decode == ML on a dense grid and a noisy stream; nearest-point vs brute force; folded decode == ML.
- **C2 (constant-time fast path):** fast-path coverage rises with SNR; folded label-table storage shrinks (6x / 10.5x).
- **C3 (packing gain):** matched unit energy + correct Eb/N0 mapping; hex `d_min` > square `d_min`; dB-gain interpolation, CI-bounded gain, and impulsive floor `~ p*(1-1/M)`.
- **C4 (symmetry-reduced exact metric):** orbit-reduced enumerator == full for C6 and D6; exact 6x op-count.
- **C6 (equivariant decode):** decoder commutes with the order-6 rotation; differential invariance to k*60 degrees.
- **C7 (radial-dual constellation):** shell-complete, phase-pair-uniform, inversion-paired about r^2=12; integer-dual shell pairs; folded decoder bitwise-ML.
- **C8 (C6xZ2 differential codec):** combined rotation+inversion differential invariance over all 12 static actions.
- **C9 (radial-dual geometry price):** smaller `(d_min, N_nn)` predicts a crossover; the paired sweep shows the low-/high-SNR sign flip.
- **C10 (symmetry-reduced design search):** D6 canonicalization reduces the candidate count while preserving the exact optimum.
- **C11 (inversion-pair block code):** inversion-paired codebook with a thinned product-distance spectrum and an exact integer consistency cross-check.
- **C12 (exact nearest-point predicate):** the `Z[sqrt(3)]` sign law and the float-filtered path agreeing bit-for-bit with the exact referee.

**For questions or issues, please contact:** nate.o.schmidt@coldhammer.net

---

**`QED`**

**Last Updated:** July 8, 2026<br>
**Version:** 1.3.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
