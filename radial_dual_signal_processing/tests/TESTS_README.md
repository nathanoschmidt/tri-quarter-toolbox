# Tri-Quarter Framework Method: Radial Dual Signal Processing: Automated Testing Framework: TESTS_README

**Each time code is deployed to production without test automation, a child gets a mullet**

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.0.0<br>
**Date:** June 23, 2026<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-8caae6.svg)](https://scipy.org/)
[![pytest](https://img.shields.io/badge/pytest-compatible-green.svg)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/core--library%20coverage-81%25-green.svg)](https://coverage.readthedocs.io/)

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
the subproject's six falsifiable claims (C1-C6), so that the experiments and the
paper rest on continuously-verified foundations rather than one-off runs.

Key Features:
- **98 total test cases** across 4 test files
- **~5 second** full test suite execution time
- **No GPU required** (CPU-only; PyTorch not needed for the suite)
- Shared path configuration (`conftest.py`) so tests import `src/` modules directly
- Exact integer/rational verification (no tolerance) for the exactness claims
- Adversarial brute-force cross-checks for the nearest-point decoder
- Mathematical property verification (rotation equivariance, energy normalization)
- Statistical-tooling checks (exact Clopper-Pearson intervals)
- C3 packing-gain helpers: log-linear SER interpolation, CI-bounded dB gain, impulsive floor
- Reproducibility metadata checks (provenance dict shape; ran_on_cuda honesty; JSON sidecar)
- Claim-mapped coverage: every test traces to a specific claim (C1-C6)

All tests are compatible with Python 3.10+ and run under `pytest`.

---

## 2. Testing Philosophy

The testing framework is built on five pillars:

**A. Exactness Before Performance**
- The headline claims (C1, C4) are *exactness* claims, so the tests assert
  bitwise/integer equality (`==`, `array_equal`), never a floating-point tolerance.
- The closed-form decoder is checked against exhaustive ML on a dense grid and a
  noisy stream; the A2 nearest-point routine is checked against a wide brute-force
  search, including adversarial deep-hole inputs.

**B. Scientific Rigor**
- Mathematical properties are verified directly: order-6 rotation equivariance of
  the sector label, the order-6 group property, unit average energy of every
  constellation, and the exact 6x orbit-reduction operation count.
- Reproducibility is enforced via fixed NumPy `default_rng` seeds.

**C. Apples-to-Apples Preconditions**
- The fairness preconditions for the C3 comparison are tested independently of the
  comparison itself: both hex and square constellations are verified to carry unit
  average energy, and the Eb/N0 -> noise-variance mapping is checked against the
  closed-form relation and the empirical noise variance.

**D. Robustness**
- Edge cases: the origin is excluded from sector tests; boundary/deep-hole points
  are explicitly exercised for the decoder; isolated-vertex handling is covered by
  the lattice graph.
- Platform independence: Windows, Linux, macOS (pure NumPy/SciPy, no GPU).

**E. Maintainability**
- Clear, self-documenting test names (`test_<thing>_<property>`).
- A single `conftest.py` handles the `src/` import path (no per-file boilerplate).
- Tests exercise the *shipped* code (including the simulation helper functions),
  not re-implementations, so a regression in the real code is caught.

---

## 3. File Structure

```
tests/
|-- conftest.py                        # Puts src/ on the import path (no fixtures needed)
|-- test_tqf_hex_signal.py             # Core library: C1 + primitives + channels + differential
|-- test_tqf_lattice_graph.py          # Lattice graph + trihexagonal six-colouring (C5)
|-- test_symmetry_and_equivariance.py  # Exact orbit reduction (C4) + decoder equivariance (C6)
|-- test_packing_gain.py               # C3 packing-gain helpers: dB gain + CI band, impulsive floor, McNemar + Holm significance
`-- TESTS_README.md                    # This file
```

### Core Infrastructure File

**conftest.py**

Purpose: shared test configuration.

Contains:
- Automatic path configuration: inserts the subproject's `src/` directory onto
  `sys.path` so test modules can `import tqf_hex_signal` / `import tqf_lattice_graph`
  (and the simulation helpers) directly, exactly as the simulations do when run
  from `src/`.

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

Core-library coverage (the two modules the unit tests target):

```bash
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_lattice_graph --cov-report=term-missing
```

HTML report:

```bash
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_lattice_graph --cov-report=html
# then open htmlcov/index.html
```

---

## 5. Test Coverage

### Overall

- **Total test cases:** 98
- **Test files:** 4 (+ `conftest.py`)
- **Full suite execution time:** ~5 seconds (one provenance test triggers a one-time torch import, ~1 s, only if PyTorch is installed)
- **GPU required:** No
- **Core-library code coverage:** ~81%

### Test Suite Breakdown by File

| Test File | Test Cases | Status | Focus Area |
|-----------|------------|--------|------------|
| test_tqf_hex_signal.py | 46 | All passing | C1 exact decode vs ML, A2 nearest-point vs brute force, unit-energy normalization, sector/rotation primitives, channel + Eb/N0 mapping, Clopper-Pearson intervals, differential hex, and provenance/reproducibility metadata |
| test_tqf_lattice_graph.py | 18 | All passing | C5 lattice graph construction, trihexagonal six-colouring properness, colour-class partition, colour-ordered relaxation sweep |
| test_symmetry_and_equivariance.py | 15 | All passing | C4 exact orbit reduction (`==`, exact 6x op-count, d_min), C6 decoder equivariance on a real hex constellation + differential round-trip |
| test_packing_gain.py | 19 | All passing | C3 packing-gain helpers (sim02): log-linear SER interpolation, CI-bounded dB gain + `ci_resolved` flag, impulsive floor `~ p*(1-1/M)` and recovered p; exact McNemar p-value + Holm-Bonferroni correction (the paired per-point significance helpers), and `run_channel` emitting the paired McNemar/Holm columns + `sim02_significance_summary.csv` |

### Claim Coverage (what each claim's invariant is pinned by)

| Claim | Invariant pinned | Representative tests |
|-------|------------------|----------------------|
| **C1** | Closed-form decode is bitwise-identical to exhaustive ML; the 3x3 window equals a wide brute-force nearest-point search (incl. deep holes) | `test_decode_fast_equals_ml_on_noisy_stream`, `test_decode_fast_equals_ml_on_dense_grid`, `test_nearest_lattice_point_matches_bruteforce` |
| **C2** | Fast-path coverage rises with SNR (the precondition for the O(1) fast path) | `test_fast_path_fraction_increases_with_snr` |
| **C3** | Matched unit average energy (apples-to-apples precondition) and correct Eb/N0 -> sigma^2 mapping; plus the dB packing-gain readout: log-linear SER interpolation, a CI-bounded gain with an honest `ci_resolved` flag, and the impulsive floor `~ p*(1-1/M)` | `test_filled_constellation_unit_energy`, `test_square_qam_unit_energy`, `test_noise_sigma_mapping`, `test_awgn_empirical_variance_matches_mapping`, `test_gain_ci_band_brackets_the_point_estimate`, `test_gain_ci_resolved_true_when_bands_disjoint`, `test_impulsive_floor_matches_formula_and_recovers_p` |
| **C4** | Orbit-reduced enumerator equals the full enumerator exactly; exact 6x operation-count reduction; d_min^2 = 1 | `test_orbit_reduced_enumerator_equals_full`, `test_exact_six_times_operation_reduction`, `test_disk_min_distance_squared_is_one` |
| **C5** | Six-colouring is proper (conflict-free schedule), verified against the edge set; the relaxation sweep denoises and is deterministic | `test_six_coloring_proper_against_edges`, `test_color_classes_partition_all_vertices`, `test_relaxation_sweep_reduces_mse` |
| **C6** | Decoder commutes with the order-6 rotation (sector index permutes by +1); differential scheme invariant to any k*60-degree offset | `test_decoder_equivariance_on_real_hex_constellation`, `test_sector_index_in_range_and_rotation_equivariant`, `test_differential_invariant_to_k_times_60deg` |
| **Repro.** | Provenance dict carries the Methods-table fields; `ran_on_cuda` can only be True with a CUDA device (keeps Simulation 04's CSV verdict honest); the JSON sidecar round-trips | `test_collect_provenance_has_methods_table_fields`, `test_collect_provenance_ran_on_cuda_requires_a_cuda_device`, `test_emit_provenance_writes_json_roundtrip` |

### Module-Specific Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| tqf_lattice_graph.py | 90% | Graph build, six-colouring + verification, colour-ordered relaxation sweep. |
| tqf_hex_signal.py | 79% | Primitives, constellations, decoder, AWGN, Clopper-Pearson, differential coding, and the provenance helpers (`collect_provenance` / `emit_provenance`) are exercised. Uncovered lines are mainly the impulsive/Rayleigh channels, the exact-`Fraction` inversion helper, bit-labeling utilities, the torch/CUDA branch of `collect_provenance` (no GPU in CI), and the `__main__` self-test -- all exercised end-to-end by the simulations. |
| **TOTAL (core library)** | **81%** | The simulation CLIs and the figure generator are validated by end-to-end runs (`run_all.ps1` / `run_all.sh`), not by unit tests; the suite focuses on the reusable library and the claim-bearing simulation helpers (C3 packing-gain helpers in `simulation_02` and the C4/C6 helpers in `simulation_03`/`simulation_05` are imported and tested directly). |

---

## 6. Interpreting Test Results

### A. Understanding pytest Output

When you run `python -m pytest tests/ -v`, you'll see output like:

```
======================================================== test session starts ========================================================
platform linux -- Python 3.12.x, pytest-8.x, pluggy-1.x
collected 98 items

tests/test_tqf_hex_signal.py::test_eisenstein_norm_equals_squared_euclidean PASSED                                          [  1%]
tests/test_tqf_hex_signal.py::test_decode_fast_equals_ml_on_noisy_stream[16] PASSED                                         [  9%]
...
tests/test_packing_gain.py::test_impulsive_floor_matches_formula_and_recovers_p[256] PASSED                                [100%]

==================================================== 98 passed in 4.80s ====================================================
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
`98 passed`. A `SKIPPED` or `ERROR` here is unexpected and worth investigating
(usually a missing dependency or a `src/` path problem -- see Troubleshooting).

### C. Reading a Failure Message

Example (an intentionally broken exactness assertion):

```python
FAILED tests/test_symmetry_and_equivariance.py::test_orbit_reduced_enumerator_equals_full[37]
______________________ test_orbit_reduced_enumerator_equals_full[37] ______________________

    def test_orbit_reduced_enumerator_equals_full(T):
        ...
>       assert np.array_equal(full, orbit)
E       assert False
E        +  where False = <function array_equal>(array([...]), array([...]))

tests/test_symmetry_and_equivariance.py:38: AssertionError
```

**Failure anatomy:**
1. **Test path + parameter:** `...::test_orbit_reduced_enumerator_equals_full[37]` (the `[37]` is the `max_norm_sq` parameter)
2. **Failing line:** the exact assertion that broke
3. **Location:** file and line number
4. **Values:** the two arrays that differ

Because the exactness tests assert `==`/`array_equal`, any failure means the code
no longer reproduces the exact quantity -- a real regression, not a tolerance drift.

### D. Coverage Reports

```bash
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_lattice_graph --cov-report=term-missing
```

Output:

```
Name                       Stmts   Miss  Cover   Missing
------------------------------------------------------------
src/tqf_hex_signal.py        338     72    79%   [lines]
src/tqf_lattice_graph.py      70      7    90%   [lines]
------------------------------------------------------------
TOTAL                        408     79    81%
```

**Interpreting coverage:**
- **Stmts:** total executable statements
- **Miss:** statements not executed by tests
- **Cover:** percentage covered
- **Missing:** uncovered line numbers (use `--cov-report=term-missing`)

### E. Common Patterns to Watch For

**Pattern 1: All tests in one file fail at collection**
- Almost always a `src/` import/path problem -- confirm `conftest.py` is present in
  `tests/` and that you ran pytest from the subproject root.

**Pattern 2: Only the exactness tests fail**
- A change altered the decoder, the orbit reduction, or the lattice geometry.
  Re-derive the affected primitive; these tests are the canary for C1/C4.

**Pattern 3: Flaky/random failures**
- This suite is deterministic (fixed seeds). A flaky failure usually means a seed
  was removed, or a tolerance was loosened where exact equality was expected.

### F. Debugging Failed Tests

```bash
# Run a single failing test very verbosely, with prints shown
python -m pytest tests/test_tqf_hex_signal.py::test_nearest_lattice_point_matches_bruteforce -vv -s

# Drop into the debugger on failure
python -m pytest tests/ --pdb

# Inspect recent changes
git diff HEAD~1 -- src tests
```

### G. Expected Test Results (Health Check)

[OK] **Healthy test suite:**
```
98 passed in ~5s
```
- All claims' invariants hold; no skips; no warnings.

[WARN] **Needs attention:**
```
94 passed, 4 failed in ~5s
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
- Exact assertions (`==`, `np.array_equal`) for exactness claims; tolerances only
  where a statistic is genuinely random (e.g. empirical noise variance).
- Reproducibility via fixed `np.random.default_rng(seed)` in every stochastic test.
- Adversarial inputs (lattice deep holes) for the decoder correctness test.
- Tests run against the shipped modules and simulation helpers, not re-implementations.

### Test Organization

- One file per concern: core library, lattice graph, symmetry/equivariance.
- Parameterization (`@pytest.mark.parametrize`) over M and orbit sizes to cover
  several regimes without duplication.
- A single `conftest.py` for path setup; no hidden global state.

### Scientific Rigor

- Mathematical properties validated directly (equivariance, group order, energy).
- Numerical stability is a non-issue for the exact claims (integer/rational arithmetic).
- Edge cases (origin, boundary/deep-hole points, isolated vertices) covered explicitly.

---

## 8. Adding New Tests

### Basic Template

```python
import numpy as np
import pytest

import tqf_hex_signal as t   # src/ is already on the path via conftest.py


def test_new_property_holds():
    """One-line statement of WHAT invariant this pins, and WHY it matters.

    WHY:  the scientific rationale (which claim it supports).
    HOW:  the method (construct input, call the shipped function, assert).
    WHAT: the expected exact/within-tolerance result.
    """
    rng = np.random.default_rng(0)                 # fixed seed -> reproducible
    con = t.build_filled_constellation(64)         # arrange
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
3. State the claim (C1-C6) the test supports in the docstring.
4. Prefer exact assertions for exactness claims; reserve `pytest.approx`/tolerances
   for genuinely random quantities.
5. Seed every stochastic test with `np.random.default_rng(<int>)`.
6. Parameterize over regimes (M, radius, orbit size) instead of copy-pasting.
7. Keep tests independent, isolated, and fast (the whole suite should stay a few seconds).

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
- PyTorch is **not** required for the test suite (Simulation 04's GPU path is not unit-tested).

### D. Coverage Tool Missing

**Problem:** `unrecognized arguments: --cov`

**Solution:**
- Install coverage plugin: `python -m pip install pytest-cov` (included in `requirements-dev.txt`).

### E. Unexpected Failures After a Change

**Problem:** Exactness tests fail after editing the decoder / orbit reduction / lattice

**Solution:**
- Read the failing assertion and map it to its claim via Section 5.
- Re-run just that test with `-vv -s` to see the differing values.
- These tests are intentionally strict (`==`): the fix is to restore the exact
  behaviour, not to loosen the assertion.

---

## 10. Reference: pytest Commands

### Basic Execution

```bash
python -m pytest tests/                                  # Run all tests
python -m pytest tests/test_tqf_hex_signal.py            # Run one file
python -m pytest tests/test_tqf_lattice_graph.py::test_six_coloring_proper_against_edges  # One test
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
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_lattice_graph
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_lattice_graph --cov-report=term-missing
python -m pytest tests/ --cov=tqf_hex_signal --cov=tqf_lattice_graph --cov-report=html
```

### Parallel Execution (optional, requires pytest-xdist)

```bash
python -m pip install pytest-xdist
python -m pytest tests/ -n auto
```

---

## 11. Recap Summary

This automated testing framework validates the `radial_dual_signal_processing`
subproject through **98 test cases** organized across **4 test modules**, pinning
the exact invariant behind each of the six claims (C1-C6). It emphasizes exactness
(integer/rational `==` for the exactness claims), reproducibility (fixed seeds),
and minimal, dependency-light infrastructure (a single `conftest.py`).

### Key Statistics

- **Total test cases:** 98
- **Test modules:** 4 (+ `conftest.py`)
- **Skipped tests:** 0
- **Execution time:** ~5 seconds (full suite; +~1 s one-time torch import if PyTorch is installed)
- **Core-library code coverage:** ~81% (`tqf_hex_signal.py` 79%, `tqf_lattice_graph.py` 90%)
- **GPU required:** No

### Key Strengths

- **Exactness-first:** the headline claims (C1, C4) are checked with bitwise/integer equality.
- **Tests the shipped code:** simulation helpers (orbit reduction, equivariance) are imported and exercised directly.
- **Adversarial correctness check:** the decoder is validated against brute force on deep-hole inputs.
- **Fast and portable:** a few seconds, CPU-only, Windows/Linux/macOS.
- **Claim-mapped:** every test traces to a specific claim (see Section 5).

### Testing Categories

1. **Primitive/unit tests (~53):** geometry, constellations, channels, statistics, colouring, packing-gain helpers, provenance.
2. **Property tests (~20):** rotation equivariance, group order, exact orbit reduction, energy normalization.
3. **Correctness-vs-reference tests (~6):** closed-form decode vs exhaustive ML; 3x3 window vs brute-force nearest point.

### Claim Coverage

- **C1 (exact decode):** decode == ML on a dense grid and a noisy stream; nearest-point vs brute force.
- **C2 (constant-time fast path):** fast-path coverage rises with SNR.
- **C3 (packing gain):** matched unit energy + correct Eb/N0 mapping (fairness preconditions); dB-gain interpolation, CI-bounded gain, and impulsive floor `~ p*(1-1/M)`.
- **C4 (symmetry-reduced exact metric):** orbit-reduced enumerator == full; exact 6x op-count.
- **C5 (conflict-free parallel recovery):** proper six-colouring; denoising relaxation sweep.
- **C6 (equivariant decode):** decoder commutes with the order-6 rotation; differential invariance to k*60 degrees.

**For questions or issues, please contact:** nate.o.schmidt@coldhammer.net

---

**`QED`**

**Last Updated:** June 23, 2026<br>
**Version:** 1.0.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
