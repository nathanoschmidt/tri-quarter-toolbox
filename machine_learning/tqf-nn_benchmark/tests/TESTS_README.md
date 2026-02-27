# TQF-NN Benchmark Tools: Automated Testing Framework: TESTS_README

**Each time code is deployed to production without test automation, a child gets a mullet**

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.1.0<br>
**Date:** February 26, 2026<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![pytest](https://img.shields.io/badge/pytest-compatible-green.svg)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen.svg)](https://coverage.readthedocs.io/)

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

This automated testing framework provides comprehensive coverage of all modules in the TQF-NN project, ensuring scientific rigor, reproducibility, and code quality.

Key Features:
- **724 total test cases** across 17 test files
- **~5-10 second** full test suite execution time
- Shared utility infrastructure (conftest.py)
- Full type hint coverage
- Self-documented tests (WHY, HOW, WHAT)
- Edge case and error handling coverage
- Mathematical property verification
- Performance benchmarks
- Integration tests
- `pytest` and `unittest` compatible

All tests are compatible with Python 3.8+ and follow PEP 8 style guidelines.

---

## 2. Testing Philosophy

The testing framework is built on five pillars:

**A. Completeness**
- Every public function has at least 3 tests: basic, edge cases, errors
- All classes have initialization, method, and property tests
- Integration tests verify end-to-end workflows

**B. Scientific Rigor**
- Mathematical properties are verified (symmetry, invariance, etc.)
- Theoretical guarantees are validated
- Numerical stability is checked
- Reproducibility is enforced

**C. Documentation**
- Every test has comprehensive docstring
- WHY: Scientific rationale
- HOW: Test methodology
- WHAT: Expected behavior
- Type hints on all variables

**D. Robustness**
- Edge cases: empty inputs, boundary values, extreme values
- Error handling: invalid types, out-of-range, exceptions
- Platform independence: Windows, Linux, macOS

**E. Maintainability**
- Clear naming conventions
- Modular test structure
- Shared utilities eliminate duplication
- Easy to extend
- Self-contained fixtures
- Single source of truth

---

## 3. File Structure

```
tests/
|-- conftest.py                             # Shared test utilities and fixtures
|-- pytest.ini                              # pytest configuration
|-- run_tests.py                            # Master test runner
|-- test_cli.py                             # Command-line interface
|-- test_config.py                          # Configuration validation
|-- test_datasets.py                        # Data loading
|-- test_dual_metrics.py                    # Dual metrics
|-- test_dual_metrics_hexagonal_lattice.py  # Hexagonal lattice foundation tests
|-- test_engine.py                          # Training engine, experiment orchestration
|-- test_evaluation.py                      # Performance metrics evaluation
|-- test_logging_utils.py                   # Logging utilities
|-- test_main.py                            # Good ole main
|-- test_models_tqf_lattice_integration.py  # TQF-ANN lattice integration tests
|-- test_output_formatters.py               # Output logging/formatting
|-- test_param_matcher.py                   # Model parameter matching/auto-tuning
|-- test_performance.py                     # Performance benchmarks
|-- test_symmetry_ops.py                    # Symmetry ops unit tests (Z6/D6/T24)
|-- test_tqf_ann.py                         # Main TQF-ANN test suite
|-- test_tqf_ann_integration.py             # TQF-ANN end-to-end integration tests
|-- test_verification_features.py           # Verification features
`-- TESTS_README.md                         # This file
```

### Core Infrastructure Files

**conftest.py**

Purpose: Shared test utilities and configuration

Contains:
- Automatic path configuration (eliminates duplication)
- Dependency validation (TORCH_AVAILABLE, CUDA_AVAILABLE, NUMPY_AVAILABLE)
- pytest fixtures (device, config_module, temporary directories)
- Shared assertions (assert_valid_probability, assert_positive_integer, etc.)
- Utility functions (count_parameters, set_seed, assert_tensor_shape)
- Base test class (TQFTestBase) for common patterns

Benefits:
- Eliminates ~100 lines of duplicated path setup across test files
- Provides single source of truth for common functionality
- Makes adding new tests faster and more consistent
- Reduces maintenance burden

**pytest.ini**

Purpose: Standardized pytest configuration

Contains:
- Test discovery patterns
- Custom markers (slow, cuda, integration, performance)
- Default command-line options
- Console output configuration

Benefits:
- Consistent test execution behavior
- Easy selective test execution
- Better organized test categories
- Improved CI/CD integration

---

## 4. Running Tests

### A. Quick Start (Recommended)

Run all tests:

```bash
python run_tests.py
```

Run specific file:

```bash
python run_tests.py --file test_config
python run_tests.py --file test_dual_metrics
```

Skip slow tests:

```bash
python run_tests.py --quick
```

Use specific device:

```bash
python run_tests.py --device cpu
python run_tests.py --device cuda
```

Generate coverage report:

```bash
python run_tests.py --coverage
```

### B. Individual Test Files

Basic execution:

```bash
python test_config.py
python test_dual_metrics.py
python test_output_formatters.py
```

Verbose mode:

```bash
python test_config.py --verbose
python test_datasets.py --verbose
```

### C. pytest (Recommended for Development)

Run all tests:

```bash
pytest
```

Run specific file:

```bash
pytest test_config.py
```

Run specific test:

```bash
pytest test_dual_metrics.py::TestDiscreteDualMetric::test_discrete_metric_direct_neighbors
```

Skip slow tests:

```bash
pytest -m "not slow"
pytest --quick
```

Run only specific marker:

```bash
pytest -m cuda --device cuda
pytest -m integration
```

With coverage:

```bash
pytest --cov=. --cov-report=html
pytest --cov=. --cov-report=term-missing
```

Parallel execution (requires pytest-xdist):

```bash
pytest -n auto
pytest -n 4
```

Verbose output:

```bash
pytest -v
pytest -vv
```

Stop on first failure:

```bash
pytest -x
```

Show print statements:

```bash
pytest -s
```

Run last failed tests:

```bash
pytest --lf
```

---

## 5. Test Coverage

### Overall Coverage

- **Total test cases:** 724 (varies based on skip conditions)
- **Test files:** 17
- **Full suite execution time:** ~5-10 seconds
- **Core module coverage:** 100%
- **Overall code coverage:** ~87%

### Test Suite Breakdown by File

| Test File | Test Cases | Status | Focus Area |
|-----------|------------|--------|------------|
| test_cli.py | 115 | All passing | CLI validation, orbit mixing flags, Z6 enhancement flags, range constant consistency |
| test_config.py | 80 | All passing | Configuration constants, range constants, defaults-within-ranges |
| test_datasets.py | 23 | All passing | Data loading and transformations |
| test_dual_metrics.py | 34 | All passing | Geometric dual metrics, k-hop neighbors |
| test_dual_metrics_hexagonal_lattice.py | 37 | All passing | Hexagonal lattice Eisenstein coordinates |
| test_engine.py | 55 | All passing | Training loop, optimizer setup, losses |
| test_evaluation.py | 55 | All passing | Performance metrics, evaluation, orbit mixing, Z6 enhancement modes, orbit consistency loss |
| test_logging_utils.py | 25 | All passing | Logging and output formatting |
| test_main.py | 10 | Varies | Main pipeline integration |
| test_models_tqf_lattice_integration.py | 27 | Slow/Skip | TQF-ANN hexagonal lattice integration |
| test_output_formatters.py | 74 | All passing | Result formatting and display |
| test_param_matcher.py | 27 | Varies | Parameter matching and auto-tuning |
| test_performance.py | 13 | All passing | Performance benchmarks |
| test_symmetry_ops.py | 53 | All passing | Symmetry ops unit tests (Z6/D6/T24) |
| test_tqf_ann.py | 47 | All passing | TQF-ANN architecture, k-hop, checkpointing, inner zone caching |
| test_tqf_ann_integration.py | 41 | Slow/Skip | TQF-ANN end-to-end integration tests |
| test_verification_features.py | 8 | All passing | Model verification utilities |

**Note:** A small number of tests are conditionally skipped or deselected based on environment and marker filters. 6 tests are unconditionally skipped due to a known Windows/CUDA 12.6 driver compatibility issue with ResNet Conv2d (3 tests) and CUDA-specific logging paths (3 tests). Tests marked `@pytest.mark.slow` are deselected when running with `-m "not slow"`.

### Module-Specific Coverage

| Module | Coverage | Test Cases | Notes |
|--------|----------|------------|-------|
| cli.py | 95% | 115 | All CLI arguments, validation, defaults, range constant consistency, Z6 enhancements |
| config.py | 95% | 81 | All constants, range constants, defaults-within-ranges, orbit mixing |
| dual_metrics.py | 92% | 53 | Metrics, k-hop precomputation, edge cases |
| models_tqf.py | 90% | 69+ | Architecture, k-hop, checkpointing, inner zone caching |
| models_baseline.py | 85% | 13+ | All baseline models (via param_matcher) |
| datasets.py | 92% | 23 | Loading, transforms, validation |
| engine.py | 80% | 48 | Training loops, evaluation, orchestration |
| evaluation.py | 90% | 55 | Metrics, statistical tests, timing, orbit mixing, Z6 enhancement modes, orbit consistency loss |
| output_formatters.py | 93% | 49 | All formatters, edge cases, constants |
| param_matcher.py | 90% | 27 | Parameter counting, model registry |
| symmetry_ops.py | 95% | 53 | Z6/D6/T24 group transformations |
| logging_utils.py | 88% | 25 | Separators, config logging, progress |

---

## 6. Interpreting Test Results

### A. Understanding pytest Output

When you run `pytest -v`, you'll see output like:

```
======================================================== test session starts ========================================================
platform win32 -- Python 3.13.9, pytest-9.0.2, pluggy-1.6.0
collected 724 items

test_cli.py::TestGetAllModelNames::test_returns_list PASSED                                                                    [  0%]
test_cli.py::TestGetAllModelNames::test_contains_tqf_ann PASSED                                                                [  0%]
...
test_verification_features.py::TestParameterCounting::test_count_parameters_tqf PASSED                                         [100%]

================================================== 718 passed, N skipped in 5-10s ==================================================
```

**Key Elements:**
- **Platform info:** Shows OS, Python version, pytest version
- **collected N items:** Total number of tests discovered
- **Test status:** PASSED, FAILED, SKIPPED, XFAIL, ERROR
- **Progress percentage:** Shows completion progress
- **Summary:** Final count of passed/failed/skipped tests and execution time

### B. Test Status Meanings

| Status | Symbol | Meaning | Action Required |
|--------|--------|---------|-----------------|
| PASSED | `.` or `PASSED` | Test executed successfully | None |
| FAILED | `F` or `FAILED` | Test assertion failed or unexpected error | Investigate and fix |
| SKIPPED | `s` or `SKIPPED` | Test intentionally skipped (e.g., missing GPU) | None (expected) |
| XFAIL | `x` | Expected failure (known issue) | Track for future fix |
| XPASS | `X` | Unexpected pass (expected to fail) | Update test expectations |
| ERROR | `E` | Error during test setup/teardown | Fix test infrastructure |

### C. Common Skip Reasons

```python
# Skipped: "ResNet Conv2d triggers CUDA access violation on Windows/CUDA 12.6"
# -> Known PyTorch/CUDA driver compatibility issue on Windows
# -> Tests manually verified: ResNet has 656,848 params (1.05% deviation from target)
# -> Not a code bug; safe to skip (3 tests: test_main.py, test_param_matcher.py x2)

# Skipped: "CUDA not available on this system"
# -> Tests that exercise CUDA-specific code paths (e.g., GPU logging output)
# -> Normal behavior on machines without GPU support; skipped at runtime

# Deselected: Tests with @pytest.mark.slow when running with -m "not slow"
# -> TQF-ANN integration tests and production-sized model tests are excluded
# -> Run full suite without -m "not slow" to include these tests
# -> Affected files: test_main.py, test_param_matcher.py,
#    test_verification_features.py, test_models_tqf_lattice_integration.py,
#    test_tqf_ann_integration.py
```

**Current skip/deselect status (as of Feb 2026):**

| Command | Passed | Skipped | Deselected | Total |
|---------|--------|---------|------------|-------|
| `pytest tests/ -v` (full suite) | 718 | 6 | 0 | 724 |
| `pytest tests/ -v -m "not slow"` | 640 | 6 | 78 | 724 |

**When to worry about skips:**
- [OK] **Normal:** Skips for known environment issues (ResNet CUDA, GPU logging)
- [WARN] **Investigate:** Skips for core functionality or unexpected missing dependencies
- [X] **Fix immediately:** Skips due to import errors or configuration issues

### D. Reading Failure Messages

Example failure output:

```python
FAILED test_config.py::TestTrainingHyperparameters::test_learning_rate_valid
________________________ TestTrainingHyperparameters.test_learning_rate_valid ________________________

self = <test_config.TestTrainingHyperparameters testMethod=test_learning_rate_valid>

    def test_learning_rate_valid(self) -> None:
        """Test that learning rate is within valid range."""
>       assert 0.0 < config.LEARNING_RATE_DEFAULT < 1.0
E       AssertionError: assert (0.0 < 0.001 < 1.0)

test_config.py:85: AssertionError
```

**Failure anatomy:**
1. **Test path:** `test_config.py::TestTrainingHyperparameters::test_learning_rate_valid`
2. **Test function:** Shows the failing test code
3. **Failure location:** `test_config.py:85` (line number)
4. **Assertion error:** Shows what failed and actual values
5. **Context:** Shows surrounding code and variables

### E. Using Verbose Modes

**Level 1: Default (`pytest`)**
```
test_config.py .........                                                  [ 10%]
```
- Shows progress dots
- Minimal output

**Level 2: Verbose (`pytest -v`)**
```
test_config.py::TestReproducibilityConstants::test_num_seeds_default_is_valid PASSED  [  1%]
test_config.py::TestReproducibilityConstants::test_seed_default_is_integer PASSED      [  2%]
```
- Shows each test name
- Displays status and progress

**Level 3: Very Verbose (`pytest -vv`)**
```
test_config.py::TestReproducibilityConstants::test_num_seeds_default_is_valid PASSED  [  1%]
  WHY: Validates NUM_SEEDS_DEFAULT is within reasonable range...
  WHAT: NUM_SEEDS_DEFAULT = 5
```
- Shows test names, status, progress
- Includes docstrings and additional context
- Displays assertion details

### F. Coverage Reports

**Terminal coverage:**
```bash
pytest --cov=. --cov-report=term-missing
```

Output:
```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
config.py                 150      8    95%   45-52
dual_metrics.py           200     20    90%   189-195, 220-225
models_tqf.py             350     42    88%   [lines]
-----------------------------------------------------
TOTAL                    2500    213    87%
```

**Interpreting coverage:**
- **Stmts:** Total executable statements
- **Miss:** Statements not executed by tests
- **Cover:** Percentage covered
- **Missing:** Line numbers not covered

**Coverage goals:**
- [OK] **>90%:** Excellent coverage
- [WARN] **80-90%:** Good coverage, room for improvement
- [X] **<80%:** Needs more tests

### G. Performance Benchmarks

Some tests measure performance:

```python
test_performance.py::TestModelPerformance::test_inference_speed_placeholder PASSED
# Inference time: 2.34ms per sample
# Throughput: 427 samples/second
```

**What to monitor:**
- Inference time should be <10ms per sample for real-time applications
- Throughput should scale linearly with batch size
- Memory usage should be stable across runs
- Training time should be reasonable (varies by model)

### H. Statistical Test Results

Some tests perform statistical comparisons:

```python
test_evaluation.py::TestStatisticalSignificance::test_different_distributions_may_be_significant PASSED
# p-value: 0.023 (< 0.05, significant difference detected)
# Effect size (Cohen's d): 0.42 (medium effect)
```

**Interpreting statistics:**
- **p-value < 0.05:** Significant difference (reject null hypothesis)
- **Effect size:** Magnitude of difference (small: 0.2, medium: 0.5, large: 0.8)
- **Power:** Probability of detecting true effect (typically >0.8)

### I. Common Patterns to Watch For

**Pattern 1: Consistent failures in one module**
```
test_dual_metrics.py::... FAILED
test_dual_metrics.py::... FAILED
test_dual_metrics.py::... FAILED
```
-> Likely a bug in the module or broken dependency

**Pattern 2: Random failures (flaky tests)**
```
Run 1: PASSED
Run 2: FAILED
Run 3: PASSED
```
-> Non-deterministic behavior; check for:
  - Missing random seed
  - Race conditions
  - Floating-point precision issues
  - Hardware-dependent behavior

**Pattern 3: All tests in a class fail**
```
test_tqf_ann.py::TestTQFANNInitialization::... FAILED
test_tqf_ann.py::TestTQFANNInitialization::... FAILED
```
-> Likely setup/fixture issue or broken initialization

**Pattern 4: Only GPU tests fail**
```
test_engine.py::test_cuda_training FAILED
test_engine.py::test_cpu_training PASSED
```
-> GPU-specific issue; check CUDA installation and memory

### J. Debugging Failed Tests

**Step 1: Run single failing test with verbose output**
```bash
pytest test_config.py::TestTrainingHyperparameters::test_learning_rate_valid -vv
```

**Step 2: Add print statements (show with -s)**
```bash
pytest test_config.py::TestTrainingHyperparameters::test_learning_rate_valid -vv -s
```

**Step 3: Use debugger (drops into pdb on failure)**
```bash
pytest --pdb test_config.py::TestTrainingHyperparameters::test_learning_rate_valid
```

**Step 4: Check recent changes**
```bash
git diff HEAD~1 test_config.py
git log --oneline -5
```

### K. Best Practices for Test Interpretation

1. **Run full suite regularly:** `pytest` (catches regressions)
2. **Check coverage periodically:** `pytest --cov=.` (identify gaps)
3. **Investigate all failures immediately** (prevents accumulation)
4. **Don't ignore skipped tests** (verify skip reasons are valid)
5. **Monitor performance trends** (detect degradation early)
6. **Review statistical test results** (ensure meaningful differences)
7. **Keep tests updated** (match code changes)
8. **Document test changes** (maintain test history)

### L. Expected Test Results (Health Check)

[OK] **Healthy test suite (full run):**
```
718 passed, 6 skipped in 5-10s
```

[OK] **Healthy test suite (fast run, `-m "not slow"`):**
```
640 passed, 6 skipped, 78 deselected in ~15s
```
- All core tests passing
- 6 skipped: 3 ResNet/CUDA driver issues + 3 CUDA-path tests (all documented)
- 78 deselected when using `-m "not slow"`: TQF-ANN integration and production model tests
- No warnings or errors

[WARN] **Needs attention:**
```
100 passed, 8 failed, 3 skipped in 2.5m
```
- Some failures detected
- Investigate and fix failing tests
- May indicate regression or environment issue

[X] **Critical issues:**
```
50 passed, 50 failed, 10 skipped, 5 errors in 1.5m
```
- Many failures and errors
- Likely broken dependency or major regression
- Fix immediately before proceeding

---

## 7. Best Practices Implemented

- All test functions have comprehensive docstrings
- Type hints on all variables and parameters
- PEP 8 compliant code style
- Self-documenting test names
- Clear assertion messages
- Proper error handling

### Test Organization

- Logical grouping of related tests
- Class-based organization for related functionality
- Fixture-based setup/teardown
- Shared utilities in conftest.py
- Consistent naming conventions

### Scientific Rigor

- Reproducibility enforced (fixed seeds)
- Mathematical properties validated
- Numerical stability checked
- Edge cases thoroughly tested
- Error conditions verified

### Performance

- Fast tests run first
- Slow tests marked and skippable
- Parallel execution supported
- Minimal redundant computation
- Efficient fixture usage

---

## 8. Adding New Tests

### Basic Template

```python
import pytest
from conftest import TQFTestBase, TORCH_AVAILABLE

class TestNewFeature(TQFTestBase):
    """Test suite for new feature.

    WHY: Validates that new feature behaves correctly.
    HOW: Tests basic functionality, edge cases, error handling.
    WHAT: Expects correct output for valid inputs, errors for invalid.
    """

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires PyTorch")
    def test_basic_functionality(self, device):
        """Test basic feature behavior.

        WHY: Ensures feature works for standard inputs.
        HOW: Create valid input, call feature, verify output.
        WHAT: Output should match expected behavior.
        """
        # Arrange (Setup)
        input_data = self.create_test_input()

        # Act (Execute)
        result = new_feature(input_data)

        # Assert (Verify)
        assert result is not None
        self.assert_tensor_shape(result, expected_shape)

    def test_edge_cases(self):
        """Test feature with edge cases."""
        # Test empty input, boundary values, etc.
        pass

    def test_error_handling(self):
        """Test feature error handling."""
        with pytest.raises(ValueError):
            new_feature(invalid_input)
```

### Guidelines

1. Use descriptive test names: `test_feature_does_what_under_condition`
2. Include comprehensive docstrings (WHY, HOW, WHAT)
3. Add type hints to all variables
4. Use fixtures from conftest.py
5. Mark slow tests with `@pytest.mark.slow`
6. Mark CUDA tests with `@pytest.mark.cuda`
7. Test at least: basic functionality, edge cases, error handling
8. Use appropriate assertions from conftest.py
9. Keep tests independent and isolated
10. Document any assumptions or limitations

---

## 9. Troubleshooting

### A. Import Errors

**Problem:** "ModuleNotFoundError: No module named 'config'"

**Solution:**
- conftest.py handles path setup automatically
- Ensure conftest.py is in the same directory as test files
- Verify path setup: `python -c "from conftest import PATHS; print(PATHS)"`
- Check project structure matches expected layout

### B. CUDA Errors

**Problem:** "CUDA out of memory" or "CUDA not available"

**Solution:**
- Use CPU device: `python run_tests.py --device cpu`
- Or with pytest: `pytest --device cpu`
- Reduce batch sizes in tests if needed
- Tests automatically skip if CUDA unavailable

### C. Missing Dependencies

**Problem:** "ImportError: cannot import name 'X'"

**Solution:**
- Install core dependencies: `pip install -r requirements.txt`
- Install PyTorch: `pip install torch torchvision`
- Install test dependencies: `pip install pytest pytest-cov`
- For parallel execution: `pip install pytest-xdist`
- Verify Python version: `python --version` (need 3.8+)

### D. pytest Not Found

**Problem:** "pytest: command not found"

**Solution:**
- Install pytest: `pip install pytest`
- Alternatively use unittest: `python test_config.py`
- Or use master runner: `python run_tests.py`

### E. Test Failures

**Problem:** Unexpected test failures or random failures

**Solution:**
- Check SEED_DEFAULT in config.py (should be 42)
- Verify deterministic mode is enabled
- Run single test to isolate: `pytest test_config.py::TestName::test_method`
- Check for dependency version conflicts
- Verify PyTorch version compatibility

### F. Coverage Issues

**Problem:** "No module named 'coverage'"

**Solution:**
- Install coverage tools: `pip install coverage pytest-cov`
- Use run_tests.py: `python run_tests.py --coverage`
- Or with pytest: `pytest --cov=. --cov-report=html`
- View report: open htmlcov/index.html

### G. Path Setup Issues

**Problem:** Tests can't find modules despite conftest.py

**Solution:**
- Verify conftest.py is in correct directory
- Check PYTHONPATH: `echo $PYTHONPATH` (Unix) or `echo %PYTHONPATH%` (Windows)
- Manually verify: `python -c "import sys; print(sys.path)"`
- Ensure source files are in expected location

### H. Parallel Execution Issues

**Problem:** Tests fail when run with pytest -n auto

**Solution:**
- Some tests may have shared state issues
- Run without parallelization: `pytest` (without -n flag)
- Check for file conflicts or temporary file issues
- Ensure tests are properly isolated

---

## 10. Reference: pytest Commands

### Basic Execution

```bash
pytest                                                                    # Run all tests
pytest test_config.py                                                     # Run specific file
pytest test_cli.py::TestParseArgsTQFSpecific                              # Run specific class
pytest -k "test_seed"                                                     # Run tests matching pattern
pytest test_cli.py::TestParseArgsTQFSpecific::test_tqf_symmetry_level_Z   # Specific test
```

### Selective Execution

```bash
pytest -m "not slow"            # Skip slow tests
pytest -m cuda                  # Run only CUDA tests
pytest -m integration           # Run only integration tests
pytest --quick                  # Skip slow tests (custom flag)
```

### Output Control

```bash
pytest -v                       # Verbose output
pytest -vv                      # Very verbose output
pytest -q                       # Quiet output
pytest -s                       # Show print statements
pytest --tb=short               # Short traceback format
pytest --tb=line                # One line per failure
```

### Execution Control

```bash
pytest -x                       # Stop on first failure
pytest --maxfail=3              # Stop after 3 failures
pytest --lf                     # Run last failed tests
pytest --ff                     # Run failed first, then rest
pytest -n auto                  # Parallel execution (all cores)
pytest -n 4                     # Parallel execution (4 workers)
```

### Coverage

```bash
pytest --cov=.                           # Coverage for all files
pytest --cov=config                      # Coverage for specific module
pytest --cov=. --cov-report=html         # HTML report
pytest --cov=. --cov-report=term-missing # Terminal with missing lines
pytest --cov=. --cov-report=xml          # XML for CI tools
```

### Device Selection

```bash
pytest --device cuda            # Use CUDA if available
pytest --device cpu             # Force CPU usage
```

### Custom Markers (defined in pytest.ini)

```python
@pytest.mark.slow               # Mark test as slow
@pytest.mark.cuda               # Requires CUDA GPU
@pytest.mark.integration        # Integration test
@pytest.mark.performance        # Performance benchmark
```

### Fixtures Available

```python
device                          # PyTorch device (cuda/cpu)
config_module                   # Config module
torch_module                    # PyTorch module
numpy_module                    # NumPy module
temp_model_dir                  # Temporary model directory
temp_results_dir                # Temporary results directory
temp_data_dir                   # Temporary data directory
```

---

## 11. Recap Summary

This automated testing framework provides robust validation of the TQF-NN project through **724 comprehensive test cases** organized across **17 test modules**. The framework emphasizes scientific rigor, maintainability, and ease of use through shared utilities (conftest.py), standardized configuration (pytest.ini), and clear documentation.

### Key Statistics

- **Total test cases:** 724
- **Skipped tests:** 6 (3 ResNet/CUDA driver + 3 CUDA-path; all documented)
- **Deselected with `-m "not slow"`:** 78 (TQF-ANN integration + production model tests)
- **Test modules:** 17
- **Execution time:** ~5-10 seconds (full suite); ~15 seconds (fast `-m "not slow"` suite)
- **Code coverage:** ~87% overall
- **Lines of test code:** ~10,000+
- **Test-to-code ratio:** ~1.2:1

### Key Strengths

- **High coverage (87%+ overall)** across all modules
- **Shared infrastructure** eliminates code duplication (~100 lines saved per file)
- **Both pytest and unittest compatible** for maximum flexibility
- **Human-readable and easy to extend** with clear patterns and examples
- **Well-documented** with comprehensive docstrings (WHY, HOW, WHAT)
- **CI/CD ready** with parallel execution support and coverage reports
- **Scientific best practices** including reproducibility, mathematical validation
- **Fast execution** (<10 seconds) enables rapid development iteration

### Testing Categories

1. **Unit tests (~624 tests):** Individual functions and components
2. **Integration tests (~75 tests):** Module interactions and workflows
3. **End-to-end tests (~25 tests):** Complete pipeline validation (marked slow, run selectively)

### Coverage Breakdown

- **Configuration & CLI:** 95% (189 tests)
- **Core algorithms:** 92% (143 tests)
- **Models & architecture:** 88% (142 tests)
- **Training engine:** 80% (48 tests)
- **Utilities & formatting:** 93% (74 tests)
- **Evaluation & metrics:** 90% (30 tests)

The framework ensures that the TQF-NN implementation maintains high quality, correctness, and reproducibility across all components. All tests follow PEP 8 style guidelines, include comprehensive type hints, and provide clear documentation for maintainability.

**For questions or issues, please contact:** nate.o.schmidt@coldhammer.net

---

**`QED`**

**Last Updated:** February 26, 2026<br>
**Version:** 1.1.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
