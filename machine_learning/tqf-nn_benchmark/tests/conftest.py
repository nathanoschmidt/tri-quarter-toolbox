"""
conftest.py - Shared Test Configuration and Utilities for TQF-NN Tests

This module provides pytest fixtures, shared utilities, and environment setup
for all TQF-NN test modules. It centralizes common test infrastructure to reduce
code duplication and ensure consistent test behavior across the test suite.

Key Features:
- Path Configuration: Automatic Python path setup for importing src/ modules
- Environment Detection: TORCH_AVAILABLE, NUMPY_AVAILABLE, CUDA_AVAILABLE flags
- Pytest Fixtures: device, lightweight_models, all_models, small_tqf_model (cached)
- Assertion Utilities: Type validation (positive_integer, valid_probability, non_negative)
- Tensor Utilities: Shape assertion, parameter counting, set_seed for reproducibility
- Test Categorization: Custom markers (@pytest.mark.slow, @pytest.mark.cuda)
- CLI Options: --device (cuda/cpu), --quick (skip slow tests)

Pytest Fixtures Provided:
    - device: torch.device for test execution (cuda/cpu)
    - lightweight_models: FC-MLP and CNN-L5 (fast initialization, module-scoped)
    - all_models: All 4 models including TQF-ANN (~15-20s init, module-scoped)
    - small_tqf_model: R=10 TQF-ANN for fast testing (~2-3s init, module-scoped)
    - temp_model_dir / temp_results_dir / temp_data_dir: Temporary directories

Test Utilities Provided:
    - assert_positive_integer: Validate positive integer configuration values
    - assert_valid_probability: Ensure probability values in [0, 1]
    - assert_non_negative: Check non-negative numeric values
    - assert_tensor_shape: Validate tensor dimensions
    - count_parameters: Count trainable parameters in PyTorch models
    - set_seed: Set reproducible random seeds across numpy/torch/random

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Date: February 2026
"""

# MIT License
#
# Copyright (c) 2026 Nathan O. Schmidt, Cold Hammer Research & Development LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

try:
    import pytest
    PYTEST_AVAILABLE: bool = True
except ImportError:
    PYTEST_AVAILABLE: bool = False


# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

def setup_test_environment() -> Dict[str, Path]:
    """Configure Python path for test imports."""
    current_file: Path = Path(__file__).resolve()

    if current_file.parent.name == 'tests':
        tests_dir: Path = current_file.parent
        project_root: Path = tests_dir.parent
    else:
        tests_dir: Path = current_file.parent
        project_root: Path = tests_dir

    src_dir: Path = project_root / 'src'
    if not src_dir.exists() or not (src_dir / 'config.py').exists():
        src_dir = project_root

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    return {'project_root': project_root, 'src_dir': src_dir, 'tests_dir': tests_dir}


PATHS: Dict[str, Path] = setup_test_environment()


# ==============================================================================
# DEPENDENCY VALIDATION
# ==============================================================================

def check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_numpy_available() -> bool:
    """Check if NumPy is available."""
    try:
        import numpy
        return True
    except ImportError:
        return False


TORCH_AVAILABLE: bool = check_torch_available()
CUDA_AVAILABLE: bool = check_cuda_available()
NUMPY_AVAILABLE: bool = check_numpy_available()

# Note: Don't warn during module import - tests will handle missing dependencies
# if not TORCH_AVAILABLE:
#     warnings.warn("PyTorch not available - some tests will be skipped", RuntimeWarning)


# ==============================================================================
# PYTEST MARKERS (if pytest available)
# ==============================================================================

if PYTEST_AVAILABLE:
    def skip_if_no_torch():
        """Pytest marker to skip tests requiring PyTorch."""
        return pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

    def skip_if_no_cuda():
        """Pytest marker to skip tests requiring CUDA."""
        return pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

    def skip_if_no_numpy():
        """Pytest marker to skip tests requiring NumPy."""
        return pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")


# ==============================================================================
# PYTEST CONFIGURATION (if pytest available)
# ==============================================================================

if PYTEST_AVAILABLE:
    def pytest_addoption(parser: Any) -> None:
        """Add custom command-line options."""
        parser.addoption("--device", action="store", default="cuda",
                        choices=["cuda", "cpu"], help="Device for tests")
        parser.addoption("--quick", action="store_true", default=False,
                        help="Skip slow tests")

    def pytest_configure(config: Any) -> None:
        """Configure test session."""
        config.addinivalue_line("markers", "slow: marks tests as slow")
        config.addinivalue_line("markers", "cuda: marks tests requiring CUDA")
        config.addinivalue_line("markers", "integration: marks integration tests")
        config.addinivalue_line("markers", "performance: marks performance benchmarks")
        if config.getoption("--quick"):
            setattr(config.option, "markexpr", "not slow")

    @pytest.fixture(scope="session")
    def device(request: Any):
        """Get device for tests."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import torch
        device_name: str = request.config.getoption("--device")
        if device_name == "cuda" and not CUDA_AVAILABLE:
            # Use CPU silently instead of warning
            device_name = "cpu"
        return torch.device(device_name)

    @pytest.fixture(scope="session")
    def config_module():
        """Load config module for tests."""
        try:
            import config
            return config
        except ImportError as e:
            pytest.fail(f"Cannot import config module: {e}")

    @pytest.fixture(scope="session")
    def torch_module():
        """Provide torch module if available."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        import torch
        return torch

    @pytest.fixture(scope="session")
    def numpy_module():
        """Provide numpy module if available."""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        import numpy
        return numpy

    @pytest.fixture
    def temp_model_dir(tmp_path: Path) -> Path:
        """Provide temporary directory for model checkpoints."""
        model_dir: Path = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        return model_dir

    @pytest.fixture
    def temp_results_dir(tmp_path: Path) -> Path:
        """Provide temporary directory for test results."""
        results_dir: Path = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        return results_dir

    @pytest.fixture
    def temp_data_dir(tmp_path: Path) -> Path:
        """Provide temporary directory for test datasets."""
        data_dir: Path = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        return data_dir

    @pytest.fixture(scope="module")
    def lightweight_models():
        """
        Fast baseline models for quick testing (module-level cache).

        Excludes TQF-ANN and ResNet to minimize initialization time.
        Use for tests that need real models but want fast execution.
        """
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        from models_baseline import get_model
        import time

        print("\n[INFO] Initializing lightweight models (one-time setup)...")
        start: float = time.time()

        models: Dict = {}

        models['FC-MLP'] = get_model('FC-MLP')
        print(f"  FC-MLP: {models['FC-MLP'].count_parameters():,} params")

        models['CNN-L5'] = get_model('CNN-L5')
        print(f"  CNN-L5: {models['CNN-L5'].count_parameters():,} params")

        elapsed: float = time.time() - start
        print(f"[INFO] Lightweight models initialized in {elapsed:.2f}s\n")

        return models

    @pytest.fixture(scope="module")
    def all_models():
        """
        All models including TQF-ANN and ResNet (module-level cache).

        WARNING: Slow initialization (~15-20 seconds) due to TQF-ANN complexity.
        Use only for comprehensive integration tests.
        """
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        from models_baseline import get_model
        import config
        import time

        print("\n[INFO] Initializing all models (one-time setup, ~15-20 seconds)...")
        start: float = time.time()

        models: Dict = {}

        # Baseline models
        models['FC-MLP'] = get_model('FC-MLP')
        print(f"  FC-MLP: {models['FC-MLP'].count_parameters():,} params")

        models['CNN-L5'] = get_model('CNN-L5')
        print(f"  CNN-L5: {models['CNN-L5'].count_parameters():,} params")

        models['ResNet-18-Scaled'] = get_model('ResNet-18-Scaled')
        print(f"  ResNet-18-Scaled: {models['ResNet-18-Scaled'].count_parameters():,} params")

        # TQF-ANN (slowest - auto-tuning + dual metrics)
        print("  TQF-ANN: Initializing (auto-tuning hidden_dim)...")
        tqf_start: float = time.time()
        models['TQF-ANN'] = get_model('TQF-ANN', R=config.TQF_TRUNCATION_R_DEFAULT)
        tqf_elapsed: float = time.time() - tqf_start
        print(f"  TQF-ANN: {models['TQF-ANN'].count_parameters():,} params (init: {tqf_elapsed:.2f}s)")

        elapsed: float = time.time() - start
        print(f"[INFO] All models initialized in {elapsed:.2f}s\n")

        return models

    @pytest.fixture(scope="module")
    def small_tqf_model():
        """
        Small TQF-ANN model for fast testing (module-level cache).

        Uses R=10, hidden_dim=32 for minimal initialization time (~2-3 seconds).
        Use for TQF-ANN specific tests that don't need production-sized models.
        """
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        from models_tqf import TQFANN
        import time

        print("\n[INFO] Initializing small TQF-ANN model (one-time setup)...")
        start: float = time.time()

        model = TQFANN(R=10, hidden_dim=32)

        elapsed: float = time.time() - start
        print(f"[INFO] Small TQF-ANN initialized: R=10, hidden_dim=32, params={model.count_parameters():,} ({elapsed:.2f}s)\n")

        return model



# ==============================================================================
# TEST UTILITY FUNCTIONS
# ==============================================================================

def assert_valid_probability(value: float, name: str = "value") -> None:
    """Assert value is valid probability in [0, 1]."""
    assert isinstance(value, (int, float)), f"{name} must be numeric"
    assert 0 <= value <= 1, f"{name} must be in [0, 1], got {value}"


def assert_positive_integer(value: int, name: str = "value") -> None:
    """Assert value is positive integer."""
    assert isinstance(value, int), f"{name} must be integer"
    assert value > 0, f"{name} must be positive, got {value}"


def assert_non_negative(value: float, name: str = "value") -> None:
    """Assert value is non-negative."""
    assert isinstance(value, (int, float)), f"{name} must be numeric"
    assert value >= 0, f"{name} must be non-negative, got {value}"


def count_parameters(model) -> int:
    """Count trainable parameters in PyTorch model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    import torch.nn as nn
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_tensor_shape(tensor, expected_shape: Tuple, msg: Optional[str] = None) -> None:
    """Assert tensor has expected shape."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    actual_shape: Tuple = tuple(tensor.shape)
    assert actual_shape == expected_shape, msg or f"Expected {expected_shape}, got {actual_shape}"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    if TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)
        if CUDA_AVAILABLE:
            torch.cuda.manual_seed_all(seed)
    if NUMPY_AVAILABLE:
        import numpy as np
        np.random.seed(seed)
    import random
    random.seed(seed)


# ==============================================================================
# BASE TEST CLASS
# ==============================================================================

class TQFTestBase:
    """Base class for TQF test cases with common utilities."""

    @staticmethod
    def get_device(prefer_cuda: bool = True):
        """Get device for testing."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        import torch
        if prefer_cuda and CUDA_AVAILABLE:
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def assert_model_output_shape(model, input_shape: Tuple, expected_output_shape: Tuple) -> None:
        """Assert model produces expected output shape."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        import torch
        device = TQFTestBase.get_device(prefer_cuda=False)
        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        actual_shape: Tuple = tuple(output.shape)
        assert actual_shape == expected_output_shape, \
            f"Expected {expected_output_shape}, got {actual_shape}"
