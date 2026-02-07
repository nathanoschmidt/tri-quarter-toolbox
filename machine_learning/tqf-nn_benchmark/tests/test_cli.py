"""
test_cli.py - Comprehensive Test Suite for CLI Argument Parsing and Validation

This module provides extensive testing of the command-line interface in cli.py,
ensuring robust argument parsing, validation, and error handling for all
configuration parameters across all model types and experimental setups.

Key Test Coverage:
- Model Selection: get_all_model_names(), single/multiple model selection, 'all' keyword
- Default Values: All CLI arguments have correct defaults matching config.py constants
- Argument Validation: Range checking for epochs, batch size, learning rate, weight decay, num_seeds
- Dataset Configuration: Train/val/test size validation, divisibility by 10 (class balance)
- Device Selection: CUDA/CPU validation and default behavior
- TQF-Specific Parameters:
  - Lattice Configuration: --tqf-R (truncation radius) range validation (2-100)
  - Architecture: --tqf-hidden-dim range validation
  - Symmetry Levels: Z6, D6, T24, none (group theory validation)
  - Fibonacci Modes: none, linear, fibonacci (golden ratio scaling)
  - Binning Schemes: --tqf-use-phi-binning (phi vs dyadic)
  - Orbit Mixing: --tqf-use-orbit-mixing flag, --tqf-adaptive-mixing-temp range validation
- Loss Function Weights (Opt-In Features):
  - Equivariance: --tqf-z6-equivariance-weight, --tqf-d6-equivariance-weight
  - Invariance: --tqf-t24-orbit-invariance-weight
  - Inversion: --tqf-inversion-loss-weight
  - Fractal: --tqf-box-counting-weight, --tqf-box-counting-scales
  - Geometry: --tqf-geometry-reg-weight
- Advanced Features:
  - Fractal Iterations: --tqf-fractal-iterations (opt-in)
  - Geometry Verification: --tqf-verify-geometry flag
- Complex Argument Combinations: Multiple TQF parameters together
- Boundary Value Testing: Min/max values for all numeric parameters
- Error Handling: Invalid values, out-of-range values, incompatible combinations
- Logging Setup: setup_logging() configuration validation

Test Organization:
- TestGetAllModelNames: Model name listing and validation
- TestParseArgsDefaults: Default value verification
- TestParseArgsModelSelection: Model selection argument parsing
- TestParseArgsValidation: General argument validation
- TestParseArgsTQFSpecific: TQF-specific parameter validation
- TestSetupLogging: Logging configuration
- TestDeviceSelection: Hardware device selection
- TestPhiBinningArgument: Golden ratio binning tests
- TestComplexArgumentCombinations: Multi-parameter scenarios
- TestInversionLossCLIFlags: Inversion loss weight validation
- TestCombinedLossFlags: Multiple loss function weight scenarios

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

import sys
import pytest
from typing import List
from unittest.mock import patch
from conftest import assert_positive_integer

# Import CLI functions
from cli import (
    get_all_model_names,
    parse_args,
    setup_logging,
    NUM_SEEDS_MIN, NUM_SEEDS_MAX,
    NUM_EPOCHS_MIN, NUM_EPOCHS_MAX,
    BATCH_SIZE_MIN, BATCH_SIZE_MAX,
    LEARNING_RATE_MIN, LEARNING_RATE_MAX,
    WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX,
    TQF_R_MIN, TQF_R_MAX,
    TQF_HIDDEN_DIM_MIN, TQF_HIDDEN_DIM_MAX,
    TQF_BOX_COUNTING_WEIGHT_MIN, TQF_BOX_COUNTING_WEIGHT_MAX,
    TQF_BOX_COUNTING_SCALES_MIN, TQF_BOX_COUNTING_SCALES_MAX,
    TQF_ADAPTIVE_MIXING_TEMP_MIN, TQF_ADAPTIVE_MIXING_TEMP_MAX
)
import config


class TestGetAllModelNames:
    """Test suite for get_all_model_names() function."""

    def test_returns_list(self) -> None:
        """
        WHY: Function must return a list type for iteration
        HOW: Call function and check type
        WHAT: Expect list instance
        """
        result: List[str] = get_all_model_names()
        assert isinstance(result, list)

    def test_contains_tqf_ann(self) -> None:
        """
        WHY: TQF-ANN is the primary model being tested
        HOW: Check if 'TQF-ANN' in returned list
        WHAT: Expect TQF-ANN present
        """
        result: List[str] = get_all_model_names()
        assert 'TQF-ANN' in result

    def test_contains_baseline_models(self) -> None:
        """
        WHY: Must include all comparison baseline models
        HOW: Check for FC-MLP, CNN-L5, ResNet-18-Scaled
        WHAT: Expect all baselines present
        """
        result: List[str] = get_all_model_names()
        assert 'FC-MLP' in result
        assert 'CNN-L5' in result
        assert 'ResNet-18-Scaled' in result

    def test_no_duplicates(self) -> None:
        """
        WHY: Duplicate model names would cause errors
        HOW: Compare list length to set length
        WHAT: Expect same length (no duplicates)
        """
        result: List[str] = get_all_model_names()
        assert len(result) == len(set(result))

    def test_all_strings(self) -> None:
        """
        WHY: Model names must be strings for CLI
        HOW: Check type of each element
        WHAT: Expect all elements are str
        """
        result: List[str] = get_all_model_names()
        assert all(isinstance(name, str) for name in result)

    def test_returns_four_models(self) -> None:
        """
        WHY: Current implementation has exactly 4 models
        HOW: Check list length
        WHAT: Expect length of 4
        """
        result: List[str] = get_all_model_names()
        assert len(result) == 4


class TestParseArgsDefaults:
    """Test default argument parsing without command-line args."""

    def test_default_models_all(self) -> None:
        """
        WHY: Default behavior should train all models
        HOW: Parse with no --models argument
        WHAT: Expect all 4 models in list
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert len(args.models) == 4
            assert 'TQF-ANN' in args.models

    def test_default_num_epochs(self) -> None:
        """
        WHY: Must have sensible default training length
        HOW: Parse args and check num_epochs
        WHAT: Expect config.MAX_EPOCHS_DEFAULT
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.num_epochs == config.MAX_EPOCHS_DEFAULT

    def test_default_batch_size(self) -> None:
        """
        WHY: Must have sensible default batch size
        HOW: Parse args and check batch_size
        WHAT: Expect config.BATCH_SIZE_DEFAULT
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.batch_size == config.BATCH_SIZE_DEFAULT

    def test_default_learning_rate(self) -> None:
        """
        WHY: Must have sensible default learning rate
        HOW: Parse args and check learning_rate
        WHAT: Expect config.LEARNING_RATE_DEFAULT
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.learning_rate == config.LEARNING_RATE_DEFAULT

    def test_default_num_seeds(self) -> None:
        """
        WHY: Default should provide statistical significance
        HOW: Parse args and check num_seeds
        WHAT: Expect config.NUM_SEEDS_DEFAULT
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.num_seeds == config.NUM_SEEDS_DEFAULT

    def test_default_tqf_symmetry_level(self) -> None:
        """
        WHY: TQF models need default symmetry setting
        HOW: Parse args and check tqf_symmetry_level
        WHAT: Expect config.TQF_SYMMETRY_LEVEL_DEFAULT (D6)
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_symmetry_level == config.TQF_SYMMETRY_LEVEL_DEFAULT

    def test_default_device_cuda(self) -> None:
        """
        WHY: Default should use GPU if available
        HOW: Parse args and check device
        WHAT: Expect 'cuda'
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.device == 'cuda'


class TestParseArgsModelSelection:
    """Test model selection argument parsing."""

    def test_single_model_selection(self) -> None:
        """
        WHY: User should be able to train single model
        HOW: Parse with --models TQF-ANN
        WHAT: Expect only TQF-ANN in list
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.models == ['TQF-ANN']

    def test_multiple_model_selection(self) -> None:
        """
        WHY: User should be able to select subset of models
        HOW: Parse with --models TQF-ANN FC-MLP
        WHAT: Expect both in list, preserve order
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN', 'FC-MLP']):
            args = parse_args()
            assert args.models == ['TQF-ANN', 'FC-MLP']

    def test_all_keyword(self) -> None:
        """
        WHY: 'all' keyword should select all models
        HOW: Parse with --models all
        WHAT: Expect all 4 models
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'all']):
            args = parse_args()
            assert len(args.models) == 4

    def test_model_order_preserved(self) -> None:
        """
        WHY: User-specified order matters for experiment sequence
        HOW: Parse with specific order
        WHAT: Expect same order in args.models
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'CNN-L5', 'TQF-ANN', 'FC-MLP']):
            args = parse_args()
            assert args.models == ['CNN-L5', 'TQF-ANN', 'FC-MLP']


class TestParseArgsValidation:
    """Test argument validation logic."""

    def test_num_seeds_within_range(self) -> None:
        """
        WHY: num_seeds must be within valid range
        HOW: Parse with valid value
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--num-seeds', '3']):
            args = parse_args()
            assert args.num_seeds == 3
            assert NUM_SEEDS_MIN <= args.num_seeds <= NUM_SEEDS_MAX

    def test_num_seeds_below_min_fails(self) -> None:
        """
        WHY: num_seeds below minimum should be rejected
        HOW: Parse with value < NUM_SEEDS_MIN
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--num-seeds', '0']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_num_seeds_above_max_fails(self) -> None:
        """
        WHY: num_seeds above maximum should be rejected
        HOW: Parse with value > NUM_SEEDS_MAX
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--num-seeds', '100']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_batch_size_within_range(self) -> None:
        """
        WHY: batch_size must be within valid range
        HOW: Parse with valid value
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--batch-size', '64']):
            args = parse_args()
            assert args.batch_size == 64
            assert BATCH_SIZE_MIN <= args.batch_size <= BATCH_SIZE_MAX

    def test_batch_size_zero_fails(self) -> None:
        """
        WHY: Batch size of 0 is invalid
        HOW: Parse with --batch-size 0
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--batch-size', '0']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_learning_rate_within_range(self) -> None:
        """
        WHY: Learning rate must be positive and reasonable
        HOW: Parse with valid value
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--learning-rate', '0.001']):
            args = parse_args()
            assert args.learning_rate == 0.001
            assert LEARNING_RATE_MIN < args.learning_rate <= LEARNING_RATE_MAX

    def test_learning_rate_zero_fails(self) -> None:
        """
        WHY: Zero learning rate means no training
        HOW: Parse with --learning-rate 0.0
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--learning-rate', '0.0']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_weight_decay_default(self) -> None:
        """
        WHY: Weight decay must have sensible default for L2 regularization
        HOW: Parse without --weight-decay argument
        WHAT: Expect config.WEIGHT_DECAY_DEFAULT (0.0001)
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.weight_decay == config.WEIGHT_DECAY_DEFAULT

    def test_weight_decay_custom_value(self) -> None:
        """
        WHY: User can specify custom weight decay value
        HOW: Parse with --weight-decay 0.0005
        WHAT: Expect 0.0005 in args
        """
        with patch('sys.argv', ['test_cli.py', '--weight-decay', '0.0005']):
            args = parse_args()
            assert args.weight_decay == 0.0005
            assert WEIGHT_DECAY_MIN <= args.weight_decay <= WEIGHT_DECAY_MAX

    def test_weight_decay_zero_valid(self) -> None:
        """
        WHY: Zero weight decay disables L2 regularization (valid use case)
        HOW: Parse with --weight-decay 0.0
        WHAT: Expect 0.0 in args (no error)
        """
        with patch('sys.argv', ['test_cli.py', '--weight-decay', '0.0']):
            args = parse_args()
            assert args.weight_decay == 0.0
            assert WEIGHT_DECAY_MIN <= args.weight_decay <= WEIGHT_DECAY_MAX

    def test_weight_decay_negative_fails(self) -> None:
        """
        WHY: Negative weight decay is mathematically invalid
        HOW: Parse with --weight-decay -0.001
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--weight-decay', '-0.001']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_weight_decay_above_max_fails(self) -> None:
        """
        WHY: Weight decay above 1.0 is unreasonably large
        HOW: Parse with --weight-decay 1.5
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--weight-decay', '1.5']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_weight_decay_boundary_min(self) -> None:
        """
        WHY: Boundary value at minimum (0.0) should be valid
        HOW: Parse with --weight-decay 0.0
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--weight-decay', str(WEIGHT_DECAY_MIN)]):
            args = parse_args()
            assert args.weight_decay == WEIGHT_DECAY_MIN

    def test_weight_decay_boundary_max(self) -> None:
        """
        WHY: Boundary value at maximum (1.0) should be valid
        HOW: Parse with --weight-decay 1.0
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--weight-decay', str(WEIGHT_DECAY_MAX)]):
            args = parse_args()
            assert args.weight_decay == WEIGHT_DECAY_MAX

    def test_num_train_divisible_by_10(self) -> None:
        """
        WHY: MNIST has 10 classes, need balanced sampling
        HOW: Parse with value divisible by 10
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--num-train', '1000']):
            args = parse_args()
            assert args.num_train % 10 == 0

    def test_num_train_not_divisible_by_10_fails(self) -> None:
        """
        WHY: Non-divisible by 10 breaks class balance
        HOW: Parse with value not divisible by 10
        WHAT: Expect SystemExit with error message
        """
        with patch('sys.argv', ['test_cli.py', '--num-train', '1005']):
            with pytest.raises(SystemExit):
                parse_args()


class TestParseArgsTQFSpecific:
    """Test TQF-specific argument parsing."""

    def test_tqf_symmetry_level_Z6(self) -> None:
        """
        WHY: Z6 is valid symmetry level
        HOW: Parse with --tqf-symmetry-level Z6
        WHAT: Expect Z6 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-symmetry-level', 'Z6']):
            args = parse_args()
            assert args.tqf_symmetry_level == 'Z6'

    def test_tqf_symmetry_level_D6(self) -> None:
        """
        WHY: D6 is valid and default symmetry level
        HOW: Parse with --tqf-symmetry-level D6
        WHAT: Expect D6 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-symmetry-level', 'D6']):
            args = parse_args()
            assert args.tqf_symmetry_level == 'D6'

    def test_tqf_symmetry_level_T24(self) -> None:
        """
        WHY: T24 is valid maximum symmetry level
        HOW: Parse with --tqf-symmetry-level T24
        WHAT: Expect T24 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-symmetry-level', 'T24']):
            args = parse_args()
            assert args.tqf_symmetry_level == 'T24'

    def test_tqf_symmetry_level_none(self) -> None:
        """
        WHY: 'none' disables symmetry for ablation studies
        HOW: Parse with --tqf-symmetry-level none
        WHAT: Expect 'none' in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-symmetry-level', 'none']):
            args = parse_args()
            assert args.tqf_symmetry_level == 'none'

    def test_tqf_symmetry_level_invalid_fails(self) -> None:
        """
        WHY: Invalid symmetry level should be rejected
        HOW: Parse with invalid value
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-symmetry-level', 'invalid']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_use_orbit_mixing_default_false(self) -> None:
        """
        WHY: Orbit mixing should be opt-in
        HOW: Parse without --tqf-use-orbit-mixing
        WHAT: Expect False
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_use_orbit_mixing == False

    def test_tqf_use_orbit_mixing_enabled(self) -> None:
        """
        WHY: User can enable orbit mixing evaluation
        HOW: Parse with --tqf-use-orbit-mixing
        WHAT: Expect True
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-use-orbit-mixing']):
            args = parse_args()
            assert args.tqf_use_orbit_mixing == True

    def test_tqf_R_within_range(self) -> None:
        """
        WHY: Truncation radius must be valid
        HOW: Parse with --tqf-R 20 --models TQF-ANN
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN', '--tqf-R', '20']):
            args = parse_args()
            assert args.tqf_R == 20
            assert TQF_R_MIN < args.tqf_R <= TQF_R_MAX

    def test_tqf_R_too_small_fails(self) -> None:
        """
        WHY: R must be greater than inversion radius (r=1)
        HOW: Parse with --tqf-R 1 --models TQF-ANN
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN', '--tqf-R', '1']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_R_minimum_boundary(self) -> None:
        """
        WHY: R=2 is minimum valid truncation radius (must be > r=1)
        HOW: Parse with --tqf-R 2 --models TQF-ANN
        WHAT: Expect successful parse with R=2
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN', '--tqf-R', '2']):
            args = parse_args()
            assert args.tqf_R == 2
            assert args.tqf_R == TQF_R_MIN

    def test_tqf_R_maximum_boundary(self) -> None:
        """
        WHY: R=100 is maximum valid truncation radius
        HOW: Parse with --tqf-R 100 --models TQF-ANN
        WHAT: Expect successful parse with R=100
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN', '--tqf-R', '100']):
            args = parse_args()
            assert args.tqf_R == 100
            assert args.tqf_R == TQF_R_MAX

    def test_tqf_R_too_large_fails(self) -> None:
        """
        WHY: R must not exceed TQF_R_MAX (100)
        HOW: Parse with --tqf-R 101 --models TQF-ANN
        WHAT: Expect SystemExit due to validation failure
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN', '--tqf-R', '101']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_hidden_dim_within_range(self) -> None:
        """
        WHY: Hidden dimension must be reasonable
        HOW: Parse with --tqf-hidden-dim 512 --models TQF-ANN
        WHAT: Expect successful parse
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN', '--tqf-hidden-dim', '512']):
            args = parse_args()
            assert args.tqf_hidden_dim == 512
            assert TQF_HIDDEN_DIM_MIN <= args.tqf_hidden_dim <= TQF_HIDDEN_DIM_MAX

    def test_tqf_fibonacci_mode_none(self) -> None:
        """
        WHY: 'none' is valid fibonacci mode (no scaling)
        HOW: Parse with --tqf-fibonacci-mode none
        WHAT: Expect 'none' in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-fibonacci-mode', 'none']):
            args = parse_args()
            assert args.tqf_fibonacci_mode == 'none'

    def test_tqf_fibonacci_mode_linear(self) -> None:
        """
        WHY: 'linear' is valid fibonacci mode (recommended)
        HOW: Parse with --tqf-fibonacci-mode linear
        WHAT: Expect 'linear' in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-fibonacci-mode', 'linear']):
            args = parse_args()
            assert args.tqf_fibonacci_mode == 'linear'

    def test_tqf_fibonacci_mode_fibonacci(self) -> None:
        """
        WHY: 'fibonacci' is valid fibonacci mode (max performance)
        HOW: Parse with --tqf-fibonacci-mode fibonacci
        WHAT: Expect 'fibonacci' in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-fibonacci-mode', 'fibonacci']):
            args = parse_args()
            assert args.tqf_fibonacci_mode == 'fibonacci'

    def test_tqf_fibonacci_mode_invalid_fails(self) -> None:
        """
        WHY: Invalid fibonacci mode should be rejected
        HOW: Parse with --tqf-fibonacci-mode invalid
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-fibonacci-mode', 'invalid']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_z6_equivariance_weight_default_none(self) -> None:
        """
        WHY: Z6 equivariance loss should be disabled by default
        HOW: Parse without --tqf-z6-equivariance-weight
        WHAT: Expect None (feature disabled)
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_z6_equivariance_weight is None

    def test_z6_equivariance_weight_enables_feature(self) -> None:
        """
        WHY: User enables Z6 equivariance loss by providing a weight
        HOW: Parse with --tqf-z6-equivariance-weight 0.01
        WHAT: Expect 0.01 (feature enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-z6-equivariance-weight', '0.01']):
            args = parse_args()
            assert args.tqf_z6_equivariance_weight == 0.01

    def test_z6_equivariance_weight_custom_value(self) -> None:
        """
        WHY: User can specify custom Z6 weight
        HOW: Parse with --tqf-z6-equivariance-weight 0.02
        WHAT: Expect 0.02 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-z6-equivariance-weight', '0.02']):
            args = parse_args()
            assert args.tqf_z6_equivariance_weight == 0.02

    def test_d6_equivariance_weight_default_none(self) -> None:
        """
        WHY: D6 equivariance loss should be disabled by default
        HOW: Parse without --tqf-d6-equivariance-weight
        WHAT: Expect None (feature disabled)
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_d6_equivariance_weight is None

    def test_d6_equivariance_weight_enables_feature(self) -> None:
        """
        WHY: User enables D6 equivariance loss by providing a weight
        HOW: Parse with --tqf-d6-equivariance-weight 0.01
        WHAT: Expect 0.01 (feature enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-d6-equivariance-weight', '0.01']):
            args = parse_args()
            assert args.tqf_d6_equivariance_weight == 0.01

    def test_d6_equivariance_weight_custom_value(self) -> None:
        """
        WHY: User can specify custom D6 weight
        HOW: Parse with --tqf-d6-equivariance-weight 0.015
        WHAT: Expect 0.015 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-d6-equivariance-weight', '0.015']):
            args = parse_args()
            assert args.tqf_d6_equivariance_weight == 0.015

    def test_t24_orbit_weight_default_none(self) -> None:
        """
        WHY: T24 orbit invariance loss should be disabled by default
        HOW: Parse without --tqf-t24-orbit-invariance-weight
        WHAT: Expect None (feature disabled)
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_t24_orbit_invariance_weight is None

    def test_t24_orbit_weight_enables_feature(self) -> None:
        """
        WHY: User enables T24 orbit invariance loss by providing a weight
        HOW: Parse with --tqf-t24-orbit-invariance-weight 0.005
        WHAT: Expect 0.005 (feature enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-t24-orbit-invariance-weight', '0.005']):
            args = parse_args()
            assert args.tqf_t24_orbit_invariance_weight == 0.005

    def test_t24_orbit_weight_custom_value(self) -> None:
        """
        WHY: User can specify custom T24 weight
        HOW: Parse with --tqf-t24-orbit-invariance-weight 0.008
        WHAT: Expect 0.008 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-t24-orbit-invariance-weight', '0.008']):
            args = parse_args()
            assert args.tqf_t24_orbit_invariance_weight == 0.008

    def test_all_equivariance_weights_combined(self) -> None:
        """
        WHY: User can enable all equivariance/invariance losses by providing weights
        HOW: Parse with all three weight parameters
        WHAT: Expect all weights set (features enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-z6-equivariance-weight', '0.01',
                                '--tqf-d6-equivariance-weight', '0.01',
                                '--tqf-t24-orbit-invariance-weight', '0.005']):
            args = parse_args()
            assert args.tqf_z6_equivariance_weight == 0.01
            assert args.tqf_d6_equivariance_weight == 0.01
            assert args.tqf_t24_orbit_invariance_weight == 0.005

    def test_tqf_box_counting_weight_default(self) -> None:
        """
        WHY: Box-counting weight has a sensible default for fractal loss
        HOW: Parse without --tqf-box-counting-weight
        WHAT: Expect config.TQF_BOX_COUNTING_WEIGHT_DEFAULT
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_box_counting_weight == config.TQF_BOX_COUNTING_WEIGHT_DEFAULT

    def test_tqf_box_counting_weight_custom_value(self) -> None:
        """
        WHY: User can specify custom box-counting weight
        HOW: Parse with --tqf-box-counting-weight 0.005
        WHAT: Expect 0.005 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-weight', '0.005']):
            args = parse_args()
            assert args.tqf_box_counting_weight == 0.005

    def test_tqf_box_counting_weight_zero_valid(self) -> None:
        """
        WHY: Zero weight disables box-counting loss (valid use case)
        HOW: Parse with --tqf-box-counting-weight 0.0
        WHAT: Expect 0.0 in args (no error)
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-weight', '0.0']):
            args = parse_args()
            assert args.tqf_box_counting_weight == 0.0
            assert TQF_BOX_COUNTING_WEIGHT_MIN <= args.tqf_box_counting_weight <= TQF_BOX_COUNTING_WEIGHT_MAX

    def test_tqf_box_counting_weight_below_min_fails(self) -> None:
        """
        WHY: Negative box-counting weight is invalid
        HOW: Parse with --tqf-box-counting-weight -0.1
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-weight', '-0.1']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_box_counting_weight_above_max_fails(self) -> None:
        """
        WHY: Excessively high box-counting weight is invalid
        HOW: Parse with --tqf-box-counting-weight 15.0 (above max of 10.0)
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-weight', '15.0']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_box_counting_scales_default(self) -> None:
        """
        WHY: Box-counting scales has a sensible default for dimension estimation
        HOW: Parse without --tqf-box-counting-scales
        WHAT: Expect config.TQF_BOX_COUNTING_SCALES_DEFAULT
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_box_counting_scales == config.TQF_BOX_COUNTING_SCALES_DEFAULT

    def test_tqf_box_counting_scales_custom_value(self) -> None:
        """
        WHY: User can specify custom number of scales
        HOW: Parse with --tqf-box-counting-scales 12
        WHAT: Expect 12 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-scales', '12']):
            args = parse_args()
            assert args.tqf_box_counting_scales == 12

    def test_tqf_box_counting_scales_within_range(self) -> None:
        """
        WHY: Scales must be within valid range for accurate estimation
        HOW: Parse with valid value
        WHAT: Expect value within [TQF_BOX_COUNTING_SCALES_MIN, TQF_BOX_COUNTING_SCALES_MAX]
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-scales', '8']):
            args = parse_args()
            assert TQF_BOX_COUNTING_SCALES_MIN <= args.tqf_box_counting_scales <= TQF_BOX_COUNTING_SCALES_MAX

    def test_tqf_box_counting_scales_below_min_fails(self) -> None:
        """
        WHY: Too few scales gives unreliable dimension estimates
        HOW: Parse with --tqf-box-counting-scales 1 (below min of 2)
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-scales', '1']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_box_counting_scales_above_max_fails(self) -> None:
        """
        WHY: Too many scales wastes computation with negligible improvement
        HOW: Parse with --tqf-box-counting-scales 25 (above max of 20)
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-box-counting-scales', '25']):
            with pytest.raises(SystemExit):
                parse_args()

    # =========================================================================
    # Adaptive Mixing Temperature Tests
    # =========================================================================

    def test_tqf_adaptive_mixing_temp_default(self) -> None:
        """
        WHY: Adaptive mixing temperature has a sensible default for orbit mixing
        HOW: Parse without --tqf-adaptive-mixing-temp
        WHAT: Expect config.TQF_ADAPTIVE_MIXING_TEMP_DEFAULT
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_adaptive_mixing_temp == config.TQF_ADAPTIVE_MIXING_TEMP_DEFAULT

    def test_tqf_adaptive_mixing_temp_custom_value(self) -> None:
        """
        WHY: User can specify custom temperature for orbit mixing
        HOW: Parse with --tqf-adaptive-mixing-temp 0.3
        WHAT: Expect 0.3 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-adaptive-mixing-temp', '0.3']):
            args = parse_args()
            assert args.tqf_adaptive_mixing_temp == 0.3

    def test_tqf_adaptive_mixing_temp_low_value(self) -> None:
        """
        WHY: Low temperature (sharp mixing) is valid use case
        HOW: Parse with --tqf-adaptive-mixing-temp 0.1
        WHAT: Expect 0.1 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-adaptive-mixing-temp', '0.1']):
            args = parse_args()
            assert args.tqf_adaptive_mixing_temp == 0.1

    def test_tqf_adaptive_mixing_temp_high_value(self) -> None:
        """
        WHY: High temperature (uniform mixing) is valid use case
        HOW: Parse with --tqf-adaptive-mixing-temp 1.0
        WHAT: Expect 1.0 in args
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-adaptive-mixing-temp', '1.0']):
            args = parse_args()
            assert args.tqf_adaptive_mixing_temp == 1.0

    def test_tqf_adaptive_mixing_temp_within_range(self) -> None:
        """
        WHY: Temperature must be within valid range for stable softmax
        HOW: Parse with valid value
        WHAT: Expect value within [TQF_ADAPTIVE_MIXING_TEMP_MIN, TQF_ADAPTIVE_MIXING_TEMP_MAX]
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-adaptive-mixing-temp', '0.5']):
            args = parse_args()
            assert TQF_ADAPTIVE_MIXING_TEMP_MIN <= args.tqf_adaptive_mixing_temp <= TQF_ADAPTIVE_MIXING_TEMP_MAX

    def test_tqf_adaptive_mixing_temp_below_min_fails(self) -> None:
        """
        WHY: Too low temperature causes numerical instability in softmax
        HOW: Parse with --tqf-adaptive-mixing-temp 0.001 (below min of 0.01)
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-adaptive-mixing-temp', '0.001']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_adaptive_mixing_temp_above_max_fails(self) -> None:
        """
        WHY: Too high temperature provides no meaningful discrimination
        HOW: Parse with --tqf-adaptive-mixing-temp 3.0 (above max of 2.0)
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-adaptive-mixing-temp', '3.0']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_tqf_adaptive_mixing_temp_with_orbit_mixing(self) -> None:
        """
        WHY: Temperature parameter is used with orbit mixing enabled
        HOW: Parse with both --tqf-use-orbit-mixing and --tqf-adaptive-mixing-temp
        WHAT: Expect both args correctly set
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-use-orbit-mixing',
                                '--tqf-adaptive-mixing-temp', '0.2']):
            args = parse_args()
            assert args.tqf_use_orbit_mixing == True
            assert args.tqf_adaptive_mixing_temp == 0.2


class TestSetupLogging:
    """Test logging setup function."""

    def test_setup_logging_no_error(self) -> None:
        """
        WHY: Logging setup must not raise exceptions
        HOW: Call setup_logging()
        WHAT: Expect no error
        """
        try:
            setup_logging()
        except Exception as e:
            pytest.fail(f"setup_logging() raised {type(e).__name__}: {e}")

    def test_setup_logging_configures_logger(self) -> None:
        """
        WHY: Must configure root logger
        HOW: Call setup_logging() and check logger
        WHAT: Expect logger has handlers
        """
        import logging
        setup_logging()
        logger = logging.getLogger()
        assert len(logger.handlers) > 0


class TestDeviceSelection:
    """Test device argument handling."""

    def test_device_cuda_accepted(self) -> None:
        """
        WHY: CUDA should be valid device choice
        HOW: Parse with --device cuda
        WHAT: Expect 'cuda' in args
        """
        with patch('sys.argv', ['test_cli.py', '--device', 'cuda']):
            args = parse_args()
            assert args.device == 'cuda'

    def test_device_cpu_accepted(self) -> None:
        """
        WHY: CPU should be valid fallback
        HOW: Parse with --device cpu
        WHAT: Expect 'cpu' in args
        """
        with patch('sys.argv', ['test_cli.py', '--device', 'cpu']):
            args = parse_args()
            assert args.device == 'cpu'

    def test_device_invalid_fails(self) -> None:
        """
        WHY: Invalid device names should be rejected
        HOW: Parse with --device invalid
        WHAT: Expect SystemExit
        """
        with patch('sys.argv', ['test_cli.py', '--device', 'invalid']):
            with pytest.raises(SystemExit):
                parse_args()


class TestPhiBinningArgument:
    """Test suite for --tqf-use-phi-binning argument.

    Phi binning uses golden ratio (phi ~ 1.618) for radial layer scaling instead
    of dyadic (powers of 2). This results in more radial layers with smoother
    transitions between them.
    """

    def test_tqf_use_phi_binning_default_false(self) -> None:
        """
        WHY: Phi binning is opt-in, dyadic binning is the default
        HOW: Parse without --tqf-use-phi-binning
        WHAT: Expect False (dyadic binning)
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_use_phi_binning == False

    def test_tqf_use_phi_binning_enabled(self) -> None:
        """
        WHY: User can enable phi (golden ratio) radial binning
        HOW: Parse with --tqf-use-phi-binning
        WHAT: Expect True
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-use-phi-binning']):
            args = parse_args()
            assert args.tqf_use_phi_binning == True

    def test_tqf_use_phi_binning_with_fibonacci_mode(self) -> None:
        """
        WHY: Phi binning and Fibonacci mode work together (both golden ratio related)
        HOW: Parse with --tqf-use-phi-binning and --tqf-fibonacci-mode fibonacci
        WHAT: Expect both enabled
        """
        with patch('sys.argv', ['test_cli.py', '--tqf-use-phi-binning',
                                '--tqf-fibonacci-mode', 'fibonacci']):
            args = parse_args()
            assert args.tqf_use_phi_binning == True
            assert args.tqf_fibonacci_mode == 'fibonacci'

    def test_tqf_use_phi_binning_with_symmetry_levels(self) -> None:
        """
        WHY: Phi binning should work with all symmetry levels
        HOW: Parse with --tqf-use-phi-binning and various symmetry levels
        WHAT: Expect successful parse with each symmetry level
        """
        for symmetry in ['none', 'Z6', 'D6', 'T24']:
            with patch('sys.argv', ['test_cli.py', '--tqf-use-phi-binning',
                                    '--tqf-symmetry-level', symmetry]):
                args = parse_args()
                assert args.tqf_use_phi_binning == True
                assert args.tqf_symmetry_level == symmetry

    def test_tqf_use_phi_binning_matches_config_default(self) -> None:
        """
        WHY: CLI default should match config constant
        HOW: Compare parsed default to TQF_USE_PHI_BINNING_DEFAULT
        WHAT: Expect equal values
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.tqf_use_phi_binning == config.TQF_USE_PHI_BINNING_DEFAULT


class TestComplexArgumentCombinations:
    """Test complex argument combinations."""

    def test_tqf_with_all_custom_params(self) -> None:
        """
        WHY: User should be able to customize all TQF params
        HOW: Parse with many TQF arguments
        WHAT: Expect all values correctly set
        """
        argv: List[str] = [
            'test_cli.py',
            '--models', 'TQF-ANN',
            '--tqf-R', '20',
            '--tqf-symmetry-level', 'D6',
            '--tqf-use-orbit-mixing',
            '--tqf-fractal-iterations', '10',
            '--tqf-geometry-reg-weight', '0.5'
        ]
        with patch('sys.argv', argv):
            args = parse_args()
            assert args.models == ['TQF-ANN']
            assert args.tqf_R == 20
            assert args.tqf_symmetry_level == 'D6'
            assert args.tqf_use_orbit_mixing == True
            assert args.tqf_fractal_iterations == 10
            assert args.tqf_geometry_reg_weight == 0.5

    def test_tqf_fractal_iterations_disabled_by_default(self) -> None:
        """
        WHY: --tqf-fractal-iterations should be disabled by default (opt-in feature)
        HOW: Parse without specifying the parameter
        WHAT: Expect tqf_fractal_iterations to be None (disabled)
        """
        argv: List[str] = [
            'test_cli.py',
            '--models', 'TQF-ANN'
        ]
        with patch('sys.argv', argv):
            args = parse_args()
            assert args.tqf_fractal_iterations is None  # Disabled by default

    def test_tqf_fractal_iterations_enabled_when_provided(self) -> None:
        """
        WHY: User should be able to explicitly enable fractal iterations
        HOW: Parse with --tqf-fractal-iterations value
        WHAT: Expect value to be set correctly
        """
        argv: List[str] = [
            'test_cli.py',
            '--models', 'TQF-ANN',
            '--tqf-fractal-iterations', '7'
        ]
        with patch('sys.argv', argv):
            args = parse_args()
            assert args.tqf_fractal_iterations == 7  # Enabled when provided

    def test_tqf_verify_geometry_flag_parsing(self) -> None:
        """
        WHY: --tqf-verify-geometry flag should enable geometry verification
        HOW: Parse with and without the flag
        WHAT: Expect correct boolean value set
        """
        # Test with flag enabled
        argv_enabled: List[str] = [
            'test_cli.py',
            '--models', 'TQF-ANN',
            '--tqf-verify-geometry'
        ]
        with patch('sys.argv', argv_enabled):
            args = parse_args()
            assert args.tqf_verify_geometry == True

        # Test without flag (default)
        argv_disabled: List[str] = [
            'test_cli.py',
            '--models', 'TQF-ANN'
        ]
        with patch('sys.argv', argv_disabled):
            args = parse_args()
            assert args.tqf_verify_geometry == False

    def test_tqf_geometry_reg_weight_default(self) -> None:
        """
        WHY: --tqf-geometry-reg-weight should have correct default value
        HOW: Parse without specifying the weight
        WHAT: Expect default value from config (0.0, disabled by default)
        """
        from config import TQF_GEOMETRY_REG_WEIGHT_DEFAULT

        argv: List[str] = [
            'test_cli.py',
            '--models', 'TQF-ANN'
        ]
        with patch('sys.argv', argv):
            args = parse_args()
            assert args.tqf_geometry_reg_weight == TQF_GEOMETRY_REG_WEIGHT_DEFAULT
            assert args.tqf_geometry_reg_weight == 0.0  # Disabled by default (opt-in)

    def test_baseline_models_ignore_tqf_params(self) -> None:
        """
        WHY: TQF parameters should not affect baseline models
        HOW: Parse with baseline models and TQF params
        WHAT: Expect no validation errors (TQF params ignored)
        """
        argv: List[str] = [
            'test_cli.py',
            '--models', 'FC-MLP', 'CNN-L5',
            '--tqf-R', '50'  # This should be ignored/not validated for non-TQF models
        ]
        with patch('sys.argv', argv):
            args = parse_args()
            assert 'TQF-ANN' not in args.models
            # Args should parse successfully even though TQF params present


class TestInversionLossCLIFlags:
    """Test suite for inversion loss CLI weight parameter.

    Features are enabled by providing a weight value (not None).
    Features are disabled by default (weight=None).
    """

    def test_inversion_loss_weight_default_none(self) -> None:
        """
        WHY: Inversion loss should be disabled by default (opt-in feature)
        HOW: Parse without specifying the weight
        WHAT: Expect tqf_inversion_loss_weight to be None (disabled)
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_inversion_loss_weight is None

    def test_inversion_loss_weight_enables_feature(self) -> None:
        """
        WHY: User enables inversion loss by providing a weight value
        HOW: Parse with --tqf-inversion-loss-weight 0.001
        WHAT: Expect weight set (feature enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-inversion-loss-weight', '0.001']):
            args = parse_args()
            assert args.tqf_inversion_loss_weight == 0.001

    def test_inversion_loss_weight_custom(self) -> None:
        """
        WHY: User should be able to set custom inversion loss weight
        HOW: Parse with --tqf-inversion-loss-weight 0.05
        WHAT: Expect weight to be 0.05
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-inversion-loss-weight', '0.05']):
            args = parse_args()
            assert args.tqf_inversion_loss_weight == 0.05


class TestCombinedLossFlags:
    """Test suite for combined symmetry/invariance loss weight scenarios.

    All loss features are controlled via weight parameters only.
    Features are disabled by default (weight=None) and enabled when a weight is provided.
    """

    def test_all_loss_weights_disabled_by_default(self) -> None:
        """
        WHY: All loss features should be disabled by default
        HOW: Parse without specifying any loss weights
        WHAT: Expect all weights to be None
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_inversion_loss_weight is None
            assert args.tqf_z6_equivariance_weight is None
            assert args.tqf_d6_equivariance_weight is None
            assert args.tqf_t24_orbit_invariance_weight is None

    def test_inversion_loss_enabled_with_weight(self) -> None:
        """
        WHY: User enables inversion loss by providing a weight
        HOW: Parse with inversion weight
        WHAT: Expect inversion weight set (feature enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-inversion-loss-weight', '0.02']):
            args = parse_args()
            assert args.tqf_inversion_loss_weight == 0.02

    def test_equivariance_and_invariance_losses_combined(self) -> None:
        """
        WHY: User should be able to enable both equivariance and invariance losses
        HOW: Parse with Z6 equivariance + T24 invariance weights
        WHAT: Expect both weights set (features enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-equivariance-weight', '0.01',
                               '--tqf-t24-orbit-invariance-weight', '0.005']):
            args = parse_args()
            assert args.tqf_z6_equivariance_weight == 0.01
            assert args.tqf_t24_orbit_invariance_weight == 0.005


def run_tests(verbosity: int = 2):
    """
    Run all CLI tests.

    Args:
        verbosity: pytest verbosity level (0, 1, or 2)

    Returns:
        Exit code (0 if all tests pass)
    """
    import pytest
    import sys
    args: List[str] = [__file__, f'-{"v" * verbosity}']
    return pytest.main(args)


if __name__ == '__main__':
    exit_code: int = run_tests()
    sys.exit(exit_code)
