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
  - Binning Schemes: dyadic radial binning
- Loss Function Weights (Opt-In Features):
  - Invariance: --tqf-t24-orbit-invariance-weight
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

    def test_default_device_auto(self) -> None:
        """
        WHY: Default should be 'auto' (resolved to cuda/cpu later in main.py)
        HOW: Parse args and check device
        WHAT: Expect 'auto'
        """
        with patch('sys.argv', ['test_cli.py']):
            args = parse_args()
            assert args.device == 'auto'


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

    # NOTE: --tqf-box-counting-weight and --tqf-box-counting-scales tests removed —
    # fractal loss features removed from codebase



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
        ]
        with patch('sys.argv', argv):
            args = parse_args()
            assert args.models == ['TQF-ANN']
            assert args.tqf_R == 20

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
            assert args.tqf_t24_orbit_invariance_weight is None

    def test_invariance_loss_enabled(self) -> None:
        """
        WHY: User should be able to enable T24 invariance loss
        HOW: Parse with T24 invariance weight
        WHAT: Expect weight set (feature enabled)
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-t24-orbit-invariance-weight', '0.005']):
            args = parse_args()
            assert args.tqf_t24_orbit_invariance_weight == 0.005


class TestZ6AugmentationFlag:
    """Test suite for --z6-data-augmentation flag (store_true, default False)."""

    def test_z6_augmentation_disabled_by_default(self) -> None:
        """
        WHY: Z6 augmentation should be disabled by default (conflicts with orbit mixing)
        HOW: Parse without specifying the flag
        WHAT: Expect z6_data_augmentation = False
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.z6_data_augmentation is False

    def test_z6_augmentation_enabled(self) -> None:
        """
        WHY: User should be able to enable Z6 augmentation for rotation robustness
        HOW: Parse with --z6-data-augmentation
        WHAT: Expect z6_data_augmentation = True
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--z6-data-augmentation']):
            args = parse_args()
            assert args.z6_data_augmentation is True


class TestOrbitMixingFlags:
    """Test suite for orbit mixing CLI flags."""

    def test_orbit_mixing_disabled_by_default(self) -> None:
        """
        WHY: All orbit mixing modes should be disabled by default
        HOW: Parse without specifying orbit mixing flags
        WHAT: Expect all orbit mixing booleans = False
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_use_z6_orbit_mixing is False
            assert args.tqf_use_d6_orbit_mixing is False
            assert args.tqf_use_t24_orbit_mixing is False

    def test_z6_orbit_mixing_enabled(self) -> None:
        """
        WHY: User should be able to enable Z6 orbit mixing
        HOW: Parse with --tqf-use-z6-orbit-mixing
        WHAT: Expect tqf_use_z6_orbit_mixing = True
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-use-z6-orbit-mixing']):
            args = parse_args()
            assert args.tqf_use_z6_orbit_mixing is True

    def test_d6_orbit_mixing_enabled(self) -> None:
        """
        WHY: User should be able to enable D6 orbit mixing
        HOW: Parse with --tqf-use-d6-orbit-mixing
        WHAT: Expect tqf_use_d6_orbit_mixing = True
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-use-d6-orbit-mixing']):
            args = parse_args()
            assert args.tqf_use_d6_orbit_mixing is True

    def test_t24_orbit_mixing_enabled(self) -> None:
        """
        WHY: User should be able to enable T24 orbit mixing
        HOW: Parse with --tqf-use-t24-orbit-mixing
        WHAT: Expect tqf_use_t24_orbit_mixing = True
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-use-t24-orbit-mixing']):
            args = parse_args()
            assert args.tqf_use_t24_orbit_mixing is True

    def test_default_temperatures(self) -> None:
        """
        WHY: Default temperatures should match spec (rotation=0.5, reflection=0.5, inversion=0.7)
        HOW: Parse without specifying temperatures
        WHAT: Expect correct default values
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_temp_rotation == 0.5
            assert args.tqf_d6_orbit_mixing_temp_reflection == 0.5
            assert args.tqf_t24_orbit_mixing_temp_inversion == 0.7

    def test_custom_temperatures(self) -> None:
        """
        WHY: User should be able to set custom temperatures
        HOW: Parse with explicit temperature values
        WHAT: Expect values match what was provided
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-temp-rotation', '0.5',
                               '--tqf-d6-orbit-mixing-temp-reflection', '0.8',
                               '--tqf-t24-orbit-mixing-temp-inversion', '1.0']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_temp_rotation == 0.5
            assert args.tqf_d6_orbit_mixing_temp_reflection == 0.8
            assert args.tqf_t24_orbit_mixing_temp_inversion == 1.0

    def test_temperature_validation_too_low(self) -> None:
        """
        WHY: Temperature below 0.01 should be rejected
        HOW: Parse with temperature = 0.001
        WHAT: Expect SystemExit from validation
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-temp-rotation', '0.001']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_temperature_validation_too_high(self) -> None:
        """
        WHY: Temperature above 10.0 should be rejected
        HOW: Parse with temperature = 15.0
        WHAT: Expect SystemExit from validation
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-t24-orbit-mixing-temp-inversion', '15.0']):
            with pytest.raises(SystemExit):
                parse_args()


class TestCLIConfigRangeConstantConsistency:
    """Test that CLI range constants are the same objects as config.py constants.

    WHY: Range constants were moved from cli.py to config.py as the single
         source of truth. cli.py now imports them. If someone accidentally
         redefines a constant locally in cli.py, the CLI validation and
         config.py assertions could use different values silently.
    HOW: Import the same names from both modules and verify identity.
    WHAT: Every range constant imported by cli.py must equal config.py's value.
    """

    def test_training_range_constants_match(self) -> None:
        """Test training hyperparameter ranges match between CLI and config."""
        import cli
        pairs = [
            'NUM_SEEDS_MIN', 'NUM_SEEDS_MAX',
            'NUM_EPOCHS_MIN', 'NUM_EPOCHS_MAX',
            'BATCH_SIZE_MIN', 'BATCH_SIZE_MAX',
            'LEARNING_RATE_MIN', 'LEARNING_RATE_MAX',
            'WEIGHT_DECAY_MIN', 'WEIGHT_DECAY_MAX',
            'LABEL_SMOOTHING_MIN', 'LABEL_SMOOTHING_MAX',
            'PATIENCE_MIN', 'PATIENCE_MAX',
            'MIN_DELTA_MIN', 'MIN_DELTA_MAX',
            'LEARNING_RATE_WARMUP_EPOCHS_MIN', 'LEARNING_RATE_WARMUP_EPOCHS_MAX',
        ]
        for name in pairs:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val}) != config.{name} ({config_val})"

    def test_dataset_range_constants_match(self) -> None:
        """Test dataset size ranges match between CLI and config."""
        import cli
        pairs = [
            'NUM_TRAIN_MIN', 'NUM_TRAIN_MAX',
            'NUM_VAL_MIN', 'NUM_VAL_MAX',
            'NUM_TEST_ROT_MIN', 'NUM_TEST_ROT_MAX',
            'NUM_TEST_UNROT_MIN', 'NUM_TEST_UNROT_MAX',
        ]
        for name in pairs:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val}) != config.{name} ({config_val})"

    def test_tqf_architecture_range_constants_match(self) -> None:
        """Test TQF architecture ranges match between CLI and config."""
        import cli
        pairs = [
            'TQF_R_MIN', 'TQF_R_MAX',
            'TQF_HIDDEN_DIM_MIN', 'TQF_HIDDEN_DIM_MAX',
        ]
        for name in pairs:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val}) != config.{name} ({config_val})"

    def test_orbit_mixing_range_constants_match(self) -> None:
        """Test orbit mixing temperature ranges match between CLI and config."""
        import cli
        pairs = [
            'TQF_ORBIT_MIXING_TEMP_MIN', 'TQF_ORBIT_MIXING_TEMP_MAX',
        ]
        for name in pairs:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val}) != config.{name} ({config_val})"

    def test_loss_weight_range_constants_match(self) -> None:
        """Test loss weight ranges match between CLI and config."""
        import cli
        pairs = [
            'TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN', 'TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX',
        ]
        for name in pairs:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val}) != config.{name} ({config_val})"

    def test_default_constants_match(self) -> None:
        """Test that CLI default values match config.py defaults."""
        import cli
        defaults = [
            'Z6_DATA_AUGMENTATION_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_TEMP_ROTATION_DEFAULT',
            'TQF_D6_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT',
            'TQF_T24_ORBIT_MIXING_TEMP_INVERSION_DEFAULT',
        ]
        for name in defaults:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val}) != config.{name} ({config_val})"


class TestZ6OrbitMixingEnhancements:
    """Tests for the mark3 Z6 orbit mixing enhancement flags.

    These flags extend the base --tqf-use-z6-orbit-mixing feature with:
    - confidence mode (max_logit | margin)
    - aggregation mode (logits | probs | log_probs)
    - top-K variant selection
    - adaptive temperature scaling
    - higher-quality rotation (bicubic, border padding, pad-rotate-crop)
    - non-rotation training augmentation
    - orbit consistency self-distillation loss
    """

    # ── Defaults ────────────────────────────────────────────────────────────────

    def test_z6_confidence_mode_default(self) -> None:
        """
        WHY: confidence_mode should default to 'max_logit' (backward compat)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_confidence_mode == 'max_logit'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_confidence_mode == 'max_logit'

    def test_z6_aggregation_mode_default(self) -> None:
        """
        WHY: aggregation_mode should default to 'logits' (backward compat)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_aggregation_mode == 'logits'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_aggregation_mode == 'logits'

    def test_z6_top_k_default_none(self) -> None:
        """
        WHY: top_k should default to None (use all 6 variants)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_top_k is None
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_top_k is None

    def test_z6_adaptive_temp_default_false(self) -> None:
        """
        WHY: adaptive_temp should default to False (backward compat)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_adaptive_temp is False
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_adaptive_temp is False

    def test_z6_adaptive_temp_alpha_default(self) -> None:
        """
        WHY: adaptive_temp_alpha should default to 1.0
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_adaptive_temp_alpha == 1.0
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_adaptive_temp_alpha == 1.0

    def test_z6_rotation_mode_default(self) -> None:
        """
        WHY: rotation_mode should default to 'bilinear' (backward compat)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_rotation_mode == 'bilinear'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_rotation_mode == 'bilinear'

    def test_z6_rotation_padding_mode_default(self) -> None:
        """
        WHY: rotation_padding_mode should default to 'zeros' (backward compat)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_rotation_padding_mode == 'zeros'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_rotation_padding_mode == 'zeros'

    def test_z6_rotation_pad_default_zero(self) -> None:
        """
        WHY: rotation_pad should default to 0 (no padding; backward compat)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_mixing_rotation_pad == 0
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_rotation_pad == 0

    def test_non_rotation_aug_default_false(self) -> None:
        """
        WHY: non-rotation augmentation should default to False
        HOW: Parse without the flag
        WHAT: Expect args.non_rotation_data_augmentation is False
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.non_rotation_data_augmentation is False

    def test_z6_orbit_consistency_weight_default_none(self) -> None:
        """
        WHY: orbit_consistency_weight should default to None (feature disabled)
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_consistency_weight is None
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_consistency_weight is None

    def test_z6_orbit_consistency_rotations_default(self) -> None:
        """
        WHY: orbit_consistency_rotations should default to 2
        HOW: Parse without the flag
        WHAT: Expect args.tqf_z6_orbit_consistency_rotations == 2
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN']):
            args = parse_args()
            assert args.tqf_z6_orbit_consistency_rotations == 2

    # ── Accepted values ─────────────────────────────────────────────────────────

    def test_z6_confidence_mode_margin(self) -> None:
        """
        WHY: 'margin' is a valid confidence mode
        HOW: Parse with --tqf-z6-orbit-mixing-confidence-mode margin
        WHAT: Expect args.tqf_z6_orbit_mixing_confidence_mode == 'margin'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-confidence-mode', 'margin']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_confidence_mode == 'margin'

    def test_z6_aggregation_mode_probs(self) -> None:
        """
        WHY: 'probs' is a valid aggregation mode
        HOW: Parse with --tqf-z6-orbit-mixing-aggregation-mode probs
        WHAT: Expect args.tqf_z6_orbit_mixing_aggregation_mode == 'probs'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-aggregation-mode', 'probs']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_aggregation_mode == 'probs'

    def test_z6_aggregation_mode_log_probs(self) -> None:
        """
        WHY: 'log_probs' is a valid aggregation mode
        HOW: Parse with --tqf-z6-orbit-mixing-aggregation-mode log_probs
        WHAT: Expect args.tqf_z6_orbit_mixing_aggregation_mode == 'log_probs'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-aggregation-mode', 'log_probs']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_aggregation_mode == 'log_probs'

    def test_z6_top_k_valid_value(self) -> None:
        """
        WHY: top_k=4 is a valid selection (>= 2 and <= 6)
        HOW: Parse with --tqf-z6-orbit-mixing-top-k 4 --tqf-use-z6-orbit-mixing
        WHAT: Expect args.tqf_z6_orbit_mixing_top_k == 4
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-use-z6-orbit-mixing',
                               '--tqf-z6-orbit-mixing-top-k', '4']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_top_k == 4

    def test_z6_adaptive_temp_enabled(self) -> None:
        """
        WHY: User should be able to enable adaptive temperature
        HOW: Parse with --tqf-z6-orbit-mixing-adaptive-temp
        WHAT: Expect args.tqf_z6_orbit_mixing_adaptive_temp is True
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-adaptive-temp']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_adaptive_temp is True

    def test_z6_rotation_mode_bicubic(self) -> None:
        """
        WHY: 'bicubic' is a valid rotation mode
        HOW: Parse with --tqf-z6-orbit-mixing-rotation-mode bicubic
        WHAT: Expect args.tqf_z6_orbit_mixing_rotation_mode == 'bicubic'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-rotation-mode', 'bicubic']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_rotation_mode == 'bicubic'

    def test_z6_rotation_padding_mode_border(self) -> None:
        """
        WHY: 'border' is a valid padding mode
        HOW: Parse with --tqf-z6-orbit-mixing-rotation-padding-mode border
        WHAT: Expect args.tqf_z6_orbit_mixing_rotation_padding_mode == 'border'
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-rotation-padding-mode', 'border']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_rotation_padding_mode == 'border'

    def test_z6_rotation_pad_valid(self) -> None:
        """
        WHY: pad=4 is a valid rotation pad value (0-8)
        HOW: Parse with --tqf-z6-orbit-mixing-rotation-pad 4
        WHAT: Expect args.tqf_z6_orbit_mixing_rotation_pad == 4
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-rotation-pad', '4']):
            args = parse_args()
            assert args.tqf_z6_orbit_mixing_rotation_pad == 4

    def test_non_rotation_aug_enabled(self) -> None:
        """
        WHY: User should be able to enable non-rotation augmentation
        HOW: Parse with --non-rotation-data-augmentation
        WHAT: Expect args.non_rotation_data_augmentation is True
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--non-rotation-data-augmentation']):
            args = parse_args()
            assert args.non_rotation_data_augmentation is True

    def test_z6_orbit_consistency_weight_valid(self) -> None:
        """
        WHY: User should be able to enable orbit consistency loss
        HOW: Parse with --tqf-z6-orbit-consistency-weight 0.01
        WHAT: Expect args.tqf_z6_orbit_consistency_weight == 0.01
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-consistency-weight', '0.01']):
            args = parse_args()
            assert args.tqf_z6_orbit_consistency_weight == pytest.approx(0.01)

    def test_z6_orbit_consistency_rotations_valid(self) -> None:
        """
        WHY: User should be able to set orbit consistency rotation count
        HOW: Parse with --tqf-z6-orbit-consistency-rotations 3
        WHAT: Expect args.tqf_z6_orbit_consistency_rotations == 3
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-consistency-rotations', '3']):
            args = parse_args()
            assert args.tqf_z6_orbit_consistency_rotations == 3

    # ── Validation: out-of-range ─────────────────────────────────────────────────

    def test_z6_top_k_too_low_rejected(self) -> None:
        """
        WHY: top_k < 2 should be rejected (need at least 2 variants to be useful)
        HOW: Parse with --tqf-z6-orbit-mixing-top-k 1 --tqf-use-z6-orbit-mixing
        WHAT: Expect SystemExit from validation
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-use-z6-orbit-mixing',
                               '--tqf-z6-orbit-mixing-top-k', '1']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_z6_adaptive_temp_alpha_too_low_rejected(self) -> None:
        """
        WHY: adaptive_temp_alpha below 0.1 should be rejected
        HOW: Parse with --tqf-z6-orbit-mixing-adaptive-temp-alpha 0.05
        WHAT: Expect SystemExit from validation
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-adaptive-temp-alpha', '0.05']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_z6_rotation_pad_too_high_rejected(self) -> None:
        """
        WHY: rotation_pad > 8 should be rejected
        HOW: Parse with --tqf-z6-orbit-mixing-rotation-pad 10
        WHAT: Expect SystemExit from validation
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-mixing-rotation-pad', '10']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_z6_orbit_consistency_weight_too_low_rejected(self) -> None:
        """
        WHY: orbit_consistency_weight must be >= 0.0001
        HOW: Parse with --tqf-z6-orbit-consistency-weight 0.00001
        WHAT: Expect SystemExit from validation
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-consistency-weight', '0.00001']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_z6_orbit_consistency_rotations_too_high_rejected(self) -> None:
        """
        WHY: orbit_consistency_rotations must be <= 5
        HOW: Parse with --tqf-z6-orbit-consistency-rotations 6
        WHAT: Expect SystemExit from validation
        """
        with patch('sys.argv', ['test_cli.py', '--models', 'TQF-ANN',
                               '--tqf-z6-orbit-consistency-rotations', '6']):
            with pytest.raises(SystemExit):
                parse_args()

    # ── Constant consistency ─────────────────────────────────────────────────────

    def test_z6_enhancement_default_constants_match_config(self) -> None:
        """
        WHY: All mark3 default values must equal config.py defaults
        HOW: Import constants from both cli and config, compare
        WHAT: All Z6 enhancement defaults match
        """
        import cli
        defaults = [
            'TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT',
            'TQF_Z6_ORBIT_MIXING_ROTATION_PAD_DEFAULT',
            'NON_ROTATION_DATA_AUGMENTATION_DEFAULT',
            'TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_DEFAULT',
        ]
        for name in defaults:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val!r}) != config.{name} ({config_val!r})"

    def test_z6_enhancement_range_constants_match_config(self) -> None:
        """
        WHY: All mark3 range constants must equal config.py ranges
        HOW: Import constants from both cli and config, compare
        WHAT: All Z6 enhancement range constants match
        """
        import cli
        ranges = [
            'TQF_Z6_ORBIT_MIXING_TOP_K_MIN', 'TQF_Z6_ORBIT_MIXING_TOP_K_MAX',
            'TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MIN',
            'TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MAX',
            'TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MIN',
            'TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MAX',
            'TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MIN',
            'TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MAX',
            'TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MIN',
            'TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MAX',
        ]
        for name in ranges:
            cli_val = getattr(cli, name)
            config_val = getattr(config, name)
            assert cli_val == config_val, \
                f"cli.{name} ({cli_val!r}) != config.{name} ({config_val!r})"


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
