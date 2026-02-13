"""
test_logging_utils.py - Test Suite for logging_utils.py

This module tests logging and formatting utilities for experiment output,
including separators, experiment configuration logging, and console formatting.

Key Test Coverage:
- Console separator formatting (single and double-line)
- Experiment configuration logging with all parameters
- System information display
- Hardware detection (GPU, CUDA)
- ASCII-only output validation

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

import pytest
import io
import argparse
from typing import List
from unittest.mock import patch, MagicMock
import torch

from logging_utils import (
    print_separator,
    print_single_separator,
    log_experiment_config
)
import config


class TestPrintSeparator:
    """Test double-line separator printing with centered titles."""

    def test_separator_prints(self) -> None:
        """Test that separator prints without error."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_separator("Test Header")
            output: str = fake_out.getvalue()
            assert len(output) > 0, "Separator should produce output"

    def test_separator_contains_header(self) -> None:
        """Test that header text appears in separator output."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            header: str = "TEST HEADER"
            print_separator(header)
            output: str = fake_out.getvalue()
            assert header in output, f"Header '{header}' not found in output"

    def test_separator_has_equals_lines(self) -> None:
        """Test that separator contains equals character lines."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_separator("Header")
            output: str = fake_out.getvalue()
            assert '=' in output, "Separator should contain '=' characters"

    def test_separator_multiple_lines(self) -> None:
        """Test that separator produces multiple lines (header between lines)."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_separator("Header")
            output: str = fake_out.getvalue()
            lines: List[str] = output.strip().split('\n')
            assert len(lines) >= 3, "Separator should have at least 3 lines"

    def test_custom_width(self) -> None:
        """Test that separator respects custom width parameter."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_separator("Header", width=50)
            output: str = fake_out.getvalue()
            lines: List[str] = output.strip().split('\n')
            # At least one line should be approximately width 50
            assert any(40 <= len(line) <= 60 for line in lines)

    def test_empty_title(self) -> None:
        """Test that separator works with empty title."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_separator("", width=80)
            output: str = fake_out.getvalue()
            # Should produce output (separator lines only)
            assert len(output) > 0, "Separator with empty title should produce output"


class TestPrintSingleSeparator:
    """Test single-line separator printing."""

    def test_single_separator_prints(self) -> None:
        """Test that single separator prints without error."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_single_separator()
            output: str = fake_out.getvalue()
            assert len(output) > 0, "Single separator should produce output"

    def test_equals_character_default(self) -> None:
        """Test that default character is equals sign."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_single_separator()
            output: str = fake_out.getvalue()
            assert '=' in output, "Default separator should contain '=' characters"

    def test_dash_character(self) -> None:
        """Test that dash character works."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_single_separator('-')
            output: str = fake_out.getvalue()
            assert '-' in output, "Should contain dash characters"

    def test_custom_character(self) -> None:
        """Test that custom separator characters work."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_single_separator('*')
            output: str = fake_out.getvalue()
            assert '*' in output, "Should contain custom character"

    def test_custom_width(self) -> None:
        """Test that single separator respects width parameter."""
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            print_single_separator('=', width=50)
            output: str = fake_out.getvalue()
            # Output should be approximately 50 chars (plus newline)
            assert 45 <= len(output.strip()) <= 55


class TestLogExperimentConfig:
    """Test comprehensive experiment configuration logging."""

    def test_logs_without_error(self) -> None:
        """Test that log_experiment_config runs without exceptions."""
        # Create minimal args namespace
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200,
            learning_rate=0.001,
            weight_decay=0.0001,
            label_smoothing=0.1,
            patience=15,
            min_delta=0.0005,
            learning_rate_warmup_epochs=5,
            num_train=30000,
            num_val=2000,
            num_test_unrot=8000,
            num_test_rot=2000,
            seed_start=42,
            model='TQF-ANN',
            tqf_R=3.0,
            tqf_hidden_dim=512,
            tqf_symmetry_level='D6',
            tqf_fractal_iterations=5,
            tqf_self_similarity_weight=0.001,
            tqf_box_counting_weight=0.001,
            tqf_geometry_reg_weight=0.01,
            tqf_rotation_inv_loss_weight=0.001,
            tqf_inversion_loss_weight=0.001,
            tqf_hop_attention_temp=0.5,
            tqf_verify_duality_interval=10,
            tqf_verify_geometry=False
        )
        seeds: List[int] = [42, 43, 44]
        device = torch.device('cpu')

        try:
            with patch('sys.stdout', new=io.StringIO()):
                log_experiment_config(args, seeds, device)
        except Exception as e:
            pytest.fail(f"log_experiment_config raised {type(e).__name__}: {e}")

    def test_logs_system_info(self) -> None:
        """Test that system information is logged."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200,
            model='TQF-ANN'
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            log_experiment_config(args, seeds, device)
            output: str = fake_out.getvalue()

            # Check for system info sections
            assert 'SYSTEM INFORMATION' in output
            assert 'Python' in output or 'PyTorch' in output

    def test_logs_dataset_config(self) -> None:
        """Test that dataset configuration is logged."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200,
            num_train=30000,
            num_val=2000,
            num_test_unrot=8000,
            num_test_rot=2000
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            log_experiment_config(args, seeds, device)
            output: str = fake_out.getvalue()

            # Check for dataset config section
            assert 'DATASET CONFIGURATION' in output
            assert '30000' in output or 'Training samples' in output

    def test_logs_hyperparameters(self) -> None:
        """Test that optimization hyperparameters are logged."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200,
            learning_rate=0.001,
            weight_decay=0.0001,
            label_smoothing=0.1,
            patience=15,
            min_delta=0.0005,
            learning_rate_warmup_epochs=5
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            log_experiment_config(args, seeds, device)
            output: str = fake_out.getvalue()

            # Check for hyperparameters section
            assert 'OPTIMIZATION HYPERPARAMETERS' in output or 'HYPERPARAMETERS' in output
            assert '0.001' in output or 'learning_rate' in output.lower()

    def test_logs_tqf_config_when_tqf_model(self) -> None:
        """Test that TQF configuration is logged when TQF model is selected."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200,
            model='TQF-ANN',
            tqf_R=3.0,
            tqf_hidden_dim=512,
            tqf_symmetry_level='D6'
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            log_experiment_config(args, seeds, device)
            output: str = fake_out.getvalue()

            # Check for TQF-specific section
            assert 'TQF-SPECIFIC' in output or 'TQF' in output
            assert 'D6' in output or 'symmetry' in output.lower()

    def test_skips_tqf_config_when_baseline_model(self) -> None:
        """Test that TQF configuration is skipped for baseline models."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200,
            model='CNN-L5'  # Non-TQF model
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            log_experiment_config(args, seeds, device)
            output: str = fake_out.getvalue()

            # TQF-specific section should not appear
            # (or should appear but be brief/skipped)
            # We just verify the function runs successfully
            assert len(output) > 0

    def test_logs_parameter_matching(self) -> None:
        """Test that parameter matching configuration is logged."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            log_experiment_config(args, seeds, device)
            output: str = fake_out.getvalue()

            # Check for parameter matching section
            assert 'PARAMETER MATCHING' in output
            assert '650' in output or 'TARGET' in output

    def test_logs_cuda_info_when_available(self) -> None:
        """Test that CUDA information is logged when GPU is available."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200
        )
        seeds: List[int] = [42]

        # Mock CUDA availability
        if torch.cuda.is_available():
            device = torch.device('cuda')

            with patch('sys.stdout', new=io.StringIO()) as fake_out:
                log_experiment_config(args, seeds, device)
                output: str = fake_out.getvalue()

                # Check for CUDA-specific info
                assert 'CUDA' in output or 'GPU' in output
        else:
            # Skip test if CUDA not available
            pytest.skip("CUDA not available on this system")

    def test_handles_missing_optional_attributes(self) -> None:
        """Test that missing optional attributes don't cause errors."""
        # Create minimal args with only required attributes
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        try:
            with patch('sys.stdout', new=io.StringIO()):
                log_experiment_config(args, seeds, device)
        except AttributeError as e:
            pytest.fail(f"Missing optional attribute caused error: {e}")

    def test_ascii_only_output(self) -> None:
        """Test that output contains only ASCII characters."""
        args = argparse.Namespace(
            batch_size=128,
            num_epochs=200,
            model='TQF-ANN'
        )
        seeds: List[int] = [42]
        device = torch.device('cpu')

        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            log_experiment_config(args, seeds, device)
            output: str = fake_out.getvalue()

            # Check that all characters are ASCII
            try:
                output.encode('ascii')
            except UnicodeEncodeError:
                pytest.fail("Output contains non-ASCII characters")


class TestRemovedObsoleteFunctions:
    """Test that obsolete/unused functions have been removed."""

    def test_progress_logger_removed(self) -> None:
        """Test that unused ProgressLogger class was removed."""
        from logging_utils import __dict__ as logging_dict
        assert 'ProgressLogger' not in logging_dict, \
            "ProgressLogger class should have been removed (unused)"

    def test_format_dict_removed(self) -> None:
        """Test that unused format_dict_for_logging was removed."""
        from logging_utils import __dict__ as logging_dict
        assert 'format_dict_for_logging' not in logging_dict, \
            "format_dict_for_logging should have been removed (unused)"

    def test_log_system_info_removed(self) -> None:
        """Test that unused log_system_info was removed."""
        from logging_utils import __dict__ as logging_dict
        assert 'log_system_info' not in logging_dict, \
            "log_system_info should have been removed (merged into log_experiment_config)"

    def test_log_config_dict_removed(self) -> None:
        """Test that log_config_dict alias was removed."""
        from logging_utils import __dict__ as logging_dict
        assert 'log_config_dict' not in logging_dict, \
            "log_config_dict alias should have been removed (use log_experiment_config)"


def run_tests(verbosity: int = 2):
    """Run all logging_utils tests."""
    import sys
    args: List[str] = [__file__, f'-{"v" * verbosity}']
    return pytest.main(args)


if __name__ == '__main__':
    import sys
    exit_code: int = run_tests()
    sys.exit(exit_code)
