"""
run_tests.py - Optimized Test Runner for TQF-NN Project

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Date: February 2026

USAGE:
    python run_tests.py                    # Run all tests
    python run_tests.py --file test_config # Run specific file
    python run_tests.py --quick            # Skip slow tests
    python run_tests.py --device cpu       # Use CPU
    python run_tests.py --coverage         # Generate coverage report
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
import os
import argparse
import subprocess
from typing import List, Optional


def run_test_file(test_file: str, verbose: bool = False, device: str = 'cuda') -> int:
    """
    Run a specific test file.

    Args:
        test_file: Name of test file (without .py extension)
        verbose: Enable verbose output
        device: Device to use (cuda/cpu)

    Returns:
        Return code from test execution
    """
    test_dir: str = os.path.dirname(os.path.abspath(__file__))
    cmd: List[str] = [sys.executable, f"{test_file}.py"]

    if verbose:
        cmd.append('--verbose')

    if device and test_file in ['test_tqf_ann', 'test_performance']:
        cmd.extend(['--device', device])

    print(f"\n{'='*80}")
    print(f"RUNNING: {test_file}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, cwd=test_dir)
    return result.returncode


def run_all_tests(verbose: bool = False, quick: bool = False, device: str = 'cuda') -> None:
    """
    Run all test suites.

    Args:
        verbose: Enable verbose output
        quick: Skip slow performance tests
        device: Device to use
    """
    test_files: List[str] = [
        'test_cli',
        'test_config',
        'test_datasets',
        'test_dual_metrics',
        'test_engine',
        'test_evaluation',
        'test_logging_utils',
        'test_main',
        'test_output_formatters'
        'test_param_matcher',
        'test_tqf_ann',
        'test_verification_features',
    ]

    if not quick:
        test_files.append('test_performance')

    failed_tests: List[str] = []
    passed_tests: List[str] = []

    for test_file in test_files:
        return_code: int = run_test_file(test_file, verbose, device)
        if return_code != 0:
            failed_tests.append(test_file)
        else:
            passed_tests.append(test_file)

    print(f"\n{'='*80}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(test_files)} | Passed: {len(passed_tests)} | Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for test_file in failed_tests:
            print(f"  - {test_file}")
        print(f"{'='*80}\n")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED!")
        print(f"{'='*80}\n")
        sys.exit(0)


def run_with_coverage(verbose: bool = False, device: str = 'cuda') -> None:
    """Run tests with coverage analysis."""
    try:
        import coverage
    except ImportError:
        print("ERROR: coverage package not installed")
        print("Install with: pip install coverage")
        sys.exit(1)

    print("Running tests with coverage analysis...")
    test_dir: str = os.path.dirname(os.path.abspath(__file__))

    cmd: List[str] = [
        sys.executable, '-m', 'coverage', 'run',
        f'--source={test_dir}', 'test_tqf_ann.py'
    ]

    if device:
        cmd.extend(['--device', device])

    result = subprocess.run(cmd, cwd=test_dir)

    if result.returncode == 0:
        print("\nGenerating coverage report...")
        subprocess.run([sys.executable, '-m', 'coverage', 'report'], cwd=test_dir)
        subprocess.run([sys.executable, '-m', 'coverage', 'html'], cwd=test_dir)
        print(f"\nHTML report: {os.path.join(test_dir, 'htmlcov', 'index.html')}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TQF-NN Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--file', type=str, help='Specific test file to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Skip slow tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    print("=" * 80)
    print("TQF-NN TEST SUITE RUNNER")
    print("=" * 80)
    print(f"Device: {args.device} | Quick: {args.quick}")
    print("=" * 80)

    if args.coverage:
        run_with_coverage(args.verbose, args.device)
    elif args.file:
        return_code: int = run_test_file(args.file, args.verbose, args.device)
        sys.exit(return_code)
    else:
        run_all_tests(args.verbose, args.quick, args.device)


if __name__ == '__main__':
    main()
