"""Test runner for VODE test suite.

Discovers and runs all tests in the vode/tests/ directory.
Provides clear output showing pass/fail status.

Usage:
    python run_tests.py
    python run_tests.py -v  # Verbose output
"""

import sys
import pytest
from pathlib import Path


def main():
    """Run all tests in the tests directory."""
    # Get the directory containing this script (tests directory)
    tests_dir = Path(__file__).parent

    print("=" * 70)
    print("VODE Test Suite")
    print("=" * 70)
    print(f"Test directory: {tests_dir}")
    print()

    # Configure pytest arguments
    pytest_args = [
        str(tests_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        "-ra",  # Show summary of all test outcomes
    ]

    # Add any command-line arguments
    if len(sys.argv) > 1:
        pytest_args.extend(sys.argv[1:])

    print("Running tests...")
    print("-" * 70)
    print()

    # Run pytest
    exit_code = pytest.main(pytest_args)

    print()
    print("=" * 70)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
