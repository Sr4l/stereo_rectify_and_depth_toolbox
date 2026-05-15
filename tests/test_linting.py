#!/usr/bin/env python3
"""
Linting test suite for Stereo Camera Calibration & Depth Toolbox.
Runs ruff over the codebase to enforce code quality standards.
"""

import subprocess
import sys
import os


def get_python_files():
    """Get list of Python files to lint, excluding third-party/submodule code."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exclude_dirs = {
        "RAFT-Stereo",       # External submodule (Princeton VL)
        "__pycache__",       # Python cache
        "venv",              # Virtual environment
        ".venv",             # Virtual environment
        ".git",              # Git directory
        "models",            # Model weights
        "examples",          # Example data
        "asstes",            # Assets (note: typo in original name)
    }

    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for f in files:
            if f.endswith(".py"):
                python_files.append(os.path.join(root, f))

    return python_files


def test_ruff_linting():
    """Run ruff check over the codebase."""
    print("Testing ruff linting...")

    # Check if ruff is available
    try:
        result = subprocess.run(
            ["ruff", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print("  ✗ Ruff is not installed. Add 'ruff' to requirements.txt and run 'pip install -r requirements.txt'")
            return False
        version_line = result.stdout.strip()
        print(f"  Ruff version: {version_line}")
    except FileNotFoundError:
        print("  ✗ Ruff command not found. Add 'ruff' to requirements.txt and run 'pip install -r requirements.txt'")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ Ruff command timed out")
        return False

    # Get Python files to check
    python_files = get_python_files()
    if not python_files:
        print("  ⚠ No Python files found to lint")
        return True

    print(f"  Checking {len(python_files)} Python files...")

    # Run ruff check
    result = subprocess.run(
        ["ruff", "check"] + python_files,
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        print("  ✗ Ruff linting failed")
        return False

    print("  ✓ Ruff linting passed")
    return True


def test_ruff_format_check():
    """Run ruff format --check to verify formatting."""
    print("Testing ruff format check...")

    # Check if ruff is available
    try:
        result = subprocess.run(
            ["ruff", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print("  ⚠ Ruff not installed, skipping format check")
            return True  # Skip rather than fail
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ⚠ Ruff not available, skipping format check")
        return True

    # Get Python files to check
    python_files = get_python_files()
    if not python_files:
        print("  ⚠ No Python files found to check")
        return True

    # Run ruff format check
    result = subprocess.run(
        ["ruff", "format", "--check"] + python_files,
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        print("  ✓ Ruff format check passed")
        return True
    else:
        # Format check returns non-zero when files would be reformatted
        if result.stdout:
            print(result.stdout)
        print("  ✗ Ruff format check failed (run 'ruff format . --diff' to see changes)")
        return False


def run_all_linting_tests():
    """Run all linting tests."""
    print("=" * 60)
    print("Linting Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_ruff_linting,
        test_ruff_format_check,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_linting_tests()
    sys.exit(0 if success else 1)