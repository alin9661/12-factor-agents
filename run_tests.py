#!/usr/bin/env python3
"""
Test runner script for the 12-factor agents test suite.

This script provides convenient commands to run different categories of tests
and generate appropriate reports.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="12-Factor Agents Test Runner")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "security", "e2e", "performance", "fast", "ci"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--html-report",
        action="store_true", 
        help="Generate HTML test report"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-branch"
        ])
    
    # Add HTML report
    if args.html_report:
        cmd.extend(["--html=reports/test_report.html", "--self-contained-html"])
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
    
    # Select tests based on type
    if args.test_type == "all":
        cmd.append("tests/")
    elif args.test_type == "unit":
        cmd.extend(["-m", "unit", "tests/unit/"])
    elif args.test_type == "integration":
        cmd.extend(["-m", "integration", "tests/integration/"])
    elif args.test_type == "security":
        cmd.extend(["-m", "security", "tests/security/"])
    elif args.test_type == "e2e":
        cmd.extend(["-m", "e2e", "tests/e2e/"])
    elif args.test_type == "performance":
        cmd.extend(["-m", "performance", "tests/performance/"])
    elif args.test_type == "fast":
        cmd.extend(["-m", "not slow and not performance", "tests/"])
    elif args.test_type == "ci":
        # CI-friendly test run (excludes slow and performance tests)
        cmd.extend([
            "-m", "not slow and not performance",
            "--maxfail=5",
            "--tb=short",
            "tests/"
        ])
    
    # Run the tests
    exit_code = run_command(cmd, f"{args.test_type.upper()} tests")
    
    if exit_code == 0:
        print(f"\n✅ {args.test_type.upper()} tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
        
        if args.html_report:
            print("\n📋 Test report generated in reports/test_report.html")
    else:
        print(f"\n❌ {args.test_type.upper()} tests failed!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())