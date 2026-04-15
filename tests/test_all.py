# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Test runner for IPSL_AID package with dynamic test discovery.

This module provides functionality to dynamically discover and run
unittest-based tests across the IPSL_AID package. It supports running
all tests or specific modules, and integrates with a custom Logger
for consistent output formatting.

Examples
--------
Run all tests in the package:
    $ python -m tests.test_all

Run tests from specific modules:
    $ python -m unittest tests.test_utils

Run specific test:
    $ python -m unittest tests.test_utils.TestEasyDict.test_empty_initialization

"""

import sys
import os
import unittest
from datetime import datetime
import argparse
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.logger import Logger


class TestRunner:
    """
    Test runner that uses the custom Logger for beautiful output.

    Parameters
    ----------
    console_output : bool, optional
        Whether to output to console. Default is True.
    file_output : bool, optional
        Whether to output to file. Default is True.
    log_file : str, optional
        Path to log file. Default is "test_results.log".

    Examples
    --------
    >>> runner = TestRunner()
    >>> runner.run_tests()
    """

    def __init__(
        self,
        console_output=True,
        file_output=False,
        log_file="test_results.log",
    ):
        self.console_output = console_output
        self.file_output = file_output
        self.log_file = log_file
        self.logger = Logger(
            console_output=console_output,
            file_output=file_output,
            log_file=log_file,
            pretty_print=True,
            record=False,
        )

    def run_tests(self, test_pattern: str = "test_*.py") -> bool:
        """
        Run all tests matching the pattern.

        Parameters
        ----------
        test_pattern : str, optional
            Pattern to match test files. Default is "test_*.py".

        Returns
        -------
        bool
            True if all tests passed, False otherwise.
        """

        self.logger.show_header("IPSL-AID Test Suite")

        # Start task
        self.logger.start_task(
            "Running Tests",
            description=f"Running tests matching pattern: {test_pattern}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Discover tests
        self.logger.info("Discovering test files...")
        start_dir = os.path.dirname(__file__)
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir, pattern=test_pattern)

        # Count tests
        test_count = suite.countTestCases()
        self.logger.info(f"Found {test_count} test cases")

        # # Create a custom test result class that uses our logger
        class LoggerTestResult(unittest.TextTestResult):
            def __init__(self, *args, logger=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.logger = logger
                self.current_test = None

            def startTest(self, test):
                super().startTest(test)
                self.logger.info(f"▶ Running: {test._testMethodName}")

            def addSuccess(self, test):
                super().addSuccess(test)
                self.current_test = test
                self.logger.success(f"✓ {test._testMethodName} passed")

            def addError(self, test, err):
                super().addError(test, err)
                self.logger.error(f"✗ {test._testMethodName} failed with error", err[1])

            def addFailure(self, test, err):
                super().addFailure(test, err)
                self.logger.error(
                    f"✗ {test._testMethodName} failed with failure", err[1]
                )

        # # Create a stream for the test runner
        stream = io.StringIO()

        # Create runner with custom result class
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=0,
            resultclass=lambda *args, **kwargs: LoggerTestResult(
                *args, logger=self.logger, **kwargs
            ),
        )

        # Run tests
        self.logger.info("Starting test execution...")
        result = runner.run(suite)

        # # Print summary
        passed = result.testsRun - len(result.failures) - len(result.errors)
        self.logger.start_task(
            "Test Summary",
            description="Test execution completed",
            total=result.testsRun,
            passed=passed,
            failures=len(result.failures),
            errors=len(result.errors),
            success_rate=f"{(passed / result.testsRun * 100):.1f}%",
        )

        # Log detailed results if there were failures
        if result.failures or result.errors:
            self.logger.warning("Detailed failure information:")
            for test, traceback in result.failures:
                self.logger.error(f"Failure in {test._testMethodName}:")
                self.logger.info(traceback)
            for test, traceback in result.errors:
                self.logger.error(f"Error in {test._testMethodName}:")
                self.logger.info(traceback)

        # Final result
        if result.wasSuccessful():
            self.logger.success("✅ All tests passed!")
            return True
        else:
            self.logger.error(
                f"❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors"
            )
            return False


def main():
    """Main entry point for running tests."""

    parser = argparse.ArgumentParser(
        description="Run IPSL-AID tests with custom logger"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="test_*.py",
        help="Pattern to match test files (default: test_*.py)",
    )

    parser.add_argument(
        "--no-console", action="store_true", help="Disable console output"
    )
    parser.add_argument("--no-file", action="store_true", help="Disable file output")

    parser.add_argument(
        "--log-file",
        type=str,
        default="test_results.log",
        help="Path to log file (default: test_results.log)",
    )

    args = parser.parse_args()

    runner = TestRunner(
        console_output=not args.no_console,
        file_output=not args.no_file,
        log_file=args.log_file,
    )

    success = runner.run_tests(test_pattern=args.pattern)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
