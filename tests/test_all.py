# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test runner for IPSL_AID package with dynamic test discovery.

This module provides functionality to dynamically discover and run
unittest-based tests across the IPSL_AID package. It supports running
all tests or specific modules, and integrates with a custom Logger
for consistent output formatting.

Examples
--------
Run all tests in the package:
    $ python test_all.py

Run tests from specific modules:
    $ python test_all.py loss network

Skip modules can be configured via the SKIP_MODULES set.
"""

import sys
import os
import unittest
import pkgutil
import importlib

# ---------------------------------------------------------------------
# Make sure IPSL_AID package is importable
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # noqa: E402

import IPSL_AID  # noqa: E402
from IPSL_AID.logger import Logger  # noqa: E402

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SKIP_MODULES = {
    "__init__",
    "main",
    "logger",
    "model_utils",
    "utils",
}


def load_module_tests(module_name, logger):
    """Load all unittest.TestCase tests from a specified module.

    Parameters
    ----------
    module_name : str
        Name of the module (without package prefix) to load tests from.
    logger : IPSL_AID.logger.Logger
        Logger instance for output messages.

    Returns
    -------
    unittest.TestSuite
        Test suite containing all discovered test cases from the module.

    Notes
    -----
    Attempts to instantiate TestCase classes with (method_name, logger)
    signature. Falls back to standard instantiation if that fails.
    Only methods starting with 'test_' are collected.
    """
    module = importlib.import_module(f"IPSL_AID.{module_name}")
    suite = unittest.TestSuite()

    for attr_name in dir(module):
        attr = getattr(module, attr_name)

        if (
            isinstance(attr, type)
            and issubclass(attr, unittest.TestCase)
            and attr is not unittest.TestCase
        ):
            test_methods = [name for name in dir(attr) if name.startswith("test_")]

            logger.info(
                f"📦 Loading {len(test_methods)} tests from "
                f"{module.__name__}.{attr.__name__}"
            )

            for method_name in test_methods:
                try:
                    suite.addTest(attr(method_name, logger))
                except TypeError:
                    # Fallback for TestCase without logger in constructor
                    suite.addTest(attr(method_name))

    return suite


def load_all_tests(logger):
    """Load tests from all eligible modules in IPSL_AID package.

    Parameters
    ----------
    logger : IPSL_AID.logger.Logger
        Logger instance for output messages.

    Returns
    -------
    unittest.TestSuite
        Combined test suite containing tests from all non-skipped modules.

    Notes
    -----
    Modules are discovered using pkgutil.iter_modules.
    Modules in SKIP_MODULES or starting with '_' are skipped.
    Package directories are also skipped.
    """
    suite = unittest.TestSuite()

    for _, module_name, is_pkg in pkgutil.iter_modules(IPSL_AID.__path__):
        if is_pkg or module_name in SKIP_MODULES or module_name.startswith("_"):
            continue

        try:
            suite.addTests(load_module_tests(module_name, logger))
        except Exception as e:
            logger.info(f"[WARN] Failed to load tests from {module_name}: {e}")

    return suite


def main():
    """Main entry point for the test runner.

    Processes command line arguments to determine which tests to run,
    executes them using unittest.TextTestRunner, and exits with
    appropriate status code.

    Returns
    -------
    None
        Exits with sys.exit(0) on success, sys.exit(1) on failure.

    Notes
    -----
    Command line behavior:
        - No arguments: Run all tests
        - With arguments: Run tests from specified modules only
    """
    logger = Logger(console_output=True, file_output=False, record=True)

    logger.show_header("IPSL-AID Test Suite")

    # -------------------------------------------------------------
    # Decide what to run
    # -------------------------------------------------------------
    if len(sys.argv) == 1:
        # Run all tests
        suite = load_all_tests(logger)
    else:
        # Run selected modules
        suite = unittest.TestSuite()
        for module_name in sys.argv[1:]:
            if module_name in SKIP_MODULES:
                logger.info(f"Skipping module '{module_name}'")
                continue
            suite.addTests(load_module_tests(module_name, logger))

    # -------------------------------------------------------------
    # Run
    # -------------------------------------------------------------
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    logger.info("-" * 70)
    if result.wasSuccessful():
        logger.info("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.info(
            f"❌ TESTS FAILED — "
            f"{len(result.failures)} failures, "
            f"{len(result.errors)} errors"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
