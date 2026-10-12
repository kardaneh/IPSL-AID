# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/


"""
Global test runner for IPSL-AID and AID-BC.

Examples
--------
Run all IPSL-AID and AID-BC tests:
    $ python -m tests.test_all

Run only the IPSL-AID test suite:
    $ python -m tests.IPSL_AID.test_runner

Run only the AID-BC test suite:
    $ python -m tests.AID_BC.test_runner
"""

import subprocess
import sys
from datetime import datetime

from IPSL_AID.logger import Logger

TEST_SUITES = [
    ("IPSL_AID", "tests.IPSL_AID.test_runner"),
    ("AID_BC", "tests.AID_BC.test_runner"),
]


class GlobalTestRunner:
    """Run all IPSL-AID and AID-BC test suites."""

    def __init__(
        self,
        console_output=True,
        file_output=False,
        log_file="test_results.log",
    ):
        self.logger = Logger(
            console_output=console_output,
            file_output=file_output,
            log_file=log_file,
            pretty_print=True,
            record=False,
        )

    def run_tests(self) -> bool:
        """Run all registered test suites."""

        self.logger.show_header("IPSL-AID Global Test Suite")

        self.logger.start_task(
            "Running Test Suites",
            description="Running IPSL_AID and AID_BC test suites",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        passed_suites = []
        failed_suites = []

        # Run each package-specific test suite sequentially
        for name, module in TEST_SUITES:
            self.logger.info(f"▶ Starting {name} test suite")

            result = subprocess.run(
                [sys.executable, "-m", module],
                check=False,
            )

            if result.returncode == 0:
                passed_suites.append(name)
                self.logger.success(f"✓ {name} test suite passed")
            else:
                failed_suites.append(name)
                self.logger.error(f"✗ {name} test suite failed")

        self.logger.start_task(
            "Global Test Summary",
            description="All test suites completed",
            total=len(TEST_SUITES),
            passed=len(passed_suites),
            failures=len(failed_suites),
            errors=0,
            success_rate=f"{(len(passed_suites) / len(TEST_SUITES) * 100):.1f}%",
        )

        if failed_suites:
            self.logger.error(f"❌ Failed test suites: {', '.join(failed_suites)}")
            return False

        self.logger.success("✅ All IPSL_AID and AID_BC test suites passed!")
        return True


def main():
    """Main entry point."""

    runner = GlobalTestRunner()
    success = runner.run_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
