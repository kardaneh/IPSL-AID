# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import logging
import unittest
import os
import tempfile
import shutil

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.logger import Logger

# ============================================================================
# Unit Tests for Logger
# ============================================================================


class TestLogger(unittest.TestCase):
    """Unit tests for Logger class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create logger
        self.test_logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_log.log")

        if self.test_logger:
            self.test_logger.info(
                f"Test setup - created temp directory: {self.temp_dir}"
            )

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_initialization_default(self):
        """Test Logger initialization with default parameters."""
        if self.test_logger:
            self.test_logger.info("Testing Logger initialization with defaults")

        logger = Logger()

        self.assertTrue(logger.console_output)
        self.assertFalse(logger.file_output)
        self.assertEqual(logger.log_file, "module_log_file.log")
        self.assertTrue(logger.pretty_print)
        self.assertFalse(logger.record)
        self.assertIsNotNone(logger.console)
        self.assertIsNotNone(logger.logger)
        self.assertEqual(logger.logger.level, logging.INFO)
        self.assertFalse(logger.logger.propagate)
        self.assertIsNotNone(logger.progress)
        self.assertIn("start_time", logger.metrics)
        self.assertIn("end_time", logger.metrics)
        self.assertIn("node_sequence", logger.metrics)
        self.assertIn("steps_completed", logger.metrics)
        self.assertIn("node_count", logger.metrics)

        if self.test_logger:
            self.test_logger.info("✅ initialization default test passed")

    def test_initialization_with_file_output(self):
        """Test Logger initialization with file output enabled."""
        if self.test_logger:
            self.test_logger.info("Testing Logger initialization with file output")

        logger = Logger(
            console_output=True,
            file_output=True,
            log_file=self.log_file,
            pretty_print=True,
            record=False,
        )

        self.assertTrue(logger.file_output)
        self.assertEqual(logger.log_file, self.log_file)
        self.assertTrue(os.path.exists(os.path.dirname(self.log_file)))

        # Check that file handler was added
        file_handlers = [
            h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(file_handlers), 1)

        if self.test_logger:
            self.test_logger.info("✅ initialization with file output test passed")

    def test_initialization_with_record(self):
        """Test Logger initialization with record mode enabled."""
        if self.test_logger:
            self.test_logger.info("Testing Logger initialization with record mode")

        logger = Logger(record=True)

        self.assertTrue(logger.record)
        self.assertTrue(hasattr(logger.console, "record"))
        self.assertTrue(logger.console.record)

        if self.test_logger:
            self.test_logger.info("✅ initialization with record test passed")

    def test_initialization_console_only(self):
        """Test Logger initialization with console only."""
        if self.test_logger:
            self.test_logger.info("Testing Logger initialization console only")

        logger = Logger(console_output=True, file_output=False)

        self.assertTrue(logger.console_output)
        self.assertFalse(logger.file_output)
        self.assertEqual(len(logger.logger.handlers), 0)  # No file handlers

        if self.test_logger:
            self.test_logger.info("✅ initialization console only test passed")

    def test_initialization_file_only(self):
        """Test Logger initialization with file only."""
        if self.test_logger:
            self.test_logger.info("Testing Logger initialization file only")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        self.assertFalse(logger.console_output)
        self.assertTrue(logger.file_output)

        file_handlers = [
            h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(file_handlers), 1)

        if self.test_logger:
            self.test_logger.info("✅ initialization file only test passed")

    # ------------------------------------------------------------------------
    # Clear Logs Tests
    # ------------------------------------------------------------------------

    def test_clear_logs_with_record(self):
        """Test clear_logs method when record is True."""
        if self.test_logger:
            self.test_logger.info("Testing clear_logs with record=True")

        logger = Logger(record=True)

        # Print something to console
        logger.info("Test message")

        # Clear logs
        logger.clear_logs()

        # No assertion, just verify no errors
        if self.test_logger:
            self.test_logger.info("✅ clear_logs with record test passed")

    def test_clear_logs_without_record(self):
        """Test clear_logs method when record is False."""
        if self.test_logger:
            self.test_logger.info("Testing clear_logs with record=False")

        logger = Logger(record=False)

        # Clear logs (should do nothing but not error)
        logger.clear_logs()

        if self.test_logger:
            self.test_logger.info("✅ clear_logs without record test passed")

    # ------------------------------------------------------------------------
    # Header Tests
    # ------------------------------------------------------------------------

    def test_show_header_console(self):
        """Test show_header method with console output."""
        if self.test_logger:
            self.test_logger.info("Testing show_header with console output")

        logger = Logger(console_output=True, file_output=False)
        logger.show_header("TestModule")

        self.assertEqual(logger.module_name, "TestModule")

        if self.test_logger:
            self.test_logger.info("✅ show_header console test passed")

    def test_show_header_file(self):
        """Test show_header method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing show_header with file output")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.show_header("TestModule")

        # Check that log file was written
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("Starting Module: TestModule", content)

        if self.test_logger:
            self.test_logger.info("✅ show_header file test passed")

    def test_show_header_both(self):
        """Test show_header method with both console and file output."""
        if self.test_logger:
            self.test_logger.info("Testing show_header with both outputs")

        logger = Logger(console_output=True, file_output=True, log_file=self.log_file)

        logger.show_header("TestModule")

        self.assertTrue(os.path.exists(self.log_file))

        if self.test_logger:
            self.test_logger.info("✅ show_header both test passed")

    # ------------------------------------------------------------------------
    # Start Task Tests
    # ------------------------------------------------------------------------

    def test_start_task_minimal(self):
        """Test start_task method with minimal parameters."""
        if self.test_logger:
            self.test_logger.info("Testing start_task with minimal parameters")

        logger = Logger(console_output=True, file_output=False)
        logger.start_task("TestTask")

        if self.test_logger:
            self.test_logger.info("✅ start_task minimal test passed")

    def test_start_task_with_description(self):
        """Test start_task method with description."""
        if self.test_logger:
            self.test_logger.info("Testing start_task with description")

        logger = Logger(console_output=True, file_output=False)
        logger.start_task("TestTask", description="This is a test task")

        if self.test_logger:
            self.test_logger.info("✅ start_task with description test passed")

    def test_start_task_with_metadata(self):
        """Test start_task method with metadata."""
        if self.test_logger:
            self.test_logger.info("Testing start_task with metadata")

        logger = Logger(console_output=True, file_output=False)
        logger.start_task(
            "TestTask",
            description="Test task",
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
        )

        if self.test_logger:
            self.test_logger.info("✅ start_task with metadata test passed")

    def test_start_task_file_output(self):
        """Test start_task method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing start_task with file output")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.start_task(
            "TestTask", description="File output test", param1="value1", param2=42
        )

        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("TASK STARTED: TestTask", content)
            self.assertIn("File output test", content)

        if self.test_logger:
            self.test_logger.info("✅ start_task file output test passed")

    # ------------------------------------------------------------------------
    # Logging Level Tests
    # ------------------------------------------------------------------------

    def test_info_console(self):
        """Test info method with console output."""
        if self.test_logger:
            self.test_logger.info("Testing info method with console")

        logger = Logger(console_output=True, file_output=False)
        logger.info("This is an info message")

        if self.test_logger:
            self.test_logger.info("✅ info console test passed")

    def test_info_file(self):
        """Test info method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing info method with file")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.info("File info message")

        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("File info message", content)

        if self.test_logger:
            self.test_logger.info("✅ info file test passed")

    def test_warning_console(self):
        """Test warning method with console output."""
        if self.test_logger:
            self.test_logger.info("Testing warning method with console")

        logger = Logger(console_output=True, file_output=False)
        logger.warning("This is a warning message")

        if self.test_logger:
            self.test_logger.info("✅ warning console test passed")

    def test_warning_file(self):
        """Test warning method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing warning method with file")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.warning("File warning message")

        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("WARNING", content)
            self.assertIn("File warning message", content)

        if self.test_logger:
            self.test_logger.info("✅ warning file test passed")

    def test_success_console(self):
        """Test success method with console output."""
        if self.test_logger:
            self.test_logger.info("Testing success method with console")

        logger = Logger(console_output=True, file_output=False)
        logger.success("Operation completed successfully")

        if self.test_logger:
            self.test_logger.info("✅ success console test passed")

    def test_success_file(self):
        """Test success method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing success method with file")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.success("File success message")

        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("SUCCESS: File success message", content)

        if self.test_logger:
            self.test_logger.info("✅ success file test passed")

    def test_step_console(self):
        """Test step method with console output."""
        if self.test_logger:
            self.test_logger.info("Testing step method with console")

        logger = Logger(console_output=True, file_output=False)
        logger.step("DataLoading", "Loading training data")

        if self.test_logger:
            self.test_logger.info("✅ step console test passed")

    def test_step_file(self):
        """Test step method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing step method with file")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.step("Preprocessing", "Normalizing images")

        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("Step: Preprocessing - Normalizing images", content)

        if self.test_logger:
            self.test_logger.info("✅ step file test passed")

    # ------------------------------------------------------------------------
    # Error and Exception Tests
    # ------------------------------------------------------------------------

    def test_error_without_exception(self):
        """Test error method without exception."""
        if self.test_logger:
            self.test_logger.info("Testing error method without exception")

        logger = Logger(console_output=True, file_output=False)
        logger.error("An error occurred")

        if self.test_logger:
            self.test_logger.info("✅ error without exception test passed")

    def test_error_with_exception(self):
        """Test error method with exception."""
        if self.test_logger:
            self.test_logger.info("Testing error method with exception")

        logger = Logger(console_output=True, file_output=False)

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error("An error occurred during processing", e)

        if self.test_logger:
            self.test_logger.info("✅ error with exception test passed")

    def test_error_file_output(self):
        """Test error method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing error method with file output")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.error("File error message")

        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("ERROR", content)
            self.assertIn("File error message", content)

        if self.test_logger:
            self.test_logger.info("✅ error file output test passed")

    def test_exception_without_exception(self):
        """Test exception method without exception object."""
        if self.test_logger:
            self.test_logger.info("Testing exception method without exception")

        logger = Logger(console_output=True, file_output=False)
        logger.exception("An exception occurred")

        if self.test_logger:
            self.test_logger.info("✅ exception without exception test passed")

    def test_exception_with_exception(self):
        """Test exception method with exception object."""
        if self.test_logger:
            self.test_logger.info("Testing exception method with exception")

        logger = Logger(console_output=True, file_output=False)

        try:
            x = 1 / 0  # noqa: F841
        except ZeroDivisionError as e:
            logger.exception("Division by zero error", e)

        if self.test_logger:
            self.test_logger.info("✅ exception with exception test passed")

    def test_exception_file_output(self):
        """Test exception method with file output."""
        if self.test_logger:
            self.test_logger.info("Testing exception method with file output")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        try:
            raise KeyError("Missing key")
        except KeyError as e:
            logger.exception("Key error in dictionary", e)

        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("ERROR", content)
            self.assertIn("Key error in dictionary", content)

        if self.test_logger:
            self.test_logger.info("✅ exception file output test passed")

    # ------------------------------------------------------------------------
    # Metrics Tests
    # ------------------------------------------------------------------------

    def test_log_metrics_empty(self):
        """Test log_metrics with empty metrics."""
        if self.test_logger:
            self.test_logger.info("Testing log_metrics with empty metrics")

        logger = Logger(console_output=True, file_output=False)
        logger.metrics["node_count"] = {}
        logger.metrics["node_times"] = {}

        logger.log_metrics()

        if self.test_logger:
            self.test_logger.info("✅ log_metrics empty test passed")

    def test_log_metrics_with_data(self):
        """Test log_metrics with populated metrics."""
        if self.test_logger:
            self.test_logger.info("Testing log_metrics with data")

        logger = Logger(console_output=True, file_output=False)

        # Add some metrics
        logger.metrics["node_count"] = {
            "preprocessing": 10,
            "training": 5,
            "validation": 5,
        }
        logger.metrics["node_times"] = {
            "preprocessing": 12.5,
            "training": 45.2,
            "validation": 8.3,
        }

        logger.log_metrics()

        if self.test_logger:
            self.test_logger.info("✅ log_metrics with data test passed")

    def test_log_metrics_file_output(self):
        """Test log_metrics with file output."""
        if self.test_logger:
            self.test_logger.info("Testing log_metrics with file output")

        logger = Logger(console_output=False, file_output=True, log_file=self.log_file)

        logger.metrics["node_count"] = {"node1": 5, "node2": 3}
        logger.metrics["node_times"] = {"node1": 10.2, "node2": 7.8}

        logger.log_metrics()

        with open(self.log_file, "r") as f:
            content = f.read()
            self.assertIn("Pipeline Metrics Summary", content)
            self.assertIn("node1", content)
            self.assertIn("node2", content)

        if self.test_logger:
            self.test_logger.info("✅ log_metrics file output test passed")

    # ------------------------------------------------------------------------
    # Integration Tests
    # ------------------------------------------------------------------------

    def test_full_logging_cycle(self):
        """Test a complete logging cycle with all methods."""
        if self.test_logger:
            self.test_logger.info("Testing full logging cycle")

        logger = Logger(
            console_output=True, file_output=True, log_file=self.log_file, record=True
        )

        # Header
        logger.show_header("IntegrationTest")

        # Tasks
        logger.start_task("DataPrep", description="Preparing data", source="ERA5")

        # Steps
        logger.step("Loading", "Loading NetCDF files")
        logger.info("Loaded 10 files")

        logger.step("Processing", "Applying filters")
        logger.success("Filtering completed")

        # Warnings
        logger.warning("Some missing values detected")

        # Another task
        logger.start_task("Training", description="Model training", epochs=50)
        logger.step("Epoch 1", "Loss: 0.45")
        logger.step("Epoch 2", "Loss: 0.32")

        # Error handling
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("Training interrupted", e)

        # Metrics
        logger.metrics["node_count"] = {"prep": 1, "train": 2}
        logger.metrics["node_times"] = {"prep": 5.2, "train": 120.5}
        logger.log_metrics()

        # Verify file was written
        self.assertTrue(os.path.exists(self.log_file))

        if self.test_logger:
            self.test_logger.info("✅ full logging cycle test passed")

    def test_progress_bar(self):
        """Test progress bar functionality."""
        if self.test_logger:
            self.test_logger.info("Testing progress bar")

        logger = Logger(console_output=True, file_output=False)

        # Just verify progress object exists and can be used
        self.assertIsNotNone(logger.progress)

        # Add a task
        task_id = logger.progress.add_task("Testing", total=100)
        logger.progress.update(task_id, advance=50)

        if self.test_logger:
            self.test_logger.info("✅ progress bar test passed")

    def test_logger_with_unicode(self):
        """Test logger with unicode characters."""
        if self.test_logger:
            self.test_logger.info("Testing logger with unicode")

        logger = Logger(console_output=True, file_output=True, log_file=self.log_file)

        unicode_messages = [
            "Test with émoji 🚀",
            "Café français",
            "数据科学",
            "Γειά σου Κόσμε",
        ]

        for msg in unicode_messages:
            logger.info(msg)

        if self.test_logger:
            self.test_logger.info("✅ unicode test passed")

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        if self.test_logger:
            self.test_logger.info(
                f"Test teardown - removed temp directory: {self.temp_dir}"
            )


def run_tests():
    """Run all Logger tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test class
    suite.addTests(loader.loadTestsFromTestCase(TestLogger))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
