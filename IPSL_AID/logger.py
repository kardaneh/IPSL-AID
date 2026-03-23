# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import logging
import traceback
from datetime import datetime
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
import unittest
import os
import tempfile
import shutil


class Logger:
    def __init__(
        self,
        console_output=True,
        file_output=False,
        log_file="module_log_file.log",
        pretty_print=True,
        record=False,
    ):
        self.console_output = console_output
        self.file_output = file_output
        self.log_file = log_file
        self.pretty_print = pretty_print
        self.record = record

        self.console = Console(record=self.record)
        self.logger = logging.getLogger("ModuleLogger")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Plain text handler for file output only (no RichHandler for console)
        if self.file_output:
            file_handler = logging.FileHandler(
                self.log_file, mode="w", encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            # Use a simple formatter for file output
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
            transient=True,
        )

        self.metrics = {
            "start_time": None,
            "end_time": None,
            "node_sequence": [],
            "steps_completed": 0,
            "node_count": {},
        }

    def clear_logs(self):
        """Clear the stored Rich logs if record=True."""
        if self.record and hasattr(self, "console") and self.console:
            self.console.clear()

    def show_header(self, module_name):
        """Display startup banner."""
        self.module_name = module_name
        if self.console_output:
            self.console.print(
                Panel(
                    f"[bold red]🚀 Starting Module:[/bold red] [cyan]{self.module_name}[/cyan]",
                    title="IPSL AI Downscaling Tool",
                    border_style="bright_blue",
                )
            )
        # Also log to file
        if self.file_output:
            self.logger.info(f"🚀 Starting Module: {self.module_name}")

    def start_task(self, task_name: str, description: str = "", **meta):
        """Display a clearly formatted 'task start' message with good spacing."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Construct sections with spacing between them
        header = Text(f"🚀 {task_name}", style="bold cyan")
        desc = Text(f"📝 {description}", style="yellow") if description else None
        time_text = Text(f"🕒 {timestamp}", style="dim")

        meta_lines = []
        for key, value in meta.items():
            meta_lines.append(f"🔹 [white]{key.upper()}:[/white] {value}")

        components = [header]
        if desc:
            components.append(desc)
        components.append(Text(""))  # blank line
        components.append(time_text)
        components.append(Text(""))  # blank line
        if meta_lines:
            components.extend(Text.from_markup(line) for line in meta_lines)
            components.append(Text(""))

        content = Group(*components)

        if self.console_output:
            self.console.print(
                Panel(
                    content,
                    title="[bold red]TASK STARTED[/bold red]",
                    border_style="red",
                    expand=False,
                    padding=(1, 4),  # (top-bottom, left-right)
                )
            )

        # Log to file
        if self.file_output:
            meta_str = ", ".join([f"{k.upper()}: {v}" for k, v in meta.items()])
            self.logger.info(f"TASK STARTED: {task_name} - {description} - {meta_str}")

    def log_metrics(self):
        """Log pipeline metrics"""
        if self.console_output:
            table = Table(title="📊 Pipeline Metrics", show_lines=True)
            table.add_column("Node", style="cyan")
            table.add_column("Count", justify="center")
            table.add_column("Total Time (s)", justify="right")

            for node, count in self.metrics["node_count"].items():
                total_time = self.metrics["node_times"].get(node, 0)
                table.add_row(node, str(count), f"{total_time:.2f}")
            table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{sum(self.metrics['node_count'].values())}[/bold]",
                f"[bold]{sum(self.metrics['node_times'].values()):.2f}[/bold]",
            )

            self.console.print(
                Panel(table, title="Metrics Summary", border_style="bright_blue")
            )

        # Log metrics to file
        if self.file_output:
            self.logger.info("Pipeline Metrics Summary:")
            for node, count in self.metrics["node_count"].items():
                total_time = self.metrics["node_times"].get(node, 0)
                self.logger.info(
                    f"  {node}: Count={count}, Total Time={total_time:.2f}s"
                )
            self.logger.info(
                f"  Total: Count={sum(self.metrics['node_count'].values())}, Total Time={sum(self.metrics['node_times'].values()):.2f}s"
            )

    def info(self, message):
        """Formatted info message"""
        if self.console_output:
            self.console.print(f"[bold cyan][INFO][/bold cyan] {message}")
        if self.file_output:
            self.logger.info(message)

    def warning(self, message):
        """Formatted warning message"""
        if self.console_output:
            self.console.print(
                f"[bold yellow][WARNING][/bold yellow] :warning: {message}"
            )
        if self.file_output:
            self.logger.warning(message)

    def success(self, message):
        """Custom success level (not default logging level)"""
        if self.console_output:
            self.console.print(f":white_check_mark: [bold green]{message}[/bold green]")
        if self.file_output:
            self.logger.info(f"SUCCESS: {message}")

    def step(self, step_name, message):
        """Highlight pipeline step events"""
        if self.console_output:
            self.console.print(
                f"[bold magenta]▶ Step: {step_name}[/bold magenta] — {message}"
            )
        if self.file_output:
            self.logger.info(f"Step: {step_name} - {message}")

    def _format_traceback_panels(self, exception: Exception):
        """Format traceback as a series of Rich panels for readability."""
        tb = exception.__traceback__
        extracted_tb = traceback.extract_tb(tb)
        panels = []

        for i, frame in enumerate(extracted_tb):
            file_name = frame.filename
            line_no = frame.lineno
            func_name = frame.name
            code_line = (frame.line or "").strip()

            # Create base Text block (no markup parsing)
            header = Text()
            header.append(f"File: {file_name}\n", style="bold cyan")
            header.append(f"Line: {line_no} | Function: {func_name}\n", style="dim")

            if code_line:
                # Use from_markup for the highlighted code
                code_text = Text.from_markup(
                    f"Code: [italic yellow]{code_line}[/italic yellow]"
                )
                header.append(code_text)

            frame_panel = Panel(
                header,
                title=f"[Frame {i + 1}]",
                border_style="bright_blue",
                expand=False,
            )
            panels.append(frame_panel)

        exception_info = Panel(
            Text.from_markup(
                f"[bold red]{type(exception).__name__}[/bold red]: {exception}"
            ),
            title="[bold red]Exception Raised[/bold red]",
            border_style="red",
        )

        return Panel(
            Group(*panels, exception_info),
            title="[bold red]Traceback[/bold red]",
            border_style="red",
            expand=False,
        )

    def exception(self, message, exception=None):
        """Display a formatted exception message with visual stack trace."""
        if exception:
            if self.file_output:
                self.logger.error(f"{message} - {exception}")
            if self.console_output:
                tb_panels = self._format_traceback_panels(exception)
                main_panel = Panel(
                    Group(
                        Text.from_markup(f"[bold red]{message}[/bold red]\n"), tb_panels
                    ),
                    title="[bold red]EXCEPTION[/bold red]",
                    border_style="red",
                )
                self.console.print(main_panel)
        else:
            if self.file_output:
                self.logger.error(message)
            if self.console_output:
                self.console.print(
                    Panel(
                        f"[bold red]{message}[/bold red]",
                        title="[bold red]EXCEPTION[/bold red]",
                        border_style="red",
                    )
                )

    def error(self, message, exception=None):
        """Display a formatted error log, optionally including exception trace."""
        if exception:
            if self.file_output:
                self.logger.error(f"{message} - {exception}")
            if self.console_output:
                tb = traceback.format_exc()
                self.console.print(
                    Panel(
                        f"[bold red]{message}[/bold red]\n\n"
                        f"[red]Error:[/red] [bold]{type(exception).__name__}[/bold]: {str(exception)}\n\n"
                        f"[dim]{tb}[/dim]",
                        title="[bold red]ERROR[/bold red]",
                        border_style="red",
                    )
                )
        else:
            if self.file_output:
                self.logger.error(message)
            if self.console_output:
                self.console.print(
                    Panel(
                        f"[bold red]{message}[/bold red]",
                        title="[bold red]ERROR[/bold red]",
                        border_style="red",
                    )
                )


# ============================================================================
# Unit Tests for Logger
# ============================================================================


class TestLogger(unittest.TestCase):
    """Unit tests for Logger class."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.test_logger = logger

    def setUp(self):
        """Set up test fixtures."""
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
