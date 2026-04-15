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
