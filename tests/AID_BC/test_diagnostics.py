# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os

from datetime import datetime
import matplotlib as mpl
import numpy as np
import unittest
import torch
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.diagnostics import (
    plot_validation_pdfs,
    plot_power_spectra,
    plot_qq_quantiles,
    plot_surface,
)

# python -m unittest tests.test_diagnostics

# ---------------------------------------------
# COMPLETE MATPLOTLIB STYLE CONFIGURATION
# ---------------------------------------------
params = {
    # DPI & figure settings
    # "figure.dpi": 150,
    # "savefig.dpi": 300,
    # Fonts
    "font.family": "DejaVu Sans",
    "mathtext.rm": "arial",
    "font.size": 12,  # General font size (affects ax.text())
    "font.style": "normal",  # 'normal', 'italic', 'oblique'
    "font.weight": "normal",  # 'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'
    "font.stretch": "normal",  # Font stretch
    # Line properties
    "lines.linewidth": 2,
    "lines.dashed_pattern": [4, 2],
    "lines.dashdot_pattern": [6, 3, 2, 3],
    "lines.dotted_pattern": [2, 3],
    # Axis labels and titles
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    # Tick settings
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.fontsize": 10,
    "legend.loc": "best",
    "legend.frameon": False,
    # Text properties
    "text.color": "black",  # Default text color
    "text.usetex": False,  # LaTeX rendering
    "text.hinting": "auto",  # Text hinting
    "text.antialiased": True,  # Text anti-aliasing
    "text.latex.preamble": "",  # LaTeX preamble
}


mpl.rcParams.update(params)

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================


class PlotConfig:
    """Central configuration for all plotting functions."""

    # General settings
    DEFAULT_SAVE_DIR = "./results"
    DEFAULT_FIGSIZE_MULTIPLIER = 4

    # Color schemes
    COLORMAPS = {
        "temperature": "rainbow",
        "temp": "rainbow",
        "2t": "rainbow",
        "zonal": "BrBG_r",
        "10u": "BrBG_r",
        "meridional": "BrBG_r",
        "10v": "BrBG_r",
        "tp": "Blues",
        "TP": "Blues",
        "precipitation": "Blues",
        "dewpoint": "rainbow",
        "d2m": "rainbow",
        "surface temperature": "rainbow",
        "st": "rainbow",
        "pressure": "viridis",
        "pres": "viridis",
        "humidity": "Greens",
        "humid": "Greens",
        "wind": "coolwarm",
        "speed": "coolwarm",
        "mae": "Reds",
        "error": "Reds",
        "divergence": "seismic",
        "curl": "seismic",
        "ssr": "seismic",
        "default": "viridis",
    }

    # Fixed visualization ranges for error diagnostics
    FIXED_DIFF_RANGES = {
        "T2M": (-5.0, 5.0),  # K
        "temperature": (-5.0, 5.0),
        "2t": (-5.0, 5.0),
        "VAR_2T": (-5.0, 5.0),
        "U10": (-5.0, 5.0),  # m/s
        "10u": (-5.0, 5.0),
        "meridional": (-5.0, 5.0),
        "VAR_10U": (-5.0, 5.0),
        "V10": (-5.0, 5.0),  # m/s
        "10v": (-5.0, 5.0),
        "VAR_10V": (-5.0, 5.0),
        "TP": (-0.5, 0.5),  # mm/h
        "tp": (-0.5, 0.5),
        "VAR_TP": (-0.5, 0.5),
        "VAR_D2M": (-5.0, 5.0),  # K
        "VAR_ST": (-5.0, 5.0),  # K
    }

    # Geographic features
    COASTLINE_w = 0.5
    BORDER_w = 0.5
    LAKE_w = 0.5
    BORDER_STYLE = "--"

    # Colorbar settings
    COLORBAR_h = 0.02
    COLORBAR_PAD = 0.05

    @classmethod
    def get_colormap(cls, variable_name):
        """Get appropriate colormap for a variable."""
        var_lower = variable_name.lower()
        for key, cmap in cls.COLORMAPS.items():
            if key in var_lower:
                return cmap
        return cls.COLORMAPS["default"]

    @classmethod
    def get_plot_name(cls, variable_name):
        """Convert variable name to readable plot name."""
        # Remove common prefixes
        name = variable_name.replace("VAR_", "").replace("var_", "")

        # Special cases
        if name == "2T":
            return "Temperature [K]"
        elif name == "10U":
            return "Zonal Wind [m/s]"
        elif name == "10V":
            return "Meridional Wind [m/s]"
        elif name == "MSLP":
            return "Sea Level Pressure"
        elif name == "T2M":
            return "2m Temperature [K]"
        elif name == "U10":
            return "10m Zonal Wind [m/s]"
        elif name == "V10":
            return "10m Meridional Wind [m/s]"
        elif name == "TP":
            return "Precipitation [mm/h]"
        elif name == "tp":
            return "Precipitation [mm/h]"
        elif name == "D2M":
            return "Dewpoint [K]"
        elif name == "ST":
            return "Surface Temperature [K]"

        # General conversion
        name = name.replace("_", " ")
        return name.title()

    @classmethod
    def convert_units(cls, variable_name, data):
        """
        Safe unit conversion when required.
        - NEVER modifies input
        - Returns a new array only if conversion is needed
        """
        name = variable_name.lower()
        if name in ["tp", "var_tp", "precipitation"]:
            return data * 1000.0  # m to mm
        return data

    @staticmethod
    def get_fixed_diff_range(var_name):
        """Get fixed visualization range for signed differences (Prediction − Truth)."""
        return PlotConfig.FIXED_DIFF_RANGES.get(var_name, None)


# ============================================================================
# Plotting Functions Test Suite
# ============================================================================


class TestPlottingFunctions(unittest.TestCase):
    """Unit tests for plotting functions with visible output for styling adjustment."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = "./test_plots"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        # Generate realistic synthetic test data
        np.random.seed(42)
        self.batch_size = 50
        self.num_vars = 4
        self.h = 64
        self.w = 64

        if self.logger:
            self.logger.info(
                f"Test setup complete - output directory: {self.output_dir}"
            )
            self.logger.info(
                f"Batch size: {self.batch_size}, Variables: {self.num_vars}, Resolution: {self.h}x{self.w}"
            )

        # Create correlated data for realistic plots
        x = np.linspace(0, 4 * np.pi, self.w)
        y = np.linspace(0, 4 * np.pi, self.h)
        X, Y = np.meshgrid(x, y)

        patterns = [
            np.sin(X) * np.cos(Y),
            np.exp(-0.1 * (X - 10) ** 2 - 0.1 * (Y - 10) ** 2),
            X * Y / 100,
            np.sin(0.5 * X) * np.cos(0.5 * Y) + 0.5 * np.sin(2 * X) * np.cos(2 * Y),
        ]

        self.predictions = np.zeros((self.batch_size, self.num_vars, self.h, self.w))
        self.targets = np.zeros((self.batch_size, self.num_vars, self.h, self.w))
        self.coarse_inputs = np.zeros((self.batch_size, self.num_vars, self.h, self.w))

        for i in range(self.num_vars):
            base_pattern = patterns[i % len(patterns)]
            for b in range(self.batch_size):
                noise_pred = np.random.normal(0, 0.1, (self.h, self.w))
                noise_target = np.random.normal(0, 0.1, (self.h, self.w))
                noise_coarse = np.random.normal(0, 0.2, (self.h, self.w))

                scale = 1.0 + 0.1 * np.random.random()
                offset = 0.1 * np.random.random()

                self.predictions[b, i] = base_pattern * scale + offset + noise_pred
                self.targets[b, i] = (
                    base_pattern * (scale + 0.05) + offset + 0.05 + noise_target
                )
                self.coarse_inputs[b, i] = (
                    base_pattern * (scale - 0.1) + offset - 0.1 + noise_coarse
                )

        self.variable_names = [
            "Temp",
            "Press",
            "Humid",
            "Wind",
        ]

        # Create lat/lon arrays for spatial tests
        self.lat = np.linspace(-90, 90, self.h)
        self.lon = np.linspace(-180, 180, self.w)

        # Create comprehensive metrics history
        self.valid_metrics_history = {}
        metrics = ["rmse", "mae", "r2"]

        for var in self.variable_names:
            var_key = var.split(" ")[0]
            for metric in metrics:
                base_val_pred = 0.8 if metric == "r2" else 1.0
                base_val_coarse = 0.6 if metric == "r2" else 1.5
                decay = np.linspace(0, 0.3, 10)

                if metric == "r2":
                    self.valid_metrics_history[f"{var_key}_pred_vs_fine_{metric}"] = (
                        base_val_pred + decay
                    )
                    self.valid_metrics_history[f"{var_key}_coarse_vs_fine_{metric}"] = (
                        base_val_coarse + decay * 0.5
                    )
                else:
                    self.valid_metrics_history[f"{var_key}_pred_vs_fine_{metric}"] = (
                        base_val_pred - decay
                    )
                    self.valid_metrics_history[f"{var_key}_coarse_vs_fine_{metric}"] = (
                        base_val_coarse - decay * 0.5
                    )

        # Add average metrics
        for metric in metrics:
            self.valid_metrics_history[f"average_pred_vs_fine_{metric}"] = (
                0.1 + np.linspace(0, 0.2, 10)
            )
            self.valid_metrics_history[f"average_coarse_vs_fine_{metric}"] = (
                0.7 + np.linspace(0, 0.2, 10)
            )

        # Loss histories
        self.train_loss_history = np.exp(
            -np.linspace(0, 2, 20)
        ) + 0.1 * np.random.random(20)
        self.valid_loss_history = np.exp(
            -np.linspace(0, 1.5, 20)
        ) + 0.15 * np.random.random(20)

    # ============================================================================
    # SINGLE COMPREHENSIVE TEST FOR EACH DIAGNOSTIC METHOD
    # ============================================================================

    def test_validation_pdfs_comprehensive(self):
        """Comprehensive test for validation PDF plots."""
        if self.logger:
            self.logger.info("Testing validation PDF plots comprehensively")

        # Test 1: Standard configuration with coarse inputs
        expected_path = plot_validation_pdfs(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_pdfs_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Without coarse inputs
        expected_path = plot_validation_pdfs(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=None,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_pdfs_no_coarse.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)

        expected_path = plot_validation_pdfs(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_pdfs_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All validation PDF tests passed")

    def test_power_spectra_comprehensive(self):
        """Comprehensive test for power spectra plots."""
        if self.logger:
            self.logger.info("Testing power spectra plots comprehensively")

        dlat = np.abs(self.lat[1] - self.lat[0])
        dlon = np.abs(self.lon[1] - self.lon[0])

        # Test 1: Standard configuration
        expected_path = plot_power_spectra(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            dlat=dlat,
            dlon=dlon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="power_spectra_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Without coarse inputs
        expected_path = plot_power_spectra(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=None,
            dlat=dlat,
            dlon=dlon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="power_spectra_no_coarse.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)
        expected_path = plot_power_spectra(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            dlat=dlat,
            dlon=dlon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="power_spectra_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All power spectra tests passed")

    def test_qq_quantiles_comprehensive(self):
        """Comprehensive test for QQ-quantiles plots."""
        if self.logger:
            self.logger.info("Testing QQ-quantiles plots comprehensively")

        # Test 1: Standard configuration with all parameters
        expected_path = plot_qq_quantiles(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
            save_dir=self.output_dir,
            filename="qq_quantiles_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 4: Single variable (edge case)
        expected_path = plot_qq_quantiles(
            predictions=self.predictions[:, 0:1],  # Keep only first variable
            targets=self.targets[:, 0:1],
            coarse_inputs=self.coarse_inputs[:, 0:1],
            variable_names=["Temperature (K)"],
            quantiles=[0.90, 0.95, 0.99],
            save_dir=self.output_dir,
            filename="qq_quantiles_single_var.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 5: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)

        expected_path = plot_qq_quantiles(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
            save_dir=self.output_dir,
            filename="qq_quantiles_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All QQ-quantiles tests passed")

    def test_plot_surface_comprehensive(self):
        """Comprehensive test for surface plots."""
        if self.logger:
            self.logger.info("Testing surface plots comprehensively")

        # Test case 1: Standard configuration
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Create synthetic data
        batch_size = 1
        n_vars = 3
        h, w = 48, 68

        # Create spatial patterns
        x = np.linspace(0, 3 * np.pi, w)
        y = np.linspace(0, 3 * np.pi, h)
        X, Y = np.meshgrid(x, y)

        # Initialize arrays
        coarse_inputs = np.zeros((batch_size, n_vars, h, w))
        targets = np.zeros((batch_size, n_vars, h, w))
        pred = np.zeros((batch_size, n_vars, h, w))

        base_patterns = [
            np.sin(X / 2) * np.cos(Y / 2),
            np.exp(-0.01 * (X - 24) ** 2 - 0.01 * (Y - 24) ** 2),
            X * Y / 200,
        ]

        for i in range(n_vars):
            base_pattern = base_patterns[i % len(base_patterns)]
            pattern = base_pattern * 20 + 280  # Temperature-like

            coarse_inputs[0, i] = pattern + np.random.randn(h, w) * 2
            targets[0, i] = pattern + np.random.randn(h, w) * 1
            pred[0, i] = targets[0, i] + np.random.randn(h, w) * 0.3

        variable_names = ["Temp", "Press", "Humid"]
        timestamp = datetime(2024, 1, 1, 12, 0)

        # Test with numpy arrays
        expected_path = plot_surface(
            coarse_inputs=coarse_inputs,
            targets=targets,
            predictions=pred,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_surface_standard.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test with PyTorch tensors
        coarse_inputs_tensor = torch.from_numpy(coarse_inputs.copy())
        targets_tensor = torch.from_numpy(targets.copy())
        pred_tensor = torch.from_numpy(pred.copy())

        expected_path = plot_surface(
            coarse_inputs=coarse_inputs_tensor,
            targets=targets_tensor,
            predictions=pred_tensor,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_surface_torch.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All surface plot tests passed")


def run_tests():
    """Run all plotting tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPlottingFunctions))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
