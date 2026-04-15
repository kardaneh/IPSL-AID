# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston, Pierre Chapel, Rosie Eade
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os

os.environ.setdefault(
    "CARTOPY_DATA_DIR",
    "/leonardo_work/EUHPC_D27_095/cartopy_data",
)

from datetime import datetime
import matplotlib as mpl
import numpy as np
import unittest
import torch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.logger import Logger
from IPSL_AID.diagnostics import (
    plot_validation_hexbin,
    plot_comparison_hexbin,
    plot_validation_pdfs,
    plot_power_spectra,
    plot_spatiotemporal_histograms,
    plot_surface,
    plot_ensemble_surface,
    plot_zoom_comparison,
    plot_global_surface_robinson,
    plot_MAE_map,
    plot_error_map,
    plot_spread_skill_ratio_map,
    plot_spread_skill_ratio_hexbin,
    plot_mean_divergence_map,
    plot_mean_curl_map,
    plot_dry_frequency_map,
    dry_frequency_map,
    get_divergence,
    get_curl,
    plot_metric_histories,
    plot_loss_histories,
    plot_average_metrics,
    plot_metrics_heatmap,
    plot_qq_quantiles,
    plot_validation_mvcorr,
    plot_validation_mvcorr_space,
    plot_temporal_series_comparison,
    ranks,
    plot_ranks,
    spread_skill_ratio,
)

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

    FIXED_DIFF_RANGES_ERRORS = {
        "VAR_2T": (0, 0.01),  # K
        "VAR_10U": (0, 3.0),  # m/s
        "VAR_10V": (0, 3.0),  # m/s
        "VAR_TP": (0, 0.5),  # mm/h
        "VAR_D2M": (0, 1.0),  # K
        "VAR_ST": (0, 1.0),  # K
        "Temp": (0, 3.0),
        "Press": (0, 3.0),
        "Humid": (0, 3.0),
        "Wind": (0, 3.0),
    }

    FIXED_MAE_RANGES = {
        "T2M": (0.0, 3.0),
        "temperature": (0.0, 3.0),
        "2t": (0.0, 3.0),
        "VAR_2T": (0.0, 3.0),
        "U10": (0.0, 3.0),
        "10u": (0.0, 3.0),
        "meridional": (0.0, 3.0),
        "VAR_10U": (0.0, 3.0),
        "V10": (0.0, 3.0),
        "10v": (0.0, 3.0),
        "VAR_10V": (0.0, 3.0),
        "TP": (0.0, 1.0),
        "tp": (0.0, 1.0),
        "VAR_TP": (0.0, 1.0),
        "VAR_D2M": (0.0, 3.0),
        "VAR_ST": (0.0, 3.0),
    }

    FIXED_SSR_RANGES = {
        "T2M": (0.0, 3.0),
        "temperature": (0.0, 3.0),
        "2t": (0.0, 3.0),
        "VAR_2T": (0.0, 3.0),
        "U10": (0.0, 3.0),
        "10u": (0.0, 3.0),
        "meridional": (0.0, 3.0),
        "VAR_10U": (0.0, 3.0),
        "V10": (0.0, 3.0),
        "10v": (0.0, 3.0),
        "VAR_10V": (0.0, 3.0),
        "TP": (0.0, 3.0),
        "tp": (0.0, 3.0),
        "VAR_TP": (0.0, 1.0),
        "VAR_D2M": (0.0, 3.0),
        "VAR_ST": (0.0, 3.0),
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

    @staticmethod
    def get_fixed_diff_range_errors(var_name):
        """Get fixed visualization range for error map."""
        return PlotConfig.FIXED_DIFF_RANGES_ERRORS.get(var_name, None)

    @staticmethod
    def get_fixed_mae_range(var_name):
        """Get fixed visualization range for Mean Absolute Error (MAE)."""
        return PlotConfig.FIXED_MAE_RANGES.get(var_name, None)

    @staticmethod
    def get_fixed_ssr_range(var_name):
        """Get fixed visualization range for Spread Skill Ratio (SSR)."""
        return PlotConfig.FIXED_SSR_RANGES.get(var_name, None)


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

    def test_validation_hexbin_comprehensive(self):
        """Comprehensive test for validation hexbin plots."""
        if self.logger:
            self.logger.info("Testing validation hexbin plots comprehensively")

        # Test 1: Standard configuration
        expected_path = plot_validation_hexbin(
            predictions=self.predictions,
            targets=self.targets,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_hexbin_standard.png",
            figsize_multiplier=5,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_comparison_hexbin(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            filename="comparison_hexbin_standard.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)

        expected_path = plot_validation_hexbin(
            predictions=predictions_tensor,
            targets=targets_tensor,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_hexbin_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_comparison_hexbin(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            filename="comparison_hexbin_torch.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Single variable
        single_pred = self.predictions[:, 0:1, :, :]
        single_target = self.targets[:, 0:1, :, :]
        single_coarse = self.coarse_inputs[:, 0:1, :, :]

        expected_path = plot_validation_hexbin(
            predictions=single_pred,
            targets=single_target,
            variable_names=[self.variable_names[0]],
            save_dir=self.output_dir,
            filename="validation_hexbin_single.png",
            figsize_multiplier=6,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_comparison_hexbin(
            predictions=single_pred,
            targets=single_target,
            coarse_inputs=single_coarse,
            variable_names=[self.variable_names[0]],
            filename="comparison_hexbin_single.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All validation hexbin tests passed")

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

    def test_spatiotemporal_histograms_comprehensive(self):
        """Comprehensive test for spatiotemporal histograms."""
        if self.logger:
            self.logger.info("Testing spatiotemporal histograms comprehensively")

        class MockSteps:
            latitude = 180
            longitude = 360

        steps = MockSteps()

        # Test 1: Dense data
        tindex_lim = (0, 365)
        # n_samples = 2000
        centers = []
        tindices = []

        clusters = [
            {
                "lat_range": (30, 60),
                "lon_range": (200, 250),
                "time_range": (0, 100),
                "n": 500,
            },
            {
                "lat_range": (10, 40),
                "lon_range": (100, 150),
                "time_range": (100, 200),
                "n": 400,
            },
            {
                "lat_range": (50, 80),
                "lon_range": (300, 350),
                "time_range": (200, 300),
                "n": 600,
            },
            {
                "lat_range": (0, 30),
                "lon_range": (50, 100),
                "time_range": (300, 365),
                "n": 500,
            },
        ]

        for cluster in clusters:
            for _ in range(cluster["n"]):
                lat = np.random.randint(
                    cluster["lat_range"][0], cluster["lat_range"][1]
                )
                lon = np.random.randint(
                    cluster["lon_range"][0], cluster["lon_range"][1]
                )
                tindex = np.random.randint(
                    cluster["time_range"][0], cluster["time_range"][1]
                )
                centers.append((lat, lon))
                tindices.append(tindex)

        expected_path = plot_spatiotemporal_histograms(
            steps=steps,
            tindex_lim=tindex_lim,
            centers=centers,
            tindices=tindices,
            mode="train",
            filename="spatiotemporal_dense_",
            save_dir=self.output_dir,
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All spatiotemporal histogram tests passed")

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

    def test_plot_ensemble_surface_comprehensive(self):
        """Comprehensive test for ensemble surface plots."""
        if self.logger:
            self.logger.info("Testing ensemble surface plots comprehensively")

        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Ensemble configuration
        N_ens = 5
        n_vars = 3
        H, W = 48, 68

        x = np.linspace(0, 3 * np.pi, W)
        y = np.linspace(0, 3 * np.pi, H)
        X, Y = np.meshgrid(x, y)

        # different types of spatial patterns
        base_patterns = [
            np.sin(X / 2) * np.cos(Y / 2),  # sinusoidal pattern
            np.exp(-0.01 * (X - 24) ** 2 - 0.01 * (Y - 24) ** 2),  # Gaussian blob
            X * Y / 200,  # linear gradient
        ]

        # ensemble array: [N_ens, n_vars, H, W]
        predictions_ens = np.zeros((N_ens, n_vars, H, W))

        # Generate ensemble members
        for k in range(N_ens):
            for i in range(n_vars):
                # Select a base spatial pattern for each variable
                base = base_patterns[i % len(base_patterns)]
                # Scale to realistic physical values
                signal = base * 20 + 280

                # Add Gaussian noise to simulate ensemble spread
                predictions_ens[k, i] = signal + np.random.randn(H, W) * (0.5 + k * 0.2)

        variable_names = ["Temp", "Press", "Humid"]
        timestamp = datetime(2024, 1, 1, 12, 0)

        # Test 1: numpy
        expected_path = plot_ensemble_surface(
            predictions_ens=predictions_ens,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=variable_names,
            timestamp=timestamp,
            filename="plot_ensemble_surface_numpy.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(os.path.exists(expected_path))
        self.assertGreater(os.path.getsize(expected_path), 0)

        # Test 2: torch
        expected_path = plot_ensemble_surface(
            predictions_ens=torch.from_numpy(predictions_ens),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=variable_names,
            timestamp=timestamp,
            filename="plot_ensemble_surface_torch.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(os.path.exists(expected_path))
        self.assertGreater(os.path.getsize(expected_path), 0)

        if self.logger:
            self.logger.info("✅ All ensemble surface plot tests passed")

    def test_plot_zoom_comparison_comprehensive(self):
        """Comprehensive test for zoom comparison plots."""
        if self.logger:
            self.logger.info("Testing zoom comparison plots comprehensively")

        # Grid
        lat_1d = np.linspace(-90, 90, 144)
        lon_1d = np.linspace(0, 360, 360, endpoint=False)

        batch_size = 1
        n_vars = 3
        H, W = 144, 360

        targets = np.zeros((batch_size, n_vars, H, W))
        predictions = np.zeros((batch_size, n_vars, H, W))

        for i in range(n_vars):
            base = np.ones((H, W)) * (280 + i * 5)

            targets[0, i] = base + np.random.randn(H, W) * 1.0
            predictions[0, i] = targets[0, i] + np.random.randn(H, W) * 0.5

        variable_names = ["Temp", "Press", "Humid"]

        zoom_box = {
            "lat_min": -23,
            "lat_max": 13,
            "lon_min": 255,
            "lon_max": 345,
        }

        # Test 1: numpy
        expected_path = plot_zoom_comparison(
            predictions=predictions,
            targets=targets,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=variable_names,
            filename="plot_zoom_numpy.png",
            save_dir=self.output_dir,
            zoom_box=zoom_box,
        )

        self.assertTrue(os.path.exists(expected_path))
        self.assertGreater(os.path.getsize(expected_path), 0)

        # Test 2: torch
        expected_path = plot_zoom_comparison(
            predictions=torch.from_numpy(predictions),
            targets=torch.from_numpy(targets),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=variable_names,
            filename="plot_zoom_torch.png",
            save_dir=self.output_dir,
            zoom_box=zoom_box,
        )

        self.assertTrue(os.path.exists(expected_path))
        self.assertGreater(os.path.getsize(expected_path), 0)

        if self.logger:
            self.logger.info("✅ All zoom comparison plot tests passed")

    def test_plot_global_surface_robinson_comprehensive(self):
        """Comprehensive test for global Robinson surface plots."""
        if self.logger:
            self.logger.info("Testing global Robinson surface plots comprehensively")

        # GLOBAL domain
        H, W = 90, 180
        lat_1d = np.linspace(-90, 90, H)
        lon_1d = np.linspace(-180, 180, W)

        batch_size = 1
        n_vars = 3

        # Create synthetic global patterns
        x = np.linspace(-np.pi, np.pi, W)
        y = np.linspace(-np.pi / 2, np.pi / 2, H)
        X, Y = np.meshgrid(x, y)

        coarse_inputs = np.zeros((batch_size, n_vars, H, W))
        targets = np.zeros((batch_size, n_vars, H, W))
        pred = np.zeros((batch_size, n_vars, H, W))

        base_patterns = [np.sin(X) * np.cos(Y), np.cos(2 * X) * np.sin(Y), X * Y]

        for i in range(n_vars):
            pattern = base_patterns[i] * 10 + 280
            coarse_inputs[0, i] = pattern + np.random.randn(H, W) * 2
            targets[0, i] = pattern + np.random.randn(H, W)
            pred[0, i] = targets[0, i] + np.random.randn(H, W) * 0.3

        variable_names = ["Temp", "Press", "Humid"]
        timestamp = datetime(2024, 1, 1, 12, 0)

        # Test with numpy arrays
        expected_path = plot_global_surface_robinson(
            predictions=pred,
            targets=targets,
            coarse_inputs=coarse_inputs,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_global_robinson_standard.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(os.path.exists(expected_path))

        # Test with PyTorch tensors
        expected_path = plot_global_surface_robinson(
            predictions=torch.from_numpy(pred),
            targets=torch.from_numpy(targets),
            coarse_inputs=torch.from_numpy(coarse_inputs),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_global_robinson_torch.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(os.path.exists(expected_path))

        if self.logger:
            self.logger.info("✅ All global Robinson surface plot tests passed")

    def test_plot_mae_map_comprehensive(self):
        """Comprehensive test for time-averaged MAE spatial map plots."""
        if self.logger:
            self.logger.info("Testing MAE map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, :, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_MAE_map(
            predictions=predictions,
            targets=targets,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_mae_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        expected_path = plot_MAE_map(
            predictions=torch.from_numpy(predictions),
            targets=torch.from_numpy(targets),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_mae_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Single variable
        expected_path = plot_MAE_map(
            predictions=predictions[:, 0:1],
            targets=targets[:, 0:1],
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=[self.variable_names[0]],
            save_dir=self.output_dir,
            filename="validation_mae_map_single_var.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All MAE map plot tests passed")

    def test_plot_error_map_comprehensive(self):
        """Comprehensive test for time-averaged ERROR spatial map plots."""
        if self.logger:
            self.logger.info("Testing ERROR map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, :, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_error_map(
            predictions=predictions,
            targets=targets,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_error_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        expected_path = plot_error_map(
            predictions=torch.from_numpy(predictions),
            targets=torch.from_numpy(targets),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_error_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Single variable
        expected_path = plot_error_map(
            predictions=predictions[:, 0:1],
            targets=targets[:, 0:1],
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=[self.variable_names[0]],
            save_dir=self.output_dir,
            filename="validation_error_map_single_var.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All ERROR map plot tests passed")

    def test_plot_spread_skill_ratio_map_comprehensive(self):
        """Comprehensive test for time-averaged MAE spatial map plots."""
        if self.logger:
            self.logger.info("Testing SSR map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        predictions = self.predictions[:, :, :48, :68]
        # Transform predictions into ensemble by adding noise:
        T, C, h, w = predictions.shape
        noise = np.random.normal(size=(10, T, C, h, w))
        predictions_ensemble = predictions + noise
        targets = self.targets[:, :, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_spread_skill_ratio_map(
            predictions=predictions_ensemble,
            targets=targets,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_ssr_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        # Test 2: PyTorch tensors
        expected_path = plot_spread_skill_ratio_map(
            predictions=torch.from_numpy(predictions_ensemble),
            targets=torch.from_numpy(targets),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_ssr_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Single variable
        expected_path = plot_spread_skill_ratio_map(
            predictions=predictions_ensemble[:, :, 0:1],
            targets=targets[:, 0:1],
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=[self.variable_names[0]],
            save_dir=self.output_dir,
            filename="validation_ssr_map_single_var.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All spread skill ratio map plot tests passed")

    def test_plot_spread_skill_ratio_hexbin_comprehensive(self):
        """Comprehensive test for spread skill ratio hexbin plots"""
        if self.logger:
            self.logger.info("Testing SSR hexbin plots comprehensively")

        # Matching spatial resolution
        predictions = self.predictions[:, :, :48, :68]
        # Transform predictions into ensemble by adding noise:
        T, C, h, w = predictions.shape
        noise = np.random.normal(size=(10, T, C, h, w))
        predictions_ensemble = predictions + noise
        targets = self.targets[:, :, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_spread_skill_ratio_hexbin(
            predictions=predictions_ensemble,
            targets=targets,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_ssr_hexbin_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        # Test 2: PyTorch tensors
        expected_path = plot_spread_skill_ratio_hexbin(
            predictions=torch.from_numpy(predictions_ensemble),
            targets=torch.from_numpy(targets),
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_ssr_hexbin_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Single variable
        expected_path = plot_spread_skill_ratio_hexbin(
            predictions=predictions_ensemble[:, :, 0:1],
            targets=targets[:, 0:1],
            variable_names=[self.variable_names[0]],
            save_dir=self.output_dir,
            filename="validation_ssr_hexbin_single_var.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All spread skill ratio hexbin plot tests passed")

    def test_plot_mean_divergence_map_comprehensive(self):
        """Comprehensive test for mean divergence map plots."""
        if self.logger:
            self.logger.info("Testing divergence map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        u_pred = self.predictions[:, 0, :48, :68]
        v_pred = self.predictions[:, 1, :48, :68]
        u_target = self.targets[:, 0, :48, :68]
        v_target = self.targets[:, 1, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_mean_divergence_map(
            u_pred,
            v_pred,
            u_target,
            v_target,
            spacing=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_mean_divergence_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        expected_path = plot_mean_divergence_map(
            torch.from_numpy(u_pred),
            torch.from_numpy(v_pred),
            torch.from_numpy(u_target),
            torch.from_numpy(v_target),
            spacing=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_mean_divergence_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        if self.logger:
            self.logger.info("✅ All mean divergence map plot tests passed")

    def test_plot_mean_curl_map_comprehensive(self):
        """Comprehensive test for mean curl map plots."""
        if self.logger:
            self.logger.info("Testing curl map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        u_pred = self.predictions[:, 0, :48, :68]
        v_pred = self.predictions[:, 1, :48, :68]
        u_target = self.targets[:, 0, :48, :68]
        v_target = self.targets[:, 1, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_mean_curl_map(
            u_pred,
            v_pred,
            u_target,
            v_target,
            spacing=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_mean_curl_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        expected_path = plot_mean_curl_map(
            torch.from_numpy(u_pred),
            torch.from_numpy(v_pred),
            torch.from_numpy(u_target),
            torch.from_numpy(v_target),
            spacing=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_mean_curl_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        if self.logger:
            self.logger.info("✅ All mean curl map plot tests passed")

    def test_plot_dry_frequency_map_comprehensive(self):
        """Comprehensive test for dry frequency map plots."""
        if self.logger:
            self.logger.info("Testing dry frequency map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, :, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_dry_frequency_map(
            predictions=predictions[:, 0, :, :],
            targets=targets[:, 0, :, :],
            threshold=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_dry_frequency_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        expected_path = plot_dry_frequency_map(
            predictions=torch.from_numpy(predictions[:, 0, :, :]),
            targets=torch.from_numpy(targets[:, 0, :, :]),
            threshold=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_dry_frequency_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        if self.logger:
            self.logger.info("✅ All dry frequency map plot tests passed")

    def test_dry_frequency_map(self):
        """Comprehensive test for the dry frequency map compute function."""
        if self.logger:
            self.logger.info(
                "Testing dry frequency map compute function comprehensively"
            )
        predictions = self.predictions[:, :, :48, :68]
        # Test 1 : standard numpy inputs
        arr = dry_frequency_map(predictions[:, 0, :, :], 1)
        self.assertTrue(arr.shape == predictions.shape[-2:])
        # Test 1 : torch tensors
        arr = dry_frequency_map(torch.from_numpy(predictions[:, 0, :, :]), 1)
        self.assertTrue(arr.shape == predictions.shape[-2:])
        if self.logger:
            self.logger.info("✅ All dry frequency tests passed")

    def test_divergence(self):
        """Comprehensive test for the divergence compute function."""
        if self.logger:
            self.logger.info("Testing divergence compute function comprehensively")
        u = self.predictions[:, 0, :48, :68]
        v = self.predictions[:, 1, :48, :68]
        # test 1 : standard numpy inputs :
        div = get_divergence(u, v, spacing=1)
        self.assertTrue(div.shape == u.shape)
        # test 2 : torch inputs :
        div_torch = get_divergence(torch.from_numpy(u), torch.from_numpy(v), spacing=1)
        self.assertTrue(div_torch.shape == u.shape)
        if self.logger:
            self.logger.info("✅ All divergence tests passed")

    def test_curl(self):
        """Comprehensive test for the curl compute function."""
        if self.logger:
            self.logger.info("Testing curl compute function comprehensively")
        u = self.predictions[:, 0, :48, :68]
        v = self.predictions[:, 1, :48, :68]
        # test 1 : standard numpy inputs :
        curl = get_curl(u, v, spacing=1)
        self.assertTrue(curl.shape == u.shape)
        # test 2 : torch inputs :
        curl_torch = get_curl(torch.from_numpy(u), torch.from_numpy(v), spacing=1)
        self.assertTrue(curl_torch.shape == u.shape)
        if self.logger:
            self.logger.info("✅ All curl tests passed")

    def test_metric_plots_comprehensive(self):
        """Comprehensive test for metric plots."""
        if self.logger:
            self.logger.info("Testing metric plots comprehensively")

        # Test 1: Metric histories
        expected_path = plot_metric_histories(
            valid_metrics_history=self.valid_metrics_history,
            variable_names=[name.split(" ")[0] for name in self.variable_names],
            metric_names=["rmse", "mae", "r2"],
            save_dir=self.output_dir,
            filename="metric_histories_comprehensive",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Loss histories
        expected_path = plot_loss_histories(
            train_loss_history=self.train_loss_history,
            valid_loss_history=self.valid_loss_history,
            save_dir=self.output_dir,
            filename="loss_histories_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Average metrics
        expected_path = plot_average_metrics(
            valid_metrics_history=self.valid_metrics_history,
            metric_names=["rmse", "mae", "r2"],
            save_dir=self.output_dir,
            filename="average_metrics_standard.png",
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All metric plot tests passed")

    def test_plot_metrics_heatmap_comprehensive(self):
        """Comprehensive test for validation metrics heatmap."""

        if self.logger:
            self.logger.info("Testing metrics heatmap")

        # Local dummy MetricTracker
        class DummyMetricTracker:
            def __init__(self, values):
                self.values = np.asarray(values)
                self.count = len(self.values)

            def getmean(self):
                return float(np.mean(self.values)) if self.count > 0 else np.nan

        # Fake MetricTracker-based metrics dict
        valid_metrics_trackers = {}

        for var in self.variable_names:
            var_key = var.split(" ")[0]
            for metric in ["rmse", "mae", "r2"]:
                # Reuse the existing synthetic histories
                history = self.valid_metrics_history[f"{var_key}_pred_vs_fine_{metric}"]
                valid_metrics_trackers[f"{var_key}_pred_vs_fine_{metric}"] = (
                    DummyMetricTracker(history)
                )

        expected_path = plot_metrics_heatmap(
            valid_metrics_history=valid_metrics_trackers,
            variable_names=[name.split(" ")[0] for name in self.variable_names],
            metric_names=["rmse", "mae", "r2"],
            save_dir=self.output_dir,
            filename="metrics_heatmap_comprehensive",
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ Metrics heatmap test passed")

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

    def test_mv_correlation(self):
        """Test for correlation over the time dimension for pairs of variables.
        Test for correlation over the spatial dimensions.
        """

        # Define lat lon grid
        w = self.predictions.shape[2]
        h = self.predictions.shape[3]
        dlat = 20
        dlon = 40
        lat1 = 30
        lon1 = -120
        lon, lat = np.meshgrid(
            np.linspace(lon1, lon1 + dlon, w), np.linspace(lat1, lat1 + dlat, h)
        )

        # Test 1: Standard configuration Numpy arrays for correlation over time dimension
        expected_path = plot_validation_mvcorr(
            predictions=self.predictions,
            targets=self.targets,
            lat=lat,
            lon=lon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_mvcorr_numpy.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_validation_mvcorr(
            predictions=self.predictions,
            targets=self.targets,
            lat=lat,
            lon=lon,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="comparison_mvcorr_numpy.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Standard configuration PyTorch tensors for correlation over time dimension
        coarse_tensor = torch.from_numpy(self.coarse_inputs.copy())
        fine_tensor = torch.from_numpy(self.targets.copy())
        pred_tensor = torch.from_numpy(self.predictions.copy())

        expected_path = plot_validation_mvcorr(
            predictions=pred_tensor,
            targets=fine_tensor,
            lat=lat,
            lon=lon,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="comparison_mvcorr_torch.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Standard configuration Numpy arrays for correlation over space dimensions
        expected_path = plot_validation_mvcorr_space(
            predictions=self.predictions,
            targets=self.targets,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="comparison_mv_corr_space_numpy.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 4: Standard configuration Numpy arrays for correlation over space dimensions
        expected_path = plot_validation_mvcorr_space(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="comparison_mvcorr_space_torch.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All correlation plots tests passed")

    def test_temporal_series_comparison_comprehensive(self):
        """Comprehensive test for spatially averaged temporal series comparison."""

        if self.logger:
            self.logger.info("Testing temporal series comparison comprehensively")

        # Here we reinterpret batch_size as time dimension T
        T = 100
        C = self.num_vars
        H = self.h
        W = self.w

        # Create synthetic temporal signal
        time = np.linspace(0, 4 * np.pi, T)

        predictions = np.zeros((T, C, H, W))
        targets = np.zeros((T, C, H, W))

        for c in range(C):
            for t in range(T):
                seasonal_signal = np.sin(time[t]) * (c + 1)

                spatial_pattern = (
                    np.sin(np.linspace(0, 2 * np.pi, W))[None, :]
                    * np.cos(np.linspace(0, 2 * np.pi, H))[:, None]
                )

                targets[t, c] = seasonal_signal + spatial_pattern
                predictions[t, c] = (
                    seasonal_signal + spatial_pattern + np.random.normal(0, 0.1, (H, W))
                )

        # Test 1: numpy inputs
        expected_path = plot_temporal_series_comparison(
            predictions=predictions,
            targets=targets,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="temporal_series_numpy.png",
        )

        self.assertTrue(
            os.path.exists(expected_path),
            f"File not found: {expected_path}",
        )

        # Test 2: torch tensors
        expected_path_torch = plot_temporal_series_comparison(
            predictions=torch.from_numpy(predictions),
            targets=torch.from_numpy(targets),
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="temporal_series_torch.png",
        )

        self.assertTrue(
            os.path.exists(expected_path_torch),
            f"File not found: {expected_path_torch}",
        )

        # Test 3: shape mismatch error
        with self.assertRaises(ValueError):
            plot_temporal_series_comparison(
                predictions=predictions,
                targets=targets[:, :, :-1, :],  # wrong shape
                variable_names=self.variable_names,
                save_dir=self.output_dir,
                filename="temporal_series_error.png",
            )

        if self.logger:
            self.logger.info("✅ All temporal series comparison tests passed")

    def test_ranks(self):
        """Comprehensive test for the ranks compute function."""
        if self.logger:
            self.logger.info("Testing ranks compute function comprehensively")
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, 0, :48, :68]
        ensemble_size = 10
        # Test 1 : standard numpy inputs
        predictions_repeated = np.repeat(
            predictions[None, :, 0, :, :], axis=0, repeats=ensemble_size
        )
        arr = ranks(predictions=predictions_repeated, targets=targets)
        self.assertTrue(arr.shape == targets.flatten().shape)
        # Test 2 : torch tensors
        predictions_repeated_torch = torch.from_numpy(predictions_repeated)
        targets_torch = torch.from_numpy(targets)
        arr_torch = ranks(predictions=predictions_repeated_torch, targets=targets_torch)
        self.assertTrue(arr_torch.shape == targets.flatten().shape)

        if self.logger:
            self.logger.info("✅ All ranks tests passed")

    def test_plot_ranks(self):
        """Comprehensive test for the ranks plot function."""
        if self.logger:
            self.logger.info("Testing plot_ranks function comprehensively")
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, :, :48, :68]
        ensemble_size = 10
        # Test 1 : standard numpy inputs
        predictions_repeated = np.repeat(
            predictions[None, :, :, :, :], axis=0, repeats=ensemble_size
        )
        expected_path = plot_ranks(
            predictions=predictions_repeated,
            targets=targets,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="ranks.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        # Test 2 : torch inputs
        expected_path_torch = plot_ranks(
            predictions=torch.from_numpy(predictions_repeated),
            targets=torch.from_numpy(targets),
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="ranks_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path_torch), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All ranks plot tests passed")

    def tearDown(self):
        """Clean up after tests."""
        # Note: We don't remove the output directory so you can inspect the plots
        if self.logger:
            self.logger.info(
                f"Plotting tests completed - plots available in: {self.output_dir}"
            )


class TestSSRFunction(unittest.TestCase):
    """Unit tests for crps_ensemble_all function."""

    def setUp(self):
        """Set up test fixtures."""

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up spread_skill_ratio function test fixtures")

    def test_ssr_basic(self):
        """Test SSR with simple known values."""
        if self.logger:
            self.logger.info("Testing SSR basic functionality")
        np.random.seed(0)  # set random seed for reproducibility.
        true = np.zeros((100, 1, 10, 10))  # shape [T,C,h,w]
        pred_ens = np.random.normal(
            loc=0, scale=0.1, size=(10, 100, 1, 10, 10)
        )  # N_ens = 10
        ssr = spread_skill_ratio(
            predictions=pred_ens, targets=true, variable_names=None, pixel_wise=False
        )[0]  # get element of single element array.
        # expected SSR should be ~3.15 :
        # let $X_{i,r}$ be the prediction for ensemble member r.
        # as targets = mean of predictions, RMSE should be very close to variance of predictions (= 0.01)
        # deriving the expected value of the spread is more complicated :
        # $ spread = \frac{11}{10} \sqrt{ \frac{1}{N} \sum_{i,r} (X_{i,r} - \bar X_i)^2 } $
        # We can inject the definition of $\bar X_i = \frac{1}{R} \sum_{r'} X_{i,r'} $
        # and develop the squared term :
        # $ X_{i,r} \frac{R-1}{R} - \frac{1}{R} \sum_{r' != r} X_{i,r'} $
        # we can replace the mean by the expectance operator, and develop the squared term :
        # $ R \times \mathbb{E} [ ( X_{i,r} \frac{R-1}{R} - \frac{1}{R} \sum_{r' != r} X_{i,r'} )^2 ]  $
        # Since all X_{i,r'} are iid, we can develop the square inside the expected value, the covariance terms will be 0
        # and we are left with :
        # $ R \times [ \mathbb{V}(X_{i,r}) (\frac{R-1}{R})^2 + \frac{R-1}{R^2} \mathbb{V}(X_{i,r'}) ] $
        # plugging in R = 10, we get :
        # $ 10 * 0.9 * \mathbb{X}$
        # this term is under a square root and multiplied by the corrective factor to give the spread :
        # spread = \sqrt{1.1} \times \sqrt{\mathbb{V}(X)} \times 3
        # So, SSR = spread / RMSE = \sqrt{1.1} * 3 ~ 3.15

        # SSR must be finite and non-negative
        self.assertGreaterEqual(ssr, 0.0)
        self.assertAlmostEqual(ssr, 3.15, places=1)
        if self.logger:
            self.logger.info(f"SSR computed : {ssr:.2f} vs SSR expected : ~ 3.15")
            self.logger.info("✅ SSR basic test passed")

    def test_ssr_one_when_perfect_prediction(self):
        """Test SSR is supposed to be 1 when the predictions follow the same distribution as the truth."""
        if self.logger:
            self.logger.info("Testing SSR perfect prediction")

        np.random.seed(0)  # set random seed for reproducibility.
        true = np.random.normal(
            loc=0, scale=0.1, size=(100, 1, 10, 10)
        )  # shape [T,C,h,w]
        pred_ens = np.random.normal(
            loc=0, scale=0.1, size=(10, 100, 1, 10, 10)
        )  # N_ens = 10
        ssr = spread_skill_ratio(
            predictions=pred_ens, targets=true, variable_names=None, pixel_wise=False
        )[0]  # get element of single element array.

        self.assertAlmostEqual(ssr, 1.0, places=1)

        if self.logger:
            self.logger.info(f"SSR computed : {ssr:.2f} vs SSR expected : ~ 1.0")
            self.logger.info("✅ SSR perfect prediction test passed")


def run_tests():
    """Run all plotting tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPlottingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestSSRFunction))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
