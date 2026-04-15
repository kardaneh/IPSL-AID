# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston, Pierre Chapel
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import pandas as pd
import xarray as xr
import numpy as np
import tempfile
import unittest
import shutil
import torch
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.utils import EasyDict
from IPSL_AID.logger import Logger
from IPSL_AID.dataset import (
    DataPreprocessor,
    stats,
    coarse_down_up,
    gaussian_filter,
)

# ============================================================================
# Test Utilities
# ============================================================================


def create_dummy_netcdf(temp_dir, year=2020, has_constants=False):
    """Create a dummy NetCDF file for testing."""
    lat = np.linspace(-90, 90, 36)
    lon = np.linspace(-180, 180, 72)
    time = pd.date_range(f"{year}-01-01", f"{year}-01-03", freq="6h")

    data_vars = {
        "VAR_2T": (
            ("time", "latitude", "longitude"),
            np.random.randn(len(time), len(lat), len(lon)) * 10 + 285,
        ),
        "VAR_10U": (
            ("time", "latitude", "longitude"),
            np.random.randn(len(time), len(lat), len(lon)) * 3 + 0.5,
        ),
        "VAR_10V": (
            ("time", "latitude", "longitude"),
            np.random.randn(len(time), len(lat), len(lon)) * 3 - 0.1,
        ),
        "VAR_TP": (
            ("time", "latitude", "longitude"),
            np.abs(np.random.randn(len(time), len(lat), len(lon)) * 0.0002 + 0.00009),
        ),
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
        },
    )

    if has_constants:
        const_ds = xr.Dataset(
            {
                "z": (
                    ("latitude", "longitude"),
                    np.random.randn(len(lat), len(lon)) * 100 + 500,
                ),
                "lsm": (
                    ("latitude", "longitude"),
                    np.random.randint(0, 2, (len(lat), len(lon))),
                ),
            },
            coords={"latitude": lat, "longitude": lon, "time": [time[0]]},
        )
        const_path = os.path.join(temp_dir, "constants.nc")
        const_ds.to_netcdf(const_path)
        return ds, const_path

    return ds


def create_dummy_statistics_json(temp_dir):
    """Create dummy statistics.json file for testing."""
    stats_dict = {
        "VAR_2T_coarse": {"vmean": 285.04, "vstd": 12.7438},
        "VAR_2T_residual": {"vmean": -0.000094627, "vstd": 1.6042},
        "VAR_2T_fine": {"vmean": 286.31, "vstd": 12.6325},
        "VAR_10U_coarse": {"vmean": 0.44536, "vstd": 3.4649},
        "VAR_10U_residual": {"vmean": -0.0013833, "vstd": 1.0221},
        "VAR_10U_fine": {"vmean": 0.36626, "vstd": 3.4527},
        "VAR_10V_coarse": {"vmean": -0.11892, "vstd": 3.7420},
        "VAR_10V_residual": {"vmean": -0.0015548, "vstd": 1.0384},
        "VAR_10V_fine": {"vmean": -0.05726, "vstd": 3.6996},
        "VAR_TP_coarse": {"vmean": 0.000094189, "vstd": 0.00026393},
        "VAR_TP_residual": {"vmean": -0.000000040417, "vstd": 0.00028678},
        "VAR_TP_fine": {"vmean": 0.000097360, "vstd": 0.00044533},
    }

    stats_path = os.path.join(temp_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f)

    return stats_path


# ============================================================================
# Unit Tests for DataPreprocessor
# ============================================================================


class TestDataPreprocessor(unittest.TestCase):
    """Unit tests for DataPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info(f"Test setup - created temp directory: {self.temp_dir}")

        # Create dummy dataset
        self.ds, self.const_path = create_dummy_netcdf(
            self.temp_dir, year=2020, has_constants=True
        )

        # Create statistics JSON
        create_dummy_statistics_json(self.temp_dir)

        # Setup configuration
        self.years = [2020]
        self.varnames_list = ["VAR_2T", "VAR_10U", "VAR_10V", "VAR_TP"]
        self.units_list = ["K", "m/s", "m/s", "m"]
        self.in_shape = (8, 16)
        self.batch_size_lat = 32
        self.batch_size_lon = 32

        # Compute steps
        self.norm_mapping, self.steps = stats(
            self.ds, self.logger, input_dir=self.temp_dir
        )

        # Create index mapping
        self.index_mapping = {var: i for i, var in enumerate(self.varnames_list)}

        # Normalization types
        self.normalization_type = {var: "standard" for var in self.varnames_list}

        # Constant variables
        self.constant_variables = ["z", "lsm"]

        if self.logger:
            self.logger.info(
                "Test setup complete - DataPreprocessor test fixtures ready"
            )

    # ------------------------------------------------------------------------
    # Stats Function Tests
    # ------------------------------------------------------------------------

    def test_stats_computation_without_existing_json(self):
        """Test stats computation when no statistics.json exists."""
        if self.logger:
            self.logger.info("Testing stats computation without existing JSON")

        # Create a fresh dataset and directory without stats.json
        temp_dir_no_stats = tempfile.mkdtemp()
        ds_no_stats = create_dummy_netcdf(temp_dir_no_stats, year=2020)

        norm_mapping = {}
        result_norm, steps = stats(
            ds_no_stats,
            self.logger,
            input_dir=temp_dir_no_stats,
            norm_mapping=norm_mapping,
        )

        # Check that norm_mapping was populated with constants
        self.assertIn("VAR_2T_coarse", result_norm)
        self.assertIn("VAR_2T_fine", result_norm)
        self.assertAlmostEqual(result_norm["VAR_2T_coarse"].vmean, 286.2744, places=4)
        self.assertAlmostEqual(result_norm["VAR_2T_coarse"].vstd, 12.4757, places=4)

        # Check coordinate stats
        self.assertIn("latitude", result_norm)
        self.assertIn("longitude", result_norm)

        # Check steps
        self.assertEqual(steps["latitude"], 36)
        self.assertEqual(steps["longitude"], 72)
        self.assertEqual(steps["time"], 9)

        shutil.rmtree(temp_dir_no_stats)

        if self.logger:
            self.logger.info("✅ Stats computation without JSON test passed")

    def test_stats_computation_with_existing_json(self):
        """Test stats computation when statistics.json exists."""
        if self.logger:
            self.logger.info("Testing stats computation with existing JSON")

        norm_mapping = {}
        result_norm, _ = stats(
            self.ds, self.logger, input_dir=self.temp_dir, norm_mapping=norm_mapping
        )

        # Check that stats were loaded from JSON
        self.assertEqual(result_norm["VAR_2T_coarse"].vmean, 285.04)
        self.assertEqual(result_norm["VAR_2T_residual"].vstd, 1.6042)

        if self.logger:
            self.logger.info("✅ Stats computation with JSON test passed")

    # ------------------------------------------------------------------------
    # CoarseDownUp Function Tests
    # ------------------------------------------------------------------------
    def test_coarse_down_up_with_torch_tensors(self):
        """Test coarse_down_up with torch tensor inputs."""
        if self.logger:
            self.logger.info("Testing coarse_down_up with torch tensors")

        channels = 4
        h_fine, w_fine = 144, 144
        coarse_shape = (16, 32)

        fine_filtered = torch.randn(channels, h_fine, w_fine)
        fine_batch = torch.randn(channels, h_fine, w_fine)

        coarse_up = coarse_down_up(fine_filtered, fine_batch, input_shape=coarse_shape)

        self.assertEqual(coarse_up.shape, (channels, h_fine, w_fine))

        if self.logger:
            self.logger.info(
                f"✅ Coarse_down_up test passed - output shapes: {coarse_up.shape}"
            )

    def test_coarse_down_up_with_numpy_arrays(self):
        """Test coarse_down_up with numpy array inputs."""
        if self.logger:
            self.logger.info("Testing coarse_down_up with numpy arrays")

        channels = 4
        h_fine, w_fine = 144, 144

        fine_filtered = np.random.randn(channels, h_fine, w_fine).astype(np.float32)
        fine_batch = np.random.randn(channels, h_fine, w_fine).astype(np.float32)

        coarse_up = coarse_down_up(fine_filtered, fine_batch)

        self.assertIsInstance(coarse_up, torch.Tensor)
        self.assertEqual(coarse_up.shape, (channels, h_fine, w_fine))

        if self.logger:
            self.logger.info("✅ Coarse_down_up with numpy arrays test passed")

    # ------------------------------------------------------------------------
    # GaussianFilter Function Tests
    # ------------------------------------------------------------------------

    def test_gaussian_filter_basic(self):
        """Test basic Gaussian filtering operation."""
        if self.logger:
            self.logger.info("Testing Gaussian filter basic operation")

        H, W = 144, 144
        image = np.random.randn(H, W).astype(np.float32)
        dH = dW = 0.5
        cutoff_H = cutoff_W = 0.1

        filtered = gaussian_filter(
            image, dW, dH, cutoff_W, cutoff_H, epsilon=0.01, margin=8
        )

        self.assertEqual(filtered.shape, image.shape)
        self.assertLess(np.var(filtered), np.var(image))

        if self.logger:
            self.logger.info(
                f"✅ Gaussian filter test passed - variance reduced from {np.var(image):.4f} to {np.var(filtered):.4f}"
            )

    def test_gaussian_filter_different_epsilon(self):
        """Test Gaussian filter with different epsilon values."""
        if self.logger:
            self.logger.info("Testing Gaussian filter with different epsilon values")

        H, W = 144, 144
        image = np.random.randn(H, W).astype(np.float32)
        dH = dW = 0.5
        cutoff_H = cutoff_W = 0.1

        filtered_smooth = gaussian_filter(
            image, dW, dH, cutoff_W, cutoff_H, epsilon=0.001, margin=8
        )
        filtered_sharp = gaussian_filter(
            image, dW, dH, cutoff_W, cutoff_H, epsilon=0.1, margin=8
        )

        self.assertGreater(np.var(filtered_sharp), np.var(filtered_smooth))

        if self.logger:
            self.logger.info(
                f"✅ Gaussian filter epsilon test passed - sharp var: {np.var(filtered_sharp):.4f}, smooth var: {np.var(filtered_smooth):.4f}"
            )

    # ------------------------------------------------------------------------
    # DataPreprocessor Initialization Tests
    # ------------------------------------------------------------------------

    def test_preprocessor_initialization_train_mode(self):
        """Test DataPreprocessor initialization in train mode."""
        if self.logger:
            self.logger.info("Testing DataPreprocessor initialization - TRAIN mode")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Verify dimensions
        self.assertEqual(preprocessor.H, 36)
        self.assertEqual(preprocessor.W, 72)
        self.assertEqual(len(preprocessor.time_batchs), 9)  # Full time range
        self.assertEqual(preprocessor.sbatch, 4)
        self.assertEqual(preprocessor.tbatch, 2)

        # Verify constant variables
        self.assertIsNotNone(preprocessor.const_vars)
        self.assertEqual(preprocessor.const_vars.shape, (2, 36, 72))

        # Verify time normalization
        self.assertIsNotNone(preprocessor.year_norm)
        self.assertIsNotNone(preprocessor.doy_norm)
        self.assertIsNotNone(preprocessor.hour_norm)

        if self.logger:
            self.logger.info(
                f"✅ Train mode initialization test passed - H={preprocessor.H}, W={preprocessor.W}"
            )

    def test_preprocessor_initialization_train_regional_mode(self):
        """Test DataPreprocessor initialization in train regional mode."""
        if self.logger:
            self.logger.info(
                "Testing DataPreprocessor initialization - TRAIN REGIONAL mode"
            )

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train_regional",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            region_center=(10.0, 50.0),
            region_size=(32, 64),
            logger=self.logger,
        )

        # Verify that region_center and region_size attributes exist :
        self.assertEqual(preprocessor.region_center, (10.0, 50.0))
        self.assertEqual(preprocessor.region_size, (32, 64))

        # Verify that train_slices attribute exists and contains the right locations:
        self.assertTrue(hasattr(preprocessor, "train_slices"))
        self.assertIsNotNone(preprocessor.train_slices)
        self.assertEqual(len(preprocessor.train_slices), 2)

        if self.logger:
            self.logger.info("✅ Regional train mode initialization test passed")

    def test_preprocessor_initialization_validation_mode(self):
        """Test DataPreprocessor initialization in validation mode."""
        if self.logger:
            self.logger.info(
                "Testing DataPreprocessor initialization - VALIDATION mode"
            )

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="validation",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Verify evaluation slices
        self.assertIsNotNone(preprocessor.eval_slices)
        expected_slices = (36 // 32) * (72 // 32)  # 1 * 2 = 2
        self.assertEqual(len(preprocessor.eval_slices), expected_slices)
        self.assertEqual(preprocessor.sbatch, expected_slices)

        if self.logger:
            self.logger.info(
                f"✅ Validation mode initialization test passed - generated {len(preprocessor.eval_slices)} evaluation slices"
            )

    def test_preprocessor_initialization_cos_sin_normalization(self):
        """Test DataPreprocessor with cos_sin time normalization."""
        if self.logger:
            self.logger.info("Testing DataPreprocessor with cos_sin time normalization")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="cos_sin",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Verify time normalization features
        self.assertIsNotNone(preprocessor.doy_sin)
        self.assertIsNotNone(preprocessor.doy_cos)
        self.assertIsNotNone(preprocessor.hour_sin)
        self.assertIsNotNone(preprocessor.hour_cos)

        if self.logger:
            self.logger.info("✅ Cos_sin time normalization test passed")

    def test_preprocessor_batch_size_validation(self):
        """Test that batch size validation raises appropriate errors."""
        if self.logger:
            self.logger.info("Testing batch size validation")

        with self.assertRaises(ValueError):
            DataPreprocessor(
                years=self.years,
                loaded_dfs=self.ds,
                constants_file_path=self.const_path,
                varnames_list=self.varnames_list,
                units_list=self.units_list,
                in_shape=self.in_shape,
                batch_size_lat=50,  # > 36
                batch_size_lon=self.batch_size_lon,
                steps=self.steps,
                tbatch=2,
                sbatch=4,
                debug=True,
                mode="train",
                run_type="train",
                time_normalization="linear",
                norm_mapping=self.norm_mapping,
                index_mapping=self.index_mapping,
                normalization_type=self.normalization_type,
                constant_variables=self.constant_variables,
                logger=self.logger,
            )

        if self.logger:
            self.logger.info("✅ Batch size validation test passed")

    # ------------------------------------------------------------------------
    # DataPreprocessor Method Tests
    # ------------------------------------------------------------------------

    def test_new_epoch_method(self):
        """Test new_epoch method resets random centers."""
        if self.logger:
            self.logger.info("Testing new_epoch method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Set some random centers
        preprocessor.random_centers = [(10, 20), (15, 30), (20, 40), (25, 50)]
        preprocessor.new_epoch()

        # Verify centers were reset
        self.assertEqual(preprocessor.random_centers, [None] * 4)
        self.assertEqual(preprocessor.last_tbatch_index, -1)

        if self.logger:
            self.logger.info("✅ New_epoch method test passed")

    def test_get_center_indices_from_latlon(self):
        """Test conversion from geographic coordinates to grid indices."""
        if self.logger:
            self.logger.info("Testing get_center_indices_from_latlon")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="inference_regional",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            region_center=(10.0, 50.0),
            region_size=(32, 64),
            logger=self.logger,
        )

        lat_val, lon_val = preprocessor.region_center

        lat_idx, lon_idx = preprocessor.get_center_indices_from_latlon(lat_val, lon_val)

        lat_array = self.ds.latitude.values
        lon_array = self.ds.longitude.values

        self.assertTrue(0 <= lat_idx < len(lat_array))
        self.assertTrue(0 <= lon_idx < len(lon_array))

        # verify closest point
        self.assertEqual(lat_idx, np.abs(lat_array - lat_val).argmin())
        self.assertEqual(lon_idx, np.abs(lon_array - lon_val).argmin())

        if self.logger:
            self.logger.info(
                f"✅ get_center_indices_from_latlon test passed - indices: ({lat_idx}, {lon_idx})"
            )

    def test_generate_random_batch_centers(self):
        """Test random batch centers generation."""
        if self.logger:
            self.logger.info("Testing generate_random_batch_centers")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        centers = preprocessor.generate_random_batch_centers(5)

        self.assertEqual(len(centers), 5)

        half_lat = preprocessor.batch_size_lat // 2
        for lat_center, lon_center in centers:
            self.assertGreaterEqual(lat_center, half_lat)
            self.assertLess(lat_center, preprocessor.H - half_lat)
            self.assertGreaterEqual(lon_center, 0)
            self.assertLess(lon_center, preprocessor.W)

        if self.logger:
            self.logger.info(
                f"✅ Generate random batch centers test passed - generated {len(centers)} centers"
            )

    def test_extract_batch(self):
        """Test spatial batch extraction."""
        if self.logger:
            self.logger.info("Testing extract_batch method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Create dummy data
        data = np.random.randn(36, 72)
        ilat, ilon = 18, 36

        block, indices = preprocessor.extract_batch(data, ilat, ilon)
        lat_start, lat_end, lon_start, lon_end = indices

        self.assertEqual(block.shape[-2:], (32, 32))
        self.assertEqual(lat_end - lat_start, 32)
        self.assertEqual(lon_end - lon_start, 32)

        if self.logger:
            self.logger.info(
                f"✅ Extract_batch test passed - block shape: {block.shape}"
            )

    def test_build_fine_coarse_blocks(self):
        """Test build_fine_coarse_blocks consistency (no filter)."""
        if self.logger:
            self.logger.info("Testing build_fine_coarse_blocks")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=1,
            sbatch=1,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            apply_filter=False,
            logger=self.logger,
        )

        # Build full feature tensor (C, H, W)
        npfeatures_full = np.zeros(
            (len(self.varnames_list), preprocessor.H, preprocessor.W)
        )

        sample_ds = self.ds.isel(time=0)
        for var in self.varnames_list:
            iv = self.index_mapping[var]
            npfeatures_full[iv] = sample_ds[var].values

        # Safe center without borders
        lat_center = preprocessor.H // 2
        lon_center = preprocessor.W // 2

        fine, fine_filtered, coarse, indices = preprocessor.build_fine_coarse_blocks(
            npfeatures_full, lat_center, lon_center
        )

        # Shapes
        expected_shape = (
            len(self.varnames_list),
            self.batch_size_lat,
            self.batch_size_lon,
        )

        self.assertEqual(fine.shape, expected_shape)
        self.assertEqual(coarse.shape, expected_shape)

        # No filter should be None
        self.assertIsNone(fine_filtered)

        # Indices consistency
        lat_start, lat_end, lon_start, lon_end = indices
        self.assertEqual(lat_end - lat_start, self.batch_size_lat)
        self.assertEqual(lon_end - lon_start, self.batch_size_lon)

        if self.logger:
            self.logger.info("✅ build_fine_coarse_blocks test passed")

    def test_generate_evaluation_slices(self):
        """Test evaluation slices generation."""
        if self.logger:
            self.logger.info("Testing generate_evaluation_slices")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="validation",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        slices = preprocessor.generate_evaluation_slices()

        n_blocks_lat = 36 // 32
        n_blocks_lon = 72 // 32
        self.assertEqual(len(slices), n_blocks_lat * n_blocks_lon)

        if self.logger:
            self.logger.info(
                f"✅ Generate evaluation slices test passed - created {len(slices)} slices"
            )

    def test_generate_region_slices(self):
        """Test regional slice generation."""
        if self.logger:
            self.logger.info("Testing generate_region_slices")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="inference_regional",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            region_center=(10.0, 50.0),
            region_size=(32, 64),
            logger=self.logger,
        )

        lat_val, lon_val = preprocessor.region_center
        lat_center, lon_center = preprocessor.get_center_indices_from_latlon(
            lat_val, lon_val
        )

        region_size_lat, region_size_lon = preprocessor.region_size

        slices = preprocessor.generate_region_slices(
            lat_center, lon_center, region_size_lat, region_size_lon
        )

        n_blocks_lat = region_size_lat // preprocessor.batch_size_lat
        n_blocks_lon = region_size_lon // preprocessor.batch_size_lon

        # correct number of slices
        self.assertEqual(len(slices), n_blocks_lat * n_blocks_lon)

        # verify slice sizes and bounds
        for lat_start, lat_end, lon_start, lon_end in slices:
            self.assertEqual(lat_end - lat_start, preprocessor.batch_size_lat)
            self.assertEqual(lon_end - lon_start, preprocessor.batch_size_lon)

            self.assertTrue(0 <= lat_start < preprocessor.H)
            self.assertTrue(0 < lat_end <= preprocessor.H)

        if self.logger:
            self.logger.info(
                f"✅ Generate region slices test passed - created {len(slices)} slices"
            )

    # ------------------------------------------------------------------------
    # Normalization Tests
    # ------------------------------------------------------------------------

    def test_normalize_methods(self):
        """Test different normalization methods."""
        if self.logger:
            self.logger.info("Testing normalize methods")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        stats_obj = EasyDict(
            {
                "vmin": 0.0,
                "vmax": 100.0,
                "vmean": 50.0,
                "vstd": 25.0,
                "median": 50.0,
                "iqr": 30.0,
                "q1": 35.0,
                "q3": 65.0,
            }
        )

        data = torch.tensor([25.0, 50.0, 75.0])

        # Test minmax
        result_minmax = preprocessor.normalize(data, stats_obj, "minmax")
        expected_minmax = torch.tensor([0.25, 0.5, 0.75])
        torch.testing.assert_close(result_minmax, expected_minmax)

        # Test standard
        result_standard = preprocessor.normalize(data, stats_obj, "standard")
        expected_standard = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(result_standard, expected_standard)

        # Test invalid
        with self.assertRaises(ValueError):
            preprocessor.normalize(data, stats_obj, "invalid_type")

        if self.logger:
            self.logger.info("✅ Normalize methods test passed")

    def test_normalize_time_linear(self):
        """Test time normalization with linear method."""
        if self.logger:
            self.logger.info("Testing normalize_time with linear method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        time_features = preprocessor.normalize_time(0)

        self.assertIn("year_norm", time_features)
        self.assertIn("doy_norm", time_features)
        self.assertIn("hour_norm", time_features)

        if self.logger:
            self.logger.info("✅ Normalize_time linear test passed")

    def test_normalize_time_cos_sin(self):
        """Test time normalization with cos_sin method."""
        if self.logger:
            self.logger.info("Testing normalize_time with cos_sin method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="cos_sin",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        time_features = preprocessor.normalize_time(0)

        self.assertIn("year_norm", time_features)
        self.assertIn("doy_sin", time_features)
        self.assertIn("doy_cos", time_features)
        self.assertIn("hour_sin", time_features)
        self.assertIn("hour_cos", time_features)

        # Verify sin^2 + cos^2 ≈ 1
        sin_sq_cos_sq = time_features["doy_sin"] ** 2 + time_features["doy_cos"] ** 2
        torch.testing.assert_close(sin_sq_cos_sq, torch.ones_like(sin_sq_cos_sq))

        if self.logger:
            self.logger.info("✅ Normalize_time cos_sin test passed")

    # ------------------------------------------------------------------------
    # Dataset Methods Tests
    # ------------------------------------------------------------------------

    def test_dataset_len_method(self):
        """Test __len__ method."""
        if self.logger:
            self.logger.info("Testing __len__ method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=3,
            sbatch=5,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        self.assertEqual(len(preprocessor), 9 * 5)  # all time steps * sbatch

        if self.logger:
            self.logger.info(
                f"✅ __len__ method test passed - length = {len(preprocessor)}"
            )

    def test_getitem_train_mode(self):
        """Test __getitem__ method in train mode."""
        if self.logger:
            self.logger.info("Testing __getitem__ in train mode")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=2,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        sample = preprocessor[0]

        # Verify sample structure
        self.assertIn("inputs", sample)
        self.assertIn("targets", sample)
        self.assertIn("fine", sample)
        self.assertIn("coarse", sample)
        self.assertIn("corrdinates", sample)

        # Verify shapes
        n_vars = len(self.varnames_list)
        n_const = len(self.constant_variables)
        expected_input_channels = n_vars + 2 + n_const

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))
        self.assertEqual(sample["targets"].shape, (n_vars, 32, 32))
        self.assertEqual(sample["fine"].shape, (n_vars, 32, 32))
        self.assertEqual(sample["coarse"].shape, (n_vars, 32, 32))

        if self.logger:
            self.logger.info(
                f"✅ __getitem__ train mode test passed - inputs shape: {sample['inputs'].shape}"
            )

    def test_getitem_regional_train_mode(self):
        """Test __getitem__ method in regional train mode."""
        if self.logger:
            self.logger.info("Testing __getitem__ in train mode")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=2,
            debug=True,
            mode="train",
            run_type="train_regional",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            region_center=(10.0, 50.0),
            region_size=(32, 64),
            logger=self.logger,
        )

        sample = preprocessor[0]

        # Verify sample structure
        self.assertIn("inputs", sample)
        self.assertIn("targets", sample)
        self.assertIn("fine", sample)
        self.assertIn("coarse", sample)
        self.assertIn("corrdinates", sample)

        # Verify shapes
        n_vars = len(self.varnames_list)
        n_const = len(self.constant_variables)
        expected_input_channels = n_vars + 2 + n_const

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))
        self.assertEqual(sample["targets"].shape, (n_vars, 32, 32))
        self.assertEqual(sample["fine"].shape, (n_vars, 32, 32))
        self.assertEqual(sample["coarse"].shape, (n_vars, 32, 32))

        if self.logger:
            self.logger.info(
                f"✅ __getitem__ regional train mode test passed - inputs shape: {sample['inputs'].shape}"
            )

    def test_getitem_validation_mode(self):
        """Test __getitem__ method in validation mode."""
        if self.logger:
            self.logger.info("Testing __getitem__ in validation mode")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="validation",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        sample = preprocessor[0]

        # Verify sample structure
        self.assertIn("inputs", sample)
        self.assertIn("targets", sample)

        n_vars = len(self.varnames_list)
        n_const = len(self.constant_variables)
        expected_input_channels = n_vars + 2 + n_const

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))
        self.assertEqual(sample["targets"].shape, (n_vars, 32, 32))

        if self.logger:
            self.logger.info(
                f"✅ __getitem__ validation mode test passed - inputs shape: {sample['inputs'].shape}"
            )

    def test_getitem_with_filter_enabled(self):
        """Test __getitem__ with apply_filter=True."""
        if self.logger:
            self.logger.info("Testing __getitem__ with filter enabled")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=2,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=True,
            logger=self.logger,
        )

        sample = preprocessor[0]

        # Verify shapes still correct with filtering
        n_vars = len(self.varnames_list)
        n_const = len(self.constant_variables)
        expected_input_channels = n_vars + 2 + n_const

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))
        self.assertEqual(sample["targets"].shape, (n_vars, 32, 32))

        if self.logger:
            self.logger.info("✅ __getitem__ with filter enabled test passed")

    def test_invalid_coordinate_handling(self):
        """Test handling of invalid coordinate specifications."""
        if self.logger:
            self.logger.info("Testing invalid coordinate handling")

        # Remove latitude from steps
        bad_steps = EasyDict(self.steps)
        delattr(bad_steps, "latitude")
        delattr(bad_steps, "d_latitude")

        with self.assertRaises(AssertionError):
            DataPreprocessor(
                years=self.years,
                loaded_dfs=self.ds,
                constants_file_path=self.const_path,
                varnames_list=self.varnames_list,
                units_list=self.units_list,
                in_shape=self.in_shape,
                batch_size_lat=self.batch_size_lat,
                batch_size_lon=self.batch_size_lon,
                steps=bad_steps,
                tbatch=2,
                sbatch=4,
                debug=True,
                mode="train",
                run_type="train",
                time_normalization="linear",
                norm_mapping=self.norm_mapping,
                index_mapping=self.index_mapping,
                normalization_type=self.normalization_type,
                constant_variables=self.constant_variables,
                logger=self.logger,
            )

        if self.logger:
            self.logger.info("✅ Invalid coordinate handling test passed")

    def test_preprocessor_without_constants(self):
        """Test DataPreprocessor without constant variables."""
        if self.logger:
            self.logger.info("Testing DataPreprocessor without constants")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=None,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=None,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        self.assertIsNone(preprocessor.const_vars)

        sample = preprocessor[0]

        # Without constants, inputs should be n_vars + 2
        n_vars = len(self.varnames_list)
        expected_input_channels = n_vars + 2

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))

        if self.logger:
            self.logger.info(
                f"✅ No constants test passed - inputs shape: {sample['inputs'].shape}"
            )

    def test_invalid_time_normalization(self):
        """Test invalid time normalization method."""
        if self.logger:
            self.logger.info("Testing invalid time normalization")

        with self.assertRaises(ValueError):
            DataPreprocessor(
                years=self.years,
                loaded_dfs=self.ds,
                constants_file_path=self.const_path,
                varnames_list=self.varnames_list,
                units_list=self.units_list,
                in_shape=self.in_shape,
                batch_size_lat=self.batch_size_lat,
                batch_size_lon=self.batch_size_lon,
                steps=self.steps,
                tbatch=2,
                sbatch=4,
                debug=True,
                mode="train",
                run_type="train",
                time_normalization="invalid_method",
                norm_mapping=self.norm_mapping,
                index_mapping=self.index_mapping,
                normalization_type=self.normalization_type,
                constant_variables=self.constant_variables,
                logger=self.logger,
            )

        if self.logger:
            self.logger.info("✅ Invalid time normalization test passed")

    def tearDown(self):
        """Clean up after tests."""

        shutil.rmtree(self.temp_dir)
        if self.logger:
            self.logger.info(f"Test teardown - removed temp directory: {self.temp_dir}")


def run_tests():
    """Run all DataPreprocessor test."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
