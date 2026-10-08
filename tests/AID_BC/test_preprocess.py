# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.preprocess import build_path, process_year, main


# python -m unittest tests.test_preprocess


# ============================================================================
# Test utilities
# ============================================================================


def create_yearly_dummy_era5_netcdf(
    root_dir,
    year,
    variable_name="VAR_2T",
    n_time=4,
    latitude_descending=True,
):
    """
    Create a yearly dummy ERA5 NetCDF file.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Directory in which the file is created.
    year : int
        Year used in the output filename and time coordinate.
    variable_name : str, default="VAR_2T"
        Name of the ERA5 variable.
    n_time : int, default=4
        Number of six-hourly time steps.
    latitude_descending : bool, default=True
        Whether the ERA5 latitude coordinate is stored north-to-south.

    Returns
    -------
    path : str
        Path to the generated NetCDF file.
    dataset : xarray.Dataset
        In-memory dataset used to write the file.
    """
    os.makedirs(root_dir, exist_ok=True)

    if latitude_descending:
        latitude = np.linspace(90.0, -90.0, 10)
    else:
        latitude = np.linspace(-90.0, 90.0, 10)

    longitude = np.linspace(0.0, 342.0, 20)
    time = pd.date_range(
        f"{year}-01-01",
        periods=n_time,
        freq="6h",
    )

    rng = np.random.default_rng(year)
    data = rng.standard_normal(
        (
            len(time),
            len(latitude),
            len(longitude),
        )
    ).astype(np.float32)

    dataset = xr.Dataset(
        {
            variable_name: (
                (
                    "time",
                    "latitude",
                    "longitude",
                ),
                data,
            )
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    path = os.path.join(
        root_dir,
        f"samples_{year}.nc",
    )
    dataset.to_netcdf(path)

    return path, dataset


def create_yearly_dummy_cmip6_netcdf(
    root_dir,
    year,
    variable_name="VAR_2T",
    n_time=4,
    use_lat_lon=True,
):
    """
    Create a yearly dummy CMIP6 NetCDF file.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Directory in which the file is created.
    year : int
        Year used in the output filename and time coordinate.
    variable_name : str, default="VAR_2T"
        Name of the CMIP6 variable.
    n_time : int, default=4
        Number of six-hourly time steps.
    use_lat_lon : bool, default=True
        If ``True``, use CMIP-style ``lat`` and ``lon`` coordinate names.
        Otherwise, use ``latitude`` and ``longitude``.

    Returns
    -------
    path : str
        Path to the generated NetCDF file.
    dataset : xarray.Dataset
        In-memory dataset used to write the file.
    """
    os.makedirs(root_dir, exist_ok=True)

    latitude_name = "lat" if use_lat_lon else "latitude"
    longitude_name = "lon" if use_lat_lon else "longitude"

    latitude = np.linspace(-75.0, 75.0, 6)
    longitude = np.linspace(0.0, 330.0, 12)
    time = pd.date_range(
        f"{year}-01-01",
        periods=n_time,
        freq="6h",
    )

    rng = np.random.default_rng(year + 1000)
    data = rng.standard_normal(
        (
            len(time),
            len(latitude),
            len(longitude),
        )
    ).astype(np.float32)

    dataset = xr.Dataset(
        {
            variable_name: (
                (
                    "time",
                    latitude_name,
                    longitude_name,
                ),
                data,
            )
        },
        coords={
            "time": time,
            latitude_name: latitude,
            longitude_name: longitude,
        },
    )

    path = os.path.join(
        root_dir,
        f"samples_{year}.nc",
    )
    dataset.to_netcdf(path)

    return path, dataset


# ============================================================================
# Unit tests for preprocess.py
# ============================================================================


class TestPreprocess(unittest.TestCase):
    """Unit tests for the ERA5-on-CMIP6 preprocessing module."""

    def setUp(self):
        """Create temporary directories and a test logger."""
        self.temp_dir = tempfile.mkdtemp()

        self.era5_root = os.path.join(
            self.temp_dir,
            "era5",
        )

        self.cmip6_root = os.path.join(
            self.temp_dir,
            "cmip6",
        )

        self.output_dir = os.path.join(
            self.temp_dir,
            "era5_on_cmip6",
        )

        self.variable_name = "VAR_2T"

        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.logger.info(f"Test setup - created temporary directory: {self.temp_dir}")

    # ------------------------------------------------------------------------
    # Path tests
    # ------------------------------------------------------------------------

    def test_build_path(self):
        """Test yearly NetCDF path construction."""
        self.logger.info("Testing yearly NetCDF path construction")

        path = build_path(
            self.era5_root,
            2020,
        )

        self.assertEqual(
            path,
            os.path.join(
                self.era5_root,
                "samples_2020.nc",
            ),
        )

        self.logger.info("✅ Yearly NetCDF path construction test passed")

    # ------------------------------------------------------------------------
    # process_year tests
    # ------------------------------------------------------------------------

    def test_process_year_writes_era5_on_cmip6_file(self):
        """
        Test preprocessing and writing one ERA5-on-CMIP6 yearly file.
        """
        self.logger.info(
            "Testing preprocessing and writing of one ERA5-on-CMIP6 yearly file"
        )

        year = 2020

        _, era5_dataset = create_yearly_dummy_era5_netcdf(
            root_dir=self.era5_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
            latitude_descending=True,
        )

        _, cmip6_dataset = create_yearly_dummy_cmip6_netcdf(
            root_dir=self.cmip6_root,
            year=year,
            variable_name=self.variable_name,
            n_time=4,
            use_lat_lon=True,
        )

        latitude_descending = process_year(
            year=year,
            era5_root=self.era5_root,
            cmip6_root=self.cmip6_root,
            variable_name=self.variable_name,
            output_dir=self.output_dir,
            logger=self.logger,
        )

        output_file = os.path.join(
            self.output_dir,
            f"samples_{year}.nc",
        )

        self.assertTrue(latitude_descending)
        self.assertTrue(os.path.isfile(output_file))

        with xr.open_dataset(output_file) as output_dataset:
            self.assertIn(
                self.variable_name,
                output_dataset.data_vars,
            )

            output = output_dataset[self.variable_name]

            self.assertEqual(
                output.dtype,
                np.dtype("float32"),
            )

            self.assertEqual(
                output.dims,
                (
                    "time",
                    "latitude",
                    "longitude",
                ),
            )

            # Time comes from ERA5, while the spatial shape comes from CMIP6.
            self.assertEqual(
                output.sizes["time"],
                era5_dataset.sizes["time"],
            )
            self.assertEqual(
                output.sizes["latitude"],
                cmip6_dataset.sizes["lat"],
            )
            self.assertEqual(
                output.sizes["longitude"],
                cmip6_dataset.sizes["lon"],
            )

            # The CMIP6 grid is reordered to match the original ERA5
            # north-to-south latitude convention.
            expected_latitude = np.sort(cmip6_dataset["lat"].values)[::-1]

            np.testing.assert_allclose(
                output["latitude"].values,
                expected_latitude,
            )

            np.testing.assert_allclose(
                output["longitude"].values,
                np.sort(cmip6_dataset["lon"].values % 360.0),
            )

            np.testing.assert_array_equal(
                output["time"].values,
                era5_dataset["time"].values,
            )

        self.logger.info("✅ ERA5-on-CMIP6 yearly preprocessing test passed")

    def test_process_year_returns_false_for_ascending_era5_latitude(self):
        """Test detection of ascending ERA5 latitude coordinates."""
        self.logger.info("Testing detection of ascending ERA5 latitude coordinates")

        year = 2020

        create_yearly_dummy_era5_netcdf(
            root_dir=self.era5_root,
            year=year,
            variable_name=self.variable_name,
            latitude_descending=False,
        )

        create_yearly_dummy_cmip6_netcdf(
            root_dir=self.cmip6_root,
            year=year,
            variable_name=self.variable_name,
        )

        latitude_descending = process_year(
            year=year,
            era5_root=self.era5_root,
            cmip6_root=self.cmip6_root,
            variable_name=self.variable_name,
            output_dir=self.output_dir,
            logger=self.logger,
        )

        self.assertFalse(latitude_descending)

        self.logger.info("✅ Ascending ERA5 latitude detection test passed")

    # ------------------------------------------------------------------------
    # main tests
    # ------------------------------------------------------------------------

    def test_main_writes_single_year_netcdf_and_metadata(self):
        """Test one-year preprocessing through the command-line entry point."""
        self.logger.info("Testing single-year preprocessing and metadata writing")

        year = 2020

        create_yearly_dummy_era5_netcdf(
            root_dir=self.era5_root,
            year=year,
            variable_name=self.variable_name,
            latitude_descending=True,
        )

        create_yearly_dummy_cmip6_netcdf(
            root_dir=self.cmip6_root,
            year=year,
            variable_name=self.variable_name,
        )

        test_args = [
            "preprocess.py",
            "--era5_root",
            self.era5_root,
            "--cmip6_root",
            self.cmip6_root,
            "--variable",
            self.variable_name,
            "--start_year",
            str(year),
            "--end_year",
            str(year),
            "--output_dir",
            self.output_dir,
        ]

        with patch.object(
            sys,
            "argv",
            test_args,
        ):
            main()

        output_file = os.path.join(
            self.output_dir,
            f"samples_{year}.nc",
        )
        metadata_file = os.path.join(
            self.output_dir,
            "metadata.json",
        )

        self.assertTrue(os.path.isfile(output_file))
        self.assertTrue(os.path.isfile(metadata_file))

        with open(
            metadata_file,
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)

        self.assertEqual(
            metadata["variable_name"],
            self.variable_name,
        )
        self.assertTrue(metadata["latitude_descending"])
        self.assertEqual(
            metadata["source_era5_root"],
            self.era5_root,
        )
        self.assertEqual(
            metadata["source_cmip6_root"],
            self.cmip6_root,
        )
        self.assertEqual(
            metadata["start_year"],
            year,
        )
        self.assertEqual(
            metadata["end_year"],
            year,
        )

        self.logger.info(
            "✅ Single-year preprocessing and metadata writing test passed"
        )

    def test_main_writes_one_netcdf_file_per_year(self):
        """Test that multiple years are written as separate NetCDF files."""
        self.logger.info("Testing one NetCDF output file per preprocessing year")

        years = [2020, 2021]

        for year in years:
            create_yearly_dummy_era5_netcdf(
                root_dir=self.era5_root,
                year=year,
                variable_name=self.variable_name,
                latitude_descending=True,
            )

            create_yearly_dummy_cmip6_netcdf(
                root_dir=self.cmip6_root,
                year=year,
                variable_name=self.variable_name,
            )

        test_args = [
            "preprocess.py",
            "--era5_root",
            self.era5_root,
            "--cmip6_root",
            self.cmip6_root,
            "--variable",
            self.variable_name,
            "--start_year",
            str(years[0]),
            "--end_year",
            str(years[-1]),
            "--output_dir",
            self.output_dir,
        ]

        with patch.object(
            sys,
            "argv",
            test_args,
        ):
            main()

        for year in years:
            output_file = os.path.join(
                self.output_dir,
                f"samples_{year}.nc",
            )
            self.assertTrue(os.path.isfile(output_file))

            with xr.open_dataset(output_file) as dataset:
                self.assertEqual(
                    dataset[self.variable_name].sizes["time"],
                    4,
                )
                self.assertEqual(
                    str(dataset["time"].values[0])[:10],
                    f"{year}-01-01",
                )

        metadata_file = os.path.join(
            self.output_dir,
            "metadata.json",
        )

        with open(
            metadata_file,
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)

        self.assertEqual(
            metadata["start_year"],
            years[0],
        )
        self.assertEqual(
            metadata["end_year"],
            years[-1],
        )

        self.logger.info("✅ Multi-year preprocessing output test passed")

    def test_main_raises_if_latitude_orientation_changes(self):
        """
        Test rejection of inconsistent ERA5 latitude ordering across years.
        """
        self.logger.info(
            "Testing rejection of inconsistent ERA5 latitude ordering across years"
        )

        test_args = [
            "preprocess.py",
            "--era5_root",
            self.era5_root,
            "--cmip6_root",
            self.cmip6_root,
            "--variable",
            self.variable_name,
            "--start_year",
            "2020",
            "--end_year",
            "2021",
            "--output_dir",
            self.output_dir,
        ]

        with patch.object(
            sys,
            "argv",
            test_args,
        ):
            with patch(
                "AID_BC.preprocess.process_year",
                side_effect=[True, False],
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "latitude ordering changed",
                ):
                    main()

        self.logger.info("✅ Inconsistent ERA5 latitude-order rejection test passed")

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    def tearDown(self):
        """Remove all temporary test files."""
        shutil.rmtree(
            self.temp_dir,
            ignore_errors=True,
        )

        self.logger.info(
            f"Test teardown - removed temporary directory: {self.temp_dir}"
        )


def run_tests():
    """Run all preprocess tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPreprocess))

    runner = unittest.TextTestRunner(verbosity=2)

    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
