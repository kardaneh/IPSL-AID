# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
import xarray as xr
import torch
import torchvision


def resize_with_torchvision(da, target_shape):
    """
    Resize a DataArray's spatial dimensions using torchvision's bilinear
    interpolation with antialiasing, matching the resizing method used
    in DataPreprocessor.coarse_down_up during model training.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array with dims (time, latitude, longitude).
    target_shape : tuple of int
        Target (H, W) shape for latitude and longitude.

    Returns
    -------
    np.ndarray
        Resized array of shape (time, H, W). Coordinates are NOT attached;
        the caller must reassign latitude/longitude coordinates explicitly.
    """
    values = da.values.astype(np.float32)  # (time, H, W)
    tensor = torch.from_numpy(values).unsqueeze(1)  # (time, 1, H, W)

    resize_transform = torchvision.transforms.Resize(
        target_shape,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    # Remove the channel dimension
    resized = resize_transform(tensor).squeeze(1).numpy()  # (time, Hc, Wc)

    return resized


class ClimateDataset:
    """
    Climate dataset preprocessing for native CMIP6-grid bias correction.

    This class:
    - loads ERA5 and CMIP6 datasets,
    - renames coordinates to a common ERA5-style convention,
    - normalizes longitudes to [0, 360),
    - sorts latitudes consistently,
    - interpolates ERA5 onto the native CMIP6 grid when ERA5 is provided,
    - keeps CMIP6 on its native grid.

    The class supports two modes:

    1. ERA5 and CMIP6 mode
       Used for training or evaluation. ERA5 is interpolated onto the
       native CMIP6 grid and returned together with CMIP6.

    2. CMIP6-only mode
       Used for application periods where ERA5 is not available or not
       required. CMIP6 is standardized and returned on its native grid.

    Attributes
    ----------
    era5_data : xarray.DataArray or None
        ERA5 variable interpolated onto the native CMIP6 grid.

    cmip6_data : xarray.DataArray or None
        CMIP6 variable on its native grid.
    """

    def __init__(self, cmip6_path, variable_name, era5_path=None, logger=None):
        """
        Initialize the climate dataset handler.

        Parameters
        ----------
        cmip6_path : str
            Path to the CMIP6 NetCDF file.

        variable_name : str
            Name of the climate variable to process.

        era5_path : str, optional
            Path to the ERA5 NetCDF file. If None, only CMIP6 is prepared.

        logger : Logger, optional
            Custom logger instance.
        """
        self.era5_path = era5_path
        self.cmip6_path = cmip6_path
        self.variable_name = variable_name
        self.logger = logger

        self.era5 = None
        self.cmip6 = None

        self.era5_data = None
        self.cmip6_data = None

        self.era5_lat_descending = None

    def load_cmip6(self):
        """
        Load the CMIP6 dataset and check that the target variable exists.

        Raises
        ------
        ValueError
            If the requested variable is not found in the CMIP6 dataset.
        """
        if self.logger:
            self.logger.info(f"Loading CMIP6 dataset:\n{self.cmip6_path}")

        self.cmip6 = xr.open_dataset(self.cmip6_path)

        if self.variable_name not in self.cmip6:
            raise ValueError(
                f"Variable '{self.variable_name}' not found in CMIP6 dataset"
            )

    def load_era5(self):
        """
        Load the ERA5 dataset and check that the target variable exists.

        Raises
        ------
        ValueError
            If no ERA5 path is provided or if the requested variable is not
            found in the ERA5 dataset.
        """
        if self.era5_path is None:
            raise ValueError("era5_path is required for ERA5 interpolation.")

        if self.logger:
            self.logger.info(f"Loading ERA5 dataset:\n{self.era5_path}")

        self.era5 = xr.open_dataset(self.era5_path)

        if self.variable_name not in self.era5:
            raise ValueError(
                f"Variable '{self.variable_name}' not found in ERA5 dataset"
            )

    @staticmethod
    def rename_coordinates(ds):
        """
        Rename common CMIP-style coordinates to ERA5-style coordinate names.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Input dataset or data array.

        Returns
        -------
        ds : xarray.Dataset or xarray.DataArray
            Dataset or data array with standardized coordinate names.

        Raises
        ------
        ValueError
            If latitude or longitude coordinates are missing.
        """

        rename_dict = {}

        # Rename latitude/longitude coordinate if it uses the CMIP-style name "lat"/"lon"
        if "lat" in ds.coords:
            rename_dict["lat"] = "latitude"

        if "lon" in ds.coords:
            rename_dict["lon"] = "longitude"

        if rename_dict:
            ds = ds.rename(rename_dict)

        if "latitude" not in ds.coords:
            raise ValueError("Dataset has no latitude coordinate.")

        if "longitude" not in ds.coords:
            raise ValueError("Dataset has no longitude coordinate.")

        return ds

    @staticmethod
    def is_latitude_descending(ds):
        """
        Check whether latitude is ordered from north to south.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Input dataset or data array with a latitude coordinate.

        Returns
        -------
        bool
            True if latitude is descending, False otherwise.
        """
        lat = ds["latitude"].values
        return lat[0] > lat[-1]

    @staticmethod
    def normalize_longitudes(ds):
        """
        Normalize longitudes to [0, 360), sort them, and remove duplicates.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Input dataset or data array with a longitude coordinate.

        Returns
        -------
        ds : xarray.Dataset or xarray.DataArray
            Dataset or data array with normalized and sorted longitudes.
        """

        # Convert longitude values to the [0, 360) convention
        lon = ds["longitude"].values
        lon_new = lon % 360

        ds = ds.assign_coords(longitude=lon_new)

        # Sort longitudes after normalization
        ds = ds.sortby("longitude")

        # Remove duplicate longitudes if both equivalent values existed,
        # for example 0 and 360 degrees
        lon_sorted = ds["longitude"].values
        _, unique_indices = np.unique(lon_sorted, return_index=True)
        unique_indices = np.sort(unique_indices)

        ds = ds.isel(longitude=unique_indices)

        return ds

    @staticmethod
    def sort_latitude(ds, descending=False):
        """
        Sort latitude either ascending or descending.
        ERA5 often uses descending latitude.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Input dataset or data array with a latitude coordinate.

        descending : bool, default=False
            If True, sort latitude from north to south.
            If False, sort latitude from south to north.

        Returns
        -------
        ds : xarray.Dataset or xarray.DataArray
            Dataset or data array with sorted latitude.
        """

        return ds.sortby("latitude", ascending=not descending)

    @staticmethod
    def standardize_coordinates(ds, latitude_descending=False):
        """
        Apply coordinate naming and ordering conventions.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Input dataset or data array.

        latitude_descending : bool, default=False
            Whether the output latitude coordinate should be descending.

        Returns
        -------
        ds : xarray.Dataset or xarray.DataArray
            Dataset or data array with standardized coordinates.
        """

        ds = ClimateDataset.rename_coordinates(ds)
        ds = ClimateDataset.normalize_longitudes(ds)
        ds = ClimateDataset.sort_latitude(ds, descending=latitude_descending)

        return ds

    def prepare_dataset(self, latitude_descending=None):
        """
        Run the dataset preparation pipeline.

        If ERA5 is available, ERA5 is interpolated onto the native CMIP6 grid.
        This mode is used for training and evaluation.

        If ERA5 is not available, only CMIP6 is prepared. This mode is used
        for application periods, including future CMIP6 simulations.

        Parameters
        ----------
        latitude_descending : bool or None
            Latitude direction to use when preparing CMIP6 without ERA5.
            Required when era5_path is None.

            If ERA5 is provided, this argument is ignored because the original
            ERA5 latitude convention is detected automatically and used for
            the final output ordering.

        Returns
        -------
        tuple or xarray.DataArray
            If ERA5 is provided, returns:

            era5_on_cmip6 : xarray.DataArray
                ERA5 variable interpolated onto the native CMIP6 grid.

            cmip6_native : xarray.DataArray
                CMIP6 variable on its native grid.

            era5_lat_descending : bool
                Whether the original ERA5 latitude coordinate is descending.

            If ERA5 is not provided, returns:

            cmip6_native : xarray.DataArray
                CMIP6 variable on its native grid.

        Raises
        ------
        ValueError
            If latitude_descending is not provided in CMIP6-only mode.
        """

        # Case 1: ERA5 + CMIP6
        # This is used for training and for evaluation against ERA5
        if self.era5_path is not None:
            self.load_era5()
            self.load_cmip6()

            # self.era5 = self.rename_coordinates(self.era5)
            # self.cmip6 = self.rename_coordinates(self.cmip6)

            self.era5_lat_descending = self.is_latitude_descending(self.era5)

            # Interpolation is performed with latitude sorted ascending
            era5_interp_grid = self.standardize_coordinates(
                self.era5,
                latitude_descending=False,
            )

            cmip6_interp_grid = self.standardize_coordinates(
                self.cmip6,
                latitude_descending=False,
            )

            if self.logger:
                self.logger.info(
                    "Resizing ERA5 onto CMIP6 native shape using torchvision "
                    "(matching coarse_down_up resizing used during model training)"
                )

            cmip6_var = cmip6_interp_grid[self.variable_name]
            target_shape = (
                cmip6_var.sizes["latitude"],
                cmip6_var.sizes["longitude"],
            )

            era5_lat_range = (
                era5_interp_grid["latitude"].min().item(),
                era5_interp_grid["latitude"].max().item(),
            )
            era5_lon_range = (
                era5_interp_grid["longitude"].min().item(),
                era5_interp_grid["longitude"].max().item(),
            )
            cmip6_lat_range = (
                cmip6_var["latitude"].min().item(),
                cmip6_var["latitude"].max().item(),
            )
            cmip6_lon_range = (
                cmip6_var["longitude"].min().item(),
                cmip6_var["longitude"].max().item(),
            )

            if self.logger:
                self.logger.info(
                    f"ERA5 lat range: {era5_lat_range}, CMIP6 lat range: {cmip6_lat_range}"
                )
                self.logger.info(
                    f"ERA5 lon range: {era5_lon_range}, CMIP6 lon range: {cmip6_lon_range}"
                )

            era5_resized_values = resize_with_torchvision(
                era5_interp_grid[self.variable_name],
                target_shape=target_shape,
            )

            era5_on_cmip6 = xr.DataArray(
                era5_resized_values,
                dims=("time", "latitude", "longitude"),
                coords={
                    "time": era5_interp_grid["time"],
                    "latitude": cmip6_var["latitude"],
                    "longitude": cmip6_var["longitude"],
                },
                name=self.variable_name,
            )

            # CMIP6 stays on its native grid
            cmip6_native = cmip6_interp_grid[self.variable_name]

            # Reorder both arrays to match the original ERA5 latitude
            # convention used throughout the pipeline
            era5_on_cmip6 = self.sort_latitude(
                era5_on_cmip6,
                descending=self.era5_lat_descending,
            )

            cmip6_native = self.sort_latitude(
                cmip6_native,
                descending=self.era5_lat_descending,
            )

            era5_on_cmip6 = era5_on_cmip6.transpose(
                "time",
                "latitude",
                "longitude",
            )

            cmip6_native = cmip6_native.transpose(
                "time",
                "latitude",
                "longitude",
            )

            self.era5_data = era5_on_cmip6
            self.cmip6_data = cmip6_native

            if self.logger:
                self.logger.success(
                    f"Prepared ERA5/CMIP6 pair on native CMIP6 grid: "
                    f"{self.cmip6_data.shape}"
                )

            return self.era5_data, self.cmip6_data, self.era5_lat_descending

        # Case 2: CMIP6 only
        # This is used for the application period where ERA5 is not required
        if latitude_descending is None:
            raise ValueError(
                "latitude_descending is required when preparing CMIP6 without ERA5."
            )

        self.load_cmip6()

        # Standardize CMIP6 using the latitude convention detected from ERA5
        # during the training period
        self.cmip6 = self.standardize_coordinates(
            self.cmip6,
            latitude_descending=latitude_descending,
        )

        cmip6_native = self.cmip6[self.variable_name].transpose(
            "time",
            "latitude",
            "longitude",
        )

        self.cmip6_data = cmip6_native

        if self.logger:
            self.logger.success(
                f"Prepared CMIP6-only data on native grid: " f"{self.cmip6_data.shape}"
            )

        return self.cmip6_data

    def close(self):
        """
        Close opened xarray datasets.

        This should be called after the prepared DataArrays have been loaded
        into memory or are no longer needed. Closing datasets prevents too many
        NetCDF files from remaining open during multi-year processing.
        """
        if self.era5 is not None:
            self.era5.close()

        if self.cmip6 is not None:
            self.cmip6.close()
