# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Precompute ERA5 resized onto the native CMIP6 grid, once, and store it
as yearly NetCDF files.

Run this once per (era5_root, cmip6_root, variable) combination, then
point main.py at the output directory with --era5_on_cmip6_root.

Example
--------
Run the preprocessing for 2m air temperature:

python preprocess.py \
    --era5_root /data/kkingston/data/CMIP6/ERA5/data_6hourly_t2m \
    --cmip6_root /data/kkingston/data/CMIP6/CMIP6_historical/data_6hourly_tas \
    --variable VAR_2T \
    --start_year 1980 \
    --end_year 2014 \
    --output_dir /data/kkingston/data/CMIP6/ERA5_on_CMIP6/data_6hourly_t2m
"""

import argparse
import gc
import json
from pathlib import Path
import numpy as np

from AID_BC.dataset import ClimateDataset
from AID_BC.logger import Logger


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Precompute ERA5 resized onto the native CMIP6 grid, once, "
            "for reuse across bias-correction runs."
        )
    )

    parser.add_argument(
        "--era5_root",
        type=str,
        required=True,
        help="Directory containing raw ERA5 yearly NetCDF files.",
    )

    parser.add_argument(
        "--cmip6_root",
        type=str,
        required=True,
        help=(
            "Directory containing CMIP6 files that define the target "
            "native grid (typically the historical training root)."
        ),
    )

    parser.add_argument(
        "--variable",
        type=str,
        default="VAR_2T",
        help="Climate variable name.",
    )

    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        help="First year to precompute (inclusive).",
    )

    parser.add_argument(
        "--end_year",
        type=int,
        required=True,
        help="Last year to precompute (inclusive).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write precomputed samples_<year>.nc files to.",
    )

    return parser.parse_args()


def build_path(root, year):
    """
    Build the yearly NetCDF path for a data directory.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing yearly NetCDF files.
    year : int
        Year used in the ``samples_<year>.nc`` filename.

    Returns
    -------
    str
        Full path to the yearly NetCDF file.
    """
    # All input and output yearly files follow the same naming convention
    return str(Path(root) / f"samples_{year}.nc")


def process_year(
    year,
    era5_root,
    cmip6_root,
    variable_name,
    output_dir,
    logger,
):
    """
    Resize one year of ERA5 data onto the native CMIP6 grid and save it.

    The function loads one ERA5 file and one CMIP6 file, prepares the two
    datasets with AID_BC.dataset.ClimateDataset, resizes ERA5 onto
    the CMIP6 grid, and writes the resulting ERA5 field as a yearly NetCDF
    file.

    Parameters
    ----------
    year : int
        Year to preprocess.
    era5_root : str or pathlib.Path
        Directory containing raw ERA5 yearly files named
        ``samples_<year>.nc``.
    cmip6_root : str or pathlib.Path
        Directory containing CMIP6 yearly files that define the target native
        grid.
    variable_name : str
        Name of the climate variable to preprocess.
    output_dir : str or pathlib.Path
        Directory in which the precomputed ERA5-on-CMIP6 file is written.
    logger : Logger
        Logger used to report progress, retries, and output paths.

    Returns
    -------
    bool
        ``True`` if the original ERA5 latitude coordinate is descending,
        otherwise ``False``.
    """

    # Build the ERA5 and CMIP6 input paths for the current year
    era5_path = build_path(era5_root, year)
    cmip6_path = build_path(cmip6_root, year)

    # ClimateDataset loads both sources and performs the ERA5-to-CMIP6 resizing
    dataset = ClimateDataset(
        era5_path=era5_path,
        cmip6_path=cmip6_path,
        variable_name=variable_name,
        logger=logger,
    )

    (
        era5_on_cmip6,
        _cmip6_native,
        latitude_descending,
    ) = dataset.prepare_dataset()

    era5_on_cmip6 = era5_on_cmip6.astype(
        np.float32,
    ).load()

    dataset.close()

    era5_on_cmip6 = era5_on_cmip6.transpose(
        "time",
        "latitude",
        "longitude",
    )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write one precomputed ERA5-on-CMIP6 file per year
    output_file = output_path / f"samples_{year}.nc"

    logger.info(f"Writing precomputed ERA5-on-CMIP6 file:\n{output_file}")

    # Store the DataArray as a one-variable xarray Dataset
    output_dataset = era5_on_cmip6.to_dataset(name=variable_name)

    # Compressed float32 storage to reduce disk usage
    encoding = {
        variable_name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        }
    }

    output_dataset.to_netcdf(
        output_file,
        encoding=encoding,
    )

    output_dataset.close()

    # Release large in-memory arrays before processing the next year
    del era5_on_cmip6
    del output_dataset

    gc.collect()

    # Keep the original ERA5 latitude orientation in metadata
    return bool(latitude_descending)


def main():
    """
    Run ERA5-on-CMIP6 preprocessing for the requested year range.

    The function parses command-line arguments, preprocesses every year,
    verifies that ERA5 latitude orientation remains consistent across years,
    and writes a ``metadata.json`` file describing the generated dataset.

    Raises
    ------
    ValueError
        If the ERA5 latitude ordering changes between processed years.
    """

    args = parse_args()

    logger = Logger()

    logger.info(
        "Precomputing ERA5-on-CMIP6 grid for years "
        f"{args.start_year}-{args.end_year}, variable {args.variable}"
    )

    # Track the ERA5 latitude orientation detected during the first year
    latitude_descending = None

    # Process and save each year independently to limit memory usage
    for year in range(args.start_year, args.end_year + 1):
        logger.info(f"=== Year {year} ===")

        current_latitude_descending = process_year(
            year=year,
            era5_root=args.era5_root,
            cmip6_root=args.cmip6_root,
            variable_name=args.variable,
            output_dir=args.output_dir,
            logger=logger,
        )

        # Require a consistent latitude orientation across the full period
        if latitude_descending is None:
            latitude_descending = current_latitude_descending
        elif latitude_descending != current_latitude_descending:
            raise ValueError(
                "ERA5 latitude ordering changed between years "
                f"(mismatch detected at year {year}). "
                "This would make years inconsistent to use together."
            )

    # Record the preprocessing configuration needed by main.py when it loads
    # the precomputed ERA5 files
    metadata = {
        "variable_name": args.variable,
        "latitude_descending": latitude_descending,
        "source_era5_root": args.era5_root,
        "source_cmip6_root": args.cmip6_root,
        "start_year": args.start_year,
        "end_year": args.end_year,
    }

    # Save metadata next to the yearly NetCDF files
    metadata_path = Path(args.output_dir) / "metadata.json"

    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    logger.success("Precomputation complete.\n" f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
