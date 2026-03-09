# Copyright 2026 IPSL / CNRS / Sorbonne University
# Author: Kishanthan Kingston
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import pandas as pd
import xarray as xr

# from tqdm import tqdm
import time
from IPSL_AID.logger import Logger

# python generate_all_data_ERA5.py --year_start 2015 --year_end 2015 --variable 2m_temperature --rename_var VAR_2T


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments as a namespace object with attributes
        corresponding to each argument.

    Raises
    ------
    ValueError
        If the number of variables does not match the number of
        rename variables.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate ERA5 samples from monthly NetCDF files using CSV timestamps"
    )

    parser.add_argument(
        "--year_start", type=int, required=True, help="First year to process (2015)."
    )

    parser.add_argument(
        "--year_end", type=int, required=True, help="Last year to process (inclusive)."
    )

    parser.add_argument(
        "--variable",
        type=str,
        nargs="+",
        required=True,
        help=(
            "ERA5 variable names corresponding to downloaded monthly data."
            "Example: 2m_temperature 10m_u_component_of_wind"
        ),
    )

    parser.add_argument(
        "--pressure_level",
        type=str,
        nargs="+",
        required=False,
        help="Pressure levels to extract (500 750 850)",
    )

    parser.add_argument(
        "--rename_var",
        type=str,
        nargs="+",
        required=True,
        help=(
            "New variable name(s) to use in the output NetCDF file(s)."
            "Must match the number and order of --variable. "
            "Example: VAR_2T VAR_10U"
        ),
    )

    return parser.parse_args()


def main(logger):
    """
    Generate yearly ERA5 datasets from monthly NetCDF files.

    The function follows a structured workflow:

    1. Parse command-line arguments.
    2. Load a CSV file containing timestamps to extract.
    3. Loop over years and variables.
    4. Open monthly ERA5 NetCDF files using xarray.
    5. Extract requested timestamps.
    6. Concatenate monthly subsets into yearly datasets.
    7. Rename variables and write compressed NetCDF files.

    Notes
    -----
    ERA5 data is stored monthly, so timestamps are grouped by month.
    """

    args = parse_args()

    year_start = args.year_start
    year_end = args.year_end
    variables = args.variable
    pressure_levels = args.pressure_level
    rename_vars = args.rename_var

    if len(variables) != len(rename_vars):
        raise ValueError("Number of --variable must match number of --rename_var")

    rename_dict = dict(zip(variables, rename_vars))

    # Paths
    # CSV containing timestamps to extract
    csv_path = (
        "/leonardo_work/EUHPC_D27_095/kkingston/IPSL-AID/data/dates_hours_1980_2022.csv"
    )
    # Root directory containing downloaded monthly ERA5 data
    base_data_root = "/leonardo_work/EUHPC_D27_095/kkingston/IPSL-AID/data"
    # Root directory where extracted yearly samples will be saved
    base_output_root = "/leonardo_work/EUHPC_D27_095/kkingston/IPSL-AID/data"

    # Load CSV
    logger.info("Loading CSV file")
    df = pd.read_csv(csv_path)
    # Combine date and hour columns into a full datetime column
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["hour"])
    logger.info(f"Total timestamps: {len(df)}")

    # Main loop
    for year in range(year_start, year_end + 1):
        logger.start_task(
            "Processing year", description="Generating ERA5 samples", year=year
        )

        # Filter timestamps for current year
        df_year = df[df["datetime"].dt.year == year]

        if df_year.empty:
            logger.info("No timestamps found.")
            continue

        # Group timestamps by (year, month)
        # ERA5 data is stored monthly
        grouped_month = df_year.groupby(
            [df_year["datetime"].dt.year, df_year["datetime"].dt.month]
        )

        logger.info(f"Months with data: {len(grouped_month)}")

        for variable in variables:
            logger.step("Variable", f"Processing {variable}")

            # Directory containing monthly ERA5 files for this variable
            variable_root = os.path.join(base_data_root, f"data_{variable.upper()}")

            # Output directory for this variable
            output_root = os.path.join(
                base_output_root, f"data_FOURxDaily_{variable.upper()}"
            )
            os.makedirs(output_root, exist_ok=True)

            yearly_datasets = []

            # Loop over months
            for idx, ((yyyy, month), group) in enumerate(grouped_month):
                logger.info(
                    f"\n[{idx+1}/{len(grouped_month)}] Processing month {yyyy}-{month:02d}"
                )

                mm = f"{month:02d}"

                # Construct monthly file path
                if pressure_levels:
                    level_str = "_".join(pressure_levels)
                    monthly_file = os.path.join(
                        variable_root,
                        str(yyyy),
                        f"{variable}_{level_str}_{yyyy}{mm}.nc",
                    )
                else:
                    monthly_file = os.path.join(
                        variable_root, str(yyyy), f"{variable}_{yyyy}{mm}.nc"
                    )

                if not os.path.exists(monthly_file):
                    logger.warning(f"Missing file: {monthly_file}")
                    continue

                logger.info(f"Opening file: {monthly_file}")

                t0 = time.time()
                ds = xr.open_dataset(monthly_file)

                if pressure_levels and "pressure_level" in ds.dims:
                    ds = ds.sel(pressure_level=[int(p) for p in pressure_levels])

                logger.info(f"Opened in {time.time() - t0:.2f}s")

                # Some ERA5 datasets use "valid_time" instead of "time"
                if "valid_time" in ds.dims:
                    ds = ds.rename({"valid_time": "time"})

                if "time" not in ds.dims:
                    raise RuntimeError(f"No time dimension in {monthly_file}")

                logger.info(f"Dataset dims: {ds.dims}")

                # Extract only timestamps requested in the CSV
                requested_times = group["datetime"].values
                logger.info(f"Selecting {len(requested_times)} timestamps...")

                t0 = time.time()
                ds_sel = ds.sel(time=requested_times)
                logger.info(f"Selection done in {time.time() - t0:.2f}s")
                logger.info(f"Selected timesteps: {ds_sel.time.size}")

                if ds_sel.time.size > 0:
                    yearly_datasets.append(ds_sel)

            # Progressive concatenation
            if not yearly_datasets:
                logger.info(f"No valid data for {variable} in {year}")
                continue

            logger.info("\nStarting progressive concatenation...")
            logger.info(f"Number of monthly subsets: {len(yearly_datasets)}")

            t0 = time.time()

            ds_year = yearly_datasets[0]
            logger.info(f"Initial dataset dims: {ds_year.dims}")

            for i in range(1, len(yearly_datasets)):
                logger.info(f"Concatenating subset {i+1}/{len(yearly_datasets)}...")
                ds_year = xr.concat([ds_year, yearly_datasets[i]], dim="time")

            logger.info(f"Concatenation finished in {time.time() - t0:.2f}s")

            logger.info("Sorting by time...")
            ds_year = ds_year.sortby("time")

            logger.info(f"Final dataset dims: {ds_year.dims}")

            # Rename variable
            new_name = rename_dict[variable]
            original_var = list(ds_year.data_vars)[0]
            ds_year = ds_year.rename({original_var: new_name})

            logger.info(f"Renamed {original_var} → {new_name}")

            output_file = os.path.join(output_root, f"samples_{year}.nc")

            logger.info(f"\nWriting NetCDF: {output_file}")

            t0 = time.time()

            encoding = {new_name: {"zlib": True, "complevel": 4, "dtype": "float32"}}

            ds_year.to_netcdf(output_file, encoding=encoding)

            logger.info(f"Write completed in {time.time() - t0:.2f}s")
            logger.info(f"Total timesteps: {len(ds_year.time)}")

    logger.info("\nAll requested years processed successfully.")


if __name__ == "__main__":
    logger = Logger(console_output=True)
    logger.show_header("ERA5 Dataset Generator")
    main(logger)
