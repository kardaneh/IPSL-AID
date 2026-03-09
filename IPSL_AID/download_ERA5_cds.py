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


import argparse
import os
import cdsapi
from tqdm import tqdm
import calendar
from IPSL_AID.logger import Logger

# python download_ERA5_cds.py --year_start 2015 --year_end 2015 --variable 2m_temperature


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments as a namespace object with attributes
        corresponding to each argument.

    Notes
    -----
    ERA5 variable names must match those defined in the
    Copernicus Climate Data Store catalogue.
    """

    parser = argparse.ArgumentParser()

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
        help="ERA5 variable names (example: 2m_temperature)",
    )

    parser.add_argument(
        "--pressure_level",
        type=str,
        nargs="+",
        required=False,
        help="Pressure levels in hPa (500 750 850)",
    )

    return parser.parse_args()


def main(logger):
    """
    Download ERA5 data from the Copernicus Climate Data Store.

    The function follows a structured workflow:

    1. Parse command-line arguments.
    2. Create output directories.
    3. Loop over requested years, variables, and months.
    4. Submit download requests to the CDS API.
    5. Save results as NetCDF files.

    Files are skipped if they already exist.

    Notes
    -----
    Data are downloaded at hourly resolution for all days of each month.

    The CDS API client requires a valid configuration file.
    Visit: https://cds.climate.copernicus.eu/

    The dataset used depends on whether pressure levels are requested:

    - reanalysis-era5-single-levels
    - reanalysis-era5-pressure-levels
    """

    args = parse_args()

    year_start = args.year_start
    year_end = args.year_end
    variables = args.variable
    pressure_levels = args.pressure_level

    # Base directory where data will be stored
    base_output_dir = "/leonardo_work/EUHPC_D27_095/kkingston/IPSL-AID/data/"

    # Initialize CDS API client
    # Requires ~/.cdsapirc to be configured
    client = cdsapi.Client()

    # Loop over years
    for year in range(year_start, year_end + 1):
        tqdm.write(f"\n=== Processing year {year} ===")

        for variable in variables:
            tqdm.write(f"\n--- Variable: {variable} ---")

            # Create output directory for this variable and year
            output_dir = os.path.join(base_output_dir, f"data_{variable.upper()}")
            year_dir = os.path.join(output_dir, str(year))
            os.makedirs(year_dir, exist_ok=True)

            # Loop over all 12 months
            for month in tqdm(range(1, 13), desc=f"{variable} {year}", unit="month"):
                mm = f"{month:02d}"
                yyyy = str(year)

                # Output file name: variable_YYYYMM.nc
                if pressure_levels:
                    level_str = "_".join(pressure_levels)
                    target = os.path.join(
                        year_dir, f"{variable}_{level_str}_{yyyy}{mm}.nc"
                    )
                else:
                    target = os.path.join(year_dir, f"{variable}_{yyyy}{mm}.nc")

                # Skip download if file already exists
                if os.path.exists(target):
                    continue

                # number of days in month
                n_days = calendar.monthrange(year, month)[1]

                # ERA5 monthly request
                request = {
                    "product_type": "reanalysis",
                    "variable": variable,
                    "year": yyyy,
                    "month": mm,
                    "day": [f"{d:02d}" for d in range(1, n_days + 1)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "format": "netcdf",
                }

                if pressure_levels:
                    request["pressure_level"] = pressure_levels

                # Submit request to CDS and download file
                if pressure_levels:
                    dataset = "reanalysis-era5-pressure-levels"
                else:
                    dataset = "reanalysis-era5-single-levels"

                client.retrieve(dataset, request, target)

    logger.success("ERA5 download completed successfully.")


if __name__ == "__main__":
    logger = Logger(console_output=True)
    logger.show_header("Download ERA5 from CDS")
    main(logger)
