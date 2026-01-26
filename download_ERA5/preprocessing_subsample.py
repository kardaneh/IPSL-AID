import os
import numpy as np
import xarray as xr
import random
import argparse
import pandas as pd


# CLUSTER-SPECIFIC CONFIGURATION
root_data_dir = (
    "/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/"  # Path to the existing ERA5 data
)
output_dir = "../data/"  # Output directory for processed samples

# Get arguments from argparser
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument(
    "--remove_files", action="store_true", help="Remove source files after processing"
)
args = parser.parse_args()

## Provide year and month as input to this file using args
year = args.year
month = args.month
# last_day = args.last_day
remove_files = args.remove_files

# first_day=1

month_str = f"{month:02d}"  # Format month as two digits (02)

# Variable names of surface files
# varnames= {"VAR_2T":"128_167_2t",
# "VAR_10U":"128_165_10u",
# "VAR_10V":"128_166_10v"}
variables = ["t2m", "u10", "v10"]  # 2m temperature, 10m u-wind, 10m v-wind
rename_map = {"t2m": "VAR_2T", "u10": "VAR_10U", "v10": "VAR_10V"}

# Set random seed for reproducibility, but different for each year/month
seed = year * 12 + month
print(seed)
random.seed(seed)

# Open file
first_var = variables[0]
first_path = os.path.join(
    root_data_dir, str(year), f"{first_var}.{year}{month_str}.as1e5.GLOBAL_025.nc"
)
ds = xr.open_dataset(first_path)

"""
# Select time inds randomly
time_inds = np.arange(len(ds.time), dtype=int)
random.shuffle(time_inds)
## Select 30 time inds from this month
time_inds = time_inds[:30]
"""
# Convert time coordinates from xarray to pandas datetime format
times = pd.to_datetime(ds.time.values)

# Get all unique days (as dates) present in the dataset
unique_days = np.unique(times.date)

# List to store the final selected indices
selected_inds = []

# Loop through each unique day
for day in unique_days:
    # Find all indices in the dataset corresponding to this specific day
    day_inds = np.where(times.date == day)[0]

    # Make sure there are at least 4 timestamps for the current day
    if len(day_inds) >= 4:
        # Randomly select 4 indices for this day
        selected = random.sample(list(day_inds), 4)
        # Add them to the final list of selected indices
        selected_inds.extend(selected)
    else:
        # Log a warning if the day has fewer than 4 timestamps
        print(f"Not enough timestamps for {day}, found {len(day_inds)}")

# Optional: sort the indices to keep the final selection in chronological order
selected_inds.sort()

# Use these selected indices to subset the xarray dataset
time_inds = selected_inds

# Pre-processed dataset
ds_proc = ds.isel(time=time_inds).rename(
    {first_var: rename_map[first_var]}
)  # rename variable t2m --> VAR_2T

## Open next vars and add them to the dataset.
for var in variables[1:]:
    # Open file
    file_path = os.path.join(
        root_data_dir, str(year), f"{var}.{year}{month_str}.as1e5.GLOBAL_025.nc"
    )
    ds = xr.open_dataset(file_path)

    # Pre-processed dataset and add to existing
    ds_proc2 = ds.isel(time=time_inds).rename({var: rename_map[var]})
    ds_proc = xr.merge([ds_proc, ds_proc2])

output_file = f"samples_{year}{month_str}.nc"
output_path = os.path.join(output_dir, output_file)
ds_proc.to_netcdf(output_path)
print(f"Saved: {output_path}")

if remove_files:
    print("Removing intermediate files")
    for var in variables:
        file_path = os.path.join(
            root_data_dir, str(year), f"{var}.{year}{month_str}.as1e5.GLOBAL_025.nc"
        )
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
