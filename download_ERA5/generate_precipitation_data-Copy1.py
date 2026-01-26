import os
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="Process ERA5 TP data for a year range")
parser.add_argument("--year_start", type=int, required=True, help="Start year (inclusive)")
parser.add_argument("--year_end", type=int, required=True, help="End year (inclusive)")
args = parser.parse_args()

year_start = args.year_start
year_end = args.year_end

var = "tp"
lonmin, lonmax = 0, 360
latmin, latmax = -90, 90

dir_in = "/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF"
dir_out = "/net/nfs/ssd1/kkingston/AI-Downscaling/data_tp"
os.makedirs(dir_out, exist_ok=True)

csv_path = "/net/nfs/ssd1/kkingston/AI-Downscaling/data/dates_hours_1980_2022_1xDaily.csv"

def open_and_select(file):
    if not os.path.exists(file):
        print(f"File not found: {file}")
        return None
    ds = xr.open_dataset(file)
    return ds.sel(
        longitude=slice(lonmin, lonmax),
        latitude=slice(latmax, latmin)
    )

# Load timestamp CSV
df_dates = pd.read_csv(csv_path)
df_dates["datetime"] = pd.to_datetime(df_dates["date"] + " " + df_dates["hour"])

# Main loop over years
for year in range(year_start, year_end + 1):
    print(f"\n=== Processing year {year} ===")
    output_filtered = os.path.join(dir_out, f"samples_{year}.nc")
    if os.path.exists(output_filtered):
        print(f"Already exists: {output_filtered} — skipping.")
        continue

    # Collect relevant files
    all_files = []

    # Previous Dec 31
    prev_year = year - 1
    for hour in ["600", "1800"]:
        fpath = f"{dir_in}/{prev_year}/12/{var}.{prev_year}1231.{hour}.fs1e5.GLOBAL_025.nc"
        if os.path.exists(fpath):
            all_files.append(fpath)

    # Current year
    for m in range(1, 13):
        dir_month = f"{dir_in}/{year}/{m:02d}"
        files_600 = sorted(glob(f"{dir_month}/{var}.{year}{m:02d}*.600.fs1e5.GLOBAL_025.nc"))
        files_1800 = sorted(glob(f"{dir_month}/{var}.{year}{m:02d}*.1800.fs1e5.GLOBAL_025.nc"))
        all_files.extend(files_600 + files_1800)

    # Next Jan 1
    next_year = year + 1
    fpath = f"{dir_in}/{next_year}/01/{var}.{next_year}0101.600.fs1e5.GLOBAL_025.nc"
    if os.path.exists(fpath):
        all_files.append(fpath)

    print(f"{len(all_files)} files found for {year}.")

    # Load and merge files
    datasets = []
    for f in tqdm(all_files, desc=f"Opening files for {year}"):
        ds = open_and_select(f)
        if ds is not None:
            datasets.append(ds)

    if len(datasets) == 0:
        print("No data found. Skipping year.")
        continue

    print("Starting concatenation...")
    ds_all = xr.concat(datasets, dim="time")
    print("Concatenation done.")

    print("Sorting and deduplicating time values...")
    ds_all = ds_all.sortby("time")
    #time_index = pd.to_datetime(ds_all.time.values)
    #ds_all = ds_all.isel(time=~time_index.duplicated())
    _, index = np.unique(ds_all.time.values, return_index=True)
    ds_all = ds_all.isel(time=np.sort(index))
    print("Time cleanup done.")

    print("Start - dataset filtering based on csv...")
    # Filter based on CSV
    year_dates = df_dates[df_dates["datetime"].dt.year == year]
    mask = pd.to_datetime(ds_all.time.values).isin(year_dates["datetime"])
    ds_filtered = ds_all.sel(time=ds_all.time[mask])
    ds_filtered = ds_filtered.rename({var: "VAR_TP"})
    print("End - dataset filtering based on csv...")

    # Save
    ds_filtered.to_netcdf(output_filtered)
    print(f"Saved: {output_filtered}")
    print(f"Finished processing {year}")

