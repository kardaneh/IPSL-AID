import os
import argparse

# CONFIGURATION
# year_start = 1950
# year_end = 2022

# Argument parser setup
parser = argparse.ArgumentParser(description="Generate data for a range of years.")
parser.add_argument(
    "--year_start", type=int, required=True, help="Start year (inclusive)"
)
parser.add_argument("--year_end", type=int, required=True, help="End year (inclusive)")

args = parser.parse_args()
year_start = args.year_start
year_end = args.year_end

# Loop over all years and months
for year in range(year_start, year_end + 1):
    print(f"Processing year {year}")

    for month in range(1, 13):
        print(f"Subsampling {year}-{month:02d}")
        # command = f"python preprocessing_subsample.py --year {year} --month {month}"
        command = (
            f"python preprocessing_subsample_fixed4ts.py --year {year} --month {month}"
        )
        status = os.system(command)
        if status != 0:
            print(f"Failed at {year}-{month:02d}")
            break  # Stop if something fails

    # Once all months are done, concatenate them
    print(f"Concatenating year {year}")
    concat_command = f"python preprocessing_concat_year.py --year {year}"
    os.system(concat_command)

print("All years processed.")
