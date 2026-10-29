import xarray as xr
import dask

variables_fine = ["var131", "var132", "var130", "var137"]
variables_coarse = ["u850", "v850", "t850", "prw"]
variables_covariates = ["u700", "u500", "v700", "v500", "t700", "t500", "q850", "q700", "q500"]
variables_diag = ["var59", "var228"]

files_fine = {
    "var131" : "u850_europe_6h_1985_1999_sorted.nc",
    "var132" : "v850_europe_6h_1985_1999_sorted.nc",
    "var130" : "t850_europe_6h_1985_1999_sorted.nc",
    "var137" : "TCWV_europe_6h_1985_1999_sorted.nc"
}
files_coarse = {
    "u850" : "u850_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "v850" : "v850_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "t850" : "t850_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "prw" : "prw_france_6h_1985_1999_sorted_regrid_bilinear.nc"
}
files_output_fine = {
    "var131" : "u850_europe_6h_1985_1999_sorted.zarr",
    "var132" : "v850_europe_6h_1985_1999_sorted.zarr",
    "var130" : "t850_europe_6h_1985_1999_sorted.zarr",
    "var137" : "TCWV_europe_6h_1985_1999_sorted.zarr"
}
files_output_coarse = {
    "u850" : "u850_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "v850" : "v850_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "t850" : "t850_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "prw" : "prw_france_6h_1985_1999_sorted_regrid_bilinear.zarr"
}
files_covariates = {
    "u700" : "u700_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "u500" : "u500_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "v700" : "v700_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "v500" : "v500_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "t700" : "t700_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "t500" : "t500_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "q850" : "q850_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "q700" : "q700_france_6h_1985_1999_sorted_regrid_bilinear.nc",
    "q500" : "q500_france_6h_1985_1999_sorted_regrid_bilinear.nc",
}
files_output_covariates = {
    "u700" : "u700_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "u500" : "u500_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "v700" : "v700_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "v500" : "v500_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "t700" : "t700_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "t500" : "t500_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "q850" : "q850_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "q700" : "q700_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
    "q500" : "q500_france_6h_1985_1999_sorted_regrid_bilinear.zarr",
}
files_diag = {
    "var59" : "cape_europe_6h_1985_1999_sorted.nc",
    "var228" : "precip_europe_6h_1985_1999_sorted.nc"
}
files_output_diag = {
    "var59" : "cape_europe_6h_1985_1999_sorted.zarr",
    "var228" : "precip_europe_6h_1985_1999_sorted.zarr"
}
file_static_cov = "static_cov.nc"

dir_fine = "/lustre/fsstor/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_fine_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_diag = "/lustre/fsstor/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_diag_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_covariates = "/lustre/fsstor/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"
dir_covariates_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"
dir_coarse = "/lustre/fsstor/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"
dir_coarse_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"


# for var in variables_fine :
#     file = files_fine[var]
#     file_output = files_output_fine[var]
#     with xr.open_dataset(dir_fine + file) as var_xr :
#         if "time" in list(var_xr[var].coords) :
#             var_xr = var_xr.chunk({"time": 10})
#         else :
#             var_xr = var_xr.chunk({"time_counter": 10})
#         var_xr.to_zarr(dir_fine_output + file_output)
#     print(file_output + " saved")

for var in variables_coarse :
    file = files_coarse[var]
    file_output = files_output_coarse[var]
    with xr.open_dataset(dir_coarse + file) as var_xr :
        if "time" in list(var_xr[var].coords) :
            var_xr = var_xr.chunk({"time": 10})
        else :
            var_xr = var_xr.chunk({"time_counter": 10})
        var_xr.to_zarr(dir_coarse_output + file_output)
    print(file_output + " saved")

for var in variables_covariates :
    file = files_covariates[var]
    file_output = files_output_covariates[var]
    with xr.open_dataset(dir_covariates + file) as var_xr :
        if "time" in list(var_xr[var].coords) :
            var_xr = var_xr.chunk({"time": 10})
        else :
            var_xr = var_xr.chunk({"time_counter": 10})
        var_xr.to_zarr(dir_covariates_output + file_output)
    print(file_output + " saved")


# for var in variables_diag :
#     file = files_diag[var]
#     file_no_extension = file.split(".")[0]
#     with xr.open_dataset(dir_diag + file) as var_xr :
#         if "time" in list(var_xr[var].coords) :
#             var_xr = var_xr.chunk({"time": 10})
#         else :
#             var_xr = var_xr.chunk({"time_counter": 10})
#         var_xr.to_zarr(dir_diag_output + file_no_extension + ".zarr")
#     print(file_no_extension + ".zarr" + " saved")

# with xr.open_dataset(dir_fine + file_static_cov) as var_xr :
#     file_no_extension = file_static_cov.split(".")[0]
#     if "time" in list(var_xr.coords) :
#         var_xr = var_xr.chunk({"time": 10})
#     else :
#         var_xr = var_xr.chunk({"time_counter": 10})
#     var_xr.to_zarr(dir_fine_output + file_no_extension + ".zarr")
# print(file_no_extension + ".zarr" + " saved")


######################################################################
######################################################################

# Do the same for the validation sets :

variables_fine = ["var131", "var132", "var130", "var137"]
variables_coarse = ["u850", "v850", "t850", "prw"]
variables_covariates = ["u700", "u500", "v700", "v500", "t700", "t500", "q850", "q700", "q500"]
variables_diag = ["var59", "var228"]


dir_fine = "/lustre/fsstor/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_fine_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_diag = "/lustre/fsstor/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_diag_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/"
dir_covariates = "/lustre/fsstor/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"
dir_covariates_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"
dir_coarse = "/lustre/fsstor/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"
dir_coarse_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/"

files_fine = {
    "var131" : "u850_europe_6h_2000_2014_sorted.nc",
    "var132" : "v850_europe_6h_2000_2014_sorted.nc",
    "var130" : "t850_europe_6h_2000_2014_sorted.nc",
    "var137" : "TCWV_europe_6h_2000_2014_sorted.nc"
}
files_coarse = {
    "u850" : "u850_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "v850" : "v850_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "t850" : "t850_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "prw" : "prw_france_6h_2000_2014_sorted_regrid_bilinear.nc"
}
files_output_fine = {
    "var131" : "u850_europe_6h_2000_2014_sorted.zarr",
    "var132" : "v850_europe_6h_2000_2014_sorted.zarr",
    "var130" : "t850_europe_6h_2000_2014_sorted.zarr",
    "var137" : "TCWV_europe_6h_2000_2014_sorted.zarr"
}
files_output_coarse = {
    "u850" : "u850_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "v850" : "v850_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "t850" : "t850_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "prw" : "prw_france_6h_2000_2014_sorted_regrid_bilinear.zarr"
}
files_covariates = {
    "u700" : "u700_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "u500" : "u500_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "v700" : "v700_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "v500" : "v500_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "t700" : "t700_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "t500" : "t500_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "q850" : "q850_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "q700" : "q700_france_6h_2000_2014_sorted_regrid_bilinear.nc",
    "q500" : "q500_france_6h_2000_2014_sorted_regrid_bilinear.nc",
}
files_output_covariates = {
    "u700" : "u700_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "u500" : "u500_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "v700" : "v700_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "v500" : "v500_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "t700" : "t700_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "t500" : "t500_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "q850" : "q850_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "q700" : "q700_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
    "q500" : "q500_france_6h_2000_2014_sorted_regrid_bilinear.zarr",
}
files_diag = {
    "var59" : "cape_europe_6h_2000_2014_sorted.nc",
    "var228" : "precip_europe_6h_2000_2014_sorted.nc"
}
files_diag_output = {
    "var59" : "cape_europe_6h_2000_2014_sorted.zarr",
    "var228" : "precip_europe_6h_2000_2014_sorted.zarr"
}

# for var in variables_fine :
#     file = files_fine[var]
#     file_output = files_output_fine[var]
#     with xr.open_dataset(dir_fine + file) as var_xr :
#         if "time" in list(var_xr[var].coords) :
#             var_xr = var_xr.chunk({"time": 10})
#         else :
#             var_xr = var_xr.chunk({"time_counter": 10})
#         var_xr.to_zarr(dir_fine_output + file_output)
#     print(file_output + " saved")

for var in variables_coarse :
    file = files_coarse[var]
    file_output = files_output_coarse[var]
    with xr.open_dataset(dir_coarse + file) as var_xr :
        if "time" in list(var_xr[var].coords) :
            var_xr = var_xr.chunk({"time": 10})
        else :
            var_xr = var_xr.chunk({"time_counter": 10})
        var_xr.to_zarr(dir_coarse_output + file_output)
    print(file_output + " saved")

for var in variables_covariates :
    file = files_covariates[var]
    file_output = files_output_covariates[var]
    with xr.open_dataset(dir_covariates + file) as var_xr :
        if "time" in list(var_xr[var].coords) :
            var_xr = var_xr.chunk({"time": 10})
        else :
            var_xr = var_xr.chunk({"time_counter": 10})
        var_xr.to_zarr(dir_covariates_output + file_output)
    print(file_output + " saved")

# for var in variables_diag :
#     file = files_diag[var]
#     file_output = files_diag_output[var]
#     with xr.open_dataset(dir_diag + file) as var_xr :
#         if "time" in list(var_xr[var].coords) :
#             var_xr = var_xr.chunk({"time": 10})
#         else :
#             var_xr = var_xr.chunk({"time_counter": 10})
#         var_xr.to_zarr(dir_diag_output + file_output)
#     print(file_output + " saved")