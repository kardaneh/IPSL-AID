# prepare CorrDiff : this file takes predictions made by the unet for the train periods and prepares the dataset to be used as input to train the diffusion model.
###########################
# 1 - compute and print normalisation constants
###########################
 
import xarray as xr
import numpy as np
import torch
import dask
import argparse

####################
def precip_log_transform(data_tensor) :
    """
    data_tensor is of shape (T,C,L,W)
    """
    if torch.is_tensor(data_tensor) :
        data_tensor = torch.clamp(data_tensor, min=0)
        transformed_data = torch.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = torch.log(data_tensor[:,0,:,:] + 1)
    else :
        data_tensor = np.clip(data_tensor, a_min=0, a_max=None)
        transformed_data = np.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = np.log(data_tensor[:,0,:,:] + 1)
    return transformed_data

def is_sorted(time_array) :
    return np.all(time_array[:-1] <= time_array[1:])

##########################

def prepare_corrdiff(
    dir_coarse = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA/unet_baseline_22_09_2026_corrdiff_all_train_period/results/",
    variables_coarse = ["var131", "var132", "var130", "var137", "var59", "var228"],
    variables_fine = ["var131", "var132", "var130", "var137", "var59", "var228"],
    file_prediction = "predictions_test_save_validation_member_0.nc",
    period="1985_1999" # or "2000-2014"
) :
    mean_coarses = []
    std_coarses = []
    min_coarses = []
    max_coarses = []
    mean_fines = []
    std_fines = []
    min_fines = []
    max_fines = []
    mean_residuals = []
    std_residuals = []
    min_residuals = []
    max_residuals = []

    files_fine = {
        "var131" : f"u850_europe_6h_{period}_sorted.zarr",
        "var132" : f"v850_europe_6h_{period}_sorted.zarr",
        "var130" : f"t850_europe_6h_{period}_sorted.zarr",
        "var137" : f"TCWV_europe_6h_{period}_sorted.zarr",
        "var59" : f"cape_europe_6h_{period}_sorted.zarr",
        "var228" : f"precip_europe_6h_{period}_sorted.zarr"
    }
    files_coarse = {
        "var131" : file_prediction,
        "var132" : file_prediction,
        "var130" : file_prediction,
        "var137" : file_prediction,
        "var59" : file_prediction,
        "var228" : file_prediction
    }

    norm_fine = {
        "var131" : "std",
        "var132" : "std",
        "var130" : "std",
        "var137" : "std",
        "var59" : "log",
        "var228" : "log"
    }
    norm_coarse = norm_fine

    dir_fine = "/lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/"

    slice_lon = slice(-6, 9.75)
    slice_lat = slice(54.75, 39)

    for var_coarse, var_fine in zip(variables_coarse, variables_fine) :
        file_coarse = files_coarse[var_coarse]
        file_fine = files_fine[var_fine]
        with xr.open_dataset(dir_coarse + file_coarse) as coarse_xr :
            data_var = coarse_xr[var_coarse]
            if "time" in list(data_var.coords) :
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var_coarse} : {len(data_var.time.values)}")
                print(f"time length for {var_coarse} : {len(data_var.time.values)}")
            else :
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
            if "height" in list(data_var.coords) :
                #data_var = data_var.isel(height = 0)
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                #data_var = data_var.isel(plev = 0)
                data_var = data_var.isel(plev = 0)
            data_var = data_var.sel(lon = slice_lon, lat = slice_lat)
            if "time" in list(data_var.coords):
                data_var = data_var.transpose("time", "lat", "lon")
                data_var = data_var.sortby("time", ascending=True)
            else :
                data_var = data_var.transpose("time_counter", "lat", "lon")
                data_var = data_var.sortby("time_counter", ascending=True)
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            data_coarse = data_var.values
            if var_coarse == "precip" :
                data_coarse = data_coarse #convert to mm from m?
            if "log" in norm_coarse[var_coarse] :
                data_coarse = precip_log_transform(data_coarse[:,None,:,:])
        with xr.open_zarr(dir_fine + file_fine) as var_xr :
            data_var = var_xr[var_fine]
            if "time" in list(data_var.coords) :
                print(f'dataset for {var_fine} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f'dataset for {var_fine} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var_fine} : {len(data_var.time.values)}")
                print(f"time length for {var_fine} : {len(data_var.time.values)}")
            else :
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
            if "height" in list(data_var.coords) :
                #data_var = data_var.isel(height = 0)
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                #data_var = data_var.isel(plev = 0)
                data_var = data_var.isel(plev = 0)
            data_var = data_var.sel(lon = slice_lon, lat = slice_lat)
            if "time" in list(data_var.coords):
                data_var = data_var.transpose("time", "lat", "lon")
                data_var = data_var.sortby("time", ascending=True)
            else :
                data_var = data_var.transpose("time_counter", "lat", "lon")
                data_var = data_var.sortby("time_counter", ascending=True)
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            data_fine = data_var.values
            if var_fine == "var228":
                data_fine = 1000*data_fine # convert to mm from m
            if "log" in norm_fine[var_fine] :
                data_fine = precip_log_transform(data_fine[:,None,:,:])

        residual = data_fine - data_coarse
        mean_coarses.append(
            np.nanmean(data_coarse)
        )
        min_coarses.append(
            np.nanmin(data_coarse)
        )
        max_coarses.append(
            np.nanmax(data_coarse)
        ) 
        std_coarses.append(
            np.nanstd(data_coarse)
        )
        mean_fines.append(
            np.nanmean(data_fine)
        )
        min_fines.append(
            np.nanmin(data_fine)
        )
        max_fines.append(
            np.nanmax(data_fine)
        )
        std_fines.append(
            np.nanstd(data_fine)
        )
        mean_residuals.append(
            np.nanmean(residual)
        )
        min_residuals.append(
            np.nanmin(residual)
        )
        max_residuals.append(
            np.nanmax(residual)
        )
        std_residuals.append(
            np.nanstd(residual)
        )
    print("============================")
    print(f"Train dataset {period}")
    print("============================")
    print(f"Mean of coarse : {mean_coarses}")
    print(f"Std of coarse : {std_coarses}")
    print(f"Min of coarse : {min_coarses}")
    print(f"Max of coarse : {max_coarses}")
    print(f"Mean of fine : {mean_fines}")
    print(f"Std of fine : {std_fines}")
    print(f"Min of fine : {min_fines}")
    print(f"Max of fine : {max_fines}")
    print(f"Mean of residual : {mean_residuals}")
    print(f"Std of residual : {std_residuals}")
    print(f"Min of residual : {min_residuals}")
    print(f"Max of residual : {max_residuals}")

def prepare_corrdiff_test_set(
    dir_coarse = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA/unet_baseline_22_09_2026/results/",
    variables_coarse = ["var131", "var132", "var130", "var137", "var59", "var228"],
    variables_fine = ["var131", "var132", "var130", "var137", "var59", "var228"],
    file_prediction = "predictions_test_save_validation_member_0.nc",
    period = "2000_2014"
) :
    mean_coarses = []
    std_coarses = []
    min_coarses = []
    max_coarses = []
    mean_fines = []
    std_fines = []
    min_fines = []
    max_fines = []
    mean_residuals = []
    std_residuals = []
    min_residuals = []
    max_residuals = []

    files_fine = {
        "var131" : f"u850_europe_6h_{period}_sorted.zarr",
        "var132" : f"v850_europe_6h_{period}_sorted.zarr",
        "var130" : f"t850_europe_6h_{period}_sorted.zarr",
        "var137" : f"TCWV_europe_6h_{period}_sorted.zarr",
        "var59" : f"cape_europe_6h_{period}_sorted.zarr",
        "var228" : f"precip_europe_6h_{period}_sorted.zarr"
    }
    files_coarse = {
        "var131" : file_prediction,
        "var132" : file_prediction,
        "var130" : file_prediction,
        "var137" : file_prediction,
        "var59" : file_prediction,
        "var228" : file_prediction
    }

    norm_fine = {
        "var131" : "std",
        "var132" : "std",
        "var130" : "std",
        "var137" : "std",
        "var59" : "log",
        "var228" : "log"
    }
    norm_coarse = norm_fine

    dir_fine = "/lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/"

    slice_lon = slice(-6, 9.75)
    slice_lat = slice(54.75, 39)

    for var_coarse, var_fine in zip(variables_coarse, variables_fine) :
        file_coarse = files_coarse[var_coarse]
        file_fine = files_fine[var_fine]
        with xr.open_dataset(dir_coarse + file_coarse) as coarse_xr :
            data_var = coarse_xr[var_coarse]
            if "time" in list(data_var.coords) :
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var_coarse} : {len(data_var.time.values)}")
                print(f"time length for {var_coarse} : {len(data_var.time.values)}")
            else :
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
            if "height" in list(data_var.coords) :
                #data_var = data_var.isel(height = 0)
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                #data_var = data_var.isel(plev = 0)
                data_var = data_var.isel(plev = 0)
            data_var = data_var.sel(lon = slice_lon, lat = slice_lat)
            if "time" in list(data_var.coords):
                data_var = data_var.transpose("time", "lat", "lon")
                data_var = data_var.sortby("time", ascending=True)
            else :
                data_var = data_var.transpose("time_counter", "lat", "lon")
                data_var = data_var.sortby("time_counter", ascending=True)
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            data_coarse = data_var.values
            if var_coarse == "precip" :
                data_coarse = data_coarse #convert to mm from m ?
            if "log" in norm_coarse[var_coarse] :
                data_coarse = precip_log_transform(data_coarse[:,None,:,:])
        with xr.open_zarr(dir_fine + file_fine) as var_xr :
            data_var = var_xr[var_fine]
            if "time" in list(data_var.coords) :
                print(f'dataset for {var_fine} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f'dataset for {var_fine} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var_fine} : {len(data_var.time.values)}")
                print(f"time length for {var_fine} : {len(data_var.time.values)}")
            else :
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f'dataset for {var_coarse} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
                print(f"time length for {var_coarse} : {len(data_var.time_counter.values)}")
            if "height" in list(data_var.coords) :
                #data_var = data_var.isel(height = 0)
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                #data_var = data_var.isel(plev = 0)
                data_var = data_var.isel(plev = 0)
            data_var = data_var.sel(lon = slice_lon, lat = slice_lat)
            if "time" in list(data_var.coords):
                data_var = data_var.transpose("time", "lat", "lon")
                data_var = data_var.sortby("time", ascending=True)
            else :
                data_var = data_var.transpose("time_counter", "lat", "lon")
                data_var = data_var.sortby("time_counter", ascending=True)
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            data_fine = data_var.values
            if var_fine == "var228":
                data_fine = 1000*data_fine # convert to mm from m
            if "log" in norm_fine[var_fine] :
                data_fine = precip_log_transform(data_fine[:,None,:,:])

        residual = data_fine - data_coarse
        mean_coarses.append(
            np.nanmean(data_coarse)
        )
        min_coarses.append(
            np.nanmin(data_coarse)
        )
        max_coarses.append(
            np.nanmax(data_coarse)
        ) 
        std_coarses.append(
            np.nanstd(data_coarse)
        )
        mean_fines.append(
            np.nanmean(data_fine)
        )
        min_fines.append(
            np.nanmin(data_fine)
        )
        max_fines.append(
            np.nanmax(data_fine)
        )
        std_fines.append(
            np.nanstd(data_fine)
        )
        mean_residuals.append(
            np.nanmean(residual)
        )
        min_residuals.append(
            np.nanmin(residual)
        )
        max_residuals.append(
            np.nanmax(residual)
        )
        std_residuals.append(
            np.nanstd(residual)
        )

    print("============================")
    print(f"Test dataset {period}")
    print("============================")
    print(f"Mean of coarse : {mean_coarses}")
    print(f"Std of coarse : {std_coarses}")
    print(f"Min of coarse : {min_coarses}")
    print(f"Max of coarse : {max_coarses}")
    print(f"Mean of fine : {mean_fines}")
    print(f"Std of fine : {std_fines}")
    print(f"Min of fine : {min_fines}")
    print(f"Max of fine : {max_fines}")
    print(f"Mean of residual : {mean_residuals}")
    print(f"Std of residual : {std_residuals}")
    print(f"Min of residual : {min_residuals}")
    print(f"Max of residual : {max_residuals}")

###########################
# 2 - convert nc unet prediction file to zarr and write it to $SCRATCH :
###########################
def save_zarr(
        dir_coarse = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA/unet_baseline_22_09_2026_corrdiff_all_train_period/results/",
        file_prediction = "predictions_test_save_validation_member_0.nc",
        dir_output = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA/unet_baseline_22_09_2026_corrdiff_all_train_period/results/",
        file_output = "predictions_unet_corrdiff_1985_2014.zarr",
    ) :
    with xr.open_dataset(dir_coarse + file_prediction) as coarse_xr : 
        if "time" in list(coarse_xr.coords) :
            coarse_xr = coarse_xr.chunk({"time": 10})
        else :
            coarse_xr = coarse_xr.chunk({"time_counter": 10})
        coarse_xr.to_zarr(dir_output + file_output)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(
        dir_coarse = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA/unet_baseline_03_10_2026_trained_on_1985-1999/results/",
        dir_coarse_test_set = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA/unet_baseline_24_09_2026_valid_best_epoch/results/",
        period_train = "1985_1999",
        period_test = "2000_2014",
    ) :
    prepare_corrdiff(
        dir_coarse = dir_coarse,
        variables_coarse = ["var131", "var132", "var130", "var137", "var59", "var228"],
        variables_fine = ["var131", "var132", "var130", "var137", "var59", "var228"],
        file_prediction = "predictions_test_save_validation_member_0.nc",
        period=period_train
    )
    prepare_corrdiff_test_set(
        dir_coarse = dir_coarse_test_set,
        variables_coarse = ["var131", "var132", "var130", "var137", "var59", "var228"],
        variables_fine = ["var131", "var132", "var130", "var137", "var59", "var228"],
        file_prediction = "predictions_test_save_validation_member_0.nc",
        period=period_test
    )

    # save train set to zarr :
    save_zarr(
        dir_coarse=dir_coarse,
        file_prediction = "predictions_test_save_validation_member_0.nc",
        dir_output = dir_coarse,
        file_output = f"predictions_unet_corrdiff_{period_train}.zarr",
    )
    print("saved train set to zarr")

    #save test set to zarr
    save_zarr(
        dir_coarse=dir_coarse_test_set,
        file_prediction = "predictions_test_save_validation_member_0.nc",
        dir_output = dir_coarse,
        file_output = f"predictions_unet_corrdiff_{period_test}.zarr",
    )
    print("saved test set to zarr")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_coarse")
    parser.add_argument("--dir_coarse_test_set")
    parser.add_argument("--period_train")
    parser.add_argument("--period_test")
    args = parser.parse_args()
    main(
        dir_coarse=args.dir_coarse,
        dir_coarse_test_set=args.dir_coarse_test_set,
        period_train=args.period_train,
        period_test=args.period_test,
    )