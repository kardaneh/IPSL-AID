import torch
import torchvision
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,"C:/Users/pchapel/code/IPSL-AID/")
from IPSL_AID.utils import EasyDict
from IPSL_AID.dataset import coarse_down_up, gaussian_filter 
import pandas as pd
from datetime import datetime
import os
import json

def get_mean_std_tensors(vars, norm_mapping, normalization_types, coarse_or_residual = "coarse") :
    mean_values = []
    std_values = []
    for var in vars :
        if coarse_or_residual == "residual" :
            key = var + "_residual"
        else :
            key = var
        if normalization_types[var] != "standard" :
            key = normalization_types[var] + "_" + key
        mean_value = norm_mapping[key].vmean
        std_value = norm_mapping[key].vstd
        mean_values.append(mean_value)
        std_values.append(std_value)
    return torch.tensor(mean_values), torch.tensor(std_values)



def stats(logger, input_dir, json_names = [], norm_mapping=dict()):
    """
    Load normalization statistics.

    This function loads normalization statistics from a JSON file if available.
    If no statistics file is found, it falls back to predefined constants for
    fine and coarse variables only.

    Parameters
    ----------
    ds : xarray.Dataset
        NetCDF dataset to process.
    logger : logging.Logger
        Logger instance for logging messages and statistics.
    input_dir : str
        Directory containing a statistics.json file with precomputed
        normalization statistics.*
    json_names : list of str
        Names of the files containing normalization statistics
    norm_mapping : dict, optional
        Dictionary to store computed statistics. If empty, will be populated.
        Default is empty dict.

    Returns
    -------
    norm_mapping : dict
        Dictionary mapping variable names to their computed statistics.
        For data variables: mean, std.
    steps : EasyDict
        Dictionary containing coordinate step sizes and lengths.

    Notes
    -----
    - If statistics.json is found, all statistics are loaded as-is.
    - If not found, fallback constants are used for fine and coarse variables only.
    """

    # stats_file_path = os.path.join(output_dir, "statistics.json")

    logger.info("Starting statistics computation...")

    stats_loaded = False
    # Load stats from JSON if available
    if input_dir is not None and len(json_names) != 0 :
        for file_json in json_names :
            stats_path = os.path.join(input_dir, file_json)
            if os.path.isfile(stats_path):
                logger.info(f"Loading normalization statistics from {stats_path}")

                with open(stats_path, "r") as f:
                    raw_stats = json.load(f)

                # Load all statistics from JSON into norm_mapping
                for key, values in raw_stats.items():
                    norm_mapping[key] = EasyDict(values)

                stats_loaded = True

    # Use manual stats
    if not stats_loaded:
        logger.info("No statistics.json found !")
        raise ValueError("No json file for statistics found !")

    logger.info("------------------------------------------")
    return norm_mapping

def is_sorted(time_array) :
    return np.all(time_array[:-1] <= time_array[1:])

def identity(tensor) :
    return tensor

def inverse_normalize_residual(residual_norm, normalize_residual_std, normalize_residual_mean):
    device = residual_norm.device
    return (residual_norm * (normalize_residual_std[:, np.newaxis, np.newaxis]).to(device)) + (normalize_residual_mean[:, np.newaxis, np.newaxis]).to(device)

def precip_log_transform(data_tensor) :
    """
    data_tensor is of shape (T,C,L,W)
    """
    if torch.is_tensor(data_tensor) :
        # clip data that is negative (CAPE and precip should be non negative)
        data_tensor = torch.clamp(data_tensor, min=0)
        transformed_data = torch.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = torch.log(data_tensor[:,0,:,:] + 1)
    else :
        data_tensor = np.clip(data_tensor, a_min=0, a_max=None)
        transformed_data = np.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = np.log(data_tensor[:,0,:,:] + 1)
    return transformed_data

def precip_exp_transform(data_tensor) :
    transformed_data = torch.zeros_like(data_tensor)
    transformed_data[:,0,:,:] = torch.exp(data_tensor[:,0,:,:]) - 1
    return transformed_data

def sqrt_transform(data_tensor) :
    """
    data_tensor is of shape (T,C,L,W)
    """
    if torch.is_tensor(data_tensor) :
        # clip data that is negative (TCWV should be positive)
        data_tensor = torch.clamp(data_tensor, min=0)
        transformed_data = torch.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = torch.sqrt(data_tensor[:,0,:,:])
    else :
        data_tensor = np.clip(data_tensor, a_min=0, a_max=None)
        transformed_data = np.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = np.sqrt(data_tensor[:,0,:,:])
    return transformed_data

def square_transform(data_tensor) :
    transformed_data = torch.zeros_like(data_tensor)
    transformed_data[:,0,:,:] = (data_tensor[:,0,:,:])**2
    return transformed_data

class DataPreprocessorLMDZ2ERA(torch.utils.data.Dataset):
    """
    Dataset class for lmdz data to downscale towards ERA data. Some predicted fields do not correspond to input data (like CAPE and precip), named diagnostics variables.
    Some coarse variables are not outputted : they are named dynamic covariates.
    Variables that have fine and coarse resolution will use the residual approach (the residual are given as targets to the model), the normalized diagnostics variables are forecasted directly.
    """

    def __init__(
        self,
        data_dir_coarse,
        data_dir_fine,
        data_dir_diagnostics,
        files_coarse,
        files_fine,
        files_diagnostics,
        logger,
        variables_name_coarse = ["u850", "v850", "t850", "prw"],
        variables_name_fine = ["var131", "var132", "var130", "var137"], #u,v,t,TCWV
        variables_name_diagnostics = ["var59", "var228"], #cape, précips
        path_folder_stats = "./stats/",
        json_stats_rawdata = "statistics_lmdz_nudg_6h_1_1985-2014.json",
        json_stats_residual = "statistics_era5_lmdz_nudg_6h_1_1985-2014.json",
        json_stats_diag = "statistics_era5_lmdz_nudg_6h_1_1985-2014.json",
        # normalize_rawdata_mean=torch.Tensor([3.8327718, -0.05697539, 277.3738, 17.023146]), # norm for France 1985-2014
        # normalize_rawdata_std=torch.Tensor([7.239197, 6.478215, 6.579805, 7.137549]),
        # normalize_residual_mean=torch.Tensor([-0.1413257, 0.11201838, 1.3851668, -0.62241]),
        # normalize_residual_std=torch.Tensor([2.1736937, 2.3264318, 1.6038697, 2.9346433]),
        # normalize_diagnostics_mean=torch.Tensor([1.441924, 0.26563856]),
        # normalize_diagnostics_std=torch.Tensor([1.8676592, 0.5013067]), # for log (x+1) transform for both cape and precip (in mm)
        dynamic_covariates_name= ["var59","var130","var137", "var131", "var132"], #CAPE, T850, TCWV, u850, v850
        dynamic_covariates_dir="./",
        dynamic_covariates_files=[],
        normalization_type = {},
        normalization_type_coarse = {},
        normalization_type_diagnostics = {},
        normalization_type_dynamic_cov = {},
        lon_slice_domain = slice(-6,9.75),
        lat_slice_domain = slice(54.75, 39),
        # normalize_covariates_mean=torch.Tensor([5.859726, 8.724808, -0.4640244, -1.2654291, 268.9366, 252.2054, 0.004785511, 0.0023771676, 0.00071223004, 98550.35]), #real training data South France 1985-2014 tensor([ 93.2392, 280.4401,  15.1545,   0.6444,  -0.4625])
        # normalize_covariates_std=torch.Tensor([8.168476, 10.825516, 7.781761, 11.132606, 6.0343814, 6.0387325, 0.0020607216, 0.0014883897, 0.0004916295, 3898.0361]), #real training data South France 1985-2014 tensor([323.9705,   6.4705,   8.1186,   4.3525,   5.2768])
        constant_variables_name=["LSM","topography"],
        files_constant_variables=[],
        constant_variables_dir = "./",
        normalization="linear",  # "linear" or "cos_sin"
        multiplication_factor_precip_coarse=1.0, #multiplication factor to convert to mm/6h : for LMDZ it should be 3600, for ERA it should be 1000.
        multiplication_factor_precip_fine=1000.0, #multiplication factor to convert to mm/6h : for LMDZ it should be 3600, for ERA it should be 1000.
    ):
        """
        :param data_dir: path to the dataset directory
        :param dynamic_covariates: name of the dynamic covariates used, i.e. the helping variables given to the model that will not be downscaled. If None, no covariates are used. defaults to None.
        :param dynamic_covariates_dir: path to the dataset directory containing the dynamic covariates. If None, no dynamic covariates are used. defaults to None.
        :param in_shape: shape of the low resolution images
        :param out_shape: shape of the high resolution images
        :param year_start: starting year of file named samples_{year_start}.nc
        :param year_end: ending year of file named samples_{year_end}.nc
        :param normalize_mean: channel-wise mean values estimated over all samples
        for normalizing file
        :param normalize_std: channel-wise standard deviation values estimated
        over all samples for normalizing file
        """

        self.multiplication_factor_precip_coarse = multiplication_factor_precip_coarse
        self.multiplication_factor_precip_fine = multiplication_factor_precip_fine
        self.normalization = normalization.lower()
        #add a lambda function to check that datasets are timesorted :
        # open coarse data :
        # we have one file for each variable and we open them :
        self.coarse = []
        self.transform_funcs = {}
        self.inverse_transform = {}
        self.normalization_type_coarse = normalization_type_coarse
        self.files_coarse = files_coarse
        self.variables_name_coarse = variables_name_coarse
        self.data_dir_coarse = data_dir_coarse
        self.lon_slice_domain = lon_slice_domain
        self.lat_slice_domain = lat_slice_domain
        self.ds_coarse = [xr.open_zarr(self.data_dir_coarse + file) for file in self.files_coarse]
        self.data_vars_coarse = [var_xr[var] for var_xr,var in zip(self.ds_coarse,self.variables_name_coarse)]

        for file,var in zip(self.files_coarse, self.variables_name_coarse) :
            with xr.open_zarr(self.data_dir_coarse + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_coarse + file)
                data_var = var_xr[var]
                if "time" in list(data_var.coords) :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                    print(f"time length for {var} : {len(data_var.time.values)}")
                else :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                    print(f"time length for {var} : {len(data_var.time_counter.values)}")
                # check if there is a dimension other than time,x,y in xarray:
                if "height" in list(data_var.coords) :
                    #data_var = data_var.isel(height = 0)
                    data_var = data_var.isel(height = 0)
                elif "plev" in list(data_var.coords) :
                    #data_var = data_var.isel(plev = 0)
                    data_var = data_var.isel(plev = 0)
                coarse_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
                self.input_shape = coarse_ex.shape[2:]
            if "log" in normalization_type_coarse[var] :
                self.transform_funcs[var] = precip_log_transform
            elif "sqrt" in normalization_type_coarse[var] :
                print(f"using sqrt transform for {var}")
                self.transform_funcs[var] = sqrt_transform
            else :
                self.transform_funcs[var] = identity

        # Time embeddings
        if "time" in list(data_var.coords) :
            self.time = var_xr.time
            self.time_array = var_xr.time.values
            self.stime = var_xr.time.values[0]
            self.etime = var_xr.time.values[-1]
        else :
            self.time = var_xr.time_counter
            self.time_array = var_xr.time_counter.values
            self.stime = var_xr.time_counter.values[0]
            self.etime = var_xr.time_counter.values[-1]
        self.T = len(self.time)
        T = self.T
        self.year = self.time.dt.year
        self.month = self.time.dt.month
        self.day = self.time.dt.day
        self.hour = self.time.dt.hour

        # open fine data :
        # we have one file for each variable and we open them :
        self.fine = []
        self.normalization_type = normalization_type
        self.variables_name_fine = variables_name_fine
        self.files_fine = files_fine
        self.data_dir_fine = data_dir_fine
        self.ds_fine = [xr.open_zarr(self.data_dir_fine + file) for file in self.files_fine]
        self.data_vars_fine = [var_xr[var] for var_xr,var in zip(self.ds_fine,self.variables_name_fine)]

        for file,var in zip(self.files_fine,self.variables_name_fine) :
            with xr.open_zarr(self.data_dir_fine + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_fine + file)
                data_var = var_xr[var]
                self.longitude_fine = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
                self.latitude_fine = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()
                print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var} : {len(data_var.time.values)}")
                # check if there is a dimension other than time,x,y in xarray:
                if "height" in list(data_var.coords) :
                    data_var = data_var.isel(height = 0)
                elif "plev" in list(data_var.coords) :
                    data_var = data_var.isel(plev = 0)
                #fine_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
            if "log" in normalization_type[var] :
                self.transform_funcs[var] = precip_log_transform
                self.inverse_transform[var] = precip_exp_transform
            elif "sqrt" in normalization_type[var] :
                print(f"using sqrt transform for {var}")
                self.transform_funcs[var] = sqrt_transform
                self.inverse_transform[var] = square_transform
            else :
                self.transform_funcs[var] = identity
                self.inverse_transform[var] = identity
        self.output_shape = self.input_shape

        # open static cov data :
        # we have one file for each variable and we open them :
        self.constant_variables = []
        self.files_constant_variables = files_constant_variables
        self.constant_variables_name = constant_variables_name
        self.constant_variables_dir = constant_variables_dir
        self.ds_constant = [xr.open_zarr(self.constant_variables_dir + file) for file in self.files_constant_variables]
        self.data_vars_constant = [var_xr[var] for var_xr, var in zip(self.ds_constant, self.constant_variables_name)]
        for data_var in self.data_vars_constant :
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            if "time" in list(data_var.coords) :
                data_var = data_var.isel(time = 0)
            # select lonlat zone
            data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain)
            # transpose to make sure of right order of dimensions : time,lat,lon :
            data_var = data_var.transpose("lat", "lon")
            # sort by ascending order for lon, and descending for lat
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            self.constant_variables.append(data_var.values[None,None,:,:])
        self.constant_variables = np.concatenate(self.constant_variables, axis = 1) # shape (T,nb_var,L,W)
        # check that constant_variables is in 4D : 
        if len(self.constant_variables.shape) == 3 :
            self.constant_variables = self.constant_variables[:,None,:,:]
        self.constant_variables = torch.from_numpy(self.constant_variables).float()
    
        # open dynamic cov data :
        # we have one file for each covariable and we open them :
        self.covariates_name = dynamic_covariates_name
        self.normalization_type_dynamic_cov = normalization_type_dynamic_cov
        self.dynamic_covariates_files = dynamic_covariates_files
        self.dynamic_covariates_name = dynamic_covariates_name
        self.dynamic_covariates_dir = dynamic_covariates_dir
        self.ds_covariate = [xr.open_zarr(self.dynamic_covariates_dir + file) for file in self.dynamic_covariates_files]
        self.data_vars_covariate = [var_xr[var] for var_xr,var in zip(self.ds_covariate, self.dynamic_covariates_name)]       

        for file,var in zip(self.dynamic_covariates_files,self.dynamic_covariates_name) :
            with xr.open_zarr(self.dynamic_covariates_dir + file) as var_xr:
                # var_xr = xr.open_zarr(dynamic_covariates_dir + file)
                data_var = var_xr[var]
                if "time" in list(data_var.coords) :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                    print(f"time length for {var} : {len(data_var.time.values)}")
                else :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                    print(f"time length for {var} : {len(data_var.time_counter.values)}")
            if "log" in normalization_type_dynamic_cov[var] :
                self.transform_funcs[var] = precip_log_transform
            else :
                self.transform_funcs[var] = identity

        # open diagnostics data :
        self.diagnostics_name = variables_name_diagnostics
        self.normalization_type_diagnostics = normalization_type_diagnostics
        self.files_diagnostics = files_diagnostics
        self.variables_name_diagnostics = variables_name_diagnostics
        self.data_dir_diagnostics = data_dir_diagnostics
        self.ds_diagnostic = [xr.open_zarr(self.data_dir_diagnostics + file) for file in self.files_diagnostics]
        self.data_vars_diagnostic = [var_xr[var] for var_xr,var in zip(self.ds_diagnostic, self.variables_name_diagnostics)]

        for file,var in zip(self.files_diagnostics,self.variables_name_diagnostics) :
            with xr.open_zarr(self.data_dir_diagnostics + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_fine + file)
                data_var = var_xr[var]
                print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var} : {len(data_var.time.values)}")
            if "log" in normalization_type_diagnostics[var] :
                self.transform_funcs[var] = precip_log_transform
                self.inverse_transform[var] = precip_exp_transform
            else :
                self.transform_funcs[var] = identity
                self.inverse_transform[var] = identity

        norm_mapping = stats(
            logger = logger,
            input_dir = path_folder_stats,
            json_names = [json_stats_rawdata, json_stats_residual, json_stats_diag]
        )

        normalize_rawdata_mean, normalize_rawdata_std = get_mean_std_tensors(
            vars=variables_name_coarse,
            normalization_types=normalization_type_coarse,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for coarse :")
        logger.info(variables_name_coarse)
        logger.info(normalize_rawdata_mean)
        logger.info(normalize_rawdata_std)

        self.normalize_rawdata_transform = torchvision.transforms.Normalize(
            normalize_rawdata_mean, normalize_rawdata_std
        )

        normalize_residual_mean, normalize_residual_std = get_mean_std_tensors(
            vars=variables_name_fine,
            normalization_types=normalization_type,
            norm_mapping=norm_mapping,
            coarse_or_residual="residual"
        )
        logger.info("Normalisation tensors for residuals :")
        logger.info(variables_name_fine)
        logger.info(normalize_residual_mean)
        logger.info(normalize_residual_std)

        self.normalize_residual_mean = normalize_residual_mean
        self.normalize_residual_std = normalize_residual_std
        self.normalize_residual_transform = torchvision.transforms.Normalize(
            normalize_residual_mean, normalize_residual_std
        )
        self.inverse_normalize_residual = inverse_normalize_residual

        normalize_covariates_mean, normalize_covariates_std = get_mean_std_tensors(
            vars=dynamic_covariates_name,
            normalization_types=normalization_type_dynamic_cov,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for covariates :")
        logger.info(dynamic_covariates_name)
        logger.info(normalize_covariates_mean)
        logger.info(normalize_covariates_std)

        self.normalize_covariates_transform = torchvision.transforms.Normalize(
            normalize_covariates_mean, normalize_covariates_std
        )

        # normalize constant variables :
        normalize_constant_variables_transform = torchvision.transforms.Normalize(
            torch.mean(self.constant_variables, dim = [0,2,3]), torch.std(self.constant_variables, dim = [0,2,3])
        )
        try :
            constant_variables_norm = normalize_constant_variables_transform(self.constant_variables)        
        except :
            constant_variables_norm = self.constant_variables - torch.mean(self.constant_variables, dim = [0,2,3])
        self.constant_variables_norm = constant_variables_norm
        
        # normalize diagnostic variables :
        normalize_diagnostics_mean, normalize_diagnostics_std = get_mean_std_tensors(
            vars=variables_name_diagnostics,
            normalization_types=normalization_type_diagnostics,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for diags :")
        logger.info(variables_name_diagnostics)
        logger.info(normalize_diagnostics_mean)
        logger.info(normalize_diagnostics_std)

        self.normalize_diagnostics_mean = normalize_diagnostics_mean
        self.normalize_diagnostics_std = normalize_diagnostics_std
            
        self.normalize_diagnostics_transform = torchvision.transforms.Normalize(
            self.normalize_diagnostics_mean, self.normalize_diagnostics_std
        )

        # Define limits for plotting (plus/minus 2 sigma
        self.vmin = normalize_rawdata_mean - 2 * normalize_rawdata_std
        self.vmax = normalize_rawdata_mean + 2 * normalize_rawdata_std

        self.year_norm = torch.from_numpy(
            ((self.year.to_numpy() - 1940) / 100).astype(np.float32)
        )

        # Normalize and convert to numpy (load into mem)
        if self.normalization == "linear":
            self.doy = (self.month - 1.0) * 30 + (self.day - 1.0)
            self.doy_norm = torch.from_numpy(
                (self.doy.to_numpy() / 360).astype(np.float32)
            )
            self.hour_norm = torch.from_numpy(
                (self.hour.to_numpy() / 24.0).astype(np.float32)
            )
        elif self.normalization == "cos_sin":
            date = pd.to_datetime(dict(year=self.year, month=self.month, day=self.day))
            self.doy = (date - datetime(2000, 1, 1)).dt.days
            doy_np = self.doy.to_numpy()
            hour_np = self.hour.to_numpy()
            self.doy_sin = torch.sin(2 * np.pi * torch.from_numpy(doy_np) / 365.25)
            self.doy_cos = torch.cos(2 * np.pi * torch.from_numpy(doy_np) / 365.25)
            self.hour_sin = torch.sin(2 * np.pi * torch.from_numpy(hour_np) / 24.0)
            self.hour_cos = torch.cos(2 * np.pi * torch.from_numpy(hour_np) / 24.0)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

        print("Dataset initialized.")

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.T

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        coarse = []
        coarse_transform = []
        time_index = self.time_array[index]

        for data_var, var in zip(self.data_vars_coarse, self.variables_name_coarse):
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                #data_var = data_var.isel(height = 0)
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                #data_var = data_var.isel(plev = 0)
                data_var = data_var.isel(plev = 0)
            #coarse.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            # use isel to access only right timestep without loading everything in memory
            if "time" in list(data_var.coords) :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest")
            else :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest")
            # transpose to make sure of right order of dimensions : time,lat,lon :
            data_var = data_var.transpose("lat", "lon")
            # sort by ascending order for lon, and descending for lat
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            coarse_var = data_var.values.copy()[None,None,:,:]
            if var == "precip" :
                coarse_var = self.multiplication_factor_precip_coarse*coarse_var # convert to mm from kg/(m2*s)
            coarse.append(coarse_var)
            coarse_transform.append(
                self.transform_funcs[var](
                    coarse[-1].copy()
                )
            )
        coarse = np.concatenate(coarse, axis = 1) # shape (T,nb_var,L,W)
        coarse_transform = np.concatenate(coarse_transform, axis = 1) # shape (T,nb_var,L,W)
        # replace NaNs with spatial mean for timestep of the channel
        spatial_mean_coarse = np.nanmean(coarse, axis=(2,3))[:,:,None,None]
        #coarse = np.where(np.isnan(self.coarse),0,self.coarse)
        coarse = np.where(np.isnan(coarse),spatial_mean_coarse,coarse)
        coarse = torch.from_numpy(coarse).float()
        spatial_mean_coarse_transform = np.nanmean(coarse_transform, axis=(2,3))[:,:,None,None]
        coarse_transform = np.where(np.isnan(coarse_transform),spatial_mean_coarse_transform,coarse_transform)
        coarse_transform = torch.from_numpy(coarse_transform).float()

        # open fine data :
        # we have one file for each variable and we open them :
        fine = []
        fine_transform = []
        for data_var, var_xr, var in zip(self.data_vars_fine, self.ds_fine, self.variables_name_fine) :
            longitude_fine = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).sortby("lon",ascending=True).values.copy()).float()
            latitude_fine = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).sortby("lat",ascending=False).values.copy()).float()
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            # fine.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest")
            else :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest")
            # transpose to make sure of right order of dimensions : time,lat,lon :
            data_var = data_var.transpose("lat", "lon")
            # sort by ascending order for lon, and descending for lat
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            fine_var = data_var.values.copy()[None,None,:,:]
            if var == "var228":
                fine_var = self.multiplication_factor_precip_fine*fine_var # convert to mm from m
            fine.append(fine_var)
            fine_transform.append(
                self.transform_funcs[var](
                    fine[-1].copy()
                )
            )
        fine = np.concatenate(fine, axis = 1) # shape (T,nb_var,L,W)
        spatial_mean_fine = np.nanmean(fine, axis=(2,3))[:,:,None,None]
        fine = np.where(np.isnan(fine),spatial_mean_fine,fine)
        fine = torch.from_numpy(fine).float()

        fine_transform = np.concatenate(fine_transform, axis = 1) # shape (T,nb_var,L,W)
        spatial_mean_fine_transform = np.nanmean(fine_transform, axis=(2,3))[:,:,None,None]
        fine_transform = np.where(np.isnan(fine_transform),spatial_mean_fine_transform,fine_transform)
        fine_transform = torch.from_numpy(fine_transform).float()

        # open constant data :
        constant_variables = self.constant_variables.clone()

        #open dynamic covariates :
        covariates = []
        covariates_transform = []
        for var_xr,data_var, var in zip(self.ds_covariate,self.data_vars_covariate, self.dynamic_covariates_name) :
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            #covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest")
            else :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest")
            # transpose to make sure of right order of dimensions : time,lat,lon :
            data_var = data_var.transpose("lat", "lon")
            # sort by ascending order for lon, and descending for lat
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            covariates.append(data_var.values.copy()[None,None,:,:])
            longitude_covariates = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).sortby("lon",ascending=True).values.copy()).float()
            latitude_covariates = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).sortby("lat",ascending=False).values.copy()).float()

            covariates_transform.append(
                self.transform_funcs[var](
                    covariates[-1].copy()
                )
            )
        
        covariates = np.concatenate(covariates, axis = 1) # shape (T,nb_var,L,W)
        spatial_mean_covariates = np.nanmean(covariates, axis=(2,3))[:,:,None,None]
        covariates = np.where(np.isnan(covariates),spatial_mean_covariates,covariates)
        covariates = torch.from_numpy(covariates).float()

        covariates_transform = np.concatenate(covariates_transform, axis = 1) # shape (T,nb_var,L,W)
        spatial_mean_covariates_transform = np.nanmean(covariates_transform, axis=(2,3))[:,:,None,None]
        covariates_transform = np.where(np.isnan(covariates_transform),spatial_mean_covariates_transform,covariates_transform)
        covariates_transform = torch.from_numpy(covariates_transform).float()

        # open diag data :
        diagnostics = []
        diagnostics_transform = []
        for data_var, var in zip(self.data_vars_diagnostic,self.variables_name_diagnostics) :
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            #diagnostics.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest")
            else :
                data_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest")
            # transpose to make sure of right order of dimensions : time,lat,lon :
            data_var = data_var.transpose("lat", "lon")
            # sort by ascending order for lon, and descending for lat
            data_var = data_var.sortby("lat", ascending=False)
            data_var = data_var.sortby("lon", ascending=True)
            diag_var = data_var.values.copy()[None,None,:,:]    
            if var == "var228" :
                diag_var = self.multiplication_factor_precip_fine*diag_var # convert from mm to m
            diagnostics.append(diag_var)
            diagnostics_transform.append(
                self.transform_funcs[var](
                    diagnostics[-1].copy()
                )
            )
        no_diags_flag = (len(self.variables_name_diagnostics) == 0)
        if not no_diags_flag :
            diagnostics = np.concatenate(diagnostics, axis = 1) # shape (T,nb_var,L,W)
            spatial_mean_diagnostics = np.nanmean(diagnostics, axis=(2,3))[:,:,None,None]
            diagnostics = np.where(np.isnan(diagnostics),spatial_mean_diagnostics,diagnostics)
            diagnostics = torch.from_numpy(diagnostics).float()

            diagnostics_transform = np.concatenate(diagnostics_transform, axis = 1) # shape (T,nb_var,L,W)
            spatial_mean_diagnostics_transform = np.nanmean(diagnostics_transform, axis=(2,3))[:,:,None,None]
            diagnostics_transform = np.where(np.isnan(diagnostics_transform),spatial_mean_diagnostics_transform,diagnostics_transform)
            diagnostics_transform = torch.from_numpy(diagnostics_transform).float()

        residual = fine_transform - coarse_transform

        # normalise coarse :
        coarse_norm = self.normalize_rawdata_transform(coarse_transform)
        # normalise residual :
        residual_norm = self.normalize_residual_transform(residual)
        # normalise covariates :
        covariates_norm = self.normalize_covariates_transform(covariates_transform)
        # normalise constant covariates :
        constant_variables_norm = self.constant_variables_norm.clone()
        # normalise diag :
        if not no_diags_flag :
            diagnostics_norm = self.normalize_diagnostics_transform(diagnostics_transform)

        if not no_diags_flag :
            targets_channels_name = self.variables_name_fine + self.diagnostics_name
            targets = torch.concat((residual_norm, diagnostics_norm), dim = 1)  # targets  = normalized residual (of transformed fine and coarse if applicable)
            inputs = torch.concat((coarse_norm, covariates_norm, constant_variables_norm), dim = 1)  # inputs   = normalized coarse concatenated to normalized
        else :
            targets_channels_name = self.variables_name_fine
            targets = residual_norm # targets  = normalized residual (of transformed fine and coarse if applicable)
            inputs = torch.concat((coarse_norm, covariates_norm, constant_variables_norm), dim = 1)  # inputs   = normalized coarse concatenated to normalized       
     
        if torch.any(torch.isnan(inputs)) :
            print("inputs contain NaN !")
        if torch.any(torch.isnan(targets)) :
            print("targets contain NaN !")
        if torch.any(torch.isnan(fine)) :
            print("fine contain NaN !")
        if torch.any(torch.isnan(coarse)) :
            print("coarse contain NaN !")
        if torch.any(torch.isnan(covariates)) :
            print("covariates contain NaN !")
        if not no_diags_flag :
            if torch.any(torch.isnan(diagnostics)) :
                print("diagnostics contain NaN !")
        if torch.any(torch.isnan(longitude_fine)) :
            print("longitude fine contain NaN !")
        if torch.any(torch.isnan(latitude_fine)) :
            print("latitude fine contain NaN !")
        if torch.any(torch.isnan(longitude_covariates)) :
            print("longitude covariates contain NaN !")
        if torch.any(torch.isnan(latitude_covariates)) :
            print("latitude covariates contain NaN !")
        #test for infs
        if torch.any(torch.isinf(inputs)) :
            print("inputs contain inf !")
        if torch.any(torch.isinf(targets)) :
            print("targets contain inf !")
        if torch.any(torch.isinf(fine)) :
            print("fine contain inf !")
        if torch.any(torch.isinf(coarse)) :
            print("coarse contain inf !")
        if torch.any(torch.isinf(covariates)) :
            print("covariates contain inf !")
        if not no_diags_flag :
            if torch.any(torch.isinf(diagnostics)) :
                print("diagnostics contain inf !")
        if torch.any(torch.isinf(longitude_fine)) :
            print("longitude fine contain inf !")
        if torch.any(torch.isinf(latitude_fine)) :
            print("latitude fine contain inf !")
        if torch.any(torch.isinf(longitude_covariates)) :
            print("longitude covariates contain inf !")
        if torch.any(torch.isinf(latitude_covariates)) :
            print("latitude covariates contain inf !")   

        sample = {
            "inputs": inputs[0],
            "targets": targets[0],
            "fine": fine[0],
            "coarse": coarse[0],
            "coarse_transform": coarse_transform[0],
            "residual": residual[0],
            "year": self.year_norm[index],
            "time_index": index,
            "corrdinates": {
                "lon": longitude_fine,
                "lat": latitude_fine,
            },
            "corrdinates_covariates": {
                "lon": longitude_covariates,
                "lat": latitude_covariates,
            },
        }
        if len(self.covariates_name) != 0 :
            sample["covariates"] = covariates[0]
            sample["covariates_transform"] = covariates_transform[0]
        if len(self.diagnostics_name) != 0 :
            sample["diagnostics"] = diagnostics[0]
            sample["diagnostics_transform"] = diagnostics_transform[0]

        if self.normalization == "linear":
            sample["doy"] = self.doy_norm[index]
            sample["hour"] = self.hour_norm[index]
        elif self.normalization == "cos_sin":
            sample["doy_sin"] = self.doy_sin[index]
            sample["doy_cos"] = self.doy_cos[index]
            sample["hour_sin"] = self.hour_sin[index]
            sample["hour_cos"] = self.hour_cos[index]

        return sample

    def model_output_to_pred(self, model_output, coarse_image) :
        nb_vars_fine = len(self.variables_name_fine)
        nb_diags = len(self.diagnostics_name)
        no_diags_flag = (nb_diags == 0)
        predictions = []
        residual = model_output[:,:nb_vars_fine,:,:] # first nb_vars_fine channels correspond to residual
        # denormalize :
        denorm_residual = self.inverse_normalize_residual(residual, self.normalize_residual_std, self.normalize_residual_mean)
        for i,var in enumerate(self.variables_name_fine) :
            channel_pred = self.inverse_transform[var](self.transform_funcs[var](coarse_image[:,i,:,:][:,None,:,:]) + denorm_residual[:,i,:,:][:,None,:,:])
            predictions.append(channel_pred)
        if not no_diags_flag :
            diags = model_output[:,nb_vars_fine:,:,:] # other channels correspond to diags variables
            denorm_diags = self.inverse_normalize_residual(diags, self.normalize_diagnostics_std, self.normalize_diagnostics_mean)
            for i,var in enumerate(self.diagnostics_name) :
                channel_pred = self.inverse_transform[var](denorm_diags[:,i,:,:][:,None,:,:])
                predictions.append(channel_pred)
        return torch.cat(predictions, dim=1)

    def new_epoch(self):
        pass


class DataPreprocessorERACoarse2ERA(torch.utils.data.Dataset):

    """
    Dataset class for ERA5 coarse to downscale towards ERA data. Some predicted fields do not correspond to input data (like CAPE and precip), named diagnostics variables.
    Some coarse variables are not outputted : they are named dynamic covariates.
    Variables that have fine and coarse resolution will use the residual approach (the residual are given as targets to the model), the normalized diagnostics variables are forecasted directly.
    """

    def __init__(
        self,
        data_dir_fine,
        data_dir_diagnostics,
        files_fine,
        files_diagnostics,
        logger,
        variables_name_fine = ["var131", "var132", "var130", "var137"], #u,v,t,TCWV
        variables_name_diagnostics = ["var59", "var228"], #cape, précips
        path_folder_stats = "./stats/",
        json_stats_rawdata = "statistics_era5_coarse_1985-2014.json",
        json_stats_residual = "statistics_era5_coarse_era5_fine_1985-2014.json",
        json_stats_diag = "statistics_era5_coarse_era5_fine_1985-2014.json",
        dynamic_covariates_name= ["var59","var130","var137", "var131", "var132"], #CAPE, T850, TCWV, u850, v850
        dynamic_covariates_dir="./",
        dynamic_covariates_files=[],
        normalization_type = {},
        normalization_type_diagnostics = {},
        normalization_type_dynamic_cov = {},
        lon_slice_domain = slice(-6,9.75),
        lat_slice_domain = slice(54.75, 39),
        resolution_lon_coarse = 2.5, #in degrees
        resolution_lat_coarse = 1.25, #in degrees
        constant_variables_name=["LSM","topography"],
        files_constant_variables=[],
        constant_variables_dir = "./",
        normalization="linear",  # "linear" or "cos_sin"
    ):
        """
        :param data_dir: path to the dataset directory
        :param dynamic_covariates: name of the dynamic covariates used, i.e. the helping variables given to the model that will not be downscaled. If None, no covariates are used. defaults to None.
        :param dynamic_covariates_dir: path to the dataset directory containing the dynamic covariates. If None, no dynamic covariates are used. defaults to None.
        :param in_shape: shape of the low resolution images
        :param out_shape: shape of the high resolution images
        :param year_start: starting year of file named samples_{year_start}.nc
        :param year_end: ending year of file named samples_{year_end}.nc
        :param normalize_mean: channel-wise mean values estimated over all samples
        for normalizing file
        :param normalize_std: channel-wise standard deviation values estimated
        over all samples for normalizing file
        """
        self.epsilon =  0.02
        self.margin = 8
        self.logger = logger
        self.resolution_lon_coarse = resolution_lon_coarse
        self.resolution_lat_coarse = resolution_lat_coarse
        self.normalization = normalization.lower()
        #add a lambda function to check that datasets are timesorted :
        # open coarse data :
        # we have one file for each variable and we open them :
        self.coarse = []
        self.transform_funcs = {}
        self.inverse_transform = {}
        self.lon_slice_domain = lon_slice_domain
        self.lat_slice_domain = lat_slice_domain
        # self.ds_coarse = [xr.open_zarr(self.data_dir_coarse + file) for file in self.files_coarse]
        # self.data_vars_coarse = [var_xr[var] for var_xr,var in zip(self.ds_coarse,self.variables_name_coarse)]
        self.variables_name_fine = variables_name_fine
        self.variables_name_coarse = variables_name_fine

        # for var in variables_name_fine :
            # with xr.open_zarr(self.data_dir_coarse + file) as var_xr:
            #     # var_xr = xr.open_zarr(data_dir_coarse + file)
            #     data_var = var_xr[var]
            #     if "time" in list(data_var.coords) :
            #         print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
            #         print(f"time length for {var} : {len(data_var.time.values)}")
            #     else :
            #         print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')
            #         print(f"time length for {var} : {len(data_var.time_counter.values)}")
            #     # check if there is a dimension other than time,x,y in xarray:
            #     if "height" in list(data_var.coords) :
            #         #data_var = data_var.isel(height = 0)
            #         data_var = data_var.isel(height = 0)
            #     elif "plev" in list(data_var.coords) :
            #         #data_var = data_var.isel(plev = 0)
            #         data_var = data_var.isel(plev = 0)
            #     coarse_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
            #     self.input_shape = coarse_ex.shape[2:]
            # if "log" in normalization_type_coarse[var] :
            #     self.transform_funcs[var] = precip_log_transform
            # elif "sqrt" in normalization_type_coarse[var] :
            #     print(f"using sqrt transform for {var}")
            #     self.transform_funcs[var] = sqrt_transform
            # else :
            #     self.transform_funcs[var] = identity

        # open fine data :
        # we have one file for each variable and we open them :
        self.fine = []
        self.normalization_type = normalization_type
        self.variables_name_fine = variables_name_fine
        self.files_fine = files_fine
        self.data_dir_fine = data_dir_fine
        self.ds_fine = [xr.open_zarr(self.data_dir_fine + file) for file in self.files_fine]
        self.data_vars_fine = [var_xr[var] for var_xr,var in zip(self.ds_fine,self.variables_name_fine)]

        for file,var in zip(self.files_fine,self.variables_name_fine) :
            with xr.open_zarr(self.data_dir_fine + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_fine + file)
                data_var = var_xr[var]
                self.longitude_fine = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
                self.latitude_fine = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()
                self.longitude_fine_full = torch.from_numpy(var_xr["lon"].values.copy()).float()
                self.latitude_fine_full = torch.from_numpy(var_xr["lat"].values.copy()).float()
                print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var} : {len(data_var.time.values)}")
                # check if there is a dimension other than time,x,y in xarray:
                if "height" in list(data_var.coords) :
                    data_var = data_var.isel(height = 0)
                elif "plev" in list(data_var.coords) :
                    data_var = data_var.isel(plev = 0)
                #fine_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
            fine_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
            self.input_shape = fine_ex.shape[2:]
            if "log" in normalization_type[var] :
                self.transform_funcs[var] = precip_log_transform
                self.inverse_transform[var] = precip_exp_transform
            elif "sqrt" in normalization_type[var] :
                print(f"using sqrt transform for {var}")
                self.transform_funcs[var] = sqrt_transform
                self.inverse_transform[var] = square_transform
            else :
                self.transform_funcs[var] = identity
                self.inverse_transform[var] = identity
        self.output_shape = self.input_shape

        # Shape of fine data : (time, latitude, longitude)
        # Longitude and latitude range of full domain :
        self.lon_range_full = torch.max(self.longitude_fine_full) - torch.min(self.longitude_fine_full)
        self.lat_range_full = torch.max(self.latitude_fine_full) - torch.min(self.latitude_fine_full)
        self.coarse_in_shape = ((self.lat_range_full // resolution_lat_coarse) + 1, (self.lon_range_full // resolution_lon_coarse) + 1 )
        # In shape of coarse :
        self.in_shape = (int(np.abs(self.latitude_fine_full[-1] - self.latitude_fine_full[0]).item() // resolution_lat_coarse) , int(np.abs(self.longitude_fine_full[-1] - self.longitude_fine_full[0]).item() // resolution_lon_coarse))
        #Fine resolution :
        self.resolution_lon_fine = np.abs(self.longitude_fine[1] - self.longitude_fine[0]).item()
        self.resolution_lat_fine = np.abs(self.latitude_fine[1] - self.latitude_fine[0]).item()
        self.resolution_lon_coarse = resolution_lon_coarse
        self.resolution_lat_coarse = resolution_lat_coarse
        print(f"self.resolution_lat_fine : self.dW : {self.resolution_lat_fine}")
        self.dH = self.resolution_lat_fine # for ERA data order of axes is (vars, lat, lon)
        self.dW = self.resolution_lon_fine # this corresponds to (C,H,W)
        self.dH_coarse = self.resolution_lat_coarse
        self.dW_coarse = self.resolution_lon_coarse
        # Time embeddings
        if "time" in list(data_var.coords) :
            self.time = var_xr.time
            self.time_array = var_xr.time.values
            self.stime = var_xr.time.values[0]
            self.etime = var_xr.time.values[-1]
        else :
            self.time = var_xr.time_counter
            self.time_array = var_xr.time_counter.values
            self.stime = var_xr.time_counter.values[0]
            self.etime = var_xr.time_counter.values[-1]
        self.T = len(self.time)
        T = self.T
        self.year = self.time.dt.year
        self.month = self.time.dt.month
        self.day = self.time.dt.day
        self.hour = self.time.dt.hour

        # open static cov data :
        # we have one file for each variable and we open them :
        self.constant_variables = []
        self.files_constant_variables = files_constant_variables
        self.constant_variables_name = constant_variables_name
        self.constant_variables_dir = constant_variables_dir
        self.ds_constant = [xr.open_zarr(self.constant_variables_dir + file) for file in self.files_constant_variables]
        self.data_vars_constant = [var_xr[var] for var_xr, var in zip(self.ds_constant, self.constant_variables_name)]
        for data_var in self.data_vars_constant :
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            if "time" in list(data_var.coords) :
                data_var = data_var.isel(time = 0)
            self.constant_variables.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[None,None,:,:])
        self.constant_variables = np.concatenate(self.constant_variables, axis = 1) # shape (T,nb_var,L,W)
        # check that constant_variables is in 4D : 
        if len(self.constant_variables.shape) == 3 :
            self.constant_variables = self.constant_variables[:,None,:,:]
        self.constant_variables = torch.from_numpy(self.constant_variables).float()
    
        # open dynamic cov data :
        # we have one file for each covariable and we open them :
        self.covariates_name = dynamic_covariates_name
        self.normalization_type_dynamic_cov = normalization_type_dynamic_cov
        self.dynamic_covariates_files = dynamic_covariates_files
        self.dynamic_covariates_name = dynamic_covariates_name
        self.dynamic_covariates_dir = dynamic_covariates_dir
        self.ds_covariate = [xr.open_zarr(self.dynamic_covariates_dir + file) for file in self.dynamic_covariates_files]
        self.data_vars_covariate = [var_xr[var] for var_xr,var in zip(self.ds_covariate, self.dynamic_covariates_name)]       

        for file,var in zip(self.dynamic_covariates_files,self.dynamic_covariates_name) :
            with xr.open_zarr(self.dynamic_covariates_dir + file) as var_xr:
                # var_xr = xr.open_zarr(dynamic_covariates_dir + file)
                data_var = var_xr[var]
                if "time" in list(data_var.coords) :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                    print(f"time length for {var} : {len(data_var.time.values)}")
                else :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                    print(f"time length for {var} : {len(data_var.time_counter.values)}")
            if "log" in normalization_type_dynamic_cov[var] :
                self.transform_funcs[var] = precip_log_transform
            else :
                self.transform_funcs[var] = identity

        # open diagnostics data :
        self.diagnostics_name = variables_name_diagnostics
        self.normalization_type_diagnostics = normalization_type_diagnostics
        self.files_diagnostics = files_diagnostics
        self.variables_name_diagnostics = variables_name_diagnostics
        self.data_dir_diagnostics = data_dir_diagnostics
        self.ds_diagnostic = [xr.open_zarr(self.data_dir_diagnostics + file) for file in self.files_diagnostics]
        self.data_vars_diagnostic = [var_xr[var] for var_xr,var in zip(self.ds_diagnostic, self.variables_name_diagnostics)]

        for file,var in zip(self.files_diagnostics,self.variables_name_diagnostics) :
            with xr.open_zarr(self.data_dir_diagnostics + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_fine + file)
                data_var = var_xr[var]
                print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var} : {len(data_var.time.values)}")
            if "log" in normalization_type_diagnostics[var] :
                self.transform_funcs[var] = precip_log_transform
                self.inverse_transform[var] = precip_exp_transform
            else :
                self.transform_funcs[var] = identity
                self.inverse_transform[var] = identity


        norm_mapping = stats(
            logger = logger,
            input_dir = path_folder_stats,
            json_names = [json_stats_rawdata, json_stats_residual, json_stats_diag]
        )

        normalize_rawdata_mean, normalize_rawdata_std = get_mean_std_tensors(
            vars=variables_name_fine,
            normalization_types=normalization_type,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for coarse :")
        logger.info(variables_name_fine)
        logger.info(normalize_rawdata_mean)
        logger.info(normalize_rawdata_std)

        self.normalize_rawdata_transform = torchvision.transforms.Normalize(
            normalize_rawdata_mean, normalize_rawdata_std
        )

        normalize_residual_mean, normalize_residual_std = get_mean_std_tensors(
            vars=variables_name_fine,
            normalization_types=normalization_type,
            norm_mapping=norm_mapping,
            coarse_or_residual="residual"
        )
        logger.info("Normalisation tensors for residuals :")
        logger.info(variables_name_fine)
        logger.info(normalize_residual_mean)
        logger.info(normalize_residual_std)

        self.normalize_residual_mean = normalize_residual_mean
        self.normalize_residual_std = normalize_residual_std
        self.normalize_residual_transform = torchvision.transforms.Normalize(
            normalize_residual_mean, normalize_residual_std
        )
        self.inverse_normalize_residual = inverse_normalize_residual

        normalize_covariates_mean, normalize_covariates_std = get_mean_std_tensors(
            vars=dynamic_covariates_name,
            normalization_types=normalization_type_dynamic_cov,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for covariates :")
        logger.info(dynamic_covariates_name)
        logger.info(normalize_covariates_mean)
        logger.info(normalize_covariates_std)

        self.normalize_covariates_transform = torchvision.transforms.Normalize(
            normalize_covariates_mean, normalize_covariates_std
        )

        # normalize constant variables :
        normalize_constant_variables_transform = torchvision.transforms.Normalize(
            torch.mean(self.constant_variables, dim = [0,2,3]), torch.std(self.constant_variables, dim = [0,2,3])
        )
        try :
            constant_variables_norm = normalize_constant_variables_transform(self.constant_variables)        
        except :
            constant_variables_norm = self.constant_variables - torch.mean(self.constant_variables, dim = [0,2,3])
        self.constant_variables_norm = constant_variables_norm
        
        # normalize diagnostic variables :
        normalize_diagnostics_mean, normalize_diagnostics_std = get_mean_std_tensors(
            vars=variables_name_diagnostics,
            normalization_types=normalization_type_diagnostics,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for diags :")
        logger.info(variables_name_diagnostics)
        logger.info(normalize_diagnostics_mean)
        logger.info(normalize_diagnostics_std)

        self.normalize_diagnostics_mean = normalize_diagnostics_mean
        self.normalize_diagnostics_std = normalize_diagnostics_std
            
        self.normalize_diagnostics_transform = torchvision.transforms.Normalize(
            self.normalize_diagnostics_mean, self.normalize_diagnostics_std
        )

        # Define limits for plotting (plus/minus 2 sigma
        self.vmin = normalize_rawdata_mean - 2 * normalize_rawdata_std
        self.vmax = normalize_rawdata_mean + 2 * normalize_rawdata_std

        self.year_norm = torch.from_numpy(
            ((self.year.to_numpy() - 1940) / 100).astype(np.float32)
        )

        # Normalize and convert to numpy (load into mem)
        if self.normalization == "linear":
            self.doy = (self.month - 1.0) * 30 + (self.day - 1.0)
            self.doy_norm = torch.from_numpy(
                (self.doy.to_numpy() / 360).astype(np.float32)
            )
            self.hour_norm = torch.from_numpy(
                (self.hour.to_numpy() / 24.0).astype(np.float32)
            )
        elif self.normalization == "cos_sin":
            date = pd.to_datetime(dict(year=self.year, month=self.month, day=self.day))
            self.doy = (date - datetime(2000, 1, 1)).dt.days
            doy_np = self.doy.to_numpy()
            hour_np = self.hour.to_numpy()
            self.doy_sin = torch.sin(2 * np.pi * torch.from_numpy(doy_np) / 365.25)
            self.doy_cos = torch.cos(2 * np.pi * torch.from_numpy(doy_np) / 365.25)
            self.hour_sin = torch.sin(2 * np.pi * torch.from_numpy(hour_np) / 24.0)
            self.hour_cos = torch.cos(2 * np.pi * torch.from_numpy(hour_np) / 24.0)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

        print("Dataset initialized.")

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.T

    def extract_batch(self, full_domain_data) :
        # full_domain_data is np.ndarray of shape (C,H,W)
        # this function is for ERA5 data with shape (C,H,W) where the order of axes is (nb_var, latitude, longitude)
        longitude_full = self.longitude_fine_full.numpy()
        latitude_full = self.latitude_fine_full.numpy()
        lon_min_select = np.min(self.longitude_fine.numpy())
        lon_max_select = np.max(self.longitude_fine.numpy())
        lat_min_select = np.min(self.latitude_fine.numpy())
        lat_max_select = np.max(self.latitude_fine.numpy())
        ind_w_min = np.where(longitude_full == lon_min_select)[0][0]
        ind_w_max = np.where(longitude_full == lon_max_select)[0][0]
        ind_h_min = np.where(latitude_full == lat_min_select)[0][0]
        ind_h_max = np.where(latitude_full == lat_max_select)[0][0] 
        if ind_h_min > ind_h_max :
            ind_h_min, ind_h_max = ind_h_max, ind_h_min
        if ind_w_min > ind_w_max :
            ind_w_min, ind_w_max = ind_w_max, ind_w_min
        return full_domain_data[:,ind_h_min:ind_h_max+1, ind_w_min:ind_w_max+1]

    def filter_batch(self, fine_patch, fine_block, dW_coarse = None, dH_coarse = None):
        """
        Apply Gaussian low-pass filtering for multi-scale processing.

        Parameters
        ----------
        fine_patch : np.ndarray
            Fine-resolution data of shape (C, H, W).
        fine_block : np.ndarray
            Reference block used to determine scaling factors.

        Returns
        -------
        fine_filtered : np.ndarray
            Filtered data of shape (C, H, W).

        Notes
        -----
        - Filters high-frequency components beyond the coarse grid's Nyquist.
        - Uses Gaussian filtering in the frequency domain.
        - Processes each channel independently.
        """

        # Output container
        fine_filtered = np.zeros_like(fine_patch)

        if dH_coarse is None or dW_coarse is None :
            # ----------------------------------------------------------
            # Determine scaling between fine and coarse grids
            # ----------------------------------------------------------
            scale_factor_H = fine_block.shape[-2] / self.in_shape[0]
            scale_factor_W = fine_block.shape[-1] / self.in_shape[1]

            # ----------------------------------------------------------
            # Coarse-grid physical spacing and Nyquist limits
            # ----------------------------------------------------------
            dW_coarse = self.dW * scale_factor_W
            dH_coarse = self.dH * scale_factor_H

        f_nW_coarse = 1.0 / (2.0 * dW_coarse)
        f_nH_coarse = 1.0 / (2.0 * dH_coarse)

        # ----------------------------------------------------------
        # Apply filtering per channel
        # ----------------------------------------------------------
        C = fine_patch.shape[0]

        for c in range(C):
            try:
                img = fine_patch[c]
                filt = gaussian_filter(
                    img,
                    self.dW,
                    self.dH,
                    f_nW_coarse,
                    f_nH_coarse,
                    self.epsilon,
                    self.margin,
                )

                # Save the filtered channel
                fine_filtered[c] = filt

            except Exception as e:
                self.logger.exception(f"Filtering failed at channel {c}", e)
                raise

        return fine_filtered

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        # coarse = []
        # coarse_transform = []
        # time_index = self.time_array[index]
        # for data_var, var in zip(self.data_vars_coarse, self.variables_name_coarse):
        #     # check if there is a dimension other than time,x,y in xarray:
        #     if "height" in list(data_var.coords) :
        #         #data_var = data_var.isel(height = 0)
        #         data_var = data_var.isel(height = 0)
        #     elif "plev" in list(data_var.coords) :
        #         #data_var = data_var.isel(plev = 0)
        #         data_var = data_var.isel(plev = 0)
        #     #coarse.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
        #     # use isel to access only right timestep without loading everything in memory
        #     if "time" in list(data_var.coords) :
        #         coarse_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:]
        #         if var == "precip" :
        #             coarse_var = 3600*coarse_var # convert to mm from kg/(m2*s)
        #         coarse.append(coarse_var)
        #     else :
        #         coarse_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:]
        #         if var == "precip" :
        #             coarse_var = 3600*coarse_var # convert to mm from kg/(m2*s)
        #         coarse.append(coarse_var)
        #     coarse_transform.append(
        #         self.transform_funcs[var](
        #             coarse[-1].copy()
        #         )
        #     )
        # coarse = np.concatenate(coarse, axis = 1) # shape (T,nb_var,L,W)
        # coarse_transform = np.concatenate(coarse_transform, axis = 1) # shape (T,nb_var,L,W)
        # # replace NaNs with spatial mean for timestep of the channel
        # spatial_mean_coarse = np.nanmean(coarse, axis=(2,3))[:,:,None,None]
        # #coarse = np.where(np.isnan(self.coarse),0,self.coarse)
        # coarse = np.where(np.isnan(coarse),spatial_mean_coarse,coarse)
        # coarse = torch.from_numpy(coarse).float()
        # spatial_mean_coarse_transform = np.nanmean(coarse_transform, axis=(2,3))[:,:,None,None]
        # coarse_transform = np.where(np.isnan(coarse_transform),spatial_mean_coarse_transform,coarse_transform)
        # coarse_transform = torch.from_numpy(coarse_transform).float()

        # open fine data :
        # we have one file for each variable and we open them :
        fine = []
        fine_full_domain = []
        fine_transform = []
        time_index = self.time_array[index]
        for data_var, var_xr, var in zip(self.data_vars_fine, self.ds_fine, self.variables_name_fine) :
            longitude_fine = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
            latitude_fine = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            # fine.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                fine_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:]
                fine_var_full_domain = data_var.sel(time = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228" :
                    fine_var = 1000 * fine_var # convert to mm from m
                    fine_var_full_domain = 1000 * fine_var_full_domain # convert to mm from m
                fine.append(fine_var)
                fine_full_domain.append(fine_var_full_domain)
            else :
                fine_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:]
                fine_var_full_domain = data_var.sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228":
                    fine_var = 1000*fine_var # convert to mm from m
                    fine_var_full_domain = 1000*fine_var_full_domain # convert to mm from m
                fine.append(fine_var)
                fine_full_domain.append(fine_var_full_domain)
            fine_transform.append(
                self.transform_funcs[var](
                    fine[-1].copy()
                )
            )
        fine_np = np.concatenate(fine, axis = 1) # np array shape (1,nb_var,L,W)
        # spatial_mean_fine = np.nanmean(fine_np, axis=(2,3))[:,:,None,None]
        # fine_np = np.where(np.isnan(fine_np),spatial_mean_fine,fine_np)
        fine = torch.from_numpy(fine_np).float() # torch tensor (1,nb_var,L,W)

        fine_full_domain = np.concatenate(fine_full_domain, axis = 1) # shape (1,nb_var,L,W)
        # spatial_mean_fine_full_domain = np.nanmean(fine_full_domain, axis=(2,3))[:,:,None,None]
        # fine_full_domain = np.where(np.isnan(fine_full_domain),spatial_mean_fine_full_domain,fine_full_domain)


        fine_transform = np.concatenate(fine_transform, axis = 1) # np array shape (1,nb_var,L,W)
        # spatial_mean_fine_transform = np.nanmean(fine_transform, axis=(2,3))[:,:,None,None]
        # fine_transform = np.where(np.isnan(fine_transform),spatial_mean_fine_transform,fine_transform)
        fine_transform = torch.from_numpy(fine_transform).float() # torch tensor shape (1, nb_var, L,W)

        # Create coarse data from fine_full_domain by filtering and coarse_up_down :
        fine_full_domain_filtered = self.filter_batch(
            fine_patch=fine_full_domain[0],
            fine_block=fine_np[0],
            dW_coarse=self.dW_coarse,
            dH_coarse=self.dH_coarse
        ) #np array shape (nb_var,L,W)
        coarse_full_domain = coarse_down_up(fine_full_domain_filtered, fine_full_domain, self.in_shape) #torch tensor shape (nb_var,L,W)
        coarse = self.extract_batch(coarse_full_domain)[None,:,:,:] # tensor shape (1,nb_var,L,W)

        coarse_transform = []
        for i, var in enumerate(self.variables_name_coarse):
            coarse_transform.append(
                self.transform_funcs[var](
                    coarse[0:1,i:i+1].clone()
                )
            )
        coarse_transform = torch.cat(coarse_transform, dim = 1) # shape (1,nb_var,L,W)

        # open constant data :
        constant_variables = self.constant_variables.clone()

        #open dynamic covariates :
        covariates = []
        covariates_full_domain = []
        covariates_transform = []
        for var_xr,data_var, var in zip(self.ds_covariate,self.data_vars_covariate, self.dynamic_covariates_name) :
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            #covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                #covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:])
                covariates_full_domain.append(data_var.sel(time = time_index, method = "nearest").values.copy()[None,None,:,:])
            else :
                #covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:])
                covariates_full_domain.append(data_var.sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:])
            longitude_covariates = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
            latitude_covariates = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()

            # covariates_transform.append(
            #     self.transform_funcs[var](
            #         covariates[-1].copy()
            #     )
            # )
        
        #covariates_np = np.concatenate(covariates, axis = 1) # np array shape (1,nb_var,L,W)
        # spatial_mean_covariates = np.nanmean(covariates, axis=(2,3))[:,:,None,None]
        # covariates_np = np.where(np.isnan(covariates),spatial_mean_covariates,covariates)
        #covariates = torch.from_numpy(covariates_np).float() # torch tensor shape (1, nb_var, L,W)

        covariates_full_domain = np.concatenate(covariates_full_domain, axis = 1) #np array shape (T,nb_var,L,W)
        # spatial_mean_covariates_full_domain = np.nanmean(covariates_full_domain, axis=(2,3))[:,:,None,None]
        # covariates_full_domain = np.where(np.isnan(covariates_full_domain),spatial_mean_covariates_full_domain,covariates_full_domain)
        #covariates_full_domain = torch.from_numpy(covariates_full_domain).float()

        covariates_full_domain_filtered = self.filter_batch(
            fine_patch=covariates_full_domain[0],
            fine_block=fine_np[0],
            dW_coarse=self.dW_coarse,
            dH_coarse=self.dH_coarse
        ) # np array shape (nb_var,L,W)
        covariates_full_domain_coarse = coarse_down_up(covariates_full_domain_filtered, covariates_full_domain, self.in_shape) #torch tensor shape (nb_var,L,W)
        covariates_coarse = self.extract_batch(covariates_full_domain_coarse)[None,:,:,:] # torch tensor shape (1, nb_var,L,W)

        covariates_coarse_transform = []
        for i, var in enumerate(self.dynamic_covariates_name) :
            covariates_coarse_transform.append(
                self.transform_funcs[var](
                    covariates_coarse[0:1,i:i+1].clone()
                )
            )
        covariates_coarse_transform = torch.cat(covariates_coarse_transform, dim = 1) # shape (1,nb_var,L,W)
        # spatial_mean_covariates_coarse_transform = np.nanmean(covariates_coarse_transform, axis=(2,3))[:,:,None,None]
        # covariates_coarse_transform = np.where(np.isnan(covariates_coarse_transform),spatial_mean_covariates_coarse_transform,covariates_coarse_transform)

        # open diag data :
        diagnostics = []
        diagnostics_transform = []
        for data_var, var in zip(self.data_vars_diagnostic,self.variables_name_diagnostics) :
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            #diagnostics.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                diag_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228" :
                    diag_var = 1000*diag_var # convert from mm to m
                diagnostics.append(diag_var)
            else :
                diag_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228" :
                    diag_var = 1000*diag_var # convert from mm to m
                diagnostics.append(diag_var)
            diagnostics_transform.append(
                self.transform_funcs[var](
                    diagnostics[-1].copy()
                )
            )
        no_diags_flag = (len(self.variables_name_diagnostics) == 0)
        if not no_diags_flag :
            diagnostics = np.concatenate(diagnostics, axis = 1) # shape (T,nb_var,L,W)
            # spatial_mean_diagnostics = np.nanmean(diagnostics, axis=(2,3))[:,:,None,None]
            # diagnostics = np.where(np.isnan(diagnostics),spatial_mean_diagnostics,diagnostics)
            diagnostics = torch.from_numpy(diagnostics).float()

            diagnostics_transform = np.concatenate(diagnostics_transform, axis = 1) # shape (T,nb_var,L,W)
            # spatial_mean_diagnostics_transform = np.nanmean(diagnostics_transform, axis=(2,3))[:,:,None,None]
            # diagnostics_transform = np.where(np.isnan(diagnostics_transform),spatial_mean_diagnostics_transform,diagnostics_transform)
            diagnostics_transform = torch.from_numpy(diagnostics_transform).float()

        residual = fine_transform - coarse_transform

        # normalise coarse :
        coarse_norm = self.normalize_rawdata_transform(coarse_transform)
        # normalise residual :
        residual_norm = self.normalize_residual_transform(residual)
        # normalise covariates :
        covariates_coarse_norm = self.normalize_covariates_transform(covariates_coarse_transform)
        # normalise constant covariates :
        constant_variables_norm = self.constant_variables_norm.clone()
        # normalise diag :
        if not no_diags_flag :
            diagnostics_norm = self.normalize_diagnostics_transform(diagnostics_transform)

        if not no_diags_flag :
            targets_channels_name = self.variables_name_fine + self.diagnostics_name
            targets = torch.concat((residual_norm, diagnostics_norm), dim = 1)  # targets  = normalized residual (of transformed fine and coarse if applicable)
            inputs = torch.concat((coarse_norm, covariates_coarse_norm, constant_variables_norm), dim = 1)  # inputs   = normalized coarse concatenated to normalized
        else :
            targets_channels_name = self.variables_name_fine
            targets = residual_norm # targets  = normalized residual (of transformed fine and coarse if applicable)
            inputs = torch.concat((coarse_norm, covariates_coarse_norm, constant_variables_norm), dim = 1)  # inputs   = normalized coarse concatenated to normalized       
     
        if torch.any(torch.isnan(inputs)) :
            print("inputs contain NaN !")
        if torch.any(torch.isnan(targets)) :
            print("targets contain NaN !")
        if torch.any(torch.isnan(fine)) :
            print("fine contain NaN !")
        if torch.any(torch.isnan(coarse)) :
            print("coarse contain NaN !")
        # if torch.any(torch.isnan(covariates)) :
        #     print("covariates contain NaN !")
        if not no_diags_flag :
            if torch.any(torch.isnan(diagnostics)) :
                print("diagnostics contain NaN !")
        if torch.any(torch.isnan(longitude_fine)) :
            print("longitude fine contain NaN !")
        if torch.any(torch.isnan(latitude_fine)) :
            print("latitude fine contain NaN !")
        if torch.any(torch.isnan(longitude_covariates)) :
            print("longitude covariates contain NaN !")
        if torch.any(torch.isnan(latitude_covariates)) :
            print("latitude covariates contain NaN !")
        #test for infs
        if torch.any(torch.isinf(inputs)) :
            print("inputs contain inf !")
        if torch.any(torch.isinf(targets)) :
            print("targets contain inf !")
        if torch.any(torch.isinf(fine)) :
            print("fine contain inf !")
        if torch.any(torch.isinf(coarse)) :
            print("coarse contain inf !")
        # if torch.any(torch.isinf(covariates)) :
        #     print("covariates contain inf !")
        if not no_diags_flag :
            if torch.any(torch.isinf(diagnostics)) :
                print("diagnostics contain inf !")
        if torch.any(torch.isinf(longitude_fine)) :
            print("longitude fine contain inf !")
        if torch.any(torch.isinf(latitude_fine)) :
            print("latitude fine contain inf !")
        if torch.any(torch.isinf(longitude_covariates)) :
            print("longitude covariates contain inf !")
        if torch.any(torch.isinf(latitude_covariates)) :
            print("latitude covariates contain inf !")   

        sample = {
            "inputs": inputs[0],
            "targets": targets[0],
            "fine": fine[0],
            "fine_transform": fine_transform[0],
            "residual": residual[0],
            "coarse": coarse[0],
            "coarse_transform": coarse_transform[0],
            "year": self.year_norm[index],
            "time_index": index,
            "corrdinates": {
                "lon": longitude_fine,
                "lat": latitude_fine,
            },
            "corrdinates_covariates": {
                "lon": longitude_covariates,
                "lat": latitude_covariates,
            },
        }
        if len(self.covariates_name) != 0 :
            sample["covariates"] = covariates_coarse[0]
            sample["covariates_coarse_transform"] = covariates_coarse_transform[0]
        if len(self.diagnostics_name) != 0 :
            sample["diagnostics"] = diagnostics[0]
            sample["diagnostics_transform"] = diagnostics_transform[0]

        if self.normalization == "linear":
            sample["doy"] = self.doy_norm[index]
            sample["hour"] = self.hour_norm[index]
        elif self.normalization == "cos_sin":
            sample["doy_sin"] = self.doy_sin[index]
            sample["doy_cos"] = self.doy_cos[index]
            sample["hour_sin"] = self.hour_sin[index]
            sample["hour_cos"] = self.hour_cos[index]

        return sample

    def model_output_to_pred(self, model_output, coarse_image) :
        nb_vars_fine = len(self.variables_name_fine)
        nb_diags = len(self.diagnostics_name)
        no_diags_flag = (nb_diags == 0)
        predictions = []
        residual = model_output[:,:nb_vars_fine,:,:] # first nb_vars_fine channels correspond to residual
        # denormalize :
        denorm_residual = self.inverse_normalize_residual(residual, self.normalize_residual_std, self.normalize_residual_mean)
        for i,var in enumerate(self.variables_name_fine) :
            channel_pred = self.inverse_transform[var](self.transform_funcs[var](coarse_image[:,i,:,:][:,None,:,:]) + denorm_residual[:,i,:,:][:,None,:,:])
            predictions.append(channel_pred)
        if not no_diags_flag :
            diags = model_output[:,nb_vars_fine:,:,:] # other channels correspond to diags variables
            denorm_diags = self.inverse_normalize_residual(diags, self.normalize_diagnostics_std, self.normalize_diagnostics_mean)
            for i,var in enumerate(self.diagnostics_name) :
                channel_pred = self.inverse_transform[var](denorm_diags[:,i,:,:][:,None,:,:])
                predictions.append(channel_pred)
        return torch.cat(predictions, dim=1)

    def new_epoch(self):
        pass


class DataPreprocessorCoarseERA2ERA_Corrdiff(torch.utils.data.Dataset):
    """
    Dataset class for coarse ERA to downscale towards ERA data in second phase of CorrDiff approach. Some predicted fields do not correspond to input data (like CAPE and precip), named diagnostics variables.
    Some coarse variables are not outputted : they are named dynamic covariates.
    Variables that have fine and coarse resolution will use the residual approach (the residual are given as targets to the model), the normalized diagnostics variables are forecasted directly.
    """

    def __init__(
        self,
        data_dir_coarse,
        data_dir_fine,
        data_dir_diagnostics,
        files_coarse,
        files_fine,
        files_diagnostics,
        logger,
        variables_name_coarse = ["u850", "v850", "t850", "prw"],
        variables_name_fine = ["var131", "var132", "var130", "var137"], #u,v,t,TCWV
        variables_name_diagnostics = ["var59", "var228"], #cape, précips
        path_folder_stats = "./stats/",
        json_stats_rawdata = "statistics_lmdz_nudg_6h_1_1985-2014.json",
        json_stats_residual = "statistics_era5_lmdz_nudg_6h_1_1985-2014.json",
        json_stats_diag = "statistics_era5_lmdz_nudg_6h_1_1985-2014.json",
        dynamic_covariates_name= ["var59","var130","var137", "var131", "var132"], #CAPE, T850, TCWV, u850, v850
        dynamic_covariates_dir="./",
        dynamic_covariates_files=[],
        normalization_type = {},
        normalization_type_coarse = {},
        normalization_type_diagnostics = {},
        normalization_type_dynamic_cov = {},
        lon_slice_domain = slice(-6,9.75),
        lat_slice_domain = slice(54.75, 39),
        resolution_lon_coarse = 2.5, #in degrees
        resolution_lat_coarse = 1.25, #in degrees
        constant_variables_name=["LSM","topography"],
        files_constant_variables=[],
        constant_variables_dir = "./",
        normalization="linear",  # "linear" or "cos_sin"
    ):
        """
        :param data_dir: path to the dataset directory
        :param dynamic_covariates: name of the dynamic covariates used, i.e. the helping variables given to the model that will not be downscaled. If None, no covariates are used. defaults to None.
        :param dynamic_covariates_dir: path to the dataset directory containing the dynamic covariates. If None, no dynamic covariates are used. defaults to None.
        :param in_shape: shape of the low resolution images
        :param out_shape: shape of the high resolution images
        :param year_start: starting year of file named samples_{year_start}.nc
        :param year_end: ending year of file named samples_{year_end}.nc
        :param normalize_mean: channel-wise mean values estimated over all samples
        for normalizing file
        :param normalize_std: channel-wise standard deviation values estimated
        over all samples for normalizing file
        """
        self.epsilon =  0.02
        self.margin = 8
        self.logger = logger
        self.resolution_lon_coarse = resolution_lon_coarse
        self.resolution_lat_coarse = resolution_lat_coarse
        self.normalization = normalization.lower()
        #add a lambda function to check that datasets are timesorted :
        # open coarse data :
        # we have one file for each variable and we open them :
        self.coarse = []
        self.transform_funcs = {}
        self.inverse_transform = {}
        self.normalization_type_coarse = normalization_type_coarse
        self.files_coarse = files_coarse
        self.variables_name_coarse = variables_name_coarse
        self.data_dir_coarse = data_dir_coarse
        self.lon_slice_domain = lon_slice_domain
        self.lat_slice_domain = lat_slice_domain
        self.ds_coarse = [xr.open_zarr(self.data_dir_coarse + file) for file in self.files_coarse]
        self.data_vars_coarse = [var_xr[var] for var_xr,var in zip(self.ds_coarse,self.variables_name_coarse)]

        for file,var in zip(self.files_coarse, self.variables_name_coarse) :
            with xr.open_zarr(self.data_dir_coarse + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_coarse + file)
                data_var = var_xr[var]
                if "time" in list(data_var.coords) :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                    print(f"time length for {var} : {len(data_var.time.values)}")
                else :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                    print(f"time length for {var} : {len(data_var.time_counter.values)}")
                # check if there is a dimension other than time,x,y in xarray:
                if "height" in list(data_var.coords) :
                    #data_var = data_var.isel(height = 0)
                    data_var = data_var.isel(height = 0)
                elif "plev" in list(data_var.coords) :
                    #data_var = data_var.isel(plev = 0)
                    data_var = data_var.isel(plev = 0)
                coarse_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
                self.input_shape = coarse_ex.shape[2:]
            if "log" in normalization_type_coarse[var] :
                self.transform_funcs[var] = precip_log_transform
            elif "sqrt" in normalization_type_coarse[var] :
                print(f"using sqrt transform for {var}")
                self.transform_funcs[var] = sqrt_transform
            else :
                self.transform_funcs[var] = identity

        # Time embeddings
        if "time" in list(data_var.coords) :
            self.time = var_xr.time
            self.time_array = var_xr.time.values
            self.stime = var_xr.time.values[0]
            self.etime = var_xr.time.values[-1]
        else :
            self.time = var_xr.time_counter
            self.time_array = var_xr.time_counter.values
            self.stime = var_xr.time_counter.values[0]
            self.etime = var_xr.time_counter.values[-1]
        self.T = len(self.time)
        T = self.T
        self.year = self.time.dt.year
        self.month = self.time.dt.month
        self.day = self.time.dt.day
        self.hour = self.time.dt.hour

        # open fine data :
        # we have one file for each variable and we open them :
        self.fine = []
        self.normalization_type = normalization_type
        self.variables_name_fine = variables_name_fine
        self.files_fine = files_fine
        self.data_dir_fine = data_dir_fine
        self.ds_fine = [xr.open_zarr(self.data_dir_fine + file) for file in self.files_fine]
        self.data_vars_fine = [var_xr[var] for var_xr,var in zip(self.ds_fine,self.variables_name_fine)]

        for file,var in zip(self.files_fine,self.variables_name_fine) :
            with xr.open_zarr(self.data_dir_fine + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_fine + file)
                data_var = var_xr[var]
                self.longitude_fine = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
                self.latitude_fine = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()
                self.longitude_fine_full = torch.from_numpy(var_xr["lon"].values.copy()).float()
                self.latitude_fine_full = torch.from_numpy(var_xr["lat"].values.copy()).float()
                print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var} : {len(data_var.time.values)}")
                # check if there is a dimension other than time,x,y in xarray:
                if "height" in list(data_var.coords) :
                    data_var = data_var.isel(height = 0)
                elif "plev" in list(data_var.coords) :
                    data_var = data_var.isel(plev = 0)
                #fine_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
            if "log" in normalization_type[var] :
                self.transform_funcs[var] = precip_log_transform
                self.inverse_transform[var] = precip_exp_transform
            elif "sqrt" in normalization_type[var] :
                print(f"using sqrt transform for {var}")
                self.transform_funcs[var] = sqrt_transform
                self.inverse_transform[var] = square_transform
            else :
                self.transform_funcs[var] = identity
                self.inverse_transform[var] = identity
        self.output_shape = self.input_shape

        # Get information to do coarsening of dynamic covs :
        # Shape of fine data : (time, latitude, longitude)
        # Longitude and latitude range of full domain :
        self.lon_range_full = torch.max(self.longitude_fine_full) - torch.min(self.longitude_fine_full)
        self.lat_range_full = torch.max(self.latitude_fine_full) - torch.min(self.latitude_fine_full)
        self.coarse_in_shape = ((self.lat_range_full // resolution_lat_coarse) + 1, (self.lon_range_full // resolution_lon_coarse) + 1 )
        # In shape of coarse :
        self.in_shape = (int(np.abs(self.latitude_fine_full[-1] - self.latitude_fine_full[0]).item() // resolution_lat_coarse) , int(np.abs(self.longitude_fine_full[-1] - self.longitude_fine_full[0]).item() // resolution_lon_coarse))
        #Fine resolution :
        self.resolution_lon_fine = np.abs(self.longitude_fine[1] - self.longitude_fine[0]).item()
        self.resolution_lat_fine = np.abs(self.latitude_fine[1] - self.latitude_fine[0]).item()
        self.resolution_lon_coarse = resolution_lon_coarse
        self.resolution_lat_coarse = resolution_lat_coarse
        print(f"self.resolution_lat_fine : self.dW : {self.resolution_lat_fine}")
        self.dH = self.resolution_lat_fine # for ERA data order of axes is (vars, lat, lon)
        self.dW = self.resolution_lon_fine # this corresponds to (C,H,W)
        self.dH_coarse = self.resolution_lat_coarse
        self.dW_coarse = self.resolution_lon_coarse

        # open static cov data :
        # we have one file for each variable and we open them :
        self.constant_variables = []
        self.files_constant_variables = files_constant_variables
        self.constant_variables_name = constant_variables_name
        self.constant_variables_dir = constant_variables_dir
        self.ds_constant = [xr.open_zarr(self.constant_variables_dir + file) for file in self.files_constant_variables]
        self.data_vars_constant = [var_xr[var] for var_xr, var in zip(self.ds_constant, self.constant_variables_name)]
        for data_var in self.data_vars_constant :
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            if "time" in list(data_var.coords) :
                data_var = data_var.isel(time = 0)
            self.constant_variables.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[None,None,:,:])
        self.constant_variables = np.concatenate(self.constant_variables, axis = 1) # shape (T,nb_var,L,W)
        # check that constant_variables is in 4D : 
        if len(self.constant_variables.shape) == 3 :
            self.constant_variables = self.constant_variables[:,None,:,:]
        self.constant_variables = torch.from_numpy(self.constant_variables).float()
    
        # open dynamic cov data :
        # we have one file for each covariable and we open them :
        self.covariates_name = dynamic_covariates_name
        self.normalization_type_dynamic_cov = normalization_type_dynamic_cov
        self.dynamic_covariates_files = dynamic_covariates_files
        self.dynamic_covariates_name = dynamic_covariates_name
        self.dynamic_covariates_dir = dynamic_covariates_dir
        self.ds_covariate = [xr.open_zarr(self.dynamic_covariates_dir + file) for file in self.dynamic_covariates_files]
        self.data_vars_covariate = [var_xr[var] for var_xr,var in zip(self.ds_covariate, self.dynamic_covariates_name)]       

        for file,var in zip(self.dynamic_covariates_files,self.dynamic_covariates_name) :
            with xr.open_zarr(self.dynamic_covariates_dir + file) as var_xr:
                # var_xr = xr.open_zarr(dynamic_covariates_dir + file)
                data_var = var_xr[var]
                if "time" in list(data_var.coords) :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                    print(f"time length for {var} : {len(data_var.time.values)}")
                else :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                    print(f"time length for {var} : {len(data_var.time_counter.values)}")
            if "log" in normalization_type_dynamic_cov[var] :
                self.transform_funcs[var] = precip_log_transform
            else :
                self.transform_funcs[var] = identity

        # open diagnostics data :
        self.diagnostics_name = variables_name_diagnostics
        self.normalization_type_diagnostics = normalization_type_diagnostics
        self.files_diagnostics = files_diagnostics
        self.variables_name_diagnostics = variables_name_diagnostics
        self.data_dir_diagnostics = data_dir_diagnostics
        self.ds_diagnostic = [xr.open_zarr(self.data_dir_diagnostics + file) for file in self.files_diagnostics]
        self.data_vars_diagnostic = [var_xr[var] for var_xr,var in zip(self.ds_diagnostic, self.variables_name_diagnostics)]

        for file,var in zip(self.files_diagnostics,self.variables_name_diagnostics) :
            with xr.open_zarr(self.data_dir_diagnostics + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_fine + file)
                data_var = var_xr[var]
                print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                print(f"time length for {var} : {len(data_var.time.values)}")
            if "log" in normalization_type_diagnostics[var] :
                self.transform_funcs[var] = precip_log_transform
                self.inverse_transform[var] = precip_exp_transform
            else :
                self.transform_funcs[var] = identity
                self.inverse_transform[var] = identity

        norm_mapping = stats(
            logger = logger,
            input_dir = path_folder_stats,
            json_names = [json_stats_rawdata, json_stats_residual, json_stats_diag]
        )

        normalize_rawdata_mean, normalize_rawdata_std = get_mean_std_tensors(
            vars=variables_name_coarse,
            normalization_types=normalization_type_coarse,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for coarse :")
        logger.info(variables_name_coarse)
        logger.info(normalize_rawdata_mean)
        logger.info(normalize_rawdata_std)

        self.normalize_rawdata_transform = torchvision.transforms.Normalize(
            normalize_rawdata_mean, normalize_rawdata_std
        )

        normalize_residual_mean, normalize_residual_std = get_mean_std_tensors(
            vars=variables_name_fine,
            normalization_types=normalization_type,
            norm_mapping=norm_mapping,
            coarse_or_residual="residual"
        )
        logger.info("Normalisation tensors for residuals :")
        logger.info(variables_name_fine)
        logger.info(normalize_residual_mean)
        logger.info(normalize_residual_std)

        self.normalize_residual_mean = normalize_residual_mean
        self.normalize_residual_std = normalize_residual_std
        self.normalize_residual_transform = torchvision.transforms.Normalize(
            normalize_residual_mean, normalize_residual_std
        )
        self.inverse_normalize_residual = inverse_normalize_residual

        normalize_covariates_mean, normalize_covariates_std = get_mean_std_tensors(
            vars=dynamic_covariates_name,
            normalization_types=normalization_type_dynamic_cov,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for covariates :")
        logger.info(dynamic_covariates_name)
        logger.info(normalize_covariates_mean)
        logger.info(normalize_covariates_std)

        self.normalize_covariates_transform = torchvision.transforms.Normalize(
            normalize_covariates_mean, normalize_covariates_std
        )

        # normalize constant variables :
        normalize_constant_variables_transform = torchvision.transforms.Normalize(
            torch.mean(self.constant_variables, dim = [0,2,3]), torch.std(self.constant_variables, dim = [0,2,3])
        )
        try :
            constant_variables_norm = normalize_constant_variables_transform(self.constant_variables)        
        except :
            constant_variables_norm = self.constant_variables - torch.mean(self.constant_variables, dim = [0,2,3])
        self.constant_variables_norm = constant_variables_norm
        
        # normalize diagnostic variables :
        normalize_diagnostics_mean, normalize_diagnostics_std = get_mean_std_tensors(
            vars=variables_name_diagnostics,
            normalization_types=normalization_type_diagnostics,
            norm_mapping=norm_mapping,
            coarse_or_residual="coarse"
        )
        logger.info("Normalisation tensors for diags :")
        logger.info(variables_name_diagnostics)
        logger.info(normalize_diagnostics_mean)
        logger.info(normalize_diagnostics_std)

        self.normalize_diagnostics_mean = normalize_diagnostics_mean
        self.normalize_diagnostics_std = normalize_diagnostics_std
            
        self.normalize_diagnostics_transform = torchvision.transforms.Normalize(
            self.normalize_diagnostics_mean, self.normalize_diagnostics_std
        )

        # Define limits for plotting (plus/minus 2 sigma
        self.vmin = normalize_rawdata_mean - 2 * normalize_rawdata_std
        self.vmax = normalize_rawdata_mean + 2 * normalize_rawdata_std

        self.year_norm = torch.from_numpy(
            ((self.year.to_numpy() - 1940) / 100).astype(np.float32)
        )

        # Normalize and convert to numpy (load into mem)
        if self.normalization == "linear":
            self.doy = (self.month - 1.0) * 30 + (self.day - 1.0)
            self.doy_norm = torch.from_numpy(
                (self.doy.to_numpy() / 360).astype(np.float32)
            )
            self.hour_norm = torch.from_numpy(
                (self.hour.to_numpy() / 24.0).astype(np.float32)
            )
        elif self.normalization == "cos_sin":
            date = pd.to_datetime(dict(year=self.year, month=self.month, day=self.day))
            self.doy = (date - datetime(2000, 1, 1)).dt.days
            doy_np = self.doy.to_numpy()
            hour_np = self.hour.to_numpy()
            self.doy_sin = torch.sin(2 * np.pi * torch.from_numpy(doy_np) / 365.25)
            self.doy_cos = torch.cos(2 * np.pi * torch.from_numpy(doy_np) / 365.25)
            self.hour_sin = torch.sin(2 * np.pi * torch.from_numpy(hour_np) / 24.0)
            self.hour_cos = torch.cos(2 * np.pi * torch.from_numpy(hour_np) / 24.0)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

        print("Dataset initialized.")

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.T

    def extract_batch(self, full_domain_data) :
        # full_domain_data is np.ndarray of shape (C,H,W)
        # this function is for ERA5 data with shape (C,H,W) where the order of axes is (nb_var, latitude, longitude)
        longitude_full = self.longitude_fine_full.numpy()
        latitude_full = self.latitude_fine_full.numpy()
        lon_min_select = np.min(self.longitude_fine.numpy())
        lon_max_select = np.max(self.longitude_fine.numpy())
        lat_min_select = np.min(self.latitude_fine.numpy())
        lat_max_select = np.max(self.latitude_fine.numpy())
        ind_w_min = np.where(longitude_full == lon_min_select)[0][0]
        ind_w_max = np.where(longitude_full == lon_max_select)[0][0]
        ind_h_min = np.where(latitude_full == lat_min_select)[0][0]
        ind_h_max = np.where(latitude_full == lat_max_select)[0][0] 
        if ind_h_min > ind_h_max :
            ind_h_min, ind_h_max = ind_h_max, ind_h_min
        if ind_w_min > ind_w_max :
            ind_w_min, ind_w_max = ind_w_max, ind_w_min
        return full_domain_data[:,ind_h_min:ind_h_max+1, ind_w_min:ind_w_max+1]

    def filter_batch(self, fine_patch, fine_block, dW_coarse = None, dH_coarse = None):
        """
        Apply Gaussian low-pass filtering for multi-scale processing.

        Parameters
        ----------
        fine_patch : np.ndarray
            Fine-resolution data of shape (C, H, W).
        fine_block : np.ndarray
            Reference block used to determine scaling factors.

        Returns
        -------
        fine_filtered : np.ndarray
            Filtered data of shape (C, H, W).

        Notes
        -----
        - Filters high-frequency components beyond the coarse grid's Nyquist.
        - Uses Gaussian filtering in the frequency domain.
        - Processes each channel independently.
        """

        # Output container
        fine_filtered = np.zeros_like(fine_patch)

        if dH_coarse is None or dW_coarse is None :
            # ----------------------------------------------------------
            # Determine scaling between fine and coarse grids
            # ----------------------------------------------------------
            scale_factor_H = fine_block.shape[-2] / self.in_shape[0]
            scale_factor_W = fine_block.shape[-1] / self.in_shape[1]

            # ----------------------------------------------------------
            # Coarse-grid physical spacing and Nyquist limits
            # ----------------------------------------------------------
            dW_coarse = self.dW * scale_factor_W
            dH_coarse = self.dH * scale_factor_H

        f_nW_coarse = 1.0 / (2.0 * dW_coarse)
        f_nH_coarse = 1.0 / (2.0 * dH_coarse)

        # ----------------------------------------------------------
        # Apply filtering per channel
        # ----------------------------------------------------------
        C = fine_patch.shape[0]

        for c in range(C):
            try:
                img = fine_patch[c]
                filt = gaussian_filter(
                    img,
                    self.dW,
                    self.dH,
                    f_nW_coarse,
                    f_nH_coarse,
                    self.epsilon,
                    self.margin,
                )

                # Save the filtered channel
                fine_filtered[c] = filt

            except Exception as e:
                self.logger.exception(f"Filtering failed at channel {c}", e)
                raise

        return fine_filtered

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        coarse = []
        coarse_transform = []
        time_index = self.time_array[index]

        for data_var, var in zip(self.data_vars_coarse, self.variables_name_coarse):
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                #data_var = data_var.isel(height = 0)
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                #data_var = data_var.isel(plev = 0)
                data_var = data_var.isel(plev = 0)
            #coarse.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            # use isel to access only right timestep without loading everything in memory
            if "time" in list(data_var.coords) :
                coarse_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "precip" :
                    coarse_var = coarse_var # no need to convert, coarses are from corrdiff, already in mm/6h
                coarse.append(coarse_var)
            else :
                coarse_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "precip" :
                    coarse_var = coarse_var # no need to convert, coarses are from corrdiff, already in mm/6h.
                coarse.append(coarse_var)
            coarse_transform.append(
                self.transform_funcs[var](
                    coarse[-1].copy()
                )
            )
        coarse = np.concatenate(coarse, axis = 1) # shape (T,nb_var,L,W)
        coarse_transform = np.concatenate(coarse_transform, axis = 1) # shape (T,nb_var,L,W)
        # replace NaNs with spatial mean for timestep of the channel
        spatial_mean_coarse = np.nanmean(coarse, axis=(2,3))[:,:,None,None]
        #coarse = np.where(np.isnan(self.coarse),0,self.coarse)
        coarse = np.where(np.isnan(coarse),spatial_mean_coarse,coarse)
        coarse = torch.from_numpy(coarse).float()
        spatial_mean_coarse_transform = np.nanmean(coarse_transform, axis=(2,3))[:,:,None,None]
        coarse_transform = np.where(np.isnan(coarse_transform),spatial_mean_coarse_transform,coarse_transform)
        coarse_transform = torch.from_numpy(coarse_transform).float()

        # open fine data :
        # we have one file for each variable and we open them :
        fine = []
        fine_transform = []
        for data_var, var_xr, var in zip(self.data_vars_fine, self.ds_fine, self.variables_name_fine) :
            longitude_fine = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
            latitude_fine = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            # fine.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                fine_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228" :
                    fine_var = 1000 * fine_var # convert to mm from m
                fine.append(fine_var)
            else :
                fine_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228":
                    fine_var = 1000*fine_var # convert to mm from m
                fine.append(fine_var)
            fine_transform.append(
                self.transform_funcs[var](
                    fine[-1].copy()
                )
            )
        fine = np.concatenate(fine, axis = 1) # shape (T,nb_var,L,W)
        # spatial_mean_fine = np.nanmean(fine, axis=(2,3))[:,:,None,None]
        # fine = np.where(np.isnan(fine),spatial_mean_fine,fine)
        fine = torch.from_numpy(fine).float()

        fine_transform = np.concatenate(fine_transform, axis = 1) # shape (T,nb_var,L,W)
        # spatial_mean_fine_transform = np.nanmean(fine_transform, axis=(2,3))[:,:,None,None]
        # fine_transform = np.where(np.isnan(fine_transform),spatial_mean_fine_transform,fine_transform)
        fine_transform = torch.from_numpy(fine_transform).float()

        # open constant data :
        constant_variables = self.constant_variables.clone()

        #open dynamic covariates :
        covariates = []
        covariates_full_domain = []
        covariates_transform = []
        for var_xr,data_var, var in zip(self.ds_covariate,self.data_vars_covariate, self.dynamic_covariates_name) :
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            #covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:])
                covariates_full_domain.append(data_var.sel(time = time_index, method = "nearest").values.copy()[None,None,:,:])
            else :
                covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:])
                covariates_full_domain.append(data_var.sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:])
            longitude_covariates = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
            latitude_covariates = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()

        covariates_np = np.concatenate(covariates, axis = 1) # np array shape (1,nb_var,L,W)


        covariates_full_domain = np.concatenate(covariates_full_domain, axis = 1) #np array shape (T,nb_var,L,W)
        covariates_full_domain_filtered = self.filter_batch(
            fine_patch=covariates_full_domain[0],
            fine_block=covariates_np[0],
            dW_coarse=self.dW_coarse,
            dH_coarse=self.dH_coarse
        ) # np array shape (nb_var,L,W)
        covariates_full_domain_coarse = coarse_down_up(covariates_full_domain_filtered, covariates_full_domain, self.in_shape) #torch tensor shape (nb_var,L,W)
        covariates_coarse = self.extract_batch(covariates_full_domain_coarse)[None,:,:,:] # torch tensor shape (1, nb_var,L,W)

        covariates_coarse_transform = []
        for i, var in enumerate(self.dynamic_covariates_name) :
            covariates_coarse_transform.append(
                self.transform_funcs[var](
                    covariates_coarse[0:1,i:i+1].clone()
                )
            )
        covariates_coarse_transform = torch.cat(covariates_coarse_transform, dim = 1) # shape (1,nb_var,L,W)

        # open diag data :
        diagnostics = []
        diagnostics_transform = []
        for data_var, var in zip(self.data_vars_diagnostic,self.variables_name_diagnostics) :
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            #diagnostics.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                diag_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228" :
                    diag_var = 1000*diag_var # convert from mm to m
                diagnostics.append(diag_var)
            else :
                diag_var = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).sel(time_counter = time_index, method = "nearest").values.copy()[None,None,:,:]
                if var == "var228" :
                    diag_var = 1000*diag_var # convert from mm to m
                diagnostics.append(diag_var)
            diagnostics_transform.append(
                self.transform_funcs[var](
                    diagnostics[-1].copy()
                )
            )
        no_diags_flag = (len(self.variables_name_diagnostics) == 0)
        if not no_diags_flag :
            diagnostics = np.concatenate(diagnostics, axis = 1) # shape (T,nb_var,L,W)
            spatial_mean_diagnostics = np.nanmean(diagnostics, axis=(2,3))[:,:,None,None]
            diagnostics = np.where(np.isnan(diagnostics),spatial_mean_diagnostics,diagnostics)
            diagnostics = torch.from_numpy(diagnostics).float()

            diagnostics_transform = np.concatenate(diagnostics_transform, axis = 1) # shape (T,nb_var,L,W)
            spatial_mean_diagnostics_transform = np.nanmean(diagnostics_transform, axis=(2,3))[:,:,None,None]
            diagnostics_transform = np.where(np.isnan(diagnostics_transform),spatial_mean_diagnostics_transform,diagnostics_transform)
            diagnostics_transform = torch.from_numpy(diagnostics_transform).float()

        residual = fine_transform - coarse_transform

        # normalise coarse :
        coarse_norm = self.normalize_rawdata_transform(coarse_transform)
        # normalise residual :
        residual_norm = self.normalize_residual_transform(residual)
        # normalise covariates :
        covariates_coarse_norm = self.normalize_covariates_transform(covariates_coarse_transform)
        # normalise constant covariates :
        constant_variables_norm = self.constant_variables_norm.clone()
        # normalise diag :
        if not no_diags_flag :
            diagnostics_norm = self.normalize_diagnostics_transform(diagnostics_transform)

        if not no_diags_flag :
            targets_channels_name = self.variables_name_fine + self.diagnostics_name
            targets = torch.concat((residual_norm, diagnostics_norm), dim = 1)  # targets  = normalized residual (of transformed fine and coarse if applicable)
            inputs = torch.concat((coarse_norm, covariates_coarse_norm, constant_variables_norm), dim = 1)  # inputs   = normalized coarse concatenated to normalized
        else :
            targets_channels_name = self.variables_name_fine
            targets = residual_norm # targets  = normalized residual (of transformed fine and coarse if applicable)
            inputs = torch.concat((coarse_norm, covariates_coarse_norm, constant_variables_norm), dim = 1)  # inputs   = normalized coarse concatenated to normalized       
     
        if torch.any(torch.isnan(inputs)) :
            print("inputs contain NaN !")
        if torch.any(torch.isnan(targets)) :
            print("targets contain NaN !")
        if torch.any(torch.isnan(fine)) :
            print("fine contain NaN !")
        if torch.any(torch.isnan(coarse)) :
            print("coarse contain NaN !")
        # if torch.any(torch.isnan(covariates)) :
        #     print("covariates contain NaN !")
        if not no_diags_flag :
            if torch.any(torch.isnan(diagnostics)) :
                print("diagnostics contain NaN !")
        if torch.any(torch.isnan(longitude_fine)) :
            print("longitude fine contain NaN !")
        if torch.any(torch.isnan(latitude_fine)) :
            print("latitude fine contain NaN !")
        if torch.any(torch.isnan(longitude_covariates)) :
            print("longitude covariates contain NaN !")
        if torch.any(torch.isnan(latitude_covariates)) :
            print("latitude covariates contain NaN !")
        #test for infs
        if torch.any(torch.isinf(inputs)) :
            print("inputs contain inf !")
        if torch.any(torch.isinf(targets)) :
            print("targets contain inf !")
        if torch.any(torch.isinf(fine)) :
            print("fine contain inf !")
        if torch.any(torch.isinf(coarse)) :
            print("coarse contain inf !")
        # if torch.any(torch.isinf(covariates)) :
        #     print("covariates contain inf !")
        if not no_diags_flag :
            if torch.any(torch.isinf(diagnostics)) :
                print("diagnostics contain inf !")
        if torch.any(torch.isinf(longitude_fine)) :
            print("longitude fine contain inf !")
        if torch.any(torch.isinf(latitude_fine)) :
            print("latitude fine contain inf !")
        if torch.any(torch.isinf(longitude_covariates)) :
            print("longitude covariates contain inf !")
        if torch.any(torch.isinf(latitude_covariates)) :
            print("latitude covariates contain inf !")   

        sample = {
            "inputs": inputs[0],
            "targets": targets[0],
            "fine": fine[0],
            "coarse": coarse[0],
            "year": self.year_norm[index],
            "time_index": index,
            "corrdinates": { 
                "lon": longitude_fine,
                "lat": latitude_fine,
            },
            "corrdinates_covariates": {
                "lon": longitude_covariates,
                "lat": latitude_covariates,
            },
        }
        if len(self.covariates_name) != 0 :
            sample["covariates"] = covariates_coarse[0]
        if len(self.diagnostics_name) != 0 :
            sample["diagnostics"] = diagnostics[0]

        if self.normalization == "linear":
            sample["doy"] = self.doy_norm[index]
            sample["hour"] = self.hour_norm[index]
        elif self.normalization == "cos_sin":
            sample["doy_sin"] = self.doy_sin[index]
            sample["doy_cos"] = self.doy_cos[index]
            sample["hour_sin"] = self.hour_sin[index]
            sample["hour_cos"] = self.hour_cos[index]

        return sample

    def model_output_to_pred(self, model_output, coarse_image) :
        nb_vars_fine = len(self.variables_name_fine)
        nb_diags = len(self.diagnostics_name)
        no_diags_flag = (nb_diags == 0)
        predictions = []
        residual = model_output[:,:nb_vars_fine,:,:] # first nb_vars_fine channels correspond to residual
        # denormalize :
        denorm_residual = self.inverse_normalize_residual(residual, self.normalize_residual_std, self.normalize_residual_mean)
        for i,var in enumerate(self.variables_name_fine) :
            channel_pred = self.inverse_transform[var](self.transform_funcs[var](coarse_image[:,i,:,:][:,None,:,:]) + denorm_residual[:,i,:,:][:,None,:,:])
            predictions.append(channel_pred)
        if not no_diags_flag :
            diags = model_output[:,nb_vars_fine:,:,:] # other channels correspond to diags variables
            denorm_diags = self.inverse_normalize_residual(diags, self.normalize_diagnostics_std, self.normalize_diagnostics_mean)
            for i,var in enumerate(self.diagnostics_name) :
                channel_pred = self.inverse_transform[var](denorm_diags[:,i,:,:][:,None,:,:])
                predictions.append(channel_pred)
        return torch.cat(predictions, dim=1)

    def new_epoch(self):
        pass