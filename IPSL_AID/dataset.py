import torch
import torchvision
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
#import cartopy 
import pandas as pd
from datetime import datetime

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
        transformed_data = torch.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = torch.log(data_tensor[:,0,:,:] + 1)
    else :
        transformed_data = np.zeros_like(data_tensor)
        transformed_data[:,0,:,:] = np.log(data_tensor[:,0,:,:] + 1)
    return transformed_data

def precip_exp_transform(data_tensor) :
    transformed_data = torch.zeros_like(data_tensor)
    transformed_data[:,0,:,:] = torch.exp(data_tensor[:,0,:,:]) - 1
    return transformed_data

class DataPreprocessorLMDZ2ERA_lazyload_zarr(torch.utils.data.Dataset):
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
        variables_name_coarse = ["u850", "v850", "t850", "prw"],
        variables_name_fine = ["var131", "var132", "var130", "var137"], #u,v,t,TCWV
        variables_name_diagnostics = ["var59", "var228"], #cape, précips
        normalize_rawdata_mean=torch.Tensor([3.8010375, -0.058216378, 277.36862, 17.013712]), # norm for France 1985-2014
        normalize_rawdata_std=torch.Tensor([7.3155675, 6.602474, 6.637401, 7.267456]),
        normalize_residual_mean=torch.Tensor([-0.1519282, 0.117712386, 1.3807437, -0.6129347]),
        normalize_residual_std=torch.Tensor([2.3272147, 2.5145345, 1.7226714, 3.0673473]),
        normalize_diagnostics_mean=torch.Tensor([53.797516, -8.366848]),
        normalize_diagnostics_std=torch.Tensor([236.67465, 1.2114729]),
        dynamic_covariates_name= ["var59","var130","var137", "var131", "var132"], #CAPE, T850, TCWV, u850, v850
        dynamic_covariates_dir="./",
        dynamic_covariates_files=[],
        lon_slice_domain = slice(-6,9.75),
        lat_slice_domain = slice(54.75, 39),
        normalize_covariates_mean=torch.Tensor([5.859726, 8.724808, -0.4640244, -1.2654291, 268.9366, 252.2054, 0.004785511, 0.0023771676, 0.00071223004, 98550.35]), #real training data South France 1985-2014 tensor([ 93.2392, 280.4401,  15.1545,   0.6444,  -0.4625])
        normalize_covariates_std=torch.Tensor([8.168476, 10.825516, 7.781761, 11.132606, 6.0343814, 6.0387325, 0.0020607216, 0.0014883897, 0.0004916295, 3898.0361]), #real training data South France 1985-2014 tensor([323.9705,   6.4705,   8.1186,   4.3525,   5.2768])
        constant_variables_name=["LSM","topography"],
        files_constant_variables=[],
        constant_variables_dir = "./",
        normalization="linear",  # "linear" or "cos_sin"
        precip_transform=None,
        cape_transform=None,
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

        self.normalization = normalization.lower()
        #add a lambda function to check that datasets are timesorted :
        # open coarse data :
        # we have one file for each variable and we open them :
        self.coarse = []
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
                else :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')
                # check if there is a dimension other than time,x,y in xarray:
                if "height" in list(data_var.coords) :
                    #data_var = data_var.isel(height = 0)
                    data_var = data_var.isel(height = 0)
                elif "plev" in list(data_var.coords) :
                    #data_var = data_var.isel(plev = 0)
                    data_var = data_var.isel(plev = 0)
                coarse_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
                self.input_shape = coarse_ex.shape[2:]

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
        self.transform_funcs = {}
        self.inverse_transform = {}
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
                # check if there is a dimension other than time,x,y in xarray:
                if "height" in list(data_var.coords) :
                    data_var = data_var.isel(height = 0)
                elif "plev" in list(data_var.coords) :
                    data_var = data_var.isel(plev = 0)
                #fine_ex = data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[:,None,:,:]
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
            self.constant_variables.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[None,None,:,:])
        self.constant_variables = np.concatenate(self.constant_variables, axis = 1) # shape (T,nb_var,L,W)
        # check that constant_variables is in 4D : 
        if len(self.constant_variables.shape) == 3 :
            self.constant_variables = self.constant_variables[:,None,:,:]
        self.constant_variables = torch.from_numpy(self.constant_variables).float()
    
        # open dynamic cov data :
        # we have one file for each covariable and we open them :
        self.covariates_name = dynamic_covariates_name
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
                else :
                    print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time_counter.values)}')

        # open diagnostics data :
        self.diagnostics_name = variables_name_diagnostics
        self.files_diagnostics = files_diagnostics
        self.variables_name_diagnostics = variables_name_diagnostics
        self.data_dir_diagnostics = data_dir_diagnostics
        self.precip_transform = precip_transform
        self.cape_transform = cape_transform
        self.ds_diagnostic = [xr.open_zarr(self.data_dir_diagnostics + file) for file in self.files_diagnostics]
        self.data_vars_diagnostic = [var_xr[var] for var_xr,var in zip(self.ds_diagnostic, self.variables_name_diagnostics)]

        for file,var in zip(self.files_diagnostics,self.variables_name_diagnostics) :
            with xr.open_zarr(self.data_dir_diagnostics + file) as var_xr:
                # var_xr = xr.open_zarr(data_dir_fine + file)
                data_var = var_xr[var]
                print(f'dataset for {var} is time-sorted : {is_sorted(data_var.time.values)}')
                if var == "var228" : # precip
                    if precip_transform == "log_precip" :
                        #self.diagnostics_transform.append(precip_log_transform(self.diagnostics[-1]))
                        self.transform_funcs[var] = precip_log_transform
                        self.inverse_transform[var] = precip_exp_transform
                    else :
                        #self.diagnostics_transform.append(self.diagnostics[-1])
                        self.transform_funcs[var] = identity
                        self.inverse_transform[var] = identity
                if var == "var59" :
                    if cape_transform == "log_cape" :
                        #self.diagnostics_transform.append(precip_log_transform(self.diagnostics[-1]))
                        self.transform_funcs[var] = precip_log_transform
                        self.inverse_transform[var] = precip_exp_transform
                    else :
                        #self.diagnostics_transform.append(self.diagnostics[-1])
                        self.transform_funcs[var] = identity
                        self.inverse_transform[var] = identity

        self.normalize_rawdata_transform = torchvision.transforms.Normalize(
            normalize_rawdata_mean, normalize_rawdata_std
        )
        

        self.normalize_residual_mean = normalize_residual_mean
        self.normalize_residual_std = normalize_residual_std
        self.normalize_residual_transform = torchvision.transforms.Normalize(
            normalize_residual_mean, normalize_residual_std
        )
        self.inverse_normalize_residual = inverse_normalize_residual
            
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
        for data_var in self.data_vars_coarse :
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
                coarse.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time = index).values.copy()[None,None,:,:])
            else :
                coarse.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time_counter = index).values.copy()[None,None,:,:])
        coarse = np.concatenate(coarse, axis = 1) # shape (T,nb_var,L,W)
        # replace NaNs with spatial mean for timestep of the channel
        spatial_mean_coarse = np.nanmean(coarse, axis=(2,3))[:,:,None,None]
        #coarse = np.where(np.isnan(self.coarse),0,self.coarse)
        coarse = np.where(np.isnan(coarse),spatial_mean_coarse,coarse)
        coarse = torch.from_numpy(coarse).float()

        # open fine data :
        # we have one file for each variable and we open them :
        fine = []
        for data_var, var_xr in zip(self.data_vars_fine, self.ds_fine) :
            longitude_fine = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
            latitude_fine = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()
            # check if there is a dimension other than time,x,y in xarray:
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            # fine.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                fine.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time = index).values.copy()[None,None,:,:])
            else :
                fine.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time_counter = index).values.copy()[None,None,:,:])
        fine = np.concatenate(fine, axis = 1) # shape (T,nb_var,L,W)
        spatial_mean_fine = np.nanmean(fine, axis=(2,3))[:,:,None,None]
        fine = np.where(np.isnan(fine),spatial_mean_fine,fine)
        fine = torch.from_numpy(fine).float()

        # open constant data :
        constant_variables = self.constant_variables.clone()

        #open dynamic covariates :
        covariates = []
        for var_xr,data_var in zip(self.ds_covariate,self.data_vars_covariate) :
            if "height" in list(data_var.coords) :
                data_var = data_var.isel(height = 0)
            elif "plev" in list(data_var.coords) :
                data_var = data_var.isel(plev = 0)
            #covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).values[index:index+1,None,:,:])
            if "time" in list(data_var.coords) :
                covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time = index).values.copy()[None,None,:,:])
            else :
                covariates.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time_counter = index).values.copy()[None,None,:,:])
            longitude_covariates = torch.from_numpy(var_xr["lon"].sel(lon = self.lon_slice_domain).values.copy()).float()
            latitude_covariates = torch.from_numpy(var_xr["lat"].sel(lat = self.lat_slice_domain).values.copy()).float()
        covariates = np.concatenate(covariates, axis = 1) # shape (T,nb_var,L,W)
        spatial_mean_covariates = np.nanmean(covariates, axis=(2,3))[:,:,None,None]
        covariates = np.where(np.isnan(covariates),spatial_mean_covariates,covariates)
        covariates = torch.from_numpy(covariates).float()

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
                diagnostics.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time = index).values.copy()[None,None,:,:])
            else :
                diagnostics.append(data_var.sel(lon = self.lon_slice_domain, lat = self.lat_slice_domain).isel(time_counter = index).values.copy()[None,None,:,:])
            if var == "var228" : # precip
                if self.precip_transform == "log_precip" :
                    diagnostics_transform.append(precip_log_transform(diagnostics[-1].copy()))
                else :
                    diagnostics_transform.append(diagnostics[-1].copy())
            if var == "var59" :
                if self.cape_transform == "log_cape" :
                    diagnostics_transform.append(precip_log_transform(diagnostics[-1].copy()))
                else :
                    diagnostics_transform.append(diagnostics[-1].copy())
        diagnostics = np.concatenate(diagnostics, axis = 1) # shape (T,nb_var,L,W)
        diagnostics_transform = np.concatenate(diagnostics_transform, axis = 1) # shape (T,nb_var,L,W)
        diagnostics = torch.from_numpy(diagnostics).float()
        diagnostics_transform = torch.from_numpy(diagnostics_transform).float()

        residual = fine - coarse

        # normalise coarse :
        coarse_norm = self.normalize_rawdata_transform(coarse)
        # normalise residual :
        residual_norm = self.normalize_residual_transform(residual)
        # normalise covariates :
        covariates_norm = self.normalize_covariates_transform(covariates)
        # normalise constant covariates :
        constant_variables_norm = self.constant_variables_norm.clone()
        # normalise diag :
        diagnostics_norm = self.normalize_diagnostics_transform(diagnostics_transform)

        targets_channels_name = self.variables_name_fine + self.diagnostics_name
        targets = torch.concat((residual_norm, diagnostics_norm), dim = 1)  # targets  = normalized residual (of transformed fine and coarse if applicable)
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
            "covariates": covariates[0] if self.covariates_name is not None else None,
            "diagnostics": diagnostics[0] if self.diagnostics_name is not None else None,
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
        predictions = []
        residual = model_output[:,:nb_vars_fine,:,:] # first nb_vars_fine channels correspond to residual
        diags = model_output[:,nb_vars_fine:,:,:] # other channels correspond to diags variables
        # denormalize :
        denorm_residual = self.inverse_normalize_residual(residual, self.normalize_residual_std, self.normalize_residual_mean)
        denorm_diags = self.inverse_normalize_residual(diags, self.normalize_diagnostics_std, self.normalize_diagnostics_mean)

        for i,var in enumerate(self.variables_name_fine) :
            channel_pred = self.inverse_transform[var](self.transform_funcs[var](coarse_image[:,i,:,:][:,None,:,:]) + denorm_residual[:,i,:,:][:,None,:,:])
            predictions.append(channel_pred)
        for i,var in enumerate(self.diagnostics_name) :
            channel_pred = self.inverse_transform[var](denorm_diags[:,i,:,:][:,None,:,:])
            predictions.append(channel_pred)
        return torch.cat(predictions, dim=1)

    def new_epoch(self):
        pass