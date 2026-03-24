Quickstart
==========

Basic Workflow
--------------

1. **Setup environment** (see :doc:`installation`)
2. **Configure your experiment** via setup script
3. **Test components** (see :doc:`testing_philosophy`)
4. **Run training or inference**

Running the Model
-----------------

Model execution is controlled via a **setup bash script** that:

- Defines all model, data, and training parameters
- Generates a runnable script
- Generates a SLURM submission script
- Encodes the full configuration in the output folder structure

Typical workflow:

1. Edit the setup script:
   - Select diffusion model (VE / VP / EDM / iDDPM)
   - Select architecture (e.g. ADM UNet)
   - Define variables, normalization, batch sizes
   - Choose training or inference mode

2. Generate run and sbatch scripts:

.. code-block:: bash

   ./setup

3. Submit the job:

.. code-block:: bash

   sbatch slurm/sbatch_diffusion_*.sh

This approach ensures **full reproducibility**, as every run is uniquely tagged
by its configuration.

Example Configuration
---------------------

Below is a comprehensive reference of all command line arguments accepted by the
IPSL-AID diffusion model. These arguments can be set in the setup bash script
and are passed to the Python training script.

.. list-table:: Command Line Arguments
   :header-rows: 1
   :widths: 25 15 50
   :stub-columns: 1

   * - Argument
     - Type
     - Description
   * - **Execution Mode**
     - -
     -
   * - ``--debug``
     - bool
     - Enable debug mode for reduced logging and testing (default: False)
   * - ``--run_type``
     - str
     - Run mode: ``train``, ``resume_train``, ``inference``, or ``inference_regional`` (default: train)
   * - ``--region``
     - str
     - Geographic region: ``us``, ``europe``, or ``asia`` (for regional inference)
   * - ``--inference_type``
     - str
     - Inference mode: ``direct`` (deterministic) or ``sampler`` (stochastic) (default: direct)
   * - **Data Configuration**
     - -
     -
   * - ``--datadir``
     - str
     - Main dataset directory path (**required**)
   * - ``--per_var_datadir``
     - list
     - Per-variable data directories as ``VAR=path`` pairs
   * - ``--varnames_list``
     - list
     - Variable names to train on (default: VAR_2T VAR_10U VAR_10V)
   * - ``--constant_varnames_list``
     - list
     - Constant variable names (static fields) (default: z lsm)
   * - ``--constant_varnames_file``
     - str
     - NetCDF file with constant variables (default: ERA5_const_sfc_variables.nc)
   * - ``--normalization_types``
     - list
     - Normalization per variable as ``var=type`` pairs (e.g., VAR_2T=standard)
   * - ``--units_list``
     - list
     - Units for each variable (default: K m/s m/s)
   * - ``--dynamic_covariates``
     - list
     - List of dynamic covariate names
   * - ``--dynamic_covariates_dir``
     - str
     - Directory for dynamic covariates (default: ../data_covariates/)
   * - **Time Range**
     - -
     -
   * - ``--year_start``
     - int
     - Start year for training dataset (default: 1980)
   * - ``--year_end``
     - int
     - End year for training dataset (default: 2020)
   * - ``--year_start_test``
     - int
     - Start year for test dataset (default: 2020)
   * - ``--year_end_test``
     - int
     - End year for test dataset (default: 2022)
   * - ``--time_normalization``
     - str
     - Time normalization type (e.g., linear, cos_sin) (default: linear)
   * - **Training Configuration**
     - -
     -
   * - ``--num_epochs``
     - int
     - Number of training epochs (default: 100)
   * - ``--batch_size``
     - int
     - Batch size for training (default: 8)
   * - ``--learning_rate``
     - float
     - Learning rate (default: 1e-4)
   * - ``--num_workers``
     - int
     - Number of DataLoader workers (default: 16)
   * - ``--tbatch``
     - int
     - Temporal batch length for processing (default: 1)
   * - ``--sbatch``
     - int
     - Number of spatial batches per timestamp (default: 8)
   * - ``--train_temporal_batch_mode``
     - str
     - Mode: ``full`` (whole sequence) or ``partial`` (batched) (default: partial)
   * - ``--tbatch_train``
     - int
     - Temporal batch length when mode=partial (default: 1)
   * - ``--test_temporal_batch_mode``
     - str
     - Test mode: ``full`` or ``partial`` (default: full)
   * - ``--tbatch_test``
     - int
     - Test temporal batch length (overrides if set)
   * - ``--test_spatial_batch_mode``
     - str
     - Test spatial mode: ``full`` or ``partial`` (default: full)
   * - ``--sbatch_test``
     - int
     - Test spatial batches (overrides if set)
   * - ``--batch_size_lat``
     - int
     - Latitude grid points per spatial batch (must be odd) (default: 145)
   * - ``--batch_size_lon``
     - int
     - Longitude grid points per spatial batch (must be odd) (default: 145)
   * - **Data Processing**
     - -
     -
   * - ``--epsilon``
     - float
     - Epsilon parameter for filtering (default: 0.02)
   * - ``--beta``
     - float
     - Beta parameter for loss function (default: 1.0)
   * - ``--margin``
     - int
     - Margin parameter for filtering (default: 8)
   * - **Output Configuration**
     - -
     -
   * - ``--main_folder``
     - str
     - Main output folder name (default: experiment)
   * - ``--sub_folder``
     - str
     - Sub-folder name for current run (default: experiment)
   * - ``--prefix``
     - str
     - Prefix for saved files (default: run)
   * - ``--dtype``
     - str
     - Precision: ``fp16``, ``fp32``, or ``fp64`` (default: fp32)
   * - **Model Architecture**
     - -
     -
   * - ``--arch``
     - str
     - Architecture: ``ddpmpp``, ``ncsnpp``, or ``adm`` (default: adm)
   * - ``--precond``
     - str
     - Preconditioner: ``vp``, ``ve``, ``edm``, or ``unet`` (default: edm)
   * - ``--in_channels``
     - int
     - Number of input variable channels (default: 3)
   * - ``--cond_channels``
     - int
     - Number of conditioning channels (default: 0)
   * - ``--out_channels``
     - int
     - Number of output channels (default: 3)
   * - **Checkpoint Configuration**
     - -
     -
   * - ``--save_model``
     - bool
     - Enable model checkpoint saving (default: False)
   * - ``--apply_filter``
     - bool
     - Apply fine filtering for coarse data generation (default: False)
   * - ``--save_checkpoint_name``
     - str
     - Name for saved checkpoints (default: diffusion_model_checkpoint)
   * - ``--save_per_samples``
     - int
     - Save checkpoint every N samples (default: 10000)
   * - ``--load_checkpoint_name``
     - str
     - Checkpoint file to load for resume/inference (default: model.pth.tar)
   * - **Regional Inference**
     - -
     -
   * - ``--region_center``
     - float list (2)
     - [latitude, longitude] center for regional inference
   * - ``--region_size``
     - int list (2)
     - [lat_size, lon_size] in grid points for regional inference
   * - **EDM Sampler Configuration**
     - -
     - (for stochastic inference with ``inference_type=sampler``)
   * - ``--num_steps``
     - int
     - Number of sampling steps (default: 20)
   * - ``--sigma_min``
     - float
     - Minimum noise level (default: 0.002)
   * - ``--sigma_max``
     - float
     - Maximum noise level (default: 80.0)
   * - ``--rho``
     - float
     - Exponent for time step discretization (default: 7.0)
   * - ``--s_churn``
     - float
     - Stochasticity strength (default: 40)
   * - ``--s_min``
     - float
     - Minimum noise for stochasticity (default: 0)
   * - ``--s_max``
     - float
     - Maximum noise for stochasticity (default: inf)
   * - ``--s_noise``
     - float
     - Noise scale when stochasticity enabled (default: 1.0)
   * - ``--solver``
     - str
     - ODE solver: ``heun`` or ``euler`` (default: heun)
   * - **CRPS Evaluation**
     - -
     -
   * - ``--compute_crps``
     - bool
     - Compute Continuous Ranked Probability Score (default: False)
   * - ``--crps_ensemble_size``
     - int
     - Ensemble size for CRPS calculation (default: 10)
   * - ``--crps_batch_size``
     - int
     - Batch size for CRPS computation (default: 2)

Here is an example setup script snippet with commonly used parameters:

.. code-block:: bash

   debug=true
   run_type="train"
   region=""
   save_model=true
   save_checkpoint_name="difusion_model"
   load_checkpoint_name="difusion_model"
   save_per_samples=10000

   year_start=2019
   year_end=2019
   year_start_test=2020
   year_end_test=2020

   batch_size=70
   num_epochs=1
   learning_rate=0.0001
   num_workers=8

   datadir="/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily"
   per_var_datadir=(
     "VAR_2T=/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily"
     )

   time_normalization="cos_sin"

   varnames_list=("VAR_2T")
   constant_varnames_list=("z" "lsm")
   constant_varnames_file="ERA5_const_sfc_variables.nc"
   normalization_types=("VAR_2T=standard")
   units_list=("K")
   dynamic_covariates=()
   dynamic_covariates_dir="../data_covariates/"

   sbatch=12
   tbatch=1800
   batch_size_lat=145
   batch_size_lon=361

   epsilon=0.02
   beta=1.0
   margin=8

   pretrained_path=""
   model_name=""

   dtype="fp32"
   arch="adm"
   precond="edm"
   in_channels=1
   cond_channels=5
   out_channels=1
   inference_type="sampler"

   compute_crps=false

   num_steps=10
   sigma_min=0.002
   sigma_max=80.0
   rho=7
   s_churn=40
   solver="heun"

   apply_filter=false

Data Preparation
----------------

IPSL-AID is designed to work with ERA5 reanalysis data:

1. Download ERA5 data (0.25° resolution)
2. Preprocess using the provided scripts
3. Set up train/validation/test splits (typically 2015-2019 train, 2020 validation, 2021 test)

Evaluation Metrics
------------------

The model is evaluated using:

- **Mean Absolute Error (MAE)**: Pointwise accuracy
- **Root Mean Square Error (RMSE)**: Overall deviation
- **Coefficient of Determination (R²)**: Variance explained
- **Continuous Ranked Probability Score (CRPS)**: Probabilistic performance
- **Power Spectral Density (PSD)**: Spatial scale fidelity
- **Probability Density Functions (PDFs)**: Distribution matching

Next Steps
----------

- Read the :doc:`testing_philosophy` before running large experiments
- Explore :doc:`diffusion_models` to understand model choices
- Check the :doc:`api/modules` for detailed module documentation
