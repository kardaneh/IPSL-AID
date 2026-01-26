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

.. code-block:: bash

   # Example setup_diffusion.sh snippet
   precond="EDM"
   arch="ADM_UNet"
   varnames_list=("VAR_2T" "VAR_10U" "VAR_10V" "VAR_TP")
   normalization_types=("VAR_2T=standard" "VAR_10U=standard" "VAR_10V=standard" "VAR_TP=standard")
   batch_size=32
   run_type="train"

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
