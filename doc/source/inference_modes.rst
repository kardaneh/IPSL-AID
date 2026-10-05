Inference Modes
===============

The trained model can be used in multiple inference modes:

- **Global inference** (full spatial coverage)
- **Regional inference** (single region or subdomain)
- **Direct prediction**
- **Sampler-based diffusion inference**

This flexibility makes IPSL-AID suitable for both research experiments and
downstream climate impact studies.

Global Inference
----------------

Global inference enables the generation of high-resolution predictions over
the entire globe using a block-based approach.

For global coverage, the fine-resolution reference domain is divided into
spatial blocks, which are processed independently by the trained model.
The fine-resolution reference is typically ERA5, while the coarse conditioning
input may either be derived from the fine field or provided externally, for
example from CMIP6 or HighResMIP.

To reduce discontinuities at block boundaries, IPSL-AID supports spatial
overlap between adjacent blocks. Overlapping predictions are combined using
smooth weighted blending to ensure spatial continuity across the global domain.

The global inference procedure consists of three main steps:

1. **Tiling**: Divide the global domain into spatial blocks with a configurable overlap.
2. **Processing**: Run inference independently on each block.
3. **Merging**: Reconstruct the global field using weighted blending in overlapping regions.


Spatial Overlap
^^^^^^^^^^^^^^^

The ``overlap_ratio`` parameter controls the spatial overlap between adjacent
blocks. A value of ``0.0`` disables overlap, while ``0.02`` corresponds to
a 2% overlap.

Overlapping predictions are merged using Hann-based weighting functions
to reduce discontinuities at block boundaries.

The default overlap ratio is ``0.02`` (2%). Increasing this value can
improve spatial continuity but may increase computational cost.

Configuration:

.. code-block:: yaml

   inference:
     run_type: inference
     overlap_ratio: 0.02

External Coarse Inputs
----------------------

IPSL-AID can also perform inference using an external coarse-resolution dataset,
such as CMIP6 or HighResMIP, instead of deriving the coarse input directly from
the fine-resolution reference field.

Per-variable input paths are configured with ``--per_var_datadir`` using:

- ``VAR.fine=PATH``: path to the fine-resolution reference dataset.
- ``VAR.coarse=PATH``: optional path to an external coarse-resolution dataset.

If no external coarse dataset is provided, the coarse input is derived from the
fine-resolution field using the standard downscale-upscale procedure.

The ``--already_coarse`` option controls how the external coarse field is
processed:

- ``--already_coarse false``:
  the coarse source is first downscaled to the configured coarse shape and then
  upscaled to the fine-grid resolution.
- ``--already_coarse true``:
  the downscaling step is skipped and the external coarse field is directly
  upscaled to the fine-grid resolution.

This allows several inference configurations.

ERA5-derived coarse input
^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard configuration uses ERA5 as the fine-resolution reference and
derives the coarse input internally:

.. code-block:: bash

   --per_var_datadir VAR.fine=ERA5_PATH
   --already_coarse false

CMIP6 external coarse input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMIP6 can be used as an external native coarse-resolution input while ERA5
remains the fine-resolution reference:

.. code-block:: bash

   --per_var_datadir VAR.fine=ERA5_PATH VAR.coarse=CMIP6_PATH
   --already_coarse true

In this case, the CMIP6 field is not downscaled again. It is directly upscaled
to the ERA5 fine-grid resolution before being used as the coarse conditioning
input.

HighResMIP external coarse input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HighResMIP data can also be provided as an external coarse source:

.. code-block:: bash

   --per_var_datadir VAR.fine=ERA5_PATH VAR.coarse=HiMIP_PATH
   --already_coarse false

The HighResMIP field is first downscaled to the configured coarse shape and
then upscaled to the fine-grid resolution before inference.

HighResMIP as fine-resolution input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HighResMIP may also be used as the fine-resolution reference dataset without an
external coarse source:

.. code-block:: bash

   --per_var_datadir VAR.fine=HiMIP_PATH
   --already_coarse false

In this configuration, the coarse input is derived directly from the HighResMIP
fine-resolution field.

Regional Inference
------------------

Regional inference allows running the model on a *specific geographic subset*
of the global domain instead of processing the entire globe.

This mode is particularly useful for regional studies (e.g., Europe,
North America, Southeast Asia), where only a limited area is of interest.

Conceptually, the model operates on a **spatial window** extracted from the
global grid, centered on a given location with a fixed spatial extent.

Configuration:

.. code-block:: yaml

   inference:
     run_type: inference_regional

     # Option 1: predefined region
     region: "europe"

     # Option 2: custom region (lat, lon)
     region_center: 50.0 10.0

     # Region size (lat_size, lon_size)
     region_size: 144 360

     # Supported sizes:
     # lat can be 144 or 288
     # lon can be 360 or 720

Region selection:

Two approaches are available:

- **Predefined region**:
  Use a named region (e.g., ``"us"``, ``"europe"``, ``"asia"``). The corresponding spatial
  boundaries are internally defined.

- **Custom region**:
  Specify a center point using latitude and longitude (``region_center``).
  The model will extract a region centered around this location.

Either ``region`` or ``region_center`` must be provided.

Sampling Procedure
------------------

High-resolution samples are generated by numerically solving the reverse-time
SDE. The sampler uses the second-order Heun scheme for improved accuracy:

Algorithm:
^^^^^^^^^^

1. **Initialize**: :math:`\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, t_0^2 \mathbf{1})`
2. **For each step** :math:`i = 0, \dots, N-1`:
   a. Optionally add noise increment :math:`\gamma_{\mathrm{i}}`
   b. Compute denoising direction :math:`\mathbf{d}_{\mathrm{i}}`
   c. Update latent: :math:`\mathbf{x}_{\mathrm{i+1}} = \hat{\mathbf{x}}_{\mathrm{i}} + (t_{\mathrm{i+1}} - \hat{t}_{\mathrm{i}}) \, \mathbf{d}_{\mathrm{i}}`
   d. Apply 2nd-order correction if :math:`t_{\mathrm{i+1}} \neq 0`
3. **Return**: Final denoised sample :math:`\mathbf{x}_{\rm N}`

Sampling Parameters
-------------------

.. code-block:: yaml

   sampling:
     steps: 20                    # Number of sampling steps
     sigma_min: 0.002             # Minimum noise level
     sigma_max: 80.0              # Maximum noise level
     rho: 7.0                     # Controls step distribution
     sampler: "heun"              # Integration scheme
     s_churn: 40.0                # Stochasticity parameter
     s_min: 0.05                  # Minimum stochasticity
     s_max: 50.0                  # Maximum stochasticity
     s_noise: 1.003               # Noise scale

Ensemble Generation
-------------------

For uncertainty quantification, generate multiple samples:

1. **Multiple seeds**: Different random seeds for sampling
2. **Parameter variations**: Different sampler settings
3. **Model ensembles**: Average predictions from multiple checkpoints
4. **Statistical analysis**: Compute means, variances, quantiles

Postprocessing
--------------

After generation:

1. **Add residuals**: :math:`\mathbf{y}^{\mathrm{HR}} = \mathbf{y}^{\mathrm{CU}} + \mathbf{R}'`
2. **Denormalize**: Convert from normalized to physical units
3. **Quality checks**: Validate physical constraints
4. **Format conversion**: Save in standard formats (NetCDF, GeoTIFF)

Performance Optimization
------------------------

- **Batch inference**: Process multiple time steps simultaneously
- **Memory management**: Clear intermediate results
- **GPU utilization**: Maximize GPU occupancy
- **I/O optimization**: Efficient reading/writing of large files

Real-time Applications
----------------------

For near real-time downscaling:

1. **Streaming input**: Ingest coarse forecasts
2. **Fast inference**: Optimized sampler settings
3. **Caching**: Reuse computations where possible
4. **Parallelization**: Distribute across multiple GPUs/nodes

Validation and Evaluation
-------------------------

During inference, compute:

1. **Deterministic metrics**: MAE, RMSE, R² against observations
2. **Probabilistic metrics**: CRPS, spread-skill ratio
3. **Spatial statistics**: Power spectra, variograms
4. **Extreme values**: Quantile scores, tail statistics

Example Usage
-------------

.. code-block:: python

   from IPSL_AID.evaluater import run_validation

   # Load trained model
   model = load_model("checkpoints/corresponding_expriment/best_model.pth")

   # Run global inference
   avg_val_loss, val_metrics = run_validation(
      model,
      valid_dataset,
      valid_loader,
      loss_fn,
      norm_mapping,
      normalization_type,
      index_mapping,
      args,
      steps,
      device,
      logger,
      epoch=0,
      writer=writer,
      plot_every_n_epochs=1,
      edm_sampler_steps=20,
      paths=paths,
      compute_crps=True
      )

   # Check results
   "results/corresponding_expriment/*.png"
