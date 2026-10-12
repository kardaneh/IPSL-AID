Testing Philosophy (Read This First)
====================================

Each module in IPSL-AID comes with its **own dedicated tests**.

The repository currently includes test suites for:

- ``IPSL_AID``, covering diffusion-based climate downscaling;
- ``AID_BC``, covering bias correction using Quantile Mapping and Optimal Transport.

Before running large-scale training or bias-correction experiments, users are **strongly encouraged** to:

1. Explore the test suite
2. Understand data handling and normalization
3. Validate diffusion and architecture choices
4. Tune hyperparameters using small debug runs

This approach minimizes wasted compute time and ensures correct scientific usage.

Why Testing is Essential
------------------------

Climate downscaling and bias-correction workflows involve:

- Complex neural architectures
- Sophisticated diffusion formulations
- Large-scale data processing
- Multi-GPU parallelization
- Numerical precision requirements
- Statistical bias-correction methods

Without proper testing, errors can:

1. **Waste computational resources** (days of GPU time)
2. **Produce scientifically invalid results**
3. **Mask underlying implementation bugs**
4. **Compromise reproducibility**

Recommended Workflow
--------------------

1. **Run the complete test suite**

   The global test runner executes both the ``IPSL_AID`` and ``AID_BC`` test
   suites:

   .. code-block:: bash

      python -m tests.test_all

2. **Run one test suite independently**

   .. code-block:: bash

      # IPSL_AID
      python -m tests.IPSL_AID.test_runner

      # AID_BC
      python -m tests.AID_BC.test_runner

3. **Run targeted tests**

   Individual modules, classes, or test methods can be executed directly.

   .. code-block:: bash

      # IPSL_AID module
      python -m unittest tests.IPSL_AID.test_utils

      # IPSL_AID class
      python -m unittest tests.IPSL_AID.test_utils.TestEasyDict

      # IPSL_AID single test
      python -m unittest \
        tests.IPSL_AID.test_utils.TestEasyDict.test_empty_initialization

      # AID_BC module
      python -m unittest tests.AID_BC.test_quantile_mapping

4. **Integration Tests**: Test data loading and model initialization

   .. code-block:: bash

      python tests/IPSL_AID/test_integration.py

5. **Small-scale Debug Runs**: Train on a small subset

   .. code-block:: bash

      # Use debug mode in setup script
      debug=true
      num_epochs=1
      batch_size=4

6. **Validation**: Check against known baselines

Test Coverage
-------------

The ``IPSL_AID`` test suite covers:

- Data loading and preprocessing
- Diffusion model implementations (VE, VP, EDM, iDDPM)
- UNet architectures and variants
- Loss functions and training loops
- Inference and sampling procedures
- Evaluation metrics and diagnostics

The ``AID_BC`` test suite covers:

- ERA5 and CMIP6 data loading
- ERA5-on-CMIP6 preprocessing
- Quantile Mapping
- Optimal Transport
- Bias-correction workflows
- Evaluation metrics and diagnostics

Debugging Tips
--------------

- Use `torch.autograd.set_detect_anomaly(True)` during development
- Monitor GPU memory usage with `nvidia-smi`
- Check numerical stability with gradient norms
- Validate data normalization and scaling

Remember: **Test first, scale later**. A few hours of testing can save days of wasted computation.
