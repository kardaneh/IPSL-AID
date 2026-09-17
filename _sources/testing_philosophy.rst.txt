Testing Philosophy (Read This First)
====================================

Each module in IPSL-AID comes with its **own dedicated tests**.

Before running large-scale training, users are **strongly encouraged** to:

1. Explore the test suite
2. Understand data handling and normalization
3. Validate diffusion and architecture choices
4. Tune hyperparameters using small debug runs

This approach minimizes wasted compute time and ensures correct scientific usage.

Why Testing is Essential
------------------------

Climate downscaling models involve:

- Complex neural architectures
- Sophisticated diffusion formulations
- Large-scale data processing
- Multi-GPU parallelization
- Numerical precision requirements

Without proper testing, errors can:

1. **Waste computational resources** (days of GPU time)
2. **Produce scientifically invalid results**
3. **Mask underlying implementation bugs**
4. **Compromise reproducibility**

Recommended Workflow
--------------------

1. **Unit Tests**: Run all tests

   .. code-block:: bash

      python -m tests.test_all

2. **Targeted Tests**: Run specific modules, classes, or test methods

   .. code-block:: bash

      # Module
      python -m unittest tests.test_utils
      python -m unittest tests.test_model_utils

      # Class
      python -m unittest tests.test_utils.TestEasyDict

      # Single test
      python -m unittest tests.test_utils.TestEasyDict.test_empty_initialization

3. **Integration Tests**: Test data loading and model initialization

   .. code-block:: bash

      python tests/test_integration.py

4. **Small-scale Debug Runs**: Train on a small subset

   .. code-block:: bash

      # Use debug mode in setup script
      debug=true
      num_epochs=1
      batch_size=4

5. **Validation**: Check against known baselines

Test Coverage
-------------

IPSL-AID includes tests for:

- Data loading and preprocessing
- Diffusion model implementations (VE, VP, EDM, iDDPM)
- UNet architectures and variants
- Loss functions and training loops
- Inference and sampling procedures
- Evaluation metrics and diagnostics

Debugging Tips
--------------

- Use `torch.autograd.set_detect_anomaly(True)` during development
- Monitor GPU memory usage with `nvidia-smi`
- Check numerical stability with gradient norms
- Validate data normalization and scaling

Remember: **Test first, scale later**. A few hours of testing can save days of wasted computation.
