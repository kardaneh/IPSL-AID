IPSL-AID
========

**IPSL-AID** is a high-performance research framework for **climate data downscaling**
based on **diffusion models**, and **UNet-based neural networks**, designed for **GPU clusters and HPC systems**.

The framework supports **global training**, **regional or global inference**, and
multiple **diffusion formulations** and **UNet-based architectures**, with a strong
emphasis on **reproducibility, comprehensive testing, and configurability**.

.. admonition:: Development Status
   :class: important

   **IPSL-AID is in active development.**

   This framework is currently undergoing rapid development and has not yet
   reached a stable release. Please be aware that:

   * APIs and interfaces may change without notice
   * Feature names and module structures are subject to modification
   * Core architectural decisions may be revised

   **Recommendations for users:**

   - Regularly update from the main branch
   - Rebase local branches frequently to avoid conflicts
   - Submit pull requests for broadly useful enhancements
   - Report issues for unexpected behavior or bugs.

----

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   overview
   installation
   quickstart
   testing_philosophy
   project_structure
   cartopy_configuration

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   diffusion_models
   neural_architectures
   training_strategy
   inference_modes

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules
   api/IPSL_AID

----

Getting Started
---------------

* :doc:`overview` - Framework capabilities and key features
* :doc:`installation` - Setup instructions using uv
* :doc:`quickstart` - Basic workflow and example usage
* :doc:`testing_philosophy` - Essential reading before running experiments

Core Components
---------------

* :doc:`diffusion_models` - VE, VP, EDM, iDDPM implementations
* :doc:`neural_architectures` - UNet-based architectures for climate data
* :doc:`training_strategy` - Global training with random block strategy
* :doc:`inference_modes` - Global, regional, and sampler-based inference

Project Information
-------------------

* :doc:`project_structure` - Codebase organization
* :doc:`cartopy_configuration` - Geospatial visualization setup

API Documentation
-----------------

* :doc:`api/modules` - Complete module reference
* :doc:`api/IPSL_AID` - Package-level documentation

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
