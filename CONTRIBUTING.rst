Contributing to IPSL-AID
========================

Thank you for your interest in contributing to IPSL-AID! This document provides guidelines for contributing to the project.

We want to make contributing to this project as easy and transparent as possible.

For detailed workflow instructions, refer to the documentation:

User Guide
----------

- `Overview <https://kardaneh.github.io/IPSL-AID/overview.html>`_ – Project overview and context
- `Installation <https://kardaneh.github.io/IPSL-AID/installation.html>`_ – Complete installation guide
- `Quickstart Guide <https://kardaneh.github.io/IPSL-AID/quickstart.html>`_ – Getting started with development
- `Testing Philosophy <https://kardaneh.github.io/IPSL-AID/testing_philosophy.html>`_ – How we test and validate changes (Read This First)
- `Project Structure <https://kardaneh.github.io/IPSL-AID/project_structure.html>`_ – Understanding the codebase organization
- `Pre-Push Workflow <https://kardaneh.github.io/IPSL-AID/pre_push_workflow.html>`_ – Complete step-by-step guide before pushing changes
- `Cartopy Configuration <https://kardaneh.github.io/IPSL-AID/cartopy_configuration.html>`_ – Geospatial visualization setup

Core Concepts
-------------

- `Diffusion Models <https://kardaneh.github.io/IPSL-AID/diffusion_models.html>`_ – Diffusion-specific implementations
- `Neural Architectures <https://kardaneh.github.io/IPSL-AID/neural_architectures.html>`_ – Model architecture guidelines
- `Training Strategy <https://kardaneh.github.io/IPSL-AID/training_strategy.html>`_ – Training configurations and strategies
- `Inference Modes <https://kardaneh.github.io/IPSL-AID/inference_modes.html>`_ – Inference workflows and options

API Reference
-------------

IPSL_AID / IPSL-AID

Pull Requests
-------------

We actively welcome your pull requests. To ensure a smooth contribution process, please follow these steps:

1. **Fork the repository** and create your branch from `main`

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Add your changes** – If you've added code, add the corresponding tests

3. **Update documentation** – If you've changed APIs or added new features, update the documentation accordingly

4. **Run the test suite** – Ensure all tests pass

   .. code-block:: bash

      python -m tests.test_all

5. **Run pre-commit hooks** – Make sure your code passes all pre-commit checks

   .. code-block:: bash

      pre-commit run --all-files

6. **Commit your changes** using `conventional commits <https://www.conventionalcommits.org/>`_ format:

   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation updates
   - `test:` for test additions or changes
   - `refactor:` for code refactoring
   - `perf:` for performance improvements

7. **Push to your fork** and submit a pull request

   .. code-block:: bash

      git push origin feature/your-feature-name

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- Use a descriptive branch name (e.g., `feature/add-attention-unet`, `fix/data-normalization`)
- Keep pull requests focused on a single feature or fix
- Reference any related issues in the pull request description
- Request review from relevant team members
- Respond to feedback and address requested changes promptly

Code Review
~~~~~~~~~~~

All submissions, including submissions by project maintainers, require review. We use GitHub pull requests for this purpose. A reviewer will:

- Check code quality and style
- Verify test coverage
- Ensure documentation is complete
- Test the changes locally if necessary

Merging
~~~~~~~

Pull requests are merged once they:

- Pass all continuous integration checks
- Have at least one approval from a maintainer
- Have no unresolved comments or requested changes

Questions?
~~~~~~~~~~

If you have questions about contributing, please:

- Open a GitHub issue for discussion
- Refer to the `Pre-Push Workflow <https://kardaneh.github.io/IPSL-AID/pre_push_workflow.html>`_ for detailed instructions
- Check the `Quickstart Guide <https://kardaneh.github.io/IPSL-AID/quickstart.html>`_ for setup help

---

Thank you for contributing to IPSL-AID!
