Installation
============

IPSL-AID uses **uv** for fast, portable, and reproducible environment management.

Prerequisites
-------------

- Python 3.8 or higher
- CUDA-compatible GPU (for training and inference)
- Git (for cloning the repository)

Quick Install
-------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/kardaneh/IPSL-AID.git
   cd IPSL-AID

2. Create and activate a virtual environment:

.. code-block:: bash

   uv venv --python=python3.11
   source .venv/bin/activate

3. Install the package in development mode:

.. code-block:: bash

   uv pip install -e .

This will install all dependencies defined in ``pyproject.toml`` and make the
``ipsl-aid`` command available in your environment.

Verification
------------

After installation, verify that the package is correctly installed:

.. code-block:: bash

   # Check version (may take a few minutes on first run - loading dependencies)
   ipsl-aid --version

   # Check help (may also take a few minutes on first run)
   ipsl-aid --help

   # Faster alternative for version check
   python -c "import IPSL_AID; print(IPSL_AID.__version__)"

   # Verify CUDA availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   if torch.cuda.is_available():
       print(f'CUDA version: {torch.version.cuda}')
       print(f'GPU count: {torch.cuda.device_count()}')

   # Test import
   python -c "import IPSL_AID; print('IPSL-AID imported successfully')"

.. note::
   The first execution of ``ipsl-aid --version`` or ``ipsl-aid --help`` loads all dependencies
   (PyTorch, NumPy, etc.), which may take 1-2 minutes. We are actively working on a fix to
   make these commands instantaneous. Subsequent calls will be faster due to caching.

HPC Configuration
-----------------

On HPC systems (like Leonardo), additional configuration may be necessary:

1. **Load CUDA modules** (if not automatically loaded):

.. code-block:: bash

   module load cuda/12.1

2. **Set up Cartopy data directory** (to avoid downloading during runtime):

.. code-block:: bash

   export CARTOPY_DATA_DIR="/path/to/cartopy_data"

3. **Configure environment variables** for data paths:

.. code-block:: bash

   export IPSL_AID_DATA_DIR="/path/to/your/data"
   export IPSL_AID_OUTPUT_DIR="/path/to/your/outputs"

4. **SLURM job submission**:

The package includes a bash script generator that creates SLURM submission
scripts. Example usage:

.. code-block:: bash

   # Generate SBATCH script with your configuration
   ./your_setup_script.sh

   # Submit the job
   sbatch slurm/sbatch_diffusion_*.sh

Hardware Requirements
---------------------

- **Training**: Multi-GPU systems (tested with 4× NVIDIA A100 64GB)
- **Inference**: Single GPU or multi-GPU for parallel generation
- **Storage**: Sufficient disk space for climate datasets (ERA5 ~TB scale)
- **Memory**: Adequate RAM for data loading and preprocessing

Dependencies
------------

IPSL-AID requires the following key dependencies (automatically installed):

- PyTorch >= 2.5.1
- NumPy, SciPy, Pandas, Xarray
- Matplotlib, Cartopy, Seaborn
- tqdm, Rich, TensorBoard
- CDS API client (for data download)

Optional Dependencies
---------------------

For development and documentation:

.. code-block:: bash

   # Install development dependencies
   uv pip install -e ".[dev]"

   # Install documentation dependencies
   uv pip install -e ".[docs]"

Development Installation
------------------------

For developers working on the codebase:

1. Install in editable mode with development dependencies:

.. code-block:: bash

   uv pip install -e ".[dev]"

2. Set up pre-commit hooks:

.. code-block:: bash

   pre-commit install
   pre-commit run --all-files

3. Run tests:

.. code-block:: bash

   python tests/test_all.py

Troubleshooting
---------------

**Issue**: ``ipsl-aid: command not found``

**Solution**: Ensure the virtual environment is activated:
``source .venv/bin/activate``

**Issue**: CUDA not available

**Solution**: Check PyTorch installation:
``python -c "import torch; print(torch.cuda.is_available())"``
If False, reinstall PyTorch with CUDA support.

**Issue**: Import errors

**Solution**: Verify package installation:
``uv pip list | grep ipsl``
If not found, reinstall: ``uv pip install -e .``

**Issue**: Slow startup with ``--version`` or ``--help``

**Solution**: This is expected on first run as PyTorch and other heavy modules load.
We are actively working on implementing a fast CLI handler. For quick verification,
use the alternative commands shown in the Verification section.

**Issue**: ``ModuleNotFoundError: No module named 'IPSL_AID'``

**Solution**: Ensure the package is installed and the virtual environment is activated.
Run: ``uv pip install -e .`` from the project root directory.

**Issue**: Cartopy data download fails

**Solution**: Set the Cartopy data directory to a shared location:
``export CARTOPY_DATA_DIR="/path/to/cartopy_data"``

Getting Help
------------

- **Repository**: https://github.com/kardaneh/IPSL-AID
- **Issues**: https://github.com/kardaneh/IPSL-AID/issues
- **Contact**: kardaneh@ipsl.fr
