Installation
============

IPSL-AID uses **uv** for fast, portable, and  reproducible environment management.

1. Create and activate a virtual environment:

.. code-block:: bash

   uv venv --python=python3.11
   source .venv/bin/activate

2. Install dependencies from ``pyproject.toml``:

.. code-block:: bash

   uv pip install -r pyproject.toml

3. (HPC) Ensure CUDA-compatible PyTorch is installed and visible.

Hardware Requirements
---------------------

- **Training**: Multi-GPU systems (tested for 4× NVIDIA A100 64GB)
- **Inference**: Single GPU or multi-GPU for parallel generation
- **Storage**: Sufficient disk space for climate datasets (ERA5 ~TB scale)
- **Memory**: Adequate RAM for data loading and preprocessing

HPC Configuration
-----------------

On HPC systems, additional configuration may be necessary:

- Load appropriate CUDA modules
- Ensure SLURM is configured for multi-GPU jobs
- Set up shared storage for datasets and model checkpoints

Verification
------------

After installation, verify that the environment is correctly set up:

.. code-block:: bash

   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "from IPSL_AID import main; print('IPSL-AID imported successfully')"
