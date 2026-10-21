Cartopy Configuration
=====================

For geospatial diagnostics, IPSL-AID uses `Cartopy <https://scitools.org.uk/cartopy/>`_.

On HPC systems, a shared data directory must be defined to avoid repeated downloads:

.. code-block:: python

   import cartopy
   cartopy.config['data_dir'] = "/path/to/shared/cartopy_data"

Installation
------------

Cartopy can be installed via pip or conda:

.. code-block:: bash

   # Using pip
   pip install cartopy

   # Using conda
   conda install -c conda-forge cartopy

HPC Considerations
------------------

On shared HPC systems:

1. **Set shared data directory**: Avoids duplicate downloads across users
2. **Pre-download data**: Download Natural Earth and other datasets once
3. **Set appropriate permissions**: Ensure all users can read the data

Example setup script:

.. code-block:: bash

   # Download Cartopy data to shared location
   mkdir -p /shared/cartopy_data
   python -c "import cartopy; cartopy.config['data_dir'] = '/shared/cartopy_data'"
   python -c "import cartopy.feature as cfeature; print('Testing Cartopy...')"

Common Issues and Solutions
---------------------------

1. **Missing Natural Earth data**:
   - Download manually from: https://www.naturalearthdata.com/downloads/
   - Place in Cartopy data directory

2. **Permission errors**:
   - Ensure shared directory has read permissions for all users
   - Use `chmod -R 755 /shared/cartopy_data`

3. **Proxy/firewall issues**:
   - Set HTTP_PROXY/HTTPS_PROXY environment variables
   - Use offline installation with pre-downloaded data

Usage in Diagnostics
--------------------

Cartopy is used for:

- **Spatial error maps**: MAE, RMSE across regions
- **Geographical overlays**: Coastlines, country borders
- **Projections**: Various map projections for visualization
- **Regional focus**: Zooming into specific regions of interest

Example visualization code:

.. code-block:: python

   from IPSL_AID.diagnostics import plot_MAE_map

   expected_path = plot_MAE_map(
      predictions=predictions,
      targets=targets,
      lat_1d=lat_1d,
      lon_1d=lon_1d,
      variable_names=variable_names,
      save_dir=output_dir,
      filename="validation_mae_map_standard.png"
      )

Performance Tips
----------------

- **Cache projections**: Reuse projection objects
- **Batch plotting**: Minimize figure creation overhead
- **Reduce resolution**: Use lower-resolution coastlines for faster rendering
- **Parallel rendering**: Use multiple processes for multiple plots
