# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import xarray as xr

from AID_BC.bias_corrector import (
    create_bias_corrector,
)

from AID_BC.logger import Logger
from AID_BC.evaluater import evaluate
from AID_BC.dataset import ClimateDataset


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Bias correction on the native CMIP6 grid. "
            "QM accepts exactly one climate variable; "
            "OT accepts one or several variables jointly."
        )
    )

    parser.add_argument(
        "--method",
        choices=("qm", "ot"),
        default="qm",
        help="Bias-correction method.",
    )

    parser.add_argument(
        "--split",
        choices=("none", "month"),
        default="none",
        help=(
            "Fit one global corrector ('none') or one corrector per "
            "calendar month ('month'). Monthly splitting strongly reduces "
            "the O(N^2) Sinkhorn cost."
        ),
    )

    parser.add_argument(
        "--train_start",
        type=int,
        required=True,
        help="First training year.",
    )

    parser.add_argument(
        "--train_end",
        type=int,
        required=True,
        help="Last training year.",
    )

    parser.add_argument(
        "--apply_start",
        type=int,
        required=True,
        help="First application year.",
    )

    parser.add_argument(
        "--apply_end",
        type=int,
        required=True,
        help="Last application year.",
    )

    # One argument only:
    #   QM: --variables VAR_2T
    #   OT: --variables VAR_2T VAR_10U VAR_10V
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["VAR_2T"],
        help=(
            "Climate variable names. QM requires exactly one variable. "
            "OT accepts one or several variables jointly."
        ),
    )

    parser.add_argument(
        "--era5_root",
        nargs="+",
        default=None,
        metavar="VARIABLE=PATH",
        help=(
            "Raw ERA5 roots written as VARIABLE=PATH. Provide this or "
            "--era5_on_cmip6_root, but not both."
        ),
    )

    parser.add_argument(
        "--era5_on_cmip6_root",
        nargs="+",
        default=None,
        metavar="VARIABLE=PATH",
        help=("Precomputed ERA5-on-CMIP6 roots written as VARIABLE=PATH."),
    )

    parser.add_argument(
        "--cmip6_train_root",
        nargs="+",
        required=True,
        metavar="VARIABLE=PATH",
        help=("Historical CMIP6 roots written as VARIABLE=PATH."),
    )

    parser.add_argument(
        "--cmip6_apply_root",
        nargs="+",
        required=True,
        metavar="VARIABLE=PATH",
        help=("CMIP6 application roots written as VARIABLE=PATH."),
    )

    parser.add_argument(
        "--output_dir",
        nargs="+",
        required=True,
        metavar="VARIABLE=PATH",
        help=(
            "Output directories written as VARIABLE=PATH. Each corrected "
            "variable is saved separately."
        ),
    )

    # Optimal Transport arguments
    parser.add_argument(
        "--ot_epsilon",
        type=float,
        default=0.1,
        help="Sinkhorn entropic regularization.",
    )

    parser.add_argument(
        "--ot_num_iterations",
        type=int,
        default=500,
        help="Maximum number of Sinkhorn iterations.",
    )

    parser.add_argument(
        "--ot_threshold",
        type=float,
        default=1e-3,
        help="Sinkhorn convergence threshold.",
    )

    parser.add_argument(
        "--ot_batch_size",
        type=int,
        default=32,
        help="Number of time steps transported together during application.",
    )

    parser.add_argument(
        "--ot_dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Floating-point dtype used by JAX.",
    )

    parser.add_argument(
        "--ot_no_normalization",
        action="store_true",
        help="Disable shared per-feature normalization for OT.",
    )

    return parser.parse_args()


def parse_variable_paths(items, argument_name):
    """
    Parse variable-specific paths from command-line values.

    Parameters
    ----------
    items : sequence of str or None
        Values written as VARIABLE=PATH. If None, no mapping is created.
    argument_name : str
        Name of the command-line argument, used in validation messages.

    Returns
    -------
    dict of str to str or None
        Mapping from variable names to directory paths, or None when items
        is None.

    Raises
    ------
    ValueError
        If an item does not use the VARIABLE=PATH format, contains an empty
        variable name or path, or defines the same variable more than once.
    """
    if items is None:
        return None

    paths = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"{argument_name}: expected VARIABLE=PATH, got '{item}'.")

        variable_name, path = item.split("=", 1)
        variable_name = variable_name.strip()
        path = path.strip()

        if not variable_name:
            raise ValueError(f"{argument_name}: empty variable name in '{item}'.")

        if not path:
            raise ValueError(f"{argument_name}: empty path for '{variable_name}'.")

        if variable_name in paths:
            raise ValueError(f"{argument_name}: duplicate variable '{variable_name}'.")

        paths[variable_name] = path

    return paths


def validate_variable_paths(variable_names, paths, argument_name):
    """
    Validate that paths are defined for exactly the requested variables.

    Parameters
    ----------
    variable_names : sequence of str
        Climate variables requested for the correction.
    paths : dict of str to str or None
        Mapping from variable names to directory paths.
    argument_name : str
        Name of the command-line argument, used in validation messages.

    Raises
    ------
    ValueError
        If the mapping is missing, if a requested variable has no path, or if a
        path is provided for an unrequested variable.
    """
    if paths is None:
        raise ValueError(f"{argument_name} was not provided.")

    expected = set(variable_names)
    provided = set(paths)

    missing = expected - provided
    unexpected = provided - expected

    if missing:
        raise ValueError(f"{argument_name}: missing paths for {sorted(missing)}.")

    if unexpected:
        raise ValueError(f"{argument_name}: unexpected variables {sorted(unexpected)}.")


def validate_args(args):
    """
    Validate general command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Raises
    ------
    ValueError
        If year ranges are invalid, variables are missing or duplicated, QM is
        requested with more than one variable, OT parameters are non-positive,
        or both/neither ERA5 input modes are supplied.
    """
    if args.train_start > args.train_end:
        raise ValueError(
            f"train_start must be <= train_end, got "
            f"{args.train_start} > {args.train_end}."
        )

    if args.apply_start > args.apply_end:
        raise ValueError(
            f"apply_start must be <= apply_end, got "
            f"{args.apply_start} > {args.apply_end}."
        )

    if not args.variables:
        raise ValueError("--variables must contain at least one variable.")

    if len(set(args.variables)) != len(args.variables):
        raise ValueError(f"--variables contains duplicates: {args.variables}")

    if args.method == "qm" and len(args.variables) != 1:
        raise ValueError(
            "QM is univariate and therefore requires exactly one variable, "
            f"got {len(args.variables)}: {args.variables}"
        )

    if args.ot_epsilon <= 0:
        raise ValueError(f"ot_epsilon must be positive, got {args.ot_epsilon}.")

    if args.ot_num_iterations <= 0:
        raise ValueError("ot_num_iterations must be positive.")

    if args.ot_threshold <= 0:
        raise ValueError("ot_threshold must be positive.")

    if args.ot_batch_size <= 0:
        raise ValueError("ot_batch_size must be positive.")

    if (args.era5_root is None) == (args.era5_on_cmip6_root is None):
        raise ValueError(
            "Provide exactly one of --era5_root or " "--era5_on_cmip6_root."
        )


def build_path(root, year):
    """
    Build the yearly NetCDF path for a data directory.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing yearly NetCDF files.
    year : int
        Year used in the samples_<year>.nc filename.

    Returns
    -------
    str
        Full path to the yearly NetCDF file.
    """
    # All input and output yearly files follow the same naming convention
    return str(Path(root) / f"samples_{year}.nc")


def check_same_spatial_grid(reference, target):
    """
    Check that two arrays use the same latitude-longitude grid.

    Parameters
    ----------
    reference : xarray.DataArray
        Reference array defining the expected spatial grid.
    target : xarray.DataArray
        Array whose spatial grid is compared with reference.

    Raises
    ------
    ValueError
        If latitude or longitude dimensions are missing, dimension sizes differ,
        or coordinate values are not equal within numerical tolerance.
    """
    required_dimensions = ("latitude", "longitude")

    for dimension in required_dimensions:
        if dimension not in reference.dims:
            raise ValueError(f"Reference data have no {dimension} dimension.")

        if dimension not in target.dims:
            raise ValueError(f"Target data have no {dimension} dimension.")

        if reference.sizes[dimension] != target.sizes[dimension]:
            raise ValueError(
                f"{dimension} size mismatch: "
                f"{reference.sizes[dimension]} != "
                f"{target.sizes[dimension]}."
            )

        if not np.allclose(
            reference[dimension].values,
            target[dimension].values,
        ):
            raise ValueError(f"{dimension} coordinates differ.")


def check_dataset_consistency(dataset, variable_names, label):
    """
    Validate dimensions, coordinates, and values across dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing the climate variables to validate.
    variable_names : sequence of str
        Variable names expected in dataset.
    label : str
        Readable dataset label used in error messages.

    Raises
    ------
    ValueError
        If no variables are provided, a variable is missing, required dimensions
        are absent, spatial or temporal coordinates differ, or non-finite values
        are present.
    """
    if not variable_names:
        raise ValueError("No variable name was provided.")

    first = dataset[variable_names[0]]

    for variable_name in variable_names:
        if variable_name not in dataset:
            raise ValueError(f"Variable '{variable_name}' is missing from {label}.")

        current = dataset[variable_name]

        required_dims = {"time", "latitude", "longitude"}
        if not required_dims.issubset(current.dims):
            raise ValueError(
                f"{label}/{variable_name} must contain dimensions "
                f"{sorted(required_dims)}, got {current.dims}."
            )

        check_same_spatial_grid(first, current)

        if not np.array_equal(
            first["time"].values,
            current["time"].values,
        ):
            raise ValueError(
                f"Time coordinates differ between "
                f"'{variable_names[0]}' and '{variable_name}' in {label}."
            )

        if not np.isfinite(current.values).all():
            raise ValueError(
                f"{label}/{variable_name} contains NaN or infinite values."
            )


def select_month(data, month):
    """
    Select all time steps belonging to one calendar month.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Input climate data with a datetime-like time coordinate.
    month : int
        Calendar month in the inclusive range 1 to 12.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Subset containing all selected-month samples across available years.
    """
    mask = (data["time"].dt.month == month).values
    return data.isel(time=mask)


def prepare_dataarray(data):
    """
    Convert and standardize a climate data array for downstream processing.

    The array is converted to float32, loaded into memory, and reordered to
    use the canonical dimension order time, latitude, longitude.

    Parameters
    ----------
    data : xarray.DataArray
        Climate data array to prepare.

    Returns
    -------
    xarray.DataArray
        Prepared array with float32 values loaded into memory and dimensions
        ordered as time, latitude, longitude.
    """
    return (
        data.astype(np.float32)
        .load()
        .transpose(
            "time",
            "latitude",
            "longitude",
        )
    )


def resolve_precomputed_variable_root(
    era5_on_cmip6_root,
    variable_name,
):
    """
    Resolve the precomputed ERA5 directory for one variable.

    The function supports either a common directory containing metadata.json
    and yearly files or a parent directory containing one subdirectory per
    variable.

    Parameters
    ----------
    era5_on_cmip6_root : str or pathlib.Path
        Common root or parent directory of precomputed ERA5-on-CMIP6 data.
    variable_name : str
        Climate variable whose directory is required.

    Returns
    -------
    pathlib.Path
        Directory containing the variable metadata and yearly files.
    """
    root = Path(era5_on_cmip6_root)
    variable_root = root / variable_name

    if (variable_root / "metadata.json").exists():
        return variable_root

    return root


def load_precomputed_metadata(
    era5_on_cmip6_root,
    variable_name,
    logger,
):
    """
    Load and validate metadata for precomputed ERA5-on-CMIP6 data.

    Parameters
    ----------
    era5_on_cmip6_root : str or pathlib.Path
        Common root or parent directory of precomputed ERA5 data.
    variable_name : str
        Requested climate variable.
    logger : Logger
        Logger used to report the loaded metadata.

    Returns
    -------
    metadata : dict
        Metadata read from metadata.json.
    variable_root : pathlib.Path
        Directory containing the metadata and yearly variable files.

    Raises
    ------
    FileNotFoundError
        If metadata.json cannot be found.
    ValueError
        If the metadata variable name conflicts with variable_name.
    """
    variable_root = resolve_precomputed_variable_root(
        era5_on_cmip6_root,
        variable_name,
    )

    metadata_path = variable_root / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found for variable '{variable_name}' "
            f"under {era5_on_cmip6_root}."
        )

    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)

    metadata_variable = metadata.get("variable_name")

    if metadata_variable is not None and metadata_variable != variable_name:
        raise ValueError(
            f"Precomputed metadata at {metadata_path} were built for "
            f"'{metadata_variable}', but '{variable_name}' was requested."
        )

    logger.info(
        f"Loaded precompute metadata for {variable_name} "
        f"from {metadata_path}:\n" + json.dumps(metadata, indent=2)
    )

    return metadata, variable_root


def open_dataarray(path, variable_name):
    """
    Open one variable from a NetCDF file and load it into memory.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NetCDF file.
    variable_name : str
        Variable to extract from the dataset.

    Returns
    -------
    xarray.DataArray
        Requested variable loaded into memory.

    Raises
    ------
    ValueError
        If variable_name is not present in the file.
    """
    with xr.open_dataset(path) as dataset:
        if variable_name not in dataset:
            raise ValueError(f"Variable '{variable_name}' not found in {path}.")

        data = dataset[variable_name].load()

    return data


def load_training_variable(
    train_start,
    train_end,
    cmip6_train_root,
    variable_name,
    logger,
    era5_root=None,
    era5_on_cmip6_root=None,
):
    """
    Load one training variable from ERA5 and historical CMIP6 files.

    ERA5 data can either be loaded from raw files and prepared onto the
    CMIP6 grid, or loaded from previously prepared ERA5-on-CMIP6 files.

    Parameters
    ----------
    train_start : int
        First training year.
    train_end : int
        Last training year, inclusive.
    cmip6_train_root : str or pathlib.Path
        Directory containing historical CMIP6 yearly files.
    variable_name : str
        Climate variable to load.
    logger : Logger
        Logger used to report progress.
    era5_root : str or pathlib.Path or None, optional
        Directory containing raw ERA5 yearly files.
    era5_on_cmip6_root : str or pathlib.Path or None, optional
        Directory containing precomputed ERA5-on-CMIP6 fields.

    Returns
    -------
    reference_train : xarray.DataArray
        Concatenated ERA5 training data on the CMIP6 grid.
    biased_train : xarray.DataArray
        Concatenated historical CMIP6 training data.
    latitude_descending : bool
        Whether latitude is ordered north to south.

    Raises
    ------
    ValueError
        If both or neither ERA5 input modes are provided, or if spatial grids
        or latitude orientation are inconsistent between years.
    RuntimeError
        If no training year is loaded.
    """
    # Exactly one ERA5 input mode must be selected:
    # raw ERA5 files or precomputed ERA5-on-CMIP6 files.
    if (era5_root is None) == (era5_on_cmip6_root is None):
        raise ValueError("Provide exactly one of era5_root or era5_on_cmip6_root.")

    use_precomputed = era5_on_cmip6_root is not None

    # Precomputed ERA5 files already contain the latitude orientation
    # required to match the CMIP6 grid.
    if use_precomputed:
        metadata, variable_root = load_precomputed_metadata(
            era5_on_cmip6_root,
            variable_name,
            logger,
        )
        latitude_descending = bool(metadata["latitude_descending"])
    else:
        variable_root = None
        latitude_descending = None

    # Accumulate yearly ERA5 and CMIP6 fields before concatenating them
    # along the time dimension.
    reference_years = []
    biased_years = []

    # Keep the first yearly grids as references for consistency checks.
    reference_grid = None
    biased_grid = None

    for year in range(train_start, train_end + 1):
        cmip6_path = build_path(cmip6_train_root, year)

        if use_precomputed:
            logger.info(
                f"Preparing {variable_name}, training year {year} " "(precomputed ERA5)"
            )

            era5_path = build_path(variable_root, year)

            # ERA5 is already prepared on the CMIP6 grid
            reference_year = prepare_dataarray(
                open_dataarray(
                    era5_path,
                    variable_name,
                )
            )

            climate_dataset = ClimateDataset(
                cmip6_path=cmip6_path,
                variable_name=variable_name,
                logger=logger,
            )

            # Prepare CMIP6 using the latitude orientation stored
            # in the precomputed ERA5 metadata.
            biased_year = prepare_dataarray(
                climate_dataset.prepare_dataset(
                    latitude_descending=latitude_descending,
                )
            )

        else:
            logger.info(f"Preparing {variable_name}, training year {year}")

            era5_path = build_path(era5_root, year)

            climate_dataset = ClimateDataset(
                era5_path=era5_path,
                cmip6_path=cmip6_path,
                variable_name=variable_name,
                logger=logger,
            )

            # Raw ERA5 must be prepared and aligned with the native CMIP6 grid.
            (
                reference_year,
                biased_year,
                current_latitude_descending,
            ) = climate_dataset.prepare_dataset()

            reference_year = prepare_dataarray(reference_year)

            biased_year = prepare_dataarray(biased_year)

            # Latitude orientation must remain identical across all years.
            if latitude_descending is None:
                latitude_descending = current_latitude_descending
            elif latitude_descending != current_latitude_descending:
                raise ValueError("ERA5 latitude ordering changes between years.")

        climate_dataset.close()

        # ERA5 and CMIP6 must share exactly the same spatial grid
        # before they can be used for bias correction.
        check_same_spatial_grid(
            reference_year,
            biased_year,
        )

        # Check that the grid remains unchanged from one year to the next.
        if reference_grid is None:
            reference_grid = reference_year
            biased_grid = biased_year
        else:
            check_same_spatial_grid(
                reference_grid,
                reference_year,
            )
            check_same_spatial_grid(
                biased_grid,
                biased_year,
            )

        reference_years.append(reference_year)
        biased_years.append(biased_year)

    if latitude_descending is None:
        raise RuntimeError("No training year was loaded.")

    # Merge all yearly samples into continuous training time series.
    reference_train = xr.concat(
        reference_years,
        dim="time",
    ).transpose(
        "time",
        "latitude",
        "longitude",
    )

    biased_train = xr.concat(
        biased_years,
        dim="time",
    ).transpose(
        "time",
        "latitude",
        "longitude",
    )

    # Safety check after concatenation.
    check_same_spatial_grid(
        reference_train,
        biased_train,
    )

    return (
        reference_train,
        biased_train,
        latitude_descending,
    )


def assemble_variable_dataset(
    variable_arrays,
    variable_names,
    label,
):
    """
    Assemble aligned climate variables into one dataset.

    Auxiliary scalar coordinates, such as variable-specific measurement heights,
    are moved to variable attributes before merging. Time, latitude, and longitude
    coordinates must match exactly across variables.

    Parameters
    ----------
    variable_arrays : dict of str to xarray.DataArray
        Mapping from climate variable names to data arrays.
    variable_names : sequence of str
        Ordered variables to include in the assembled dataset.
    label : str
        Human-readable dataset label used in error messages.

    Returns
    -------
    xarray.Dataset
        Dataset containing all variables with shared time and spatial coordinates.

    Raises
    ------
    ValueError
        If variables cannot be aligned exactly or fail consistency checks.
    """
    cleaned_arrays = []

    for variable_name in variable_names:
        array = variable_arrays[variable_name]

        array = array.transpose(
            "time",
            "latitude",
            "longitude",
        )

        # Preserve auxiliary coordinates as attributes before removing them.
        auxiliary_coordinates = {}

        for coordinate_name in list(array.coords):
            if coordinate_name not in {
                "time",
                "latitude",
                "longitude",
            }:
                coordinate = array.coords[coordinate_name]

                if coordinate.ndim == 0:
                    value = coordinate.values

                    if np.ndim(value) == 0:
                        value = value.item()

                    auxiliary_coordinates[coordinate_name] = value

        # Remove all non-index coordinates, for example height.
        array = array.reset_coords(
            drop=True,
        )

        # Keep their information inside the variable attributes.
        for coordinate_name, value in auxiliary_coordinates.items():
            array.attrs[f"original_coordinate_{coordinate_name}"] = value

        array.name = variable_name
        cleaned_arrays.append(array)

    try:
        aligned_arrays = xr.align(
            *cleaned_arrays,
            join="exact",
            copy=False,
        )
    except ValueError as error:
        raise ValueError(
            f"Variables in {label} do not share exactly the same "
            "time and spatial coordinates."
        ) from error

    dataset = xr.Dataset(
        {
            variable_name: array
            for variable_name, array in zip(
                variable_names,
                aligned_arrays,
            )
        }
    )

    check_dataset_consistency(
        dataset,
        variable_names,
        label,
    )

    return dataset


def load_training_data(
    train_start,
    train_end,
    era5_roots,
    era5_on_cmip6_roots,
    cmip6_train_roots,
    variable_names,
    logger,
):
    """
    Load and assemble all variables required for training.

    Parameters
    ----------
    train_start : int
        First training year.
    train_end : int
        Last training year, inclusive.
    era5_roots : dict of str to str or None
        Raw ERA5 directory for each variable, or None when precomputed ERA5 data
        are used.
    era5_on_cmip6_roots : dict of str to str or None
        Precomputed ERA5-on-CMIP6 directory for each variable, or None when raw
        ERA5 data are used.
    cmip6_train_roots : dict of str to str
        Historical CMIP6 directory for each variable.
    variable_names : sequence of str
        Ordered climate variables to load.
    logger : Logger
        Logger used to report progress.

    Returns
    -------
    reference_train : xarray.Dataset
        ERA5 training variables on the CMIP6 grid.
    biased_train : xarray.Dataset
        Historical CMIP6 training variables.
    latitude_descending : bool
        Common latitude orientation used by all variables.

    Raises
    ------
    ValueError
        If variables have inconsistent latitude orientation, grids, times, or
        values.
    """
    reference_variables = {}
    biased_variables = {}

    common_latitude_descending = None

    for variable_name in variable_names:
        cmip6_train_root = cmip6_train_roots[variable_name]

        # Select the raw ERA5 root when raw inputs are used.
        era5_root = era5_roots[variable_name] if era5_roots is not None else None

        # Select the precomputed ERA5-on-CMIP6 root when available.
        era5_on_cmip6_root = (
            era5_on_cmip6_roots[variable_name]
            if era5_on_cmip6_roots is not None
            else None
        )

        # Load the current variable using the active ERA5 input mode.
        (
            reference_variable,
            biased_variable,
            latitude_descending,
        ) = load_training_variable(
            train_start=train_start,
            train_end=train_end,
            cmip6_train_root=cmip6_train_root,
            variable_name=variable_name,
            logger=logger,
            era5_root=era5_root,
            era5_on_cmip6_root=era5_on_cmip6_root,
        )

        # All variables must use the same latitude orientation.
        if common_latitude_descending is None:
            common_latitude_descending = latitude_descending
        elif common_latitude_descending != latitude_descending:
            raise ValueError("Latitude orientation differs between variables.")

        reference_variables[variable_name] = reference_variable
        biased_variables[variable_name] = biased_variable

    # Merge all ERA5 variables into one aligned training dataset.
    reference_train = assemble_variable_dataset(
        reference_variables,
        variable_names,
        label="ERA5 training data",
    )

    # Merge all CMIP6 variables into one aligned training dataset.
    biased_train = assemble_variable_dataset(
        biased_variables,
        variable_names,
        label="CMIP6 training data",
    )

    # ERA5 and CMIP6 must share the same spatial grid before fitting.
    check_same_spatial_grid(
        reference_train[variable_names[0]],
        biased_train[variable_names[0]],
    )

    logger.success(
        "Training data ready:\n"
        f"Variables: {variable_names}\n"
        f"ERA5 sizes:  {dict(reference_train.sizes)}\n"
        f"CMIP6 sizes: {dict(biased_train.sizes)}"
    )

    return (
        reference_train,
        biased_train,
        common_latitude_descending,
    )


def load_application_data(
    year,
    era5_roots,
    era5_on_cmip6_roots,
    cmip6_apply_roots,
    variable_names,
    latitude_descending,
    logger,
):
    """
    Load and assemble ERA5 reference and CMIP6 data for one application year.

    Parameters
    ----------
    year : int
        Application year to load.
    era5_roots : dict of str to str or None
        Raw ERA5 directory for each variable.
    era5_on_cmip6_roots : dict of str to str or None
        Precomputed ERA5-on-CMIP6 directory for each variable.
    cmip6_apply_roots : dict of str to str
        CMIP6 application directory for each variable.
    variable_names : sequence of str
        Ordered climate variables to load.
    latitude_descending : bool
        Latitude orientation established during training.
    logger : Logger
        Logger used to report progress.

    Returns
    -------
    reference_apply : xarray.Dataset
        ERA5 reference data on the CMIP6 grid.
    biased_apply : xarray.Dataset
        Raw CMIP6 application data.
    """
    reference_variables = {}
    biased_variables = {}

    for variable_name in variable_names:
        cmip6_path = build_path(
            cmip6_apply_roots[variable_name],
            year,
        )

        if era5_on_cmip6_roots is not None:
            variable_root = resolve_precomputed_variable_root(
                era5_on_cmip6_roots[variable_name],
                variable_name,
            )

            era5_path = build_path(
                variable_root,
                year,
            )

            reference_variable = prepare_dataarray(
                open_dataarray(
                    era5_path,
                    variable_name,
                )
            )

            climate_dataset = ClimateDataset(
                cmip6_path=cmip6_path,
                variable_name=variable_name,
                logger=logger,
            )

            biased_variable = prepare_dataarray(
                climate_dataset.prepare_dataset(
                    latitude_descending=latitude_descending,
                )
            )

        else:
            if era5_roots is None:
                raise RuntimeError("Internal error: ERA5 roots are unavailable.")

            era5_path = build_path(
                era5_roots[variable_name],
                year,
            )

            climate_dataset = ClimateDataset(
                era5_path=era5_path,
                cmip6_path=cmip6_path,
                variable_name=variable_name,
                logger=logger,
            )

            (
                reference_variable,
                biased_variable,
                current_latitude_descending,
            ) = climate_dataset.prepare_dataset()

            if current_latitude_descending != latitude_descending:
                raise ValueError(
                    f"Latitude orientation differs for "
                    f"{variable_name} in year {year}."
                )

            reference_variable = prepare_dataarray(reference_variable)

            biased_variable = prepare_dataarray(biased_variable)

        climate_dataset.close()

        check_same_spatial_grid(
            reference_variable,
            biased_variable,
        )

        reference_variables[variable_name] = reference_variable
        biased_variables[variable_name] = biased_variable

    reference_apply = assemble_variable_dataset(
        reference_variables,
        variable_names,
        label=f"ERA5 evaluation year {year}",
    )

    biased_apply = assemble_variable_dataset(
        biased_variables,
        variable_names,
        label=f"CMIP6 application year {year}",
    )

    check_same_spatial_grid(
        reference_apply[variable_names[0]],
        biased_apply[variable_names[0]],
    )

    if not np.array_equal(
        reference_apply["time"].values,
        biased_apply["time"].values,
    ):
        raise ValueError(f"ERA5 and CMIP6 time coordinates differ for year {year}.")

    return reference_apply, biased_apply


def flatten_dataset(
    dataset,
    variable_names,
):
    """
    Flatten and concatenate climate variables into a feature matrix.

    Variables are concatenated by blocks in the order given by variable_names.
    Each block contains all latitude-longitude grid points for one variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with dimensions time, latitude, and longitude.
    variable_names : sequence of str
        Ordered variables to concatenate.

    Returns
    -------
    matrix : numpy.ndarray
        Array of shape (time, n_variables * latitude * longitude).
    metadata : dict
        Variable order and original dimensions required for reconstruction.

    Raises
    ------
    ValueError
        If the dataset fails consistency checks.
    """
    check_dataset_consistency(
        dataset,
        variable_names,
        label="flatten input",
    )

    matrices = []

    number_times = dataset.sizes["time"]
    number_latitudes = dataset.sizes["latitude"]
    number_longitudes = dataset.sizes["longitude"]
    grid_size = number_latitudes * number_longitudes

    for variable_name in variable_names:
        values = (
            dataset[variable_name]
            .transpose(
                "time",
                "latitude",
                "longitude",
            )
            .values.astype(
                np.float32,
                copy=False,
            )
        )

        matrices.append(values.reshape(number_times, grid_size))

    matrix = np.concatenate(
        matrices,
        axis=1,
    )

    metadata = {
        "variable_names": list(variable_names),
        "number_times": number_times,
        "number_latitudes": number_latitudes,
        "number_longitudes": number_longitudes,
        "grid_size": grid_size,
    }

    return matrix, metadata


def reconstruct_dataset(
    corrected_2d,
    template_dataset,
    metadata,
    method_name,
):
    """
    Reconstruct corrected climate fields from a feature matrix.

    Parameters
    ----------
    corrected_2d : numpy.ndarray
        Corrected matrix with shape
        (time, n_variables * latitude * longitude).
    template_dataset : xarray.Dataset
        Dataset providing coordinates, variable attributes, and output structure.
    metadata : dict
        Flattening metadata produced by flatten_dataset.
    method_name : str
        Name of the applied bias-correction method.

    Returns
    -------
    xarray.Dataset
        Corrected variables reconstructed on their original grid.

    Raises
    ------
    RuntimeError
        If corrected_2d does not have the expected shape.
    """
    variable_names = metadata["variable_names"]
    number_times = metadata["number_times"]
    number_latitudes = metadata["number_latitudes"]
    number_longitudes = metadata["number_longitudes"]
    grid_size = metadata["grid_size"]

    expected_shape = (
        number_times,
        len(variable_names) * grid_size,
    )

    if corrected_2d.shape != expected_shape:
        raise RuntimeError(
            "Unexpected corrected matrix shape: "
            f"{corrected_2d.shape} != {expected_shape}."
        )

    corrected_variables = {}

    for variable_index, variable_name in enumerate(variable_names):
        start = variable_index * grid_size
        stop = start + grid_size

        corrected_values = corrected_2d[
            :,
            start:stop,
        ].reshape(
            number_times,
            number_latitudes,
            number_longitudes,
        )

        template = template_dataset[variable_name].transpose(
            "time",
            "latitude",
            "longitude",
        )

        corrected_variable = template.copy(
            data=corrected_values.astype(
                np.float32,
                copy=False,
            )
        )

        corrected_variable.name = variable_name
        corrected_variable.attrs = dict(template.attrs)
        corrected_variable.attrs["bias_correction_method"] = method_name

        corrected_variable.attrs["bias_correction_mode"] = (
            "univariate" if method_name == "qm" else "multivariate"
        )

        corrected_variables[variable_name] = corrected_variable

    corrected_dataset = xr.Dataset(corrected_variables)
    corrected_dataset.attrs = dict(template_dataset.attrs)
    corrected_dataset.attrs["bias_correction_method"] = method_name
    corrected_dataset.attrs["bias_correction_variables"] = ",".join(variable_names)

    return corrected_dataset


def fit_bias_corrector(
    reference_train,
    biased_train,
    variable_names,
    corrector,
    logger,
):
    """
    Fit one bias corrector on flattened climate fields.

    For QM, command-line validation ensures that only one climate variable is
    provided. For OT, all requested variables and grid points are concatenated
    and corrected jointly.

    Parameters
    ----------
    reference_train : xarray.Dataset
        ERA5 training data on the CMIP6 grid.
    biased_train : xarray.Dataset
        Historical CMIP6 training data.
    variable_names : sequence of str
        Ordered variables used to construct the feature matrices.
    corrector : BiasCorrector
        Initialized bias-correction object.
    logger : Logger
        Logger used to report fitting diagnostics.

    Returns
    -------
    BiasCorrector
        Fitted bias corrector.

    Raises
    ------
    ValueError
        If ERA5 and CMIP6 spatial or feature dimensions are inconsistent.
    """
    reference_2d, reference_metadata = flatten_dataset(
        reference_train,
        variable_names,
    )

    biased_2d, biased_metadata = flatten_dataset(
        biased_train,
        variable_names,
    )

    if (
        reference_metadata["number_latitudes"] != biased_metadata["number_latitudes"]
        or reference_metadata["number_longitudes"]
        != biased_metadata["number_longitudes"]
    ):
        raise ValueError("ERA5 and CMIP6 spatial dimensions differ.")

    if reference_2d.shape[1] != biased_2d.shape[1]:
        raise ValueError(
            "ERA5 and CMIP6 feature dimensions differ: "
            f"{reference_2d.shape[1]} != {biased_2d.shape[1]}."
        )

    logger.info(
        f"Fitting method: {corrector.method_name.upper()}\n"
        f"Variables: {variable_names}\n"
        f"Reference matrix: {reference_2d.shape}\n"
        f"Biased matrix:    {biased_2d.shape}"
    )

    corrector.fit(
        reference=reference_2d,
        biased=biased_2d,
    )

    logger.success(f"{corrector.method_name.upper()} fitting completed")

    logger.info(
        "Corrector diagnostics:\n"
        + json.dumps(
            corrector.diagnostics,
            indent=2,
            default=str,
        )
    )

    del reference_2d
    del biased_2d
    gc.collect()

    return corrector


def fit_bias_correctors_by_month(
    reference_train,
    biased_train,
    variable_names,
    method,
    ot_kwargs,
    logger,
):
    """
    Fit one bias corrector for each calendar month.

    Parameters
    ----------
    reference_train : xarray.Dataset
        ERA5 training data covering all training years.
    biased_train : xarray.Dataset
        Historical CMIP6 training data covering all training years.
    variable_names : sequence of str
        Ordered variables used during fitting.
    method : {'qm', 'ot'}
        Bias-correction method.
    ot_kwargs : dict
        Keyword arguments passed to create_bias_corrector.
    logger : Logger
        Logger used to report monthly fitting progress.

    Returns
    -------
    dict of int to BiasCorrector
        Fitted corrector indexed by calendar month from 1 to 12.

    Raises
    ------
    RuntimeError
        If ERA5 or CMIP6 contains no training samples for a calendar month.
    """
    correctors = {}

    for month in range(1, 13):
        logger.info(f"=== Fitting corrector for month {month:02d} ===")

        reference_month = select_month(
            reference_train,
            month,
        )

        biased_month = select_month(
            biased_train,
            month,
        )

        if reference_month.sizes["time"] == 0:
            raise RuntimeError(f"No ERA5 samples found for month {month:02d}.")

        if biased_month.sizes["time"] == 0:
            raise RuntimeError(f"No CMIP6 samples found for month {month:02d}.")

        corrector = create_bias_corrector(
            method=method,
            **ot_kwargs,
        )

        corrector = fit_bias_corrector(
            reference_train=reference_month,
            biased_train=biased_month,
            variable_names=variable_names,
            corrector=corrector,
            logger=logger,
        )

        correctors[month] = corrector

        del reference_month
        del biased_month
        gc.collect()

    return correctors


def apply_bias_corrector(
    biased_apply,
    variable_names,
    corrector,
    logger,
):
    """
    Apply a fitted bias corrector to climate fields.

    Parameters
    ----------
    biased_apply : xarray.Dataset
        CMIP6 data to correct.
    variable_names : sequence of str
        Ordered variables used during fitting and application.
    corrector : BiasCorrector
        Fitted bias corrector.
    logger : Logger
        Logger used to report application diagnostics.

    Returns
    -------
    xarray.Dataset
        Corrected climate variables on the original CMIP6 grid.

    Raises
    ------
    RuntimeError
        If the transformed matrix shape differs from the input matrix shape.
    """
    biased_apply_2d, metadata = flatten_dataset(
        biased_apply,
        variable_names,
    )

    logger.info(f"Application matrix: {biased_apply_2d.shape}")

    corrected_2d = corrector.transform(biased_apply_2d)

    if corrected_2d.shape != biased_apply_2d.shape:
        raise RuntimeError(
            "The corrected matrix has an unexpected shape: "
            f"{corrected_2d.shape} != {biased_apply_2d.shape}."
        )

    corrected = reconstruct_dataset(
        corrected_2d=corrected_2d,
        template_dataset=biased_apply,
        metadata=metadata,
        method_name=corrector.method_name,
    )

    del biased_apply_2d
    del corrected_2d
    gc.collect()

    return corrected


def apply_bias_corrector_by_month(
    biased_apply,
    variable_names,
    correctors_by_month,
    logger,
):
    """
    Apply month-specific correctors and reassemble the time series.

    Parameters
    ----------
    biased_apply : xarray.Dataset
        CMIP6 data for one application year.
    variable_names : sequence of str
        Ordered variables used during fitting and application.
    correctors_by_month : dict of int to BiasCorrector
        Fitted corrector for each calendar month.
    logger : Logger
        Logger used to report application progress.

    Returns
    -------
    xarray.Dataset
        Corrected data concatenated and sorted chronologically.

    Raises
    ------
    RuntimeError
        If a required monthly corrector is missing or no month is corrected.
    """
    corrected_months = []

    for month in range(1, 13):
        biased_apply_month = select_month(
            biased_apply,
            month,
        )

        if biased_apply_month.sizes["time"] == 0:
            continue

        if month not in correctors_by_month:
            raise RuntimeError(f"No fitted corrector available for month {month:02d}.")

        logger.info(
            f"Applying month {month:02d} corrector "
            f"({biased_apply_month.sizes['time']} time steps)"
        )

        corrected_month = apply_bias_corrector(
            biased_apply=biased_apply_month,
            variable_names=variable_names,
            corrector=correctors_by_month[month],
            logger=logger,
        )

        corrected_months.append(corrected_month)

        del biased_apply_month
        gc.collect()

    if not corrected_months:
        raise RuntimeError("No application month was corrected.")

    corrected = xr.concat(
        corrected_months,
        dim="time",
    ).sortby("time")

    del corrected_months
    gc.collect()

    return corrected


def save_corrected_year(
    corrected,
    output_dirs,
    year,
    variable_names,
    logger,
):
    """
    Save each corrected variable in its own yearly NetCDF file.

    Parameters
    ----------
    corrected : xarray.Dataset
        Corrected climate variables.
    output_dirs : dict of str to str
        Output directory for each variable.
    year : int
        Year written to the samples_<year>.nc filename.
    variable_names : sequence of str
        Variables to save.
    logger : Logger
        Logger used to report output paths.

    Raises
    ------
    ValueError
        If the corrected dataset has no time samples or lacks a requested
        variable.
    """
    if corrected.sizes["time"] == 0:
        raise ValueError(f"Corrected year {year} contains no time step.")

    for variable_name in variable_names:
        if variable_name not in corrected:
            raise ValueError(f"Corrected dataset does not contain '{variable_name}'.")

        output_path = Path(output_dirs[variable_name])
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = output_path / f"samples_{year}.nc"

        logger.info(f"Writing corrected variable {variable_name}:\n" f"{output_file}")

        output_dataset = corrected[variable_name].to_dataset(name=variable_name)

        encoding = {
            variable_name: {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
            }
        }

        output_dataset.to_netcdf(
            output_file,
            encoding=encoding,
        )
        output_dataset.close()


def main():
    """
    Run the complete CMIP6 bias-correction workflow.

    The workflow validates arguments, loads and aligns training variables, fits
    one global or monthly QM/OT corrector, applies the fitted correction to each
    requested year, and saves each corrected variable in its configured output
    directory.
    """
    args = parse_args()
    validate_args(args)

    logger = Logger()
    variable_names = list(args.variables)

    era5_roots = parse_variable_paths(
        args.era5_root,
        "--era5_root",
    )
    era5_on_cmip6_roots = parse_variable_paths(
        args.era5_on_cmip6_root,
        "--era5_on_cmip6_root",
    )
    cmip6_train_roots = parse_variable_paths(
        args.cmip6_train_root,
        "--cmip6_train_root",
    )
    cmip6_apply_roots = parse_variable_paths(
        args.cmip6_apply_root,
        "--cmip6_apply_root",
    )
    output_dirs = parse_variable_paths(
        args.output_dir,
        "--output_dir",
    )

    if era5_on_cmip6_roots is not None:
        validate_variable_paths(
            variable_names,
            era5_on_cmip6_roots,
            "--era5_on_cmip6_root",
        )
    else:
        validate_variable_paths(
            variable_names,
            era5_roots,
            "--era5_root",
        )

    validate_variable_paths(
        variable_names,
        cmip6_train_roots,
        "--cmip6_train_root",
    )
    validate_variable_paths(
        variable_names,
        cmip6_apply_roots,
        "--cmip6_apply_root",
    )
    validate_variable_paths(
        variable_names,
        output_dirs,
        "--output_dir",
    )

    run_name = Path(output_dirs[variable_names[0]]).parent.name

    logger.info(f"Selected bias-correction method: {args.method.upper()}")

    logger.info(f"Selected variables: {variable_names}")

    logger.info(f"Selected split mode: {args.split}")

    if args.method == "qm":
        logger.info("QM will correct exactly one climate variable.")
    else:
        logger.info(
            "OT will jointly transport all requested variables and "
            "all spatial grid points."
        )

    ot_kwargs = dict(
        ot_epsilon=args.ot_epsilon,
        ot_num_iterations=args.ot_num_iterations,
        ot_threshold=args.ot_threshold,
        ot_batch_size=args.ot_batch_size,
        ot_dtype=args.ot_dtype,
        ot_normalize=not args.ot_no_normalization,
    )

    (
        reference_train,
        biased_train,
        latitude_descending,
    ) = load_training_data(
        train_start=args.train_start,
        train_end=args.train_end,
        era5_roots=era5_roots,
        era5_on_cmip6_roots=era5_on_cmip6_roots,
        cmip6_train_roots=cmip6_train_roots,
        variable_names=variable_names,
        logger=logger,
    )

    if args.split == "month":
        correctors = fit_bias_correctors_by_month(
            reference_train=reference_train,
            biased_train=biased_train,
            variable_names=variable_names,
            method=args.method,
            ot_kwargs=ot_kwargs,
            logger=logger,
        )
    else:
        corrector = create_bias_corrector(
            method=args.method,
            **ot_kwargs,
        )

        corrector = fit_bias_corrector(
            reference_train=reference_train,
            biased_train=biased_train,
            variable_names=variable_names,
            corrector=corrector,
            logger=logger,
        )

    del reference_train
    del biased_train
    gc.collect()

    for year in range(
        args.apply_start,
        args.apply_end + 1,
    ):
        logger.info(f"Correcting CMIP6 year {year}")

        reference_apply, biased_apply = load_application_data(
            year=year,
            era5_roots=era5_roots,
            era5_on_cmip6_roots=era5_on_cmip6_roots,
            cmip6_apply_roots=cmip6_apply_roots,
            variable_names=variable_names,
            latitude_descending=latitude_descending,
            logger=logger,
        )

        if args.split == "month":
            corrected = apply_bias_corrector_by_month(
                biased_apply=biased_apply,
                variable_names=variable_names,
                correctors_by_month=correctors,
                logger=logger,
            )
        else:
            corrected = apply_bias_corrector(
                biased_apply=biased_apply,
                variable_names=variable_names,
                corrector=corrector,
                logger=logger,
            )

        evaluate(
            reference=reference_apply,
            raw=biased_apply,
            corrected=corrected,
            variable_names=variable_names,
            method=args.method,
            year=year,
            logger=logger,
            run_name=run_name,
        )

        save_corrected_year(
            corrected=corrected,
            output_dirs=output_dirs,
            year=year,
            variable_names=variable_names,
            logger=logger,
        )

        del reference_apply
        del biased_apply
        del corrected
        gc.collect()

    logger.success("Full-grid CMIP6 bias correction completed")


if __name__ == "__main__":
    main()
