import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, PROJECT_ROOT)

project = "IPSL-AID"
copyright = f"{datetime.now().year}, IPSL Climate Modeling Center, Sorbonne University"
author = "Kazem Ardaneh and IPSL Team"
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",  # Keep commented until installed
]

# Mock heavy dependencies
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "cartopy",
    "xarray",
    "netCDF4",
    "h5netcdf",
    "matplotlib",
    "scipy",
    "numpy",
    "pandas",
    "sklearn",
    "mpltex",
    "rich",
    "tensorboard",
    "tqdm",
    "cycler",
    "markupsafe",
    "seaborn",
    "cdsapi",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []

# MathJax 3 configuration
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    },
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# Suppress warnings
suppress_warnings = [
    "intersphinx",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
