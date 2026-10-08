# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""AID-BC: AI downscaling bias correction"""

__author__ = "Kazem Ardaneh"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
__copyright__ = "2026, CNRS / IPSL / Sorbonne University"
__description__ = "Radiative Transfer Neural Networks for Climate Science"

# Import version info
from AID_BC.version import __version__, __version_info__, get_version

# Import logger
from AID_BC.logger import Logger

# Import main components for easy access

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
    "__author__",
    "__license__",
    "__copyright__",
    "Logger",
]
