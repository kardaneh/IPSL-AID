# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
IPSL-AID: AI Diffusion Models for Climate and Weather Data
"""

from IPSL_AID.version import __version__, __version_info__, get_version
from IPSL_AID.logger import Logger
from IPSL_AID.utils import EasyDict, FileUtils

__author__ = "CNRS / IPSL / Sorbonne University"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
__copyright__ = "2026, CNRS / IPSL / Sorbonne University"
__description__ = "AI Diffusion Models for Climate and Weather Data"

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
    "__author__",
    "__license__",
    "__copyright__",
    "Logger",
    "EasyDict",
    "FileUtils",
]
