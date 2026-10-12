# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
AID-BC: AI downscaling bias correction.
"""

__author__ = "Kazem Ardaneh"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
__copyright__ = "2026, CNRS / IPSL / Sorbonne University"
__description__ = "AI Downscaling Bias Correction for Climate Data"

# Import version info
from AID_BC.version import __version__, __version_info__, get_version

# Import logger
from AID_BC.logger import Logger

# Import main components for easy access
from AID_BC.dataset import ClimateDataset
from AID_BC.quantile_mapping import QM
from AID_BC.optimal_transport import OptimalTransport
from AID_BC.evaluater import evaluate

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
    "__author__",
    "__license__",
    "__copyright__",
    "__description__",
    "Logger",
    "ClimateDataset",
    "QM",
    "OptimalTransport",
    "evaluate",
]
