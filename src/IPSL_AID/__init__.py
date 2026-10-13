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
    "__description__",
    "Logger",
    "EasyDict",
    "FileUtils",
    "DataPreprocessor",
    "run_validation",
    "sampler",
    "VPLoss",
    "VELoss",
    "EDMLoss",
    "UnetLoss",
    "load_model_and_loss",
    "SongUNet",
    "DhariwalUNet",
    "VPPrecond",
    "VEPrecond",
    "iDDPMPrecond",
    "EDMPrecond",
]


def __getattr__(name):
    """Lazily import components requiring heavier dependencies."""

    if name == "DataPreprocessor":
        from IPSL_AID.dataset import DataPreprocessor

        return DataPreprocessor

    if name in {"run_validation", "sampler"}:
        from IPSL_AID.evaluater import run_validation, sampler

        return {
            "run_validation": run_validation,
            "sampler": sampler,
        }[name]

    if name in {"VPLoss", "VELoss", "EDMLoss", "UnetLoss"}:
        from IPSL_AID.loss import EDMLoss, UnetLoss, VELoss, VPLoss

        return {
            "VPLoss": VPLoss,
            "VELoss": VELoss,
            "EDMLoss": EDMLoss,
            "UnetLoss": UnetLoss,
        }[name]

    if name == "load_model_and_loss":
        from IPSL_AID.model import load_model_and_loss

        return load_model_and_loss

    if name in {
        "SongUNet",
        "DhariwalUNet",
        "VPPrecond",
        "VEPrecond",
        "iDDPMPrecond",
        "EDMPrecond",
    }:
        from IPSL_AID.networks import (
            DhariwalUNet,
            EDMPrecond,
            SongUNet,
            VEPrecond,
            VPPrecond,
            iDDPMPrecond,
        )

        return {
            "SongUNet": SongUNet,
            "DhariwalUNet": DhariwalUNet,
            "VPPrecond": VPPrecond,
            "VEPrecond": VEPrecond,
            "iDDPMPrecond": iDDPMPrecond,
            "EDMPrecond": EDMPrecond,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
