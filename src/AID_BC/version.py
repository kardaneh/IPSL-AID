# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Version information for AID_BC."""

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))


def get_version():
    """Return the version string."""
    return __version__
