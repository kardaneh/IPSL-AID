# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
from typing import Any


class EasyDict(dict):
    """
    A dictionary subclass that allows for attribute-style access to its items.
    This class extends the built-in dict and overrides the __getattr__, __setattr__, and __delattr__ methods to enable accessing dictionary keys as attributes.
    Original work: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
    Original source: https://github.com/NVlabs/edm
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class FileUtils:
    """
    Utility class for file and directory operations.
    """

    def __init__(self):
        """
        Initialize the FileUtils class. This class does not maintain any state, so the constructor is empty.
        """
        super().__init__()

    @staticmethod
    def makedir(dirs):
        """
        Create a directory if it does not exist.

        Parameters
        ----------
        dirs : str
            The path of the directory to be created.
        """
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    @staticmethod
    def makefile(dirs, filename):
        """
        Create an empty file in the specified directory.
        Parameters
        ----------
        dirs : str
            The path of the directory where the file will be created.
        filename : str
            The name of the file to be created.
        """
        filepath = os.path.join(dirs, filename)
        with open(filepath, "a"):
            pass
