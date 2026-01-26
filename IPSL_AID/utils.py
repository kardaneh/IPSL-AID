# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, List, Tuple, Union, Optional

class EasyDict(dict):
    """
    Convenience class that behaves like a dict but allows access with the attribute syntax.
    Example:
        >>> ed = EasyDict()
        >>> ed.key = 'value'
        >>> print(ed['key'])
        value
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
    A utility class for basic file and directory operations.
    This class provides static methods to create directories and files.

    Methods
    -------
    makedir(dirs: str) -> None
        Creates a directory if it does not already exist.

    makefile(dirs: str, filename: str) -> None
        Creates an empty file at the specified directory path.
    """

    def __init__(self):
        """
        Initializes the FileUtils class.
        Currently, this class does not maintain any instance attributes.
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
        Create an empty file in the given directory.

        Parameters
        ----------
        dirs : str
            The directory in which the file should be created.
        filename : str
            The name of the file to create.
        """
        filepath = os.path.join(dirs, filename)
        with open(filepath, "a"):
            pass