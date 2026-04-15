# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import tempfile
import unittest
import shutil
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.utils import EasyDict, FileUtils
from IPSL_AID.logger import Logger

# ============================================================================
# Unit Tests for EasyDict
# ============================================================================


class TestEasyDict(unittest.TestCase):
    """Unit tests for EasyDict class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Test setup - EasyDict tests")

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_empty_initialization(self):
        """Test initializing an empty EasyDict."""
        if self.logger:
            self.logger.info("Testing empty EasyDict initialization")

        ed = EasyDict()

        self.assertIsInstance(ed, dict)
        self.assertIsInstance(ed, EasyDict)
        self.assertEqual(len(ed), 0)

        if self.logger:
            self.logger.info("✅ empty initialization test passed")

    def test_initialization_with_dict(self):
        """Test initializing EasyDict with a dictionary."""
        if self.logger:
            self.logger.info("Testing EasyDict initialization with dictionary")

        data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        ed = EasyDict(data)

        self.assertEqual(ed["key1"], "value1")
        self.assertEqual(ed["key2"], 42)
        self.assertEqual(ed["key3"], [1, 2, 3])
        self.assertEqual(len(ed), 3)

        if self.logger:
            self.logger.info("✅ initialization with dict test passed")

    def test_initialization_with_kwargs(self):
        """Test initializing EasyDict with keyword arguments."""
        if self.logger:
            self.logger.info("Testing EasyDict initialization with kwargs")

        ed = EasyDict(key1="value1", key2=42, key3=[1, 2, 3])

        self.assertEqual(ed["key1"], "value1")
        self.assertEqual(ed["key2"], 42)
        self.assertEqual(ed["key3"], [1, 2, 3])
        self.assertEqual(len(ed), 3)

        if self.logger:
            self.logger.info("✅ initialization with kwargs test passed")

    # ------------------------------------------------------------------------
    # Attribute Access Tests
    # ------------------------------------------------------------------------

    def test_attribute_get_set(self):
        """Test getting and setting attributes with dot notation."""
        if self.logger:
            self.logger.info("Testing attribute get/set with dot notation")

        ed = EasyDict()

        # Set attributes
        ed.name = "test_name"
        ed.value = 100
        ed.nested = {"a": 1, "b": 2}

        # Get attributes
        self.assertEqual(ed.name, "test_name")
        self.assertEqual(ed.value, 100)
        self.assertEqual(ed.nested, {"a": 1, "b": 2})

        # Verify dictionary access also works
        self.assertEqual(ed["name"], "test_name")
        self.assertEqual(ed["value"], 100)
        self.assertEqual(ed["nested"], {"a": 1, "b": 2})

        if self.logger:
            self.logger.info("✅ attribute get/set test passed")

    def test_dict_get_set(self):
        """Test getting and setting with dictionary notation."""
        if self.logger:
            self.logger.info("Testing dictionary get/set with bracket notation")

        ed = EasyDict()

        # Set with bracket notation
        ed["name"] = "test_name"
        ed["value"] = 100

        # Get with bracket notation
        self.assertEqual(ed["name"], "test_name")
        self.assertEqual(ed["value"], 100)

        # Verify attribute access also works
        self.assertEqual(ed.name, "test_name")
        self.assertEqual(ed.value, 100)

        if self.logger:
            self.logger.info("✅ dictionary get/set test passed")

    def test_mixed_access(self):
        """Test mixing dot and bracket notation."""
        if self.logger:
            self.logger.info("Testing mixed dot and bracket notation")

        ed = EasyDict()

        # Set with dot, get with bracket
        ed.key1 = "value1"
        self.assertEqual(ed["key1"], "value1")

        # Set with bracket, get with dot
        ed["key2"] = "value2"
        self.assertEqual(ed.key2, "value2")

        if self.logger:
            self.logger.info("✅ mixed access test passed")

    def test_attribute_error_for_nonexistent(self):
        """Test that accessing nonexistent attribute raises AttributeError."""
        if self.logger:
            self.logger.info("Testing AttributeError for nonexistent attribute")

        ed = EasyDict()

        with self.assertRaises(AttributeError):
            _ = ed.nonexistent_attribute

        if self.logger:
            self.logger.info("✅ AttributeError test passed")

    def test_key_error_for_nonexistent(self):
        """Test that accessing nonexistent key raises KeyError."""
        if self.logger:
            self.logger.info("Testing KeyError for nonexistent key")

        ed = EasyDict()

        with self.assertRaises(KeyError):
            _ = ed["nonexistent_key"]

        if self.logger:
            self.logger.info("✅ KeyError test passed")

    def test_delattr(self):
        """Test deleting attributes with delattr."""
        if self.logger:
            self.logger.info("Testing delattr functionality")

        ed = EasyDict()
        ed.key = "value"

        self.assertIn("key", ed)

        del ed.key

        self.assertNotIn("key", ed)

        with self.assertRaises(AttributeError):
            _ = ed.key

        if self.logger:
            self.logger.info("✅ delattr test passed")

    def test_delitem(self):
        """Test deleting items with del."""
        if self.logger:
            self.logger.info("Testing delitem functionality")

        ed = EasyDict()
        ed["key"] = "value"

        self.assertIn("key", ed)

        del ed["key"]

        self.assertNotIn("key", ed)

        with self.assertRaises(KeyError):
            _ = ed["key"]

        if self.logger:
            self.logger.info("✅ delitem test passed")

    # ------------------------------------------------------------------------
    # Dictionary Method Tests
    # ------------------------------------------------------------------------

    def test_dict_methods(self):
        """Test that standard dictionary methods work."""
        if self.logger:
            self.logger.info("Testing dictionary methods")

        ed = EasyDict(a=1, b=2, c=3)

        # keys()
        keys = ed.keys()
        self.assertIn("a", keys)
        self.assertIn("b", keys)
        self.assertIn("c", keys)
        self.assertEqual(len(keys), 3)

        # values()
        values = ed.values()
        self.assertIn(1, values)
        self.assertIn(2, values)
        self.assertIn(3, values)

        # items()
        items = ed.items()
        self.assertEqual(len(items), 3)

        # get()
        self.assertEqual(ed.get("a"), 1)
        self.assertEqual(ed.get("nonexistent", "default"), "default")

        # update()
        ed.update({"d": 4, "e": 5})
        self.assertEqual(ed.d, 4)
        self.assertEqual(ed.e, 5)

        # pop()
        value = ed.pop("a")
        self.assertEqual(value, 1)
        self.assertNotIn("a", ed)

        if self.logger:
            self.logger.info("✅ dictionary methods test passed")

    def test_nested_easydict(self):
        """Test that nested dictionaries are not automatically converted."""
        if self.logger:
            self.logger.info("Testing nested dictionary behavior")

        ed = EasyDict()
        ed.nested = {"inner": "value"}

        # Nested dict should be a regular dict, not EasyDict
        self.assertIsInstance(ed.nested, dict)
        self.assertNotIsInstance(ed.nested, EasyDict)

        # But we can still access it
        self.assertEqual(ed.nested["inner"], "value")

        if self.logger:
            self.logger.info("✅ nested dictionary test passed")

    def test_easydict_with_easydict(self):
        """Test nesting EasyDict inside EasyDict."""
        if self.logger:
            self.logger.info("Testing nesting EasyDict inside EasyDict")

        inner = EasyDict(x=1, y=2)
        outer = EasyDict()
        outer.inner = inner

        self.assertIsInstance(outer.inner, EasyDict)
        self.assertEqual(outer.inner.x, 1)
        self.assertEqual(outer.inner.y, 2)

        if self.logger:
            self.logger.info("✅ nested EasyDict test passed")

    def tearDown(self):
        """Clean up after tests."""
        if self.logger:
            self.logger.info("Test teardown - EasyDict tests completed")


# ============================================================================
# Unit Tests for FileUtils
# ============================================================================


class TestFileUtils(unittest.TestCase):
    """Unit tests for FileUtils class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info(f"Test setup - created temp directory: {self.temp_dir}")

    # ------------------------------------------------------------------------
    # makedir Tests
    # ------------------------------------------------------------------------

    def test_makedir_new_directory(self):
        """Test creating a new directory that doesn't exist."""
        if self.logger:
            self.logger.info("Testing makedir with new directory")

        new_dir = os.path.join(self.temp_dir, "new_subdir", "nested", "path")

        self.assertFalse(os.path.exists(new_dir))

        FileUtils.makedir(new_dir)

        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))

        if self.logger:
            self.logger.info(
                f"✅ makedir new directory test passed - created: {new_dir}"
            )

    def test_makedir_existing_directory(self):
        """Test calling makedir on an existing directory."""
        if self.logger:
            self.logger.info("Testing makedir with existing directory")

        existing_dir = os.path.join(self.temp_dir, "existing_dir")
        os.makedirs(existing_dir)

        self.assertTrue(os.path.exists(existing_dir))

        # This should not raise an exception
        FileUtils.makedir(existing_dir)

        self.assertTrue(os.path.exists(existing_dir))

        if self.logger:
            self.logger.info("✅ makedir existing directory test passed")

    def test_makedir_multiple_nested_directories(self):
        """Test creating multiple nested directories at once."""
        if self.logger:
            self.logger.info("Testing makedir with multiple nested directories")

        nested_path = os.path.join(
            self.temp_dir, "level1", "level2", "level3", "level4"
        )

        FileUtils.makedir(nested_path)

        self.assertTrue(os.path.exists(nested_path))
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, "level1")))
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, "level1", "level2")))
        self.assertTrue(
            os.path.isdir(os.path.join(self.temp_dir, "level1", "level2", "level3"))
        )

        if self.logger:
            self.logger.info("✅ makedir nested directories test passed")

    # ------------------------------------------------------------------------
    # makefile Tests
    # ------------------------------------------------------------------------

    def test_makefile_new_file(self):
        """Test creating a new file in an existing directory."""
        if self.logger:
            self.logger.info("Testing makefile with new file")

        filename = "test_file.txt"
        expected_path = os.path.join(self.temp_dir, filename)

        self.assertFalse(os.path.exists(expected_path))

        FileUtils.makefile(self.temp_dir, filename)

        self.assertTrue(os.path.exists(expected_path))
        self.assertTrue(os.path.isfile(expected_path))

        # Check file size (should be empty)
        self.assertEqual(os.path.getsize(expected_path), 0)

        if self.logger:
            self.logger.info(
                f"✅ makefile new file test passed - created: {expected_path}"
            )

    def test_makefile_in_nonexistent_directory(self):
        """Test creating a file in a directory that doesn't exist."""
        if self.logger:
            self.logger.info("Testing makefile in nonexistent directory")

        nonexistent_dir = os.path.join(self.temp_dir, "does", "not", "exist")
        filename = "test.txt"

        # This should fail because the directory doesn't exist
        with self.assertRaises(FileNotFoundError):
            FileUtils.makefile(nonexistent_dir, filename)

        if self.logger:
            self.logger.info("✅ makefile nonexistent directory test passed")

    def test_makefile_existing_file(self):
        """Test creating a file that already exists."""
        if self.logger:
            self.logger.info("Testing makefile with existing file")

        filename = "existing.txt"
        filepath = os.path.join(self.temp_dir, filename)

        # Create the file first
        with open(filepath, "w") as f:
            f.write("some content")

        original_size = os.path.getsize(filepath)
        self.assertGreater(original_size, 0)

        # Call makefile - should open in append mode, not overwrite
        FileUtils.makefile(self.temp_dir, filename)

        # File should still exist and have the same content
        self.assertTrue(os.path.exists(filepath))
        new_size = os.path.getsize(filepath)
        self.assertEqual(new_size, original_size)  # Size unchanged (appended nothing)

        if self.logger:
            self.logger.info("✅ makefile existing file test passed")

    def test_makefile_multiple_files(self):
        """Test creating multiple files."""
        if self.logger:
            self.logger.info("Testing makefile with multiple files")

        filenames = ["file1.txt", "file2.log", "file3.data", "file4.csv"]

        for filename in filenames:
            FileUtils.makefile(self.temp_dir, filename)

            expected_path = os.path.join(self.temp_dir, filename)
            self.assertTrue(os.path.exists(expected_path))
            self.assertTrue(os.path.isfile(expected_path))

        # Verify all files were created
        files_in_dir = os.listdir(self.temp_dir)
        self.assertEqual(len(files_in_dir), len(filenames))
        for filename in filenames:
            self.assertIn(filename, files_in_dir)

        if self.logger:
            self.logger.info(
                f"✅ makefile multiple files test passed - created {len(filenames)} files"
            )

    # ------------------------------------------------------------------------
    # Combined Tests
    # ------------------------------------------------------------------------

    def test_makedir_then_makefile(self):
        """Test creating a directory then a file inside it."""
        if self.logger:
            self.logger.info("Testing makedir then makefile sequence")

        new_dir = os.path.join(self.temp_dir, "new_directory")
        filename = "file_in_new_dir.txt"
        expected_path = os.path.join(new_dir, filename)

        # Directory doesn't exist yet
        self.assertFalse(os.path.exists(new_dir))

        # Create directory
        FileUtils.makedir(new_dir)
        self.assertTrue(os.path.exists(new_dir))

        # Create file in that directory
        FileUtils.makefile(new_dir, filename)
        self.assertTrue(os.path.exists(expected_path))
        self.assertTrue(os.path.isfile(expected_path))

        if self.logger:
            self.logger.info("✅ makedir then makefile test passed")

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        if self.logger:
            self.logger.info(f"Test teardown - removed temp directory: {self.temp_dir}")


def run_tests():
    """Run all EasyDict and FileUtils tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEasyDict))
    suite.addTests(loader.loadTestsFromTestCase(TestFileUtils))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
