# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import datetime
import os
import sys
import torch.nn as nn
import unittest
import tempfile
import shutil
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.utils import EasyDict
from IPSL_AID.logger import Logger
from IPSL_AID.model_utils import ModelUtils

# ============================================================================
# Test Model
# ============================================================================


class TestModel(nn.Module):
    """
    A model for testing purposes that includes a mix of convolutional and linear layers,
    as well as some non-trainable parameters (buffers). This model is designed to have a
    reasonable number of parameters for testing the ModelUtils methods without being too large.
    It includes batch normalization layers to add complexity and a dropout layer to demonstrate non-trainable parameters.
    """

    def __init__(self):
        super(TestModel, self).__init__()

        # Convolutional part
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Linear part
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

        # Non-trainable parameters (buffers)
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ============================================================================
# Unit Tests for ModelUtils
# ============================================================================


class TestModelUtils(unittest.TestCase):
    """Unit tests for ModelUtils class."""

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

        # Create a single balanced model for all tests
        self.model = TestModel()

        # Create a version with frozen layers for specific tests
        self.model_with_frozen = TestModel()
        # Freeze first conv layer
        for param in self.model_with_frozen.conv1.parameters():
            param.requires_grad = False

        # Create test optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Create test device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)
        self.model_with_frozen = self.model_with_frozen.to(self.device)

        if self.logger:
            self.logger.info(f"Test setup complete - using device: {self.device}")
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(
                f"Balanced model created with {total_params:,} total parameters"
            )

    # ------------------------------------------------------------------------
    # Parameter Counting Tests
    # ------------------------------------------------------------------------

    def test_get_parameter_number(self):
        """Test parameter counting functionality."""
        if self.logger:
            self.logger.info("Testing get_parameter_number method")

        param_counts = ModelUtils.get_parameter_number(self.model, self.logger)

        # Calculate expected counts manually
        expected_total = sum(p.numel() for p in self.model.parameters())
        expected_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.assertEqual(param_counts["Total"], expected_total)
        self.assertEqual(param_counts["Trainable"], expected_trainable)
        self.assertIn("Total", param_counts)
        self.assertIn("Trainable", param_counts)

        if self.logger:
            self.logger.info(
                f"✅ get_parameter_number test passed - Total: {param_counts['Total']:,}, Trainable: {param_counts['Trainable']:,}"
            )

    def test_get_parameter_number_with_frozen_layers(self):
        """Test parameter counting with frozen layers."""
        if self.logger:
            self.logger.info("Testing get_parameter_number with frozen layers")

        param_counts = ModelUtils.get_parameter_number(
            self.model_with_frozen, self.logger
        )

        total_params = sum(p.numel() for p in self.model_with_frozen.parameters())
        trainable_params = sum(
            p.numel() for p in self.model_with_frozen.parameters() if p.requires_grad
        )

        self.assertEqual(param_counts["Total"], total_params)
        self.assertEqual(param_counts["Trainable"], trainable_params)
        self.assertLess(param_counts["Trainable"], param_counts["Total"])

        if self.logger:
            self.logger.info(
                f"✅ Frozen layers test passed - Trainable: {param_counts['Trainable']:,} < Total: {param_counts['Total']:,}"
            )

    def test_count_parameters_by_layer(self):
        """Test layer-wise parameter counting."""
        if self.logger:
            self.logger.info("Testing count_parameters_by_layer method")

        layer_params = ModelUtils.count_parameters_by_layer(self.model, self.logger)

        # Verify structure
        self.assertIsInstance(layer_params, dict)

        # Should have multiple layers (at least 8)
        self.assertGreater(len(layer_params), 8)

        # Verify counts are positive
        for count in layer_params.values():
            self.assertGreater(count, 0)

        # Verify total sum matches overall parameter count
        total_from_layers = sum(layer_params.values())
        expected_total = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_from_layers, expected_total)

        if self.logger:
            self.logger.info(
                f"✅ count_parameters_by_layer test passed - {len(layer_params)} layers counted"
            )

    # ------------------------------------------------------------------------
    # Model Inspection Tests
    # ------------------------------------------------------------------------

    def test_print_model_layers(self):
        """Test model layer printing functionality."""
        if self.logger:
            self.logger.info("Testing print_model_layers method")

        # This mainly tests that the method runs without errors
        ModelUtils.print_model_layers(self.model, self.logger)

        # With logger=None, should print to console
        ModelUtils.print_model_layers(self.model, logger=None)

        if self.logger:
            self.logger.info("✅ print_model_layers test passed")

    def test_log_model_summary_without_input_shape(self):
        """Test model summary logging without input shape."""
        if self.logger:
            self.logger.info("Testing log_model_summary without input shape")

        ModelUtils.log_model_summary(self.model, logger=self.logger)

        if self.logger:
            self.logger.info("✅ log_model_summary without input shape test passed")

    def test_log_model_summary_with_input_shape(self):
        """Test model summary logging with input shape."""
        if self.logger:
            self.logger.info("Testing log_model_summary with input shape")

        input_shape = (4, 3, 32, 32)  # Batch, Channels, Height, Width
        ModelUtils.log_model_summary(self.model, input_shape, self.logger)

        if self.logger:
            self.logger.info("✅ log_model_summary with input shape test passed")

    # ------------------------------------------------------------------------
    # Checkpoint Save/Load Tests
    # ------------------------------------------------------------------------

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        if self.logger:
            self.logger.info("Testing save_checkpoint method")

        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pth.tar")

        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": 10,
            "loss": 0.1234,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        ModelUtils.save_checkpoint(state, checkpoint_path, self.logger)

        # Verify file exists
        self.assertTrue(os.path.exists(checkpoint_path))

        # Verify file size > 0
        self.assertGreater(os.path.getsize(checkpoint_path), 0)

        if self.logger:
            self.logger.info(
                f"✅ save_checkpoint test passed - file saved: {checkpoint_path}"
            )

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        if self.logger:
            self.logger.info("Testing load_checkpoint method")

        # First save a checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pth.tar")

        original_state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": 10,
            "loss": 0.1234,
        }

        torch.save(original_state, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create new model
        new_model = TestModel().to(self.device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Load state
        ModelUtils.load_checkpoint(
            loaded_checkpoint, new_model, new_optimizer, self.logger
        )

        # Verify model weights match
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1.to(self.device), p2.to(self.device)))

        if self.logger:
            self.logger.info(
                "✅ load_checkpoint test passed - model weights restored correctly"
            )

    def test_load_checkpoint_without_optimizer(self):
        """Test loading a checkpoint without optimizer."""
        if self.logger:
            self.logger.info("Testing load_checkpoint without optimizer")

        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint_no_opt.pth.tar")

        original_state = {
            "state_dict": self.model.state_dict(),
            "epoch": 10,
            "loss": 0.1234,
        }

        torch.save(original_state, checkpoint_path)
        loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)

        new_model = TestModel().to(self.device)
        ModelUtils.load_checkpoint(
            loaded_checkpoint, new_model, optimizer=None, logger=self.logger
        )

        # Verify model weights match
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1.to(self.device), p2.to(self.device)))

        if self.logger:
            self.logger.info("✅ load_checkpoint without optimizer test passed")

    def test_load_training_checkpoint(self):
        """Test loading a comprehensive training checkpoint."""
        if self.logger:
            self.logger.info("Testing load_training_checkpoint method")

        checkpoint_path = os.path.join(self.temp_dir, "training_checkpoint.pth.tar")

        # Create comprehensive checkpoint
        checkpoint_state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": 25,
            "samples_processed": 10000,
            "batches_processed": 500,
            "best_val_loss": 0.05,
            "best_epoch": 24,
            "train_loss_history": [1.0, 0.8, 0.6, 0.4],
            "valid_loss_history": [0.9, 0.7, 0.5, 0.3],
            "valid_metrics_history": {"accuracy": [0.5, 0.6, 0.7, 0.8]},
            "val_loss": 0.3,
            "train_loss": 0.4,
            "checkpoint_type": "epoch",
            "timestamp": datetime.datetime.now().isoformat(),
        }

        torch.save(checkpoint_state, checkpoint_path)

        new_model = TestModel().to(self.device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        epoch, samples, batches, best_loss, best_epoch, checkpoint = (
            ModelUtils.load_training_checkpoint(
                checkpoint_path, new_model, new_optimizer, self.device, self.logger
            )
        )

        self.assertEqual(epoch, 25)
        self.assertEqual(samples, 10000)
        self.assertEqual(batches, 500)
        self.assertEqual(best_loss, 0.05)
        self.assertEqual(best_epoch, 24)
        self.assertIsNotNone(checkpoint)

        if self.logger:
            self.logger.info("✅ load_training_checkpoint test passed")

    def test_load_training_checkpoint_nonexistent(self):
        """Test loading a nonexistent training checkpoint."""
        if self.logger:
            self.logger.info("Testing load_training_checkpoint with nonexistent file")

        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pth.tar")

        new_model = TestModel().to(self.device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        epoch, samples, batches, best_loss, best_epoch, checkpoint = (
            ModelUtils.load_training_checkpoint(
                nonexistent_path, new_model, new_optimizer, self.device, self.logger
            )
        )

        self.assertIsNone(epoch)
        self.assertEqual(samples, 0)
        self.assertEqual(batches, 0)
        self.assertEqual(best_loss, float("inf"))
        self.assertEqual(best_epoch, 0)
        self.assertIsNone(checkpoint)

        if self.logger:
            self.logger.info("✅ load_training_checkpoint nonexistent file test passed")

    # ------------------------------------------------------------------------
    # Training Checkpoint Save Tests
    # ------------------------------------------------------------------------

    def test_save_training_checkpoint_epoch_type(self):
        """Test saving training checkpoint with epoch type."""
        if self.logger:
            self.logger.info("Testing save_training_checkpoint with epoch type")

        # Setup arguments
        args = argparse.Namespace()
        args.prefix = "test_run"
        args.save_checkpoint_name = "model"
        args.num_epochs = 50

        paths = EasyDict({"checkpoints": self.temp_dir})

        train_loss_history = [1.0, 0.8, 0.6, 0.4]
        valid_loss_history = [0.9, 0.7, 0.5, 0.3]
        valid_metrics_history = {"accuracy": [0.5, 0.6, 0.7, 0.8]}

        ModelUtils.save_training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=10,
            samples_processed=5000,
            batches_processed=250,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            valid_metrics_history=valid_metrics_history,
            best_val_loss=0.3,
            best_epoch=9,
            avg_val_loss=0.3,
            avg_epoch_loss=0.4,
            args=args,
            paths=paths,
            logger=self.logger,
            checkpoint_type="epoch",
            save_full_model=True,
        )

        # Verify files exist
        checkpoint_file = os.path.join(
            self.temp_dir, "test_run_epoch0010_model.pth.tar"
        )
        full_model_file = os.path.join(
            self.temp_dir, "test_run_epoch0010_model_full.pth"
        )
        self.assertTrue(os.path.exists(checkpoint_file))
        self.assertTrue(os.path.exists(full_model_file))

        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 10)
        self.assertEqual(checkpoint["samples_processed"], 5000)
        self.assertEqual(checkpoint["batches_processed"], 250)
        self.assertEqual(checkpoint["best_val_loss"], 0.3)
        self.assertEqual(checkpoint["best_epoch"], 9)
        self.assertEqual(checkpoint["checkpoint_type"], "epoch")

        if self.logger:
            self.logger.info(
                f"✅ epoch type checkpoint test passed - files: {checkpoint_file}"
            )

    def test_save_training_checkpoint_best_type(self):
        """Test saving training checkpoint with best type."""
        if self.logger:
            self.logger.info("Testing save_training_checkpoint with best type")

        args = argparse.Namespace()
        args.prefix = "test_run"
        args.save_checkpoint_name = "model"

        paths = EasyDict({"checkpoints": self.temp_dir})

        ModelUtils.save_training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=15,
            samples_processed=7500,
            batches_processed=375,
            train_loss_history=[],
            valid_loss_history=[],
            valid_metrics_history={},
            best_val_loss=0.25,
            best_epoch=15,
            avg_val_loss=0.25,
            avg_epoch_loss=0.3,
            args=args,
            paths=paths,
            logger=self.logger,
            checkpoint_type="best",
            save_full_model=True,
        )

        # Verify file naming
        checkpoint_file = os.path.join(self.temp_dir, "test_run_best_model.pth.tar")
        self.assertTrue(os.path.exists(checkpoint_file))

        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        self.assertEqual(checkpoint["epoch"], 15)
        self.assertEqual(checkpoint["best_val_loss"], 0.25)
        self.assertEqual(checkpoint["best_epoch"], 15)
        self.assertEqual(checkpoint["checkpoint_type"], "best")

        if self.logger:
            self.logger.info("✅ best type checkpoint test passed")

    def test_save_training_checkpoint_final_type(self):
        """Test saving training checkpoint with final type."""
        if self.logger:
            self.logger.info("Testing save_training_checkpoint with final type")

        args = argparse.Namespace()
        args.prefix = "test_run"
        args.save_checkpoint_name = "model"
        args.num_epochs = 20

        paths = EasyDict({"checkpoints": self.temp_dir})

        ModelUtils.save_training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=20,
            samples_processed=10000,
            batches_processed=500,
            train_loss_history=[],
            valid_loss_history=[],
            valid_metrics_history={},
            best_val_loss=0.2,
            best_epoch=18,
            avg_val_loss=0.22,
            avg_epoch_loss=0.21,
            args=args,
            paths=paths,
            logger=self.logger,
            checkpoint_type="final",
            save_full_model=True,
        )

        # Verify file naming includes num_epochs
        checkpoint_file = os.path.join(
            self.temp_dir, "test_run_final_model_epoch20.pth.tar"
        )
        self.assertTrue(os.path.exists(checkpoint_file))

        if self.logger:
            self.logger.info("✅ final type checkpoint test passed")

    def test_save_training_checkpoint_samples_type(self):
        """Test saving training checkpoint with samples type."""
        if self.logger:
            self.logger.info("Testing save_training_checkpoint with samples type")

        args = argparse.Namespace()
        args.prefix = "test_run"
        args.save_checkpoint_name = "model"

        paths = EasyDict({"checkpoints": self.temp_dir})

        ModelUtils.save_training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            samples_processed=2500,
            batches_processed=125,
            train_loss_history=[],
            valid_loss_history=[],
            valid_metrics_history={},
            best_val_loss=0.4,
            best_epoch=4,
            avg_val_loss=0.4,
            avg_epoch_loss=0.45,
            args=args,
            paths=paths,
            logger=self.logger,
            checkpoint_type="samples",
            save_full_model=True,
        )

        # Verify file naming includes samples
        checkpoint_file = os.path.join(
            self.temp_dir, "test_run_epoch0005_samples2500_model.pth.tar"
        )
        self.assertTrue(os.path.exists(checkpoint_file))

        if self.logger:
            self.logger.info("✅ samples type checkpoint test passed")

    # ------------------------------------------------------------------------
    # Emergency Checkpoint Tests
    # ------------------------------------------------------------------------

    def test_save_emergency_checkpoint(self):
        """Test saving emergency checkpoint."""
        if self.logger:
            self.logger.info("Testing save_emergency_checkpoint method")

        args = argparse.Namespace()
        args.prefix = "test_run"
        args.save_checkpoint_name = "model"

        paths = EasyDict({"checkpoints": self.temp_dir})

        ModelUtils.save_emergency_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=7,
            samples_processed=3500,
            batches_processed=175,
            train_loss_history=[1.0, 0.8, 0.6],
            valid_loss_history=[0.9, 0.7, 0.5],
            valid_metrics_history={"accuracy": [0.5, 0.6, 0.7]},
            args=args,
            paths=paths,
            logger=self.logger,
            reason="test_crash",
        )

        checkpoint_file = os.path.join(
            self.temp_dir, "test_run_emergency_test_crash_model.pth.tar"
        )
        # Check that files were created
        self.assertTrue(os.path.exists(checkpoint_file))

        # Verify emergency naming includes reason
        self.assertIn("emergency_test_crash", checkpoint_file)

        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        self.assertEqual(checkpoint["checkpoint_type"], "emergency_test_crash")
        self.assertEqual(checkpoint["epoch"], 7)

        if self.logger:
            self.logger.info(
                f"✅ emergency checkpoint test passed - file: {checkpoint_file}"
            )

    # ------------------------------------------------------------------------
    # Integration and Edge Case Tests
    # ------------------------------------------------------------------------

    def test_full_checkpoint_cycle(self):
        """Test complete checkpoint save-load cycle with training state."""
        if self.logger:
            self.logger.info("Testing full checkpoint cycle")

        # Setup
        args = argparse.Namespace()
        args.prefix = "cycle_test"
        args.save_checkpoint_name = "model"
        args.num_epochs = 100

        paths = EasyDict({"checkpoints": self.temp_dir})

        # Training data
        train_loss_history = [1.0, 0.9, 0.8, 0.7, 0.6]
        valid_loss_history = [0.95, 0.85, 0.75, 0.65, 0.55]
        valid_metrics_history = {"accuracy": [0.4, 0.5, 0.6, 0.7, 0.8]}

        # Save checkpoint
        ModelUtils.save_training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            samples_processed=2500,
            batches_processed=125,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            valid_metrics_history=valid_metrics_history,
            best_val_loss=0.55,
            best_epoch=5,
            avg_val_loss=0.55,
            avg_epoch_loss=0.6,
            args=args,
            paths=paths,
            logger=self.logger,
            checkpoint_type="epoch",
            save_full_model=True,
        )

        checkpoint_file = os.path.join(
            self.temp_dir, "cycle_test_epoch0005_model.pth.tar"
        )
        full_model_file = os.path.join(
            self.temp_dir, "cycle_test_epoch0005_model_full.pth"
        )

        # Check that files were created
        self.assertTrue(os.path.exists(checkpoint_file))
        self.assertTrue(os.path.exists(full_model_file))

        # Create new model and optimizer for loading
        new_model = TestModel().to(self.device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Load checkpoint
        epoch, samples, batches, best_loss, best_epoch, checkpoint = (
            ModelUtils.load_training_checkpoint(
                checkpoint_file, new_model, new_optimizer, self.device, self.logger
            )
        )

        # Verify all data was preserved
        self.assertEqual(epoch, 5)
        self.assertEqual(samples, 2500)
        self.assertEqual(batches, 125)
        self.assertEqual(best_loss, 0.55)
        self.assertEqual(best_epoch, 5)

        # Verify histories
        self.assertEqual(checkpoint["train_loss_history"], train_loss_history)
        self.assertEqual(checkpoint["valid_loss_history"], valid_loss_history)
        self.assertEqual(checkpoint["valid_metrics_history"], valid_metrics_history)

        # Verify model weights match
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1.to(self.device), p2.to(self.device)))

        if self.logger:
            self.logger.info("✅ full checkpoint cycle test passed")

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        if self.logger:
            self.logger.info(f"Test teardown - removed temp directory: {self.temp_dir}")


def run_tests():
    """Run all ModelUtils tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelUtils))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
