# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

from unittest.mock import Mock
import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.networks import VPPrecond, VEPrecond, EDMPrecond, DhariwalUNet
from IPSL_AID.loss import VPLoss, VELoss, EDMLoss, UnetLoss
from IPSL_AID.logger import Logger

# ============================================================================
# Unit Tests for losses
# ============================================================================


class TestLosses(unittest.TestCase):
    """Unit tests for diffusion models and loss functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = 3
        self.cond_channels = 7
        self.out_channels = 3
        self.label_dim = 2
        self.img_resolution = (64, 128)

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info(f"Test setup complete - using device: {self.device}")

    def test_vp_loss(self):
        """Test VP loss function."""
        if self.logger:
            self.logger.info("Testing VPLoss")

        # Create model and loss
        total_in_channels = self.in_channels + self.cond_channels
        model = VPPrecond(
            img_resolution=self.img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="SongUNet",
            model_channels=64,
            channel_mult=[1, 2],
        ).to(self.device)

        loss_fn = VPLoss()

        # Test data
        images = torch.randn(
            self.batch_size, self.in_channels, *self.img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        # Compute loss
        loss = loss_fn(model, images, conditional_img=cond_img, labels=labels)

        self.assertEqual(loss.shape, images.shape)
        self.assertGreater(loss.mean().item(), 0)
        if self.logger:
            self.logger.info(
                f"✅ VPLoss test passed - loss shape: {loss.shape}, mean: {loss.mean().item():.4f}"
            )

    def test_ve_loss(self):
        """Test VE loss function."""
        if self.logger:
            self.logger.info("Testing VELoss")

        # Create model and loss
        total_in_channels = self.in_channels + self.cond_channels
        model = VEPrecond(
            img_resolution=self.img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="SongUNet",
            model_channels=64,
            channel_mult=[1, 2],
        ).to(self.device)

        loss_fn = VELoss()

        # Test data
        images = torch.randn(
            self.batch_size, self.in_channels, *self.img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        # Compute loss
        loss = loss_fn(model, images, conditional_img=cond_img, labels=labels)

        self.assertEqual(loss.shape, images.shape)
        self.assertGreater(loss.mean().item(), 0)
        if self.logger:
            self.logger.info(
                f"✅ VELoss test passed - loss shape: {loss.shape}, mean: {loss.mean().item():.4f}"
            )

    def test_edm_loss(self):
        """Test EDM loss function."""
        if self.logger:
            self.logger.info("Testing EDMLoss")

        # Create model and loss
        total_in_channels = self.in_channels + self.cond_channels
        model = EDMPrecond(
            img_resolution=self.img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="DhariwalUNet",
            model_channels=64,
            channel_mult=[1, 2],
        ).to(self.device)

        loss_fn = EDMLoss()

        # Test data
        images = torch.randn(
            self.batch_size, self.in_channels, *self.img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        # Compute loss
        loss = loss_fn(model, images, conditional_img=cond_img, labels=labels)

        self.assertEqual(loss.shape, images.shape)
        self.assertGreater(loss.mean().item(), 0)
        if self.logger:
            self.logger.info(
                f"✅ EDMLoss test passed - loss shape: {loss.shape}, mean: {loss.mean().item():.4f}"
            )

    def test_unet_loss(self):
        """Test UnetLoss function."""
        if self.logger:
            self.logger.info("Testing UnetLoss")

        # Create UNet model (not diffusion-based)
        input_channels = 5  # Example input channels
        model = DhariwalUNet(
            img_resolution=self.img_resolution,
            in_channels=input_channels,  # No conditional channels needed
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            diffusion_model=False,
        ).to(self.device)

        loss_fn = UnetLoss()

        # Test data - UNet just reconstructs the input image
        images = torch.randn(self.batch_size, input_channels, *self.img_resolution).to(
            self.device
        )
        targets = torch.randn(
            self.batch_size, self.out_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        # Compute loss
        loss = loss_fn(model, targets, images, labels=labels)

        # Loss should be a scalar (not per-pixel like diffusion losses)
        self.assertEqual(loss.shape, ())  # Scalar tensor
        self.assertGreater(loss.item(), 0)
        if self.logger:
            self.logger.info(f"✅ UnetLoss test passed - loss value: {loss.item():.4f}")

    def test_loss_comparison(self):
        """Compare different loss functions on the same model."""
        if self.logger:
            self.logger.info("Testing loss function comparison")

        # Create model
        unet_model = DhariwalUNet(
            img_resolution=self.img_resolution,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            diffusion_model=False,
        ).to(self.device)

        total_in_channels = self.in_channels + self.cond_channels
        model = VPPrecond(
            img_resolution=self.img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="SongUNet",
            model_channels=64,
            channel_mult=[1, 2],
        ).to(self.device)

        # Create loss functions
        vp_loss = VPLoss()
        ve_loss = VELoss()
        edm_loss = EDMLoss()
        unet_loss_fn = UnetLoss()

        # Test data
        images = torch.randn(
            self.batch_size, self.in_channels, *self.img_resolution
        ).to(self.device)
        targets = torch.randn(
            self.batch_size, self.out_channels, *self.img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        # Compute losses
        vp_loss_val = vp_loss(model, images, conditional_img=cond_img, labels=labels)
        ve_loss_val = ve_loss(model, images, conditional_img=cond_img, labels=labels)
        edm_loss_val = edm_loss(model, images, conditional_img=cond_img, labels=labels)
        unet_loss_val = unet_loss_fn(unet_model, targets, images, labels=labels)

        # All losses should have same shape and be positive
        self.assertEqual(vp_loss_val.shape, ve_loss_val.shape)
        self.assertEqual(ve_loss_val.shape, edm_loss_val.shape)
        self.assertGreater(vp_loss_val.mean().item(), 0)
        self.assertGreater(ve_loss_val.mean().item(), 0)
        self.assertGreater(edm_loss_val.mean().item(), 0)
        self.assertGreater(unet_loss_val.item(), 0)

        if self.logger:
            self.logger.info("✅ Loss comparison test passed")
            self.logger.info(f"   └── VPLoss mean: {vp_loss_val.mean().item():.4f}")
            self.logger.info(f"   └── VELoss mean: {ve_loss_val.mean().item():.4f}")
            self.logger.info(f"   └── EDMLoss mean: {edm_loss_val.mean().item():.4f}")
            self.logger.info(f"   └── UnetLoss (scalar): {unet_loss_val.item():.4f}")

    def test_loss_with_augmentation(self):
        """Test loss functions with data augmentation."""
        if self.logger:
            self.logger.info("Testing loss with augmentation")

        # Mock augmentation pipe
        augment_pipe = Mock()
        augment_pipe.return_value = (
            torch.randn(self.batch_size, self.in_channels, *self.img_resolution).to(
                self.device
            ),
            torch.randint(0, 2, (self.batch_size, 1), device=self.device),
        )

        # Create model and loss
        total_in_channels = self.in_channels + self.cond_channels
        model = VPPrecond(
            img_resolution=self.img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="SongUNet",
            model_channels=64,
            channel_mult=[1, 2],
        ).to(self.device)

        loss_fn = VPLoss()

        # Test data
        images = torch.randn(
            self.batch_size, self.in_channels, *self.img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        # Compute loss with augmentation
        loss = loss_fn(
            model,
            images,
            conditional_img=cond_img,
            labels=labels,
            augment_pipe=augment_pipe,
        )

        self.assertEqual(loss.shape, images.shape)
        self.assertGreater(loss.mean().item(), 0)
        if self.logger:
            self.logger.info(
                f"✅ Loss with augmentation test passed - loss shape: {loss.shape}"
            )

    def test_loss_gradients(self):
        """Test that loss computation supports gradient computation."""
        if self.logger:
            self.logger.info("Testing loss gradients")

        # Create model and loss
        total_in_channels = self.in_channels + self.cond_channels
        model = VPPrecond(
            img_resolution=self.img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="SongUNet",
            model_channels=32,
            channel_mult=[1],
        ).to(self.device)

        loss_fn = VPLoss()

        # Test data with requires_grad
        images = torch.randn(
            self.batch_size,
            self.in_channels,
            *self.img_resolution,
            device=self.device,
            requires_grad=True,
        )
        cond_img = torch.randn(
            self.batch_size,
            self.cond_channels,
            *self.img_resolution,
            device=self.device,
        )
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        torch.cuda.empty_cache()

        # Compute loss and gradients
        loss = loss_fn(model, images, conditional_img=cond_img, labels=labels)
        total_loss = loss.mean()
        total_loss.backward()

        # Check that gradients were computed
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())

        if self.logger:
            self.logger.info(
                "✅ Loss gradients test passed - gradients computed successfully"
            )

    def tearDown(self):
        """Clean up after tests."""
        if self.logger:
            self.logger.info("Loss function tests completed successfully")


def run_tests():
    """Run all Losses test."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLosses))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
