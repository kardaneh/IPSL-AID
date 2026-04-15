# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
from IPSL_AID.utils import EasyDict
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.model import load_model_and_loss
from IPSL_AID.logger import Logger


# ============================================================================
# Unit Tests
# ============================================================================


class TestModelLoader(unittest.TestCase):
    """Unit tests for model and loss loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = 3
        self.cond_channels = 7
        self.out_channels = 3
        self.label_dim = 4
        self.img_resolution = (144, 360)

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info(f"Test setup complete - using device: {self.device}")

    def test_ddpmpp_vp_combination(self):
        """Test DDPM++ architecture with VP preconditioner."""
        if self.logger:
            self.logger.info("Testing DDPM++ + VP combination")

        opts = EasyDict(
            {
                "arch": "ddpmpp",
                "precond": "vp",
                "img_resolution": self.img_resolution,
                "in_channels": self.in_channels,
                "cond_channels": self.cond_channels,
                "out_channels": self.out_channels,
                "label_dim": self.label_dim,
                "use_fp16": False,
            }
        )

        model, loss_fn = load_model_and_loss(opts, self.logger, self.device)

        # Test forward pass
        x = torch.randn(self.batch_size, self.in_channels, *self.img_resolution).to(
            self.device
        )
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randint(0, self.label_dim, (self.batch_size, self.label_dim)).to(
            self.device
        )
        sigma = torch.tensor([0.1, 0.5], device=self.device)

        with torch.no_grad():
            output = model(x, sigma, condition_img=cond_img, class_labels=labels)

        self.assertEqual(output.shape, x.shape)

        # Test loss computation
        loss = loss_fn(model, x, conditional_img=cond_img, labels=labels)
        self.assertEqual(loss.shape, x.shape)

        if self.logger:
            self.logger.info(
                f"✅ DDPM++ + VP test passed - output shape: {output.shape}, loss shape: {loss.shape}"
            )

    def test_ncsnpp_ve_combination(self):
        """Test NCSN++ architecture with VE preconditioner."""
        if self.logger:
            self.logger.info("Testing NCSN++ + VE combination")

        opts = EasyDict(
            {
                "arch": "ncsnpp",
                "precond": "ve",
                "img_resolution": self.img_resolution,
                "in_channels": self.in_channels,
                "cond_channels": self.cond_channels,
                "out_channels": self.out_channels,
                "label_dim": self.label_dim,
                "use_fp16": False,
            }
        )

        model, loss_fn = load_model_and_loss(opts, self.logger, self.device)

        # Test forward pass
        x = torch.randn(self.batch_size, self.in_channels, *self.img_resolution).to(
            self.device
        )
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randint(0, self.label_dim, (self.batch_size, self.label_dim)).to(
            self.device
        )
        sigma = torch.tensor([0.1, 0.5], device=self.device)

        with torch.no_grad():
            output = model(x, sigma, condition_img=cond_img, class_labels=labels)

        self.assertEqual(output.shape, x.shape)

        # Test loss computation
        loss = loss_fn(model, x, conditional_img=cond_img, labels=labels)
        self.assertEqual(loss.shape, x.shape)

        if self.logger:
            self.logger.info(
                f"✅ NCSN++ + VE test passed - output shape: {output.shape}, loss shape: {loss.shape}"
            )

    def test_adm_edm_combination(self):
        """Test ADM architecture with EDM preconditioner."""
        if self.logger:
            self.logger.info("Testing ADM + EDM combination")

        opts = EasyDict(
            {
                "arch": "adm",
                "precond": "edm",
                "img_resolution": self.img_resolution,
                "in_channels": self.in_channels,
                "cond_channels": self.cond_channels,
                "out_channels": self.out_channels,
                "label_dim": self.label_dim,
                "use_fp16": False,
            }
        )

        model, loss_fn = load_model_and_loss(opts, self.logger, self.device)

        # Test forward pass
        x = torch.randn(self.batch_size, self.in_channels, *self.img_resolution).to(
            self.device
        )
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *self.img_resolution
        ).to(self.device)
        labels = torch.randint(0, self.label_dim, (self.batch_size, self.label_dim)).to(
            self.device
        )
        sigma = torch.tensor([0.1, 0.5], device=self.device)

        with torch.no_grad():
            output = model(x, sigma, condition_img=cond_img, class_labels=labels)

        self.assertEqual(output.shape, x.shape)

        # Test loss computation
        loss = loss_fn(model, x, conditional_img=cond_img, labels=labels)
        self.assertEqual(loss.shape, x.shape)

        if self.logger:
            self.logger.info(
                f"✅ ADM + EDM test passed - output shape: {output.shape}, loss shape: {loss.shape}"
            )

    def test_adm_unet_combination(self):
        """Using ADM architecture as direct U-Net without preconditioning."""
        if self.logger:
            self.logger.info("Testing ADM + UNet combination")

        input_channels = 5  # Example input channels
        opts = EasyDict(
            {
                "arch": "adm",
                "precond": "unet",
                "img_resolution": self.img_resolution,
                "in_channels": input_channels,
                "out_channels": self.out_channels,
                "label_dim": self.label_dim,
            }
        )

        model, loss_fn = load_model_and_loss(opts, self.logger, self.device)

        # Test forward pass
        x = torch.randn(self.batch_size, input_channels, *self.img_resolution).to(
            self.device
        )
        y = torch.randn(self.batch_size, self.out_channels, *self.img_resolution).to(
            self.device
        )
        labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        with torch.no_grad():
            output = model(x, class_labels=labels)

        self.assertEqual(output.shape, y.shape)

        # Test loss computation
        loss = loss_fn(model, y, x, labels=labels)
        self.assertEqual(loss.shape, ())

        if self.logger:
            self.logger.info(
                f"✅ ADM + UNet test passed - output shape: {output.shape}, loss shape: {loss.shape}"
            )

    def test_rectangular_resolution(self):
        """Test loader with rectangular resolution."""
        if self.logger:
            self.logger.info("Testing rectangular resolution")

        rectangular_res = (128, 64)
        opts = EasyDict(
            {
                "arch": "ddpmpp",
                "precond": "vp",
                "img_resolution": rectangular_res,
                "in_channels": self.in_channels,
                "cond_channels": self.cond_channels,
                "out_channels": self.out_channels,
                "label_dim": self.label_dim,
                "use_fp16": False,
            }
        )

        model, loss_fn = load_model_and_loss(opts, self.logger, self.device)

        # Test forward pass with rectangular resolution
        x = torch.randn(self.batch_size, self.in_channels, *rectangular_res).to(
            self.device
        )
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, *rectangular_res
        ).to(self.device)
        labels = torch.randint(0, self.label_dim, (self.batch_size, self.label_dim)).to(
            self.device
        )
        sigma = torch.tensor([0.1, 0.5], device=self.device)

        with torch.no_grad():
            output = model(x, sigma, condition_img=cond_img, class_labels=labels)

        self.assertEqual(output.shape, x.shape)

        if self.logger:
            self.logger.info(
                f"✅ Rectangular resolution test passed - output shape: {output.shape}"
            )

    def test_model_kwargs_override(self):
        """Test that model_kwargs can override default settings."""
        if self.logger:
            self.logger.info("Testing model_kwargs override")

        opts = EasyDict(
            {
                "arch": "ddpmpp",
                "precond": "vp",
                "img_resolution": self.img_resolution,
                "in_channels": self.in_channels,
                "cond_channels": self.cond_channels,
                "out_channels": self.out_channels,
                "label_dim": self.label_dim,
                "use_fp16": False,
                "model_kwargs": {
                    "model_channels": 64,  # Override default
                    "channel_mult": [1, 2],  # Override default
                },
            }
        )

        model, loss_fn = load_model_and_loss(opts, self.logger, self.device)

        # Verify the model has the overridden parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertLess(
            total_params, 10_000_000
        )  # Should be smaller with overridden settings

        if self.logger:
            self.logger.info(
                f"✅ Model kwargs override test passed - total params: {total_params:,}"
            )

    def test_no_conditional_channels(self):
        """Test loader without conditional channels."""
        if self.logger:
            self.logger.info("Testing without conditional channels")

        opts = EasyDict(
            {
                "arch": "ddpmpp",
                "precond": "vp",
                "img_resolution": self.img_resolution,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,  # No cond_channels specified
                "label_dim": self.label_dim,
                "use_fp16": False,
            }
        )

        model, loss_fn = load_model_and_loss(opts, self.logger, self.device)

        # Test forward pass without conditional image
        x = torch.randn(self.batch_size, self.in_channels, *self.img_resolution).to(
            self.device
        )
        labels = torch.randint(0, self.label_dim, (self.batch_size, self.label_dim)).to(
            self.device
        )
        sigma = torch.tensor([0.1, 0.5], device=self.device)

        with torch.no_grad():
            output = model(x, sigma, class_labels=labels)  # No condition_img

        self.assertEqual(output.shape, x.shape)

        if self.logger:
            self.logger.info(
                f"✅ No conditional channels test passed - output shape: {output.shape}"
            )

    def test_invalid_combinations(self):
        """Test that invalid combinations raise appropriate errors."""
        if self.logger:
            self.logger.info("Testing invalid combinations")

        # Test invalid architecture
        with self.assertRaises(ValueError):
            opts = EasyDict(
                {
                    "arch": "invalid_arch",
                    "precond": "vp",
                    "img_resolution": self.img_resolution,
                    "in_channels": self.in_channels,
                    "out_channels": self.out_channels,
                    "label_dim": self.label_dim,
                    "use_fp16": False,
                }
            )
            load_model_and_loss(opts, self.logger, self.device)

        # Test invalid preconditioner
        with self.assertRaises(ValueError):
            opts = EasyDict(
                {
                    "arch": "ddpmpp",
                    "precond": "invalid_precond",
                    "img_resolution": self.img_resolution,
                    "in_channels": self.in_channels,
                    "out_channels": self.out_channels,
                    "label_dim": self.label_dim,
                    "use_fp16": False,
                }
            )
            load_model_and_loss(opts, self.logger, self.device)

        if self.logger:
            self.logger.info("✅ Invalid combinations test passed")

    def tearDown(self):
        """Clean up after tests."""
        if self.logger:
            self.logger.info("Model tests completed successfully")


def run_tests():
    """Run all ModelLoader tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoader))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
