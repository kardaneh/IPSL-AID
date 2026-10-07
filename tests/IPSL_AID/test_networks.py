# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.logger import Logger
from IPSL_AID.networks import (
    SongUNet,
    DhariwalUNet,
    VPPrecond,
    VEPrecond,
    EDMPrecond,
)

# ============================================================================
# Unit Tests for networks
# ============================================================================


class TestDiffusionNetworks(unittest.TestCase):
    """Unit tests for diffusion network architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.in_channels = 3
        self.cond_channels = 7
        self.out_channels = 3
        self.label_dim = 2

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info(f"Test setup complete - using device: {self.device}")

    def test_song_unet_without_diffusion(self):
        """Test SongUNet without diffusion model with diffusion_model=False."""
        if self.logger:
            self.logger.info("Testing SongUNet with diffusion_model=False")

        img_resolution = 64
        total_in_channels = self.in_channels + self.cond_channels

        model = SongUNet(
            img_resolution=img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            model_channels=32,
            channel_mult=[1, 2],
            attn_resolutions=[32],
            diffusion_model=False,  # Key parameter
        ).to(self.device)

        # Test forward pass - concatenate input and conditional image
        x = torch.randn(
            self.batch_size, self.in_channels, img_resolution, img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, img_resolution, img_resolution
        ).to(self.device)
        input_img = torch.cat([x, cond_img], dim=1)

        # No noise_labels needed when diffusion_model=False
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)
        if self.logger:
            self.logger.info(
                f"Testing SongUNet with diffusion_model=False - input shape: {input_img.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            # Pass None for noise_labels or skip it
            output = model(input_img, noise_labels=None, class_labels=class_labels)

        self.assertEqual(
            output.shape,
            (self.batch_size, self.out_channels, img_resolution, img_resolution),
        )
        if self.logger:
            self.logger.info(
                f"✅ SongUNet with diffusion_model=False test passed - output shape: {output.shape}"
            )

    def test_song_unet_rectangular_without_diffusion(self):
        """Test SongUNet with rectangular resolution and diffusion_model=False."""
        if self.logger:
            self.logger.info("Testing SongUNet rectangular with diffusion_model=False")

        img_resolution = (64, 32)
        total_in_channels = self.in_channels + self.cond_channels

        model = SongUNet(
            img_resolution=img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            model_channels=32,
            channel_mult=[1, 2],
            attn_resolutions=[16],
            diffusion_model=False,  # Key parameter
        ).to(self.device)

        # Test forward pass
        x = torch.randn(self.batch_size, self.in_channels, *img_resolution).to(
            self.device
        )
        cond_img = torch.randn(self.batch_size, self.cond_channels, *img_resolution).to(
            self.device
        )
        input_img = torch.cat([x, cond_img], dim=1)
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        if self.logger:
            self.logger.info(
                f"Testing SongUNet rectangular with diffusion_model=False - input shape: {input_img.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            output = model(input_img, noise_labels=None, class_labels=class_labels)

        self.assertEqual(
            output.shape, (self.batch_size, self.out_channels, *img_resolution)
        )
        if self.logger:
            self.logger.info(
                f"✅ SongUNet rectangular with diffusion_model=False test passed - output shape: {output.shape}"
            )

    def test_song_unet_square_resolution(self):
        """Test SongUNet with square resolution and diffusion model."""
        if self.logger:
            self.logger.info(
                "Testing SongUNet with diffusion_model=True - square resolution"
            )

        img_resolution = 64
        total_in_channels = self.in_channels + self.cond_channels

        model = SongUNet(
            img_resolution=img_resolution,
            in_channels=total_in_channels,  # Use total channels including conditional
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            model_channels=32,
            channel_mult=[1, 2],
            attn_resolutions=[32],
            embedding_type="positional",
        ).to(self.device)

        # Test forward pass - concatenate input and conditional image
        x = torch.randn(
            self.batch_size, self.in_channels, img_resolution, img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, img_resolution, img_resolution
        ).to(self.device)
        input_img = torch.cat(
            [x, cond_img], dim=1
        )  # Concatenate along channel dimension

        noise_labels = torch.randn(self.batch_size).to(self.device)
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)

        if self.logger:
            self.logger.info(
                f"Testing SongUNet square with diffusion_model=True - input shape: {input_img.shape}, noise_labels shape: {noise_labels.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            output = model(input_img, noise_labels, class_labels)

        self.assertEqual(
            output.shape,
            (self.batch_size, self.out_channels, img_resolution, img_resolution),
        )
        if self.logger:
            self.logger.info(
                f"✅ SongUNet square test passed - output shape: {output.shape}"
            )

    def test_song_unet_rectangular_resolution(self):
        """Test SongUNet with rectangular resolution and diffusion model."""
        if self.logger:
            self.logger.info(
                "Testing SongUNet with diffusion_model=True - rectangular resolution"
            )

        img_resolution = (64, 32)
        total_in_channels = self.in_channels + self.cond_channels

        model = SongUNet(
            img_resolution=img_resolution,
            in_channels=total_in_channels,  # Use total channels including conditional
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            model_channels=32,
            channel_mult=[1, 2],
            attn_resolutions=[16],
            embedding_type="fourier",
        ).to(self.device)

        # Test forward pass - concatenate input and conditional image
        x = torch.randn(self.batch_size, self.in_channels, *img_resolution).to(
            self.device
        )
        cond_img = torch.randn(self.batch_size, self.cond_channels, *img_resolution).to(
            self.device
        )
        input_img = torch.cat(
            [x, cond_img], dim=1
        )  # Concatenate along channel dimension

        noise_labels = torch.randn(self.batch_size).to(self.device)
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)
        if self.logger:
            self.logger.info(
                f"Testing SongUNet rectangular with diffusion_model=True - input shape: {input_img.shape}, noise_labels shape: {noise_labels.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            output = model(input_img, noise_labels, class_labels)

        self.assertEqual(
            output.shape, (self.batch_size, self.out_channels, *img_resolution)
        )
        if self.logger:
            self.logger.info(
                f"✅ SongUNet rectangular test passed - output shape: {output.shape}"
            )

    def test_dhariwal_unet_without_diffusion(self):
        """Test DhariwalUNet without diffusion model with diffusion_model=False."""
        if self.logger:
            self.logger.info("Testing DhariwalUNet with diffusion_model=False")

        img_resolution = (128, 64)
        total_in_channels = self.in_channels + self.cond_channels
        model = DhariwalUNet(
            img_resolution=img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,  # Keep label_dim > 0 to force proper emb shape
            model_channels=32,
            channel_mult=[1, 2],
            attn_resolutions=[32, 16],
            diffusion_model=False,  # Key parameter
        ).to(self.device)

        # Test forward pass - concatenate input and conditional image
        x = torch.randn(self.batch_size, self.in_channels, *img_resolution).to(
            self.device
        )
        cond_img = torch.randn(self.batch_size, self.cond_channels, *img_resolution).to(
            self.device
        )
        input_img = torch.cat([x, cond_img], dim=1)

        # No noise_labels needed when diffusion_model=False
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)
        if self.logger:
            self.logger.info(
                f"Testing DhariwalUNet with diffusion_model=False - input shape: {input_img.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            # Pass class_labels - this forces emb to have correct batch dimension
            output = model(input_img, class_labels=class_labels)

        self.assertEqual(
            output.shape, (self.batch_size, self.out_channels, *img_resolution)
        )
        if self.logger:
            self.logger.info(
                f"✅ DhariwalUNet with diffusion_model=False test passed - output shape: {output.shape}"
            )

    def test_dhariwal_unet(self):
        """Test DhariwalUNet architecture with conditional images and diffusion model."""
        if self.logger:
            self.logger.info(
                "Testing DhariwalUNet with diffusion_model=True and conditional images"
            )

        img_resolution = (128, 64)
        total_in_channels = self.in_channels + self.cond_channels

        model = DhariwalUNet(
            img_resolution=img_resolution,
            in_channels=total_in_channels,  # Use total channels including conditional
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            model_channels=32,
            channel_mult=[1, 2],
            attn_resolutions=[32, 16],
        ).to(self.device)

        # Test forward pass - concatenate input and conditional image
        x = torch.randn(self.batch_size, self.in_channels, *img_resolution).to(
            self.device
        )
        cond_img = torch.randn(self.batch_size, self.cond_channels, *img_resolution).to(
            self.device
        )
        input_img = torch.cat(
            [x, cond_img], dim=1
        )  # Concatenate along channel dimension

        noise_labels = torch.randn(self.batch_size).to(self.device)
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)
        if self.logger:
            self.logger.info(
                f"Testing DhariwalUNet with diffusion_model=True and conditional images - input shape: {input_img.shape}, noise_labels shape: {noise_labels.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            output = model(input_img, noise_labels, class_labels)

        self.assertEqual(
            output.shape, (self.batch_size, self.out_channels, *img_resolution)
        )
        if self.logger:
            self.logger.info(
                f"✅ DhariwalUNet test passed - output shape: {output.shape}"
            )

    def test_vp_preconditioner(self):
        """Test VPPrecond with conditional images."""
        if self.logger:
            self.logger.info("Testing VPPrecond")

        img_resolution = 64
        total_in_channels = self.in_channels + self.cond_channels

        model = VPPrecond(
            img_resolution=img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="SongUNet",
            model_channels=32,
            channel_mult=[1, 2],
        ).to(self.device)

        # Test forward pass
        x = torch.randn(
            self.batch_size, self.in_channels, img_resolution, img_resolution
        ).to(self.device)
        cond_img = torch.randn(
            self.batch_size, self.cond_channels, img_resolution, img_resolution
        ).to(self.device)
        sigma = torch.rand(self.batch_size).to(
            self.device
        )  # Random sigma values for each sample
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)
        if self.logger:
            self.logger.info(
                f"Testing VPPrecond - input shape: {x.shape}, sigma shape: {sigma.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            output = model(x, sigma, condition_img=cond_img, class_labels=class_labels)

        self.assertEqual(output.shape, x.shape)
        if self.logger:
            self.logger.info(f"✅ VPPrecond test passed - output shape: {output.shape}")

    def test_ve_preconditioner(self):
        """Test VEPrecond with conditional images."""
        if self.logger:
            self.logger.info("Testing VEPrecond")

        img_resolution = (64, 32)
        total_in_channels = self.in_channels + self.cond_channels

        model = VEPrecond(
            img_resolution=img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="SongUNet",
            model_channels=32,
            channel_mult=[1, 2],
        ).to(self.device)

        # Test forward pass
        x = torch.randn(self.batch_size, self.in_channels, *img_resolution).to(
            self.device
        )
        cond_img = torch.randn(self.batch_size, self.cond_channels, *img_resolution).to(
            self.device
        )
        sigma = torch.rand(self.batch_size).to(
            self.device
        )  # Random sigma values for each sample
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)
        if self.logger:
            self.logger.info(
                f"Testing VEPrecond - input shape: {x.shape}, sigma shape: {sigma.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            output = model(x, sigma, condition_img=cond_img, class_labels=class_labels)

        self.assertEqual(output.shape, x.shape)
        if self.logger:
            self.logger.info(f"✅ VEPrecond test passed - output shape: {output.shape}")

    def test_edm_preconditioner(self):
        """Test EDMPrecond with conditional images."""
        if self.logger:
            self.logger.info("Testing EDMPrecond")

        img_resolution = (128, 64)
        total_in_channels = self.in_channels + self.cond_channels

        model = EDMPrecond(
            img_resolution=img_resolution,
            in_channels=total_in_channels,
            out_channels=self.out_channels,
            label_dim=self.label_dim,
            use_fp16=False,
            model_type="DhariwalUNet",
            model_channels=32,
            channel_mult=[1, 2],
        ).to(self.device)

        # Test forward pass
        x = torch.randn(self.batch_size, self.in_channels, *img_resolution).to(
            self.device
        )
        cond_img = torch.randn(self.batch_size, self.cond_channels, *img_resolution).to(
            self.device
        )
        sigma = torch.rand(self.batch_size).to(
            self.device
        )  # Random sigma values for each sample
        class_labels = torch.randn(self.batch_size, self.label_dim, device=self.device)
        if self.logger:
            self.logger.info(
                f"Testing EDMPrecond - input shape: {x.shape}, sigma shape: {sigma.shape}, class_labels shape: {class_labels.shape}"
            )

        with torch.no_grad():
            output = model(x, sigma, condition_img=cond_img, class_labels=class_labels)

        self.assertEqual(output.shape, x.shape)
        if self.logger:
            self.logger.info(
                f"✅ EDMPrecond test passed - output shape: {output.shape}"
            )

    def test_parameter_counts(self):
        """Test that all models have reasonable parameter counts."""
        if self.logger:
            self.logger.info("Testing parameter counts")

        configs = [
            (
                "SongUNet-Small",
                SongUNet,
                {"model_channels": 32, "channel_mult": [1, 2, 2, 2]},
            ),
            (
                "SongUNet-Medium",
                SongUNet,
                {"model_channels": 64, "channel_mult": [1, 2, 2, 2]},
            ),
            (
                "DhariwalUNet-Small",
                DhariwalUNet,
                {"model_channels": 32, "channel_mult": [1, 2, 3, 4]},
            ),
            (
                "DhariwalUNet-Medium",
                DhariwalUNet,
                {"model_channels": 64, "channel_mult": [1, 2, 3, 4]},
            ),
        ]

        for name, model_class, kwargs in configs:
            with self.subTest(model=name):
                model = model_class(
                    img_resolution=64,
                    in_channels=self.in_channels
                    + self.cond_channels,  # Use total channels
                    out_channels=self.out_channels,
                    label_dim=self.label_dim,
                    **kwargs,
                ).to(self.device)

                total_params = sum(p.numel() for p in model.parameters())
                self.assertGreater(
                    total_params, 1000
                )  # Should have at least 1K parameters
                if self.logger:
                    self.logger.info(f"✅ {name} parameter count: {total_params:,}")

    def tearDown(self):
        """Clean up after tests."""
        if self.logger:
            self.logger.info("Network tests completed successfully")


def run_tests():
    """Run all network tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDiffusionNetworks))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
