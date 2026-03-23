# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# ============================================================================
# ORIGINAL WORK (NVIDIA)
# ============================================================================
# This work is a derivative of "Elucidating the Design Space of
# Diffusion-Based Generative Models" by NVIDIA CORPORATION & AFFILIATES.
#
# Original work: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
# Original license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# Original source: https://github.com/NVlabs/edm
#
# ============================================================================
# MODIFICATIONS AND ADDITIONS (IPSL / CNRS / Sorbonne University)
# ============================================================================
# Modifications to loss functions include:
#   1. Added conditional image support to all loss functions
#      - Extended VPLoss, VELoss, EDMLoss with conditional_img parameter
#      - Modified __call__ method to pass conditional images to the network
#      - Updated documentation to reflect conditional image usage
#
#   2. Added UnetLoss class for non-diffusion UNet training
#      - Created new loss class for direct image-to-image prediction tasks
#      - Supports MSE, L1, and Smooth L1 loss types
#      - Compatible with UNet architectures (DhariwalUNet, SongUNet)
#      - Includes support for data augmentation and conditioning
#
#   3. Enhanced documentation
#      - Added comprehensive docstrings for all classes and methods
#      - Included mathematical formulas and training procedures
#      - Added usage examples and parameter descriptions
#
#   4. Added comprehensive unit tests
#      - Created TestLosses class with test methods for all loss functions
#      - Added tests for loss gradients and numerical stability
#      - Added tests with data augmentation
#      - Added loss comparison tests
#      - Added rectangular resolution support tests
#
#   5. Code quality improvements
#      - Added type hints for better code clarity
#      - Improved variable naming
#      - Added input validation where appropriate
#
# ============================================================================
# LICENSE
# ============================================================================
# This derivative work is licensed under the same terms as the original:
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# ============================================================================
# ACKNOWLEDGMENTS
# ============================================================================
# We thank the NVIDIA team for their excellent work on EDM and for making it
# available under an open license that enables further research and development.

"""
Diffusion model loss functions and testing utilities.

This module implements various loss functions for diffusion models including:
- VPLoss: Variance Preserving loss from Score-Based Generative Modeling
- VELoss: Variance Exploding loss from Score-Based Generative Modeling
- EDMLoss: Improved loss from Elucidating the Design Space of Diffusion-Based Generative Models
"""

import torch
import unittest
from unittest.mock import Mock
from IPSL_AID.networks import VPPrecond, VEPrecond, EDMPrecond, DhariwalUNet

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation


class VPLoss:
    """
    Loss function for Variance Preserving (VP) formulation diffusion models.

    This class implements the loss function for the Variance Preserving SDE
    formulation of diffusion models. It follows the continuous-time training
    objective from score-based generative modeling through stochastic differential
    equations.

    Parameters
    ----------
    beta_d : float, optional
        Maximum β parameter controlling the extent of the noise schedule.
        Larger values lead to faster noise increase. Default is 19.9.
    beta_min : float, optional
        Minimum β parameter controlling the initial slope of the noise schedule.
        Default is 0.1.
    epsilon_t : float, optional
        Minimum time value threshold to avoid numerical issues near t=0.
        Default is 1e-5.

    Attributes
    ----------
    beta_d : float
        Maximum β parameter for noise schedule.
    beta_min : float
        Minimum β parameter for noise schedule.
    epsilon_t : float
        Minimum time threshold.

    Methods
    -------
    __call__(net, images, conditional_img=None, labels=None, augment_pipe=None)
        Compute the VP loss for a batch of images.
    sigma(t)
        Compute noise level sigma for given timestep t.

    Notes
    -----
    - The loss is based on denoising score matching: E[λ(t) * ||D_θ(x_t, t) - x_0||²]
    - The weighting function λ(t) = 1/σ(t)² gives equal importance to all noise levels.
    - Time t is uniformly sampled between [epsilon_t, 1] during training.
    - This loss corresponds to training the model to predict the clean image x_0
      from noisy input x_t = x_0 + σ(t)·ε.

    References
    ----------
    - Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", 2020.
    """

    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        """
        Initialize the VPLoss function.

        Parameters
        ----------
        beta_d : float, optional
            Maximum β parameter for noise schedule.
            Default is 19.9.
        beta_min : float, optional
            Minimum β parameter for noise schedule.
            Default is 0.1.
        epsilon_t : float, optional
            Minimum time threshold.
            Default is 1e-5.
        """
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(
        self, net, images, conditional_img=None, labels=None, augment_pipe=None
    ):
        """
        Compute the VP loss for a batch of images.

        Parameters
        ----------
        net : torch.nn.Module
            The diffusion model network (typically a preconditioned U-Net).
            Should accept inputs (x, sigma, condition_img, labels, augment_labels).
        images : torch.Tensor
            Clean input images of shape (batch_size, channels, height, width).
        conditional_img : torch.Tensor, optional
            Conditional images for guided generation. Should have same spatial
            dimensions as `images`. Default is None.
        labels : torch.Tensor, optional
            Class labels for conditional generation of shape (batch_size,) or
            (batch_size, label_dim). Default is None.
        augment_pipe : callable, optional
            Data augmentation pipeline that takes images and returns augmented
            images and augmentation labels. Default is None.

        Returns
        -------
        torch.Tensor
            Loss values for each image and channel of shape
            (batch_size, channels, height, width). Typically reduced via mean()
            for training.

        Notes
        -----
        - The training procedure:
          1. Sample time t ~ Uniform[epsilon_t, 1]
          2. Compute noise level σ(t)
          3. Generate noisy images: x_t = x_0 + σ(t)·ε where ε ~ N(0, I)
          4. Compute model prediction D_θ(x_t, t)
          5. Calculate weighted MSE loss: λ(t) * ||D_θ(x_t, t) - x_0||²
        - The weight λ(t) = 1/σ(t)² ensures balanced learning across noise levels.
        - Data augmentation is applied before adding noise if augment_pipe is provided.
        """
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        """
        Compute noise level sigma for given timestep t.

        Parameters
        ----------
        t : torch.Tensor or float
            Timestep value(s) in [epsilon_t, 1].

        Returns
        -------
        torch.Tensor
            Noise level sigma corresponding to t, with same shape as input.

        Notes
        -----
        The noise schedule follows:
        σ(t) = sqrt(exp(0.5*β_d*t² + β_min*t) - 1)

        This ensures smooth transition from low to high noise levels, with
        σ(0) ≈ 0 and σ(1) determined by β_d and β_min.
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation


class VELoss:
    """
    Loss function for Variance Exploding (VE) formulation diffusion models.

    This class implements the loss function for the Variance Exploding SDE
    formulation of diffusion models. It follows the continuous-time training
    objective from score-based generative modeling through stochastic differential
    equations.

    Parameters
    ----------
    sigma_min : float, optional
        Minimum noise level. Controls the lower bound of the noise schedule.
        Smaller values allow modeling finer details. Default is 0.02.
    sigma_max : float, optional
        Maximum noise level. Controls the upper bound of the noise schedule.
        Larger values allow modeling broader structure. Default is 100.

    Attributes
    ----------
    sigma_min : float
        Minimum noise level for the geometric schedule.
    sigma_max : float
        Maximum noise level for the geometric schedule.

    Methods
    -------
    __call__(net, images, conditional_img=None, labels=None, augment_pipe=None)
        Compute the VE loss for a batch of images.

    Notes
    -----
    - The VE formulation uses a geometric noise schedule: σ(t) = σ_min * (σ_max/σ_min)^t
    - Time t is uniformly sampled between [0, 1] during training.
    - The weighting function λ(t) = 1/σ(t)² gives more emphasis to lower noise levels.
    - This corresponds to training the model to predict the clean image x_0 from
      noisy input x_t = x_0 + σ(t)·ε.
    - The geometric schedule provides a simple and effective way to span a wide
      range of noise levels with a single parameter.

    References
    ----------
    - Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", 2020.
    """

    def __init__(self, sigma_min=0.02, sigma_max=100):
        """
        Initialize the VELoss function.

        Parameters
        ----------
        sigma_min : float, optional
            Minimum noise level for the geometric schedule.
            Default is 0.02.
        sigma_max : float, optional
            Maximum noise level for the geometric schedule.
            Default is 100.
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(
        self, net, images, conditional_img=None, labels=None, augment_pipe=None
    ):
        """
        Compute the VE loss for a batch of images.

        Parameters
        ----------
        net : torch.nn.Module
            The diffusion model network (typically a preconditioned U-Net).
            Should accept inputs (x, sigma, condition_img, labels, augment_labels).
        images : torch.Tensor
            Clean input images of shape (batch_size, channels, height, width).
        conditional_img : torch.Tensor, optional
            Conditional images for guided generation. Should have same spatial
            dimensions as `images`. Default is None.
        labels : torch.Tensor, optional
            Class labels for conditional generation of shape (batch_size,) or
            (batch_size, label_dim). Default is None.
        augment_pipe : callable, optional
            Data augmentation pipeline that takes images and returns augmented
            images and augmentation labels. Default is None.

        Returns
        -------
        torch.Tensor
            Loss values for each image and channel of shape
            (batch_size, channels, height, width). Typically reduced via mean()
            for training.

        Notes
        -----
        - The training procedure:
          1. Sample time t ~ Uniform[0, 1]
          2. Compute noise level σ = σ_min * (σ_max/σ_min)^t
          3. Generate noisy images: x_t = x_0 + σ·ε where ε ~ N(0, I)
          4. Compute model prediction D_θ(x_t, t)
          5. Calculate weighted MSE loss: λ(t) * ||D_θ(x_t, t) - x_0||²
        - The weight λ(t) = 1/σ² ensures higher weighting for lower noise levels.
        - Data augmentation is applied before adding noise if augment_pipe is provided.
        - The geometric noise schedule spans orders of magnitude from σ_min to σ_max.
        """
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


class EDMLoss:
    """
    EDM (Elucidating Diffusion Models) loss function for diffusion models.

    This class implements the improved loss function from the EDM paper, which
    uses a log-normal distribution for noise level sampling and an optimized
    weighting scheme for better training stability and sample quality.

    Parameters
    ----------
    P_mean : float, optional
        Mean parameter for the log-normal distribution of sigma.
        Controls the center of the noise level distribution. Default is -1.2.
    P_std : float, optional
        Standard deviation parameter for the log-normal distribution of sigma.
        Controls the spread of the noise level distribution. Default is 1.2.
    sigma_data : float, optional
        Standard deviation of the training data. Used in the weighting function
        to balance the loss across noise levels. Default is 1.0.

    Attributes
    ----------
    P_mean : float
        Mean of log-normal distribution for sigma sampling.
    P_std : float
        Standard deviation of log-normal distribution for sigma sampling.
    sigma_data : float
        Training data standard deviation.

    Methods
    -------
    __call__(net, images, conditional_img=None, labels=None, augment_pipe=None)
        Compute the EDM loss for a batch of images.

    Notes
    -----
    - The EDM loss uses a log-normal distribution for sigma: σ ~ logNormal(P_mean, P_std)
    - The weighting function: λ(σ) = (σ² + σ_data²) / (σ·σ_data)²
    - This weighting minimizes the variance of the loss gradient, leading to
      more stable training and faster convergence.
    - The loss corresponds to training the model to predict the clean image x_0
      from noisy input x_t = x_0 + σ·ε.
    - The log-normal distribution provides a better prior for noise levels
      compared to uniform sampling.

    References
    ----------
    - Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022.
    """

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        """
        Initialize the EDMLoss function.

        Parameters
        ----------
        P_mean : float, optional
            Mean parameter for log-normal distribution.
            Default is -1.2.
        P_std : float, optional
            Standard deviation parameter for log-normal distribution.
            Default is 1.2.
        sigma_data : float, optional
            Standard deviation of training data.
            Default is 1.0.
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self, net, images, conditional_img=None, labels=None, augment_pipe=None
    ):
        """
        Compute the EDM loss for a batch of images.

        Parameters
        ----------
        net : torch.nn.Module
            The diffusion model network (typically an EDM-preconditioned U-Net).
            Should accept inputs (x, sigma, condition_img, labels, augment_labels).
        images : torch.Tensor
            Clean input images of shape (batch_size, channels, height, width).
        conditional_img : torch.Tensor, optional
            Conditional images for guided generation. Should have same spatial
            dimensions as `images`. Default is None.
        labels : torch.Tensor, optional
            Class labels for conditional generation of shape (batch_size,) or
            (batch_size, label_dim). Default is None.
        augment_pipe : callable, optional
            Data augmentation pipeline that takes images and returns augmented
            images and augmentation labels. Default is None.

        Returns
        -------
        torch.Tensor
            Loss values for each image and channel of shape
            (batch_size, channels, height, width). Typically reduced via mean()
            for training.

        Notes
        -----
        - The training procedure:
          1. Sample log(sigma) ~ Normal(P_mean, P_std)
          2. Compute noise level σ = exp(log(sigma))
          3. Generate noisy images: x_t = x_0 + σ·ε where ε ~ N(0, I)
          4. Compute model prediction D_θ(x_t, σ)
          5. Calculate weighted MSE loss: λ(σ) * ||D_θ(x_t, σ) - x_0||²
        - The weight λ(σ) = (σ² + σ_data²) / (σ·σ_data)² minimizes gradient variance.
        - Data augmentation is applied before adding noise if augment_pipe is provided.
        - The log-normal distribution provides a natural prior for noise levels,
          avoiding the need for manual schedule design.
        """
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Loss function for UNet architectures.


class UnetLoss:
    """
    Simple UNet loss function for direct image-to-image prediction.

    This loss function works with UNet models that predict images directly,
    without any diffusion noise process. It's a standard supervised loss
    for image generation/transformation tasks such as segmentation,
    denoising, super-resolution, or autoencoding.

    Parameters
    ----------
    loss_type : str, optional
        Type of loss function to use:
        - ``mse``: Mean Squared Error (L2 loss)
        - ``l1``: Mean Absolute Error (L1 loss)
        - ``smooth_l1``: Smooth L1 loss (Huber loss)
        Default is ``mse``.
    reduction : str, optional
        Reduction method for the loss:
        - ``mean``: Average the loss over all elements
        - ``sum``: Sum the loss over all elements
        - ``none``: Return loss per element
        Default is ``mean``.

    Attributes
    ----------
    loss_type : str
        Type of loss function.
    reduction : str
        Reduction method.
    loss_fn : torch.nn.Module
        PyTorch loss function instance.

    Raises
    ------
    ValueError
        If an unknown ``loss_type`` is provided.

    Notes
    -----
    - The loss computes the discrepancy between the model's output and the input image.
    - This is suitable for autoencoder-style tasks where the model learns to
      reconstruct the input.
    - For conditional generation, labels can be provided to the model.
    - Data augmentation can be applied via `augment_pipe`.
    """

    def __init__(self, loss_type="mse", reduction="mean"):
        """
        Initialize the UnetLoss function.

        Parameters
        ----------
        loss_type : str, optional
            Type of loss function.
            Default is ``mse``.
        reduction : str, optional
            Reduction method.
            Default is ``mean``.
        """
        self.loss_type = loss_type
        self.reduction = reduction

        # Initialize loss function
        if loss_type == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        elif loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss(reduction=reduction)
        elif loss_type == "smooth_l1":
            self.loss_fn = torch.nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def __call__(self, net, targets, images, labels=None, augment_pipe=None):
        """
        Compute UNet loss.

        Parameters
        ----------
        net : torch.nn.Module
            Neural network model (DhariwalUNet or SongUNet) that outputs an image
            of the same shape as input.
        images : torch.Tensor
            Input images tensor of shape (batch_size, channels, height, width).
        labels : torch.Tensor, optional
            Class labels for conditional generation of shape (batch_size,) or
            (batch_size, label_dim). Default is None.
        augment_pipe : callable, optional
            Data augmentation pipeline that takes images and returns augmented
            images and augmentation labels. Default is None.

        Returns
        -------
        torch.Tensor
            Computed loss value (scalar if reduction is ``mean`` or ``sum``,
            otherwise tensor of shape (batch_size, channels, height, width)).

        Notes
        -----
        - The model is called with the input images and optional labels.
        - The loss is computed between the model output and the (augmented) input images.
        - This setup is typical for autoencoder or denoising tasks where the
          model learns to reconstruct the input.
        """
        # Apply data augmentation if provided
        if augment_pipe is not None:
            images, augment_labels = augment_pipe(images)
        else:
            augment_labels = None

        # Get model prediction
        model_out = net(images, class_labels=labels, augment_labels=augment_labels)

        # Simple loss: compare model output with input image
        loss = self.loss_fn(model_out, targets)

        return loss


# ----------------------------------------------------------------------------
# Unit tests


class TestLosses(unittest.TestCase):
    """Unit tests for diffusion models and loss functions."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = 3
        self.cond_channels = 7
        self.out_channels = 3
        self.label_dim = 2
        self.img_resolution = (64, 128)

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


# ----------------------------------------------------------------------------
