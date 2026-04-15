# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

from IPSL_AID.utils import EasyDict

# Import all diffusion components
from IPSL_AID.networks import VPPrecond, VEPrecond, EDMPrecond, SongUNet, DhariwalUNet
from IPSL_AID.loss import VPLoss, VELoss, EDMLoss, UnetLoss


# ============================================================================
# Model + Loss Loader
# ============================================================================


def load_model_and_loss(opts, logger=None, device="cpu"):
    """
    Load a diffusion model or U-Net with corresponding loss function.

    This function initializes and configures a generative model (diffusion or
    direct U-Net) along with its corresponding loss function based on the
    provided options. It supports multiple architectures and preconditioning
    schemes.

    Parameters
    ----------
    opts : EasyDict or dict
        Configuration dictionary containing model parameters. Must include:

        - arch : str
            Architecture type: 'ddpmpp', 'ncsnpp', or 'adm'.
        - precond : str
            Preconditioning type: 'vp', 've', 'edm', or 'unet'.
        - img_resolution : int or tuple
            Image resolution (height, width).
        - in_channels : int
            Number of input channels.
        - out_channels : int
            Number of output channels.
        - label_dim : int
            Dimension of label conditioning (0 for unconditional).
        - use_fp16 : bool
            Whether to use mixed precision (FP16).
        - model_kwargs : dict, optional
            Additional model-specific parameters to override defaults.
    logger : logging.Logger, optional
        Logger instance for output messages. If None, uses print().
        Default is None.
    device : str or torch.device, optional
        Device to load the model onto ('cpu', 'cuda', etc.).
        Default is 'cpu'.

    Returns
    -------
    model : torch.nn.Module
        Initialized model instance (preconditioner or U-Net).
    loss_fn : torch.nn.Module or callable
        Corresponding loss function for the model.

    Raises
    ------
    ValueError
        If an invalid architecture or preconditioner type is specified.

    Notes
    -----
    - The function supports three main architectures:
        * DDPM++ (Song et al., 2020) with VP preconditioning
        * NCSN++ (Song et al., 2020) with VE preconditioning
        * ADM (Dhariwal & Nichol, 2021) with EDM preconditioning
    - When precond='unet', uses a direct U-Net without diffusion preconditioning.
    - Model parameters are counted and logged for transparency.
    - Default hyperparameters are provided for each architecture but can be
      overridden via opts.model_kwargs.
    """

    log = logger.info if logger else print
    opts = EasyDict(opts)
    diffusion_model = False if opts.precond == "unet" else True
    arch = opts.arch.lower()

    # --------------------------------------------------------
    # Preconditioner + matching loss
    # --------------------------------------------------------

    if opts.precond == "vp":
        precond_class = VPPrecond
        loss_class = VPLoss
        log("Using VP preconditioner & VPLoss")
    elif opts.precond == "ve":
        precond_class = VEPrecond
        loss_class = VELoss
        log("Using VE preconditioner & VELoss")
    elif opts.precond == "edm":
        precond_class = EDMPrecond
        loss_class = EDMLoss
        log("Using EDM preconditioner & EDMLoss")
    elif opts.precond == "unet":
        if arch == "adm":
            precond_class = DhariwalUNet  # Direct U-Net without preconditioning
        elif arch in ["ddpmpp", "ncsnpp"]:
            precond_class = SongUNet  # Direct U-Net without preconditioning
        else:
            raise ValueError(f"❌ Invalid arch '{opts.arch}' for direct U-Net")
        loss_class = UnetLoss
        log("Using direct U-Net & UnetLoss")
    else:
        raise ValueError(f"❌ Invalid opts.precond '{opts.precond}'")

    # --------------------------------------------------------
    # Architecture network kwargs
    # --------------------------------------------------------
    network_kwargs = EasyDict()

    if arch == "ddpmpp":
        network_kwargs.update(
            dict(
                model_type="SongUNet",
                embedding_type="positional",
                encoder_type="standard",
                decoder_type="standard",
                channel_mult_noise=1,
                resample_filter=[1, 1],
                model_channels=128,
                channel_mult=[2, 2, 2],
            )
        )
        log("Architecture DDPM++ / SongUNet selected")

    elif arch == "ncsnpp":
        network_kwargs.update(
            dict(
                model_type="SongUNet",
                embedding_type="fourier",
                encoder_type="residual",
                decoder_type="standard",
                channel_mult_noise=2,
                resample_filter=[1, 3, 3, 1],
                model_channels=128,
                channel_mult=[2, 2, 2],
            )
        )
        log("Architecture NCSN++ / SongUNet selected")

    elif arch == "adm":
        if diffusion_model:
            network_kwargs.update(
                dict(
                    model_type="DhariwalUNet",
                    model_channels=128,
                    channel_mult=[1, 2, 3, 4],
                    num_blocks=2,
                )
            )
            log("Architecture ADM / DhariwalUNet selected")
        else:
            network_kwargs.update(
                dict(
                    model_channels=128,
                    channel_mult=[1, 2, 3, 4],
                    num_blocks=2,
                    diffusion_model=False,
                )
            )
            log("Architecture ADM / DhariwalUNet selected for direct U-Net")

    else:
        raise ValueError(f"❌ Invalid opts.arch '{opts.arch}'")

    # Allow overrides from opts.model_kwargs
    if hasattr(opts, "model_kwargs"):
        log("Overriding with user model_kwargs")
        network_kwargs.update(opts.model_kwargs)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------
    log("Instantiating model...")
    if diffusion_model:
        log("Diffusion model enabled")
        total_in = opts.in_channels + (
            opts.cond_channels if "cond_channels" in opts else 0
        )
    else:
        log("Diffusion model disabled, direct U-Net, no preconditioning")

    if diffusion_model:
        model = precond_class(
            img_resolution=opts.img_resolution,
            in_channels=total_in,
            out_channels=opts.out_channels,
            label_dim=opts.label_dim,
            use_fp16=opts.use_fp16,
            **network_kwargs,
        )
    else:
        model = precond_class(
            img_resolution=opts.img_resolution,
            in_channels=opts.in_channels,
            out_channels=opts.out_channels,
            label_dim=opts.label_dim,
            **network_kwargs,
        )

    model = model.to(device)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --------------------------------------------------------
    # Comprehensive Model Information Logging
    # --------------------------------------------------------

    log("Model Summary:")
    log(f"   └── Model Type: {type(model).__name__}")
    log(f"   └── Preconditioner: {opts.precond.upper()}")
    log(f"   └── Architecture: {opts.arch.upper()}")
    if diffusion_model:
        log(
            f"   └── Input Channels: {total_in} (base: {opts.in_channels} + cond: {total_in - opts.in_channels})"
        )
    else:
        log(f"   └── Input Channels: {opts.in_channels}")
    log(f"   └── Output Channels: {opts.out_channels}")
    log(f"   └── Label Dimension: {opts.label_dim}")
    log(f"   └── Image Resolution: {opts.img_resolution}")
    if diffusion_model:
        log(f"   └── FP16 Enabled: {opts.use_fp16}")
    else:
        log("   └── FP16 Enabled: N/A for direct U-Net")
    log(f"   └── Model Parameters - Total: {total_num:,}, Trainable: {trainable_num}")

    # Log network architecture details
    log("Network Architecture:")
    for key, value in network_kwargs.items():
        log(f"   └── {key}: {value}")

    # Log device information
    device = next(model.parameters()).device
    log(f"Device: {device}")

    # Log model dtype information
    dtype = next(model.parameters()).dtype
    log(f"Model Data Type: {dtype}")

    # --------------------------------------------------------
    # Loss function instance
    # --------------------------------------------------------
    loss_fn = loss_class()
    log(f"Loss function instantiated: {loss_class.__name__}")
    log(f"  └── Loss Type: {opts.precond.upper()} Diffusion Loss")

    return model, loss_fn
