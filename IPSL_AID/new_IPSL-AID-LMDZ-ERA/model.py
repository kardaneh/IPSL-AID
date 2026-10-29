import torch
import torch.nn.functional as F
from IPSL_AID.utils import EasyDict
import unittest

# Import all diffusion components
from IPSL_AID.networks import VPPrecond, VEPrecond, EDMPrecond, SongUNet, DhariwalUNet
from IPSL_AID.loss import VPLoss, VELoss, EDMLoss, UnetLoss


# ----------------------------------------------------------------------------
# Loss function for UNet architectures.

def wasserstein1d(pred: torch.Tensor, target_quantile_values, quantiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], beta = 50) -> torch.Tensor:
    loss = torch.zeros((1,))
    loss = loss.to(device = pred.device)
    for quantile, target_quantile_value in zip(quantiles, target_quantile_values) :
        pred = pred.flatten()
        loss += torch.abs(torch.mean(torch.sigmoid(-1*beta*(pred - target_quantile_value))) - quantile).mean()
    return loss

# def occurence_intensity_loss(pred : torch.Tensor, target : torch.Tensor, threshold = 0.095, beta = 50) :
#     occurence_pred = torch.sigmoid(beta*(pred - threshold))
#     occurence_target = (target > threshold).float()
#     loss_occurence = F.binary_cross_entropy(occurence_pred, occurence_target)
#     loss_intensity = F.mse_loss(pred * occurence_target, target * occurence_target)
#     return loss_occurence + loss_intensity

def occurence_intensity_loss(pred : torch.Tensor, target : torch.Tensor, threshold = 0.095, beta = 50) :
    occurence_pred = beta*(pred - threshold)
    occurence_target = (target > threshold).float()
    loss_occurence = F.binary_cross_entropy_with_logits(occurence_pred, occurence_target)
    loss_intensity = F.mse_loss(pred * occurence_target, target * occurence_target)
    return loss_occurence + loss_intensity

class loss_custom(torch.nn.Module) :
    def __init__(
            self,
            weight_facile = 1,
            weight_cape = 3, 
            weight_precip = 3,
            weight_precip_wasserstein = 0,
            weight_occurence_intensity_loss = 0,
            target_quantile_values = None,
            quantiles = None,
            beta = 50,
        ):
        super(loss_custom, self).__init__()
        self.target_quantile_values = target_quantile_values
        self.quantiles = quantiles
        self.beta = beta
        total_weight = weight_facile + weight_cape + weight_precip + weight_precip_wasserstein + weight_occurence_intensity_loss
        self.loss_facile = torch.nn.MSELoss(reduction="mean")
        self.weight_facile = weight_facile / total_weight
        self.weight_cape = weight_cape / total_weight
        self.weight_precip = weight_precip / total_weight
        self.weight_precip_wasserstein = weight_precip_wasserstein / total_weight
        self.weight_occurence_intensity_loss = weight_occurence_intensity_loss / total_weight

    def forward(self, predictions, target) :
        # ordre des variables : [u850, v850, T850, TCWV, cape, precip] :
        loss_1 = self.weight_facile * self.loss_facile(predictions[:,0:4], target[:,0:4])
        loss_cape = self.weight_cape * self.loss_facile(predictions[:,4], target[:,4])
        loss_precip = self.weight_precip * self.loss_facile(predictions[:,5], target[:,5])
        loss_finale = loss_1 + loss_cape + loss_precip
        if self.weight_precip_wasserstein != 0 :
            loss_precip_wasserstein = self.weight_precip_wasserstein * wasserstein1d(predictions[:,5], self.target_quantile_values, self.quantiles, self.beta)
            loss_finale = loss_finale + loss_precip_wasserstein
        if self.weight_occurence_intensity_loss != 0 :
            loss_occ_int = self.weight_occurence_intensity_loss * occurence_intensity_loss(pred=predictions[:,5], target=target[:,5])
            loss_finale = loss_finale + loss_occ_int
        return loss_finale
    
class loss_diffusion_custom(torch.nn.Module) :
    def __init__(
            self,
            weight_facile = 1,
            weight_cape = 3, 
            weight_precip = 3,
            weight_precip_wasserstein = 0,
            weight_occurence_intensity_loss = 0,
            target_quantile_values = None,
            quantiles = None,
            beta = 50,
        ):
        super(loss_diffusion_custom, self).__init__()
        self.target_quantile_values = target_quantile_values
        self.quantiles = quantiles
        self.beta = beta
        #total_weight = weight_facile + weight_cape + weight_precip + weight_precip_wasserstein + weight_occurence_intensity_loss
        total_weight = 1 # TODO change back when done testing
        self.weight_facile = weight_facile / total_weight
        self.weight_cape = weight_cape / total_weight
        self.weight_precip = weight_precip / total_weight
        self.weight_precip_wasserstein = weight_precip_wasserstein / total_weight
        self.weight_occurence_intensity_loss = weight_occurence_intensity_loss / total_weight

    def forward(self, predictions, target) :
        # ordre des variables : [u850, v850, T850, TCWV, cape, precip] :
        loss_1 = self.weight_facile * ((predictions[:,0:4] - target[:,0:4])**2)
        loss_cape = self.weight_cape * ((predictions[:,4:5] - target[:,4:5])**2)
        loss_precip = self.weight_precip * ((predictions[:,5:6] - target[:,5:6])**2)
        loss_finale = torch.cat([loss_1,loss_cape,loss_precip], dim=1)
        return loss_finale

# This class is custom. Here are the modifs compared to vanilla IPSL-AID :
# - put more weight on cape and precip

class UnetLoss_custom:
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

    def __init__(self,
        weight_facile = 1,
        weight_cape = 1,
        weight_precip = 1,
        weight_precip_wasserstein = 1,
        weight_occurence_intensity_loss = 1,
        target_quantile_values = None,
        quantiles = None,
        beta = 50
        ):
        """
        Initialize the UnetLoss function.
        Parameters
        ----------
        weight_cape : float, optional
            Weight to give to CAPE in loss
            Default is 3.
        weight_precip : float, optional
            Weight to give to precip in loss
            Default is 3.            
        """
        # Initialize loss function
        self.loss_fn = loss_custom(
            weight_facile=weight_facile,
            weight_cape=weight_cape,
            weight_precip=weight_precip,
            weight_precip_wasserstein = weight_precip_wasserstein,
            weight_occurence_intensity_loss= weight_occurence_intensity_loss,
            target_quantile_values = target_quantile_values,
            quantiles = quantiles,
            beta = beta
        )

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

#####
# Custom loss for diffusion
####

class EDMCustomLoss:
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

    def __init__(
        self,
        weight_facile = 1,
        weight_cape = 1,
        weight_precip = 1,
        weight_precip_wasserstein = 1,
        weight_occurence_intensity_loss = 1,
        target_quantile_values = None,
        quantiles = None,
        beta = 50,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=1.0
        ):
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
        # Initialize loss function
        self.loss_fn = loss_diffusion_custom(
            weight_facile=weight_facile,
            weight_cape=weight_cape,
            weight_precip=weight_precip,
            weight_precip_wasserstein = weight_precip_wasserstein,
            weight_occurence_intensity_loss= weight_occurence_intensity_loss,
            target_quantile_values = target_quantile_values,
            quantiles = quantiles,
            beta = beta
        )

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
        loss = weight * self.loss_fn(predictions = D_yn, target = y)
        return loss

# ============================================================================
# Model + Loss Loader
# ============================================================================


def load_model_and_loss(
    opts,
    custom_loss = False,
    logger=None,
    device="cpu",
    weight_facile = 1,
    weight_cape = 1,
    weight_precip = 1,
    weight_precip_wasserstein = 1,
    weight_occurence_intensity_loss = 1,
    target_quantile_values = None,
    quantiles = None,
    beta = 50
    ):
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
        if custom_loss :
            loss_class = EDMCustomLoss
        else :
            loss_class = EDMLoss
        log("Using EDM preconditioner & EDMLoss")
    elif opts.precond == "unet":
        if arch == "adm":
            precond_class = DhariwalUNet  # Direct U-Net without preconditioning
        elif arch in ["ddpmpp", "ncsnpp"]:
            precond_class = SongUNet  # Direct U-Net without preconditioning
        else:
            raise ValueError(f"❌ Invalid arch '{opts.arch}' for direct U-Net")
        if custom_loss :
            loss_class = UnetLoss_custom
            log("Using U-Net & UnetLoss_custom loss")
        else :
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
    if not custom_loss :
        loss_fn = loss_class()
    else :
        loss_fn = loss_class(
            weight_facile = weight_facile,
            weight_cape=weight_cape,
            weight_precip=weight_precip,
            weight_precip_wasserstein=weight_precip_wasserstein,
            weight_occurence_intensity_loss = weight_occurence_intensity_loss,
            target_quantile_values=target_quantile_values,
            quantiles=quantiles,
            beta=beta
        )
    log(f"Loss function instantiated: {loss_class.__name__}")
    log(f"  └── Loss Type: {opts.precond.upper()} Diffusion Loss")

    return model, loss_fn
