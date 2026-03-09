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

import numpy as np
import torch
import unittest
from torch.nn.functional import silu

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    """
    Initialize weights using various initialization methods.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the weight tensor to initialize.
    mode : str
        The initialization method to use. Options are:
        - 'xavier_uniform': Xavier uniform initialization
        - 'xavier_normal': Xavier normal initialization
        - 'kaiming_uniform': Kaiming uniform initialization (also known as He initialization)
        - 'kaiming_normal': Kaiming normal initialization (also known as He initialization)
    fan_in : int
        Number of input units in the weight tensor.
    fan_out : int
        Number of output units in the weight tensor.

    Returns
    -------
    torch.Tensor
        A tensor of the specified shape with values initialized according
        to the chosen method.

    Raises
    ------
    ValueError
        If an invalid initialization mode is provided.
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    """
    A linear (fully connected) layer with customizable weight initialization.

    This layer applies a linear transformation to the incoming data: ``y = x W^T + b``.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If set to False, the layer will not learn an additive bias.
        Default is True.
    init_mode : str, optional
        Weight initialization method. Options are:
        - ``xavier_uniform``: Xavier uniform initialization
        - ``xavier_normal``: Xavier normal initialization
        - ``kaiming_uniform``: Kaiming uniform initialization (He initialization)
        - ``kaiming_normal``: Kaiming normal initialization (He initialization)
        Default is ``kaiming_normal``.
    init_weight : float or int, optional
        Scaling factor for the initialized weights.
        Default is 1.
    init_bias : float or int, optional
        Scaling factor for the initialized bias.
        Default is 0.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The learnable weights of the layer of shape (out_features, in_features).
    bias : torch.nn.Parameter or None
        The learnable bias of the layer of shape (out_features,). If bias=False,
        this attribute is set to None.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        """
        Initialize the Linear layer.

        Parameters
        ----------
        in_features : int
            Size of each input sample.
        out_features : int
            Size of each output sample.
        bias : bool, optional
            If set to False, the layer will not learn an additive bias.
            Default is True.
        init_mode : str, optional
            Weight initialization method.
            Default is 'kaiming_normal'.
        init_weight : float or int, optional
            Scaling factor for the initialized weights.
            Default is 1.
        init_bias : float or int, optional
            Scaling factor for the initialized bias.
            Default is 0.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        """
        Forward pass of the linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features) or
            ``(batch_size, *, in_features)`` where ``*`` means any number of
            additional dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features) or
            ``(batch_size, *, out_features)``.

        Notes
        -----
        The operation performed is: ``output = x @ weight^T + bias``.
        The bias is added in-place for efficiency when possible.
        """
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    """
    2D convolutional layer with optional upsampling, downsampling, and fused resampling.

    This layer implements a 2D convolution that can optionally include upsampling
    or downsampling operations with configurable resampling filters. It supports
    both separate and fused resampling modes for efficiency.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel : int
        Size of the convolutional kernel (square kernel).
    bias : bool, optional
        If set to False, the layer will not learn an additive bias.
        Default is True.
    up : bool, optional
        If True, upsample the input by a factor of 2 before convolution.
        Cannot be True if `down` is also True.
        Default is False.
    down : bool, optional
        If True, downsample the output by a factor of 2 after convolution.
        Cannot be True if `up` is also True.
        Default is False.
    resample_filter : list, optional
        Coefficients of the 1D resampling filter that will be turned into a 2D filter.
        Default is [1, 1] (bilinear filter).
    fused_resample : bool, optional
        If True, fuse the resampling operation with the convolution for efficiency.
        Default is False.
    init_mode : str, optional
        Weight initialization method. Options are:
        - 'xavier_uniform': Xavier uniform initialization
        - 'xavier_normal': Xavier normal initialization
        - 'kaiming_uniform': Kaiming uniform initialization (He initialization)
        - 'kaiming_normal': Kaiming normal initialization (He initialization)
        Default is 'kaiming_normal'.
    init_weight : float or int, optional
        Scaling factor for the initialized weights.
        Default is 1.
    init_bias : float or int, optional
        Scaling factor for the initialized bias.
        Default is 0.

    Attributes
    ----------
    weight : torch.nn.Parameter or None
        The learnable weights of the convolution of shape
        (out_channels, in_channels, kernel, kernel). If kernel is 0, this is None.
    bias : torch.nn.Parameter or None
        The learnable bias of the convolution of shape (out_channels,).
        If kernel is 0 or bias is False, this is None.
    resample_filter : torch.Tensor or None
        The 2D resampling filter used for upsampling or downsampling.
        Registered as a buffer (non-learnable parameter).

    Raises
    ------
    AssertionError
        If both `up` and `down` are set to True.

    Notes
    -----
    - When `kernel` is 0, no convolution is performed, only resampling if enabled.
    - The resampling filter is created by taking the outer product of the 1D filter
      with itself to create a separable 2D filter, then normalized.
    - Fused resampling combines the resampling and convolution operations into
      single operations for better performance.

    Examples
    --------
    >>> # Standard convolution
    >>> conv = Conv2d(3, 16, kernel=3)
    >>> x = torch.randn(4, 3, 32, 32)
    >>> out = conv(x)
    >>> out.shape
    torch.Size([4, 16, 32, 32])

    >>> # Convolution with downsampling
    >>> conv_down = Conv2d(3, 16, kernel=3, down=True)
    >>> out = conv_down(x)
    >>> out.shape
    torch.Size([4, 16, 16, 16])

    >>> # Convolution with upsampling
    >>> conv_up = Conv2d(3, 16, kernel=3, up=True)
    >>> out = conv_up(x)
    >>> out.shape
    torch.Size([4, 16, 64, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        """
        Initialize the Conv2d layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel : int
            Size of the convolutional kernel.
        bias : bool, optional
            Whether to include a bias term.
            Default is True.
        up : bool, optional
            Whether to upsample the input.
            Default is False.
        down : bool, optional
            Whether to downsample the output.
            Default is False.
        resample_filter : list, optional
            Coefficients of the 1D resampling filter.
            Default is [1, 1].
        fused_resample : bool, optional
            Whether to fuse resampling with convolution.
            Default is False.
        init_mode : str, optional
            Weight initialization method.
            Default is 'kaiming_normal'.
        init_weight : float or int, optional
            Scaling factor for weight initialization.
            Default is 1.
        init_bias : float or int, optional
            Scaling factor for bias initialization.
            Default is 0.
        """
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        """
        Forward pass of the Conv2d layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, out_height, out_width).
            If `up` is True, spatial dimensions are doubled.
            If `down` is True, spatial dimensions are halved.

        Notes
        -----
        The method handles four main cases:
        1. Fused upsampling + convolution
        2. Fused convolution + downsampling
        3. Separate up/down sampling followed by convolution
        4. Standard convolution only
        """
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    """
    Group Normalization layer.

    This layer implements Group Normalization, which divides channels into groups
    and computes within each group the mean and variance for normalization.
    It is particularly effective for small batch sizes and often used as an
    alternative to Batch Normalization.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    num_groups : int, optional
        Number of groups to divide the channels into. Must be a divisor of
        the number of channels. The actual number of groups may be reduced
        to satisfy `min_channels_per_group`.
        Default is 32.
    min_channels_per_group : int, optional
        Minimum number of channels per group. If the division would result
        in fewer channels per group, the number of groups is reduced.
        Default is 4.
    eps : float, optional
        A small constant added to the denominator for numerical stability.
        Default is 1e-5.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable scale parameter of shape (num_channels,).
        Initialized to ones.
    bias : torch.nn.Parameter
        Learnable bias parameter of shape (num_channels,).
        Initialized to zeros.

    Notes
    -----
    - Group Normalization is independent of batch size, making it suitable
      for variable batch sizes and small batch training.
    - The number of groups is automatically adjusted to ensure each group has
      at least `min_channels_per_group` channels.
    - This layer uses PyTorch's built-in `torch.nn.functional.group_norm`.
    """

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        """
        Initialize the GroupNorm layer.

        Parameters
        ----------
        num_channels : int
            Number of input channels.
        num_groups : int, optional
            Desired number of groups.
            Default is 32.
        min_channels_per_group : int, optional
            Minimum channels per group.
            Default is 4.
        eps : float, optional
            Small constant for numerical stability.
            Default is 1e-5.
        """
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        Forward pass of the GroupNorm layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_channels, height, width).

        Returns
        -------
        torch.Tensor
            Normalized tensor of same shape as input.

        Notes
        -----
        The normalization is performed across spatial dimensions and within
        each group of channels, maintaining the original mean and variance
        statistics per group.
        """
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    """
    Custom autograd function for scaled dot-product attention weight computation.

    This function computes attention weights using scaled dot-product attention:
    w = softmax(Q·K^T / √d_k), where d_k is the dimension of the key vectors.
    It implements both forward and backward passes for gradient computation.

    Notes
    -----
    - This is a stateless operation that uses torch.autograd.Function for custom backward.
    - The forward pass computes attention weights in float32 for numerical stability.
    - The backward pass computes gradients using the chain rule for softmax and matrix multiplication.
    - This implementation is optimized for memory efficiency during backward pass.
    """

    @staticmethod
    def forward(ctx, q, k):
        """
        Forward pass for attention weight computation.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFunction
            Context object to save tensors for backward pass.
        q : torch.Tensor
            Query tensor of shape (batch_size, channels, query_length).
        k : torch.Tensor
            Key tensor of shape (batch_size, channels, key_length).

        Returns
        -------
        torch.Tensor
            Attention weights of shape (batch_size, query_length, key_length).
            Each row represents attention distribution for a query position.

        Notes
        -----
        - Computes w = softmax(Q·K^T / √d_k) where d_k = k.shape[1] (channel dimension).
        - Uses float32 for computation to maintain numerical stability.
        - Saves q, k, and w in context for backward pass.
        """
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        """
        Backward pass for attention weight computation.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFunction
            Context object containing saved tensors from forward pass.
        dw : torch.Tensor
            Gradient of loss with respect to attention weights.
            Shape: (batch_size, query_length, key_length).

        Returns
        -------
        dq : torch.Tensor
            Gradient with respect to query tensor.
            Shape: (batch_size, channels, query_length).
        dk : torch.Tensor
            Gradient with respect to key tensor.
            Shape: (batch_size, channels, key_length).

        Notes
        -----
        - Uses the saved tensors q, k, w from forward pass.
        - Computes gradient of softmax using PyTorch's internal softmax_backward.
        - Applies chain rule for the scaled dot-product operation.
        - Maintains original dtypes of input tensors.
        """
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    """
    U-Net block with optional attention, up/down sampling, and adaptive scaling.

    This block implements a residual block commonly used in U-Net architectures
    for diffusion models and image-to-image translation. It supports:
    - Residual connections with optional skip scaling
    - Adaptive scaling/shifting via conditioning embeddings
    - Multi-head self-attention mechanisms
    - Upsampling or downsampling operations
    - Dropout for regularization

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_channels : int
        Number of embedding (conditioning) channels.
    up : bool, optional
        If True, upsample the input by a factor of 2.
        Default is False.
    down : bool, optional
        If True, downsample the output by a factor of 2.
        Default is False.
    attention : bool, optional
        If True, include multi-head self-attention in the block.
        Default is False.
    num_heads : int, optional
        Number of attention heads. If None, computed as out_channels // channels_per_head.
        Default is None.
    channels_per_head : int, optional
        Number of channels per attention head when num_heads is None.
        Default is 64.
    dropout : float, optional
        Dropout probability applied after the first activation.
        Default is 0.
    skip_scale : float, optional
        Scaling factor applied to the residual connection.
        Default is 1.
    eps : float, optional
        Epsilon value for GroupNorm layers for numerical stability.
        Default is 1e-5.
    resample_filter : list, optional
        Coefficients for the resampling filter used in up/down sampling.
        Default is [1, 1].
    resample_proj : bool, optional
        If True, use a 1x1 convolution in the skip connection when resampling.
        Default is False.
    adaptive_scale : bool, optional
        If True, use both scale and shift parameters from the embedding.
        If False, use only shift parameters.
        Default is True.
    init : dict, optional
        Initialization parameters for most convolutional layers.
        Default is empty dict.
    init_zero : dict, optional
        Initialization parameters for final convolutional layers (zero initialization).
        Default is {'init_weight': 0}.
    init_attn : dict, optional
        Initialization parameters for attention layers.
        If None, uses the same as `init`.
        Default is None.

    Attributes
    ----------
    norm0, norm1, norm2 : GroupNorm
        Group normalization layers.
    conv0, conv1 : Conv2d
        Convolutional layers.
    affine : Linear
        Linear layer for conditioning embedding.
    skip : Conv2d or None
        Skip connection projection (1x1 conv) if input and output channels differ or resampling.
    qkv, proj : Conv2d
        Attention query-key-value and projection layers (if attention is enabled).

    Notes
    -----
    - The block follows a pre-activation residual structure: norm -> activation -> conv.
    - When `adaptive_scale=True`, the conditioning embedding provides both scale and shift parameters.
    - The attention mechanism uses multi-head self-attention within the spatial dimensions.
    - The skip connection is automatically added when input/output channels differ or when resampling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        """
        Initialize the UNetBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        emb_channels : int
            Number of embedding channels.
        up : bool, optional
            Enable upsampling.
        down : bool, optional
            Enable downsampling.
        attention : bool, optional
            Enable attention mechanism.
        num_heads : int, optional
            Number of attention heads.
        channels_per_head : int, optional
            Channels per attention head.
        dropout : float, optional
            Dropout probability.
        skip_scale : float, optional
            Scaling factor for skip connection.
        eps : float, optional
            Epsilon for GroupNorm.
        resample_filter : list, optional
            Filter for resampling.
        resample_proj : bool, optional
            Use projection in skip connection when resampling.
        adaptive_scale : bool, optional
            Use adaptive scaling from embedding.
        init : dict, optional
            Initialization parameters.
        init_zero : dict, optional
            Zero initialization parameters.
        init_attn : dict, optional
            Attention initialization parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        """
        Forward pass of the UNetBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).
        emb : torch.Tensor
            Conditioning embedding of shape (batch_size, emb_channels).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, out_height, out_width).

        Notes
        -----
        The forward pass consists of:
        1. Initial normalization and convolution (with optional up/down sampling)
        2. Adaptive scaling/shifting from conditioning embedding
        3. Second normalization, dropout, and convolution
        4. Skip connection with scaling
        5. Optional multi-head self-attention
        """
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    """
    Sinusoidal positional embedding for sequences or timesteps.

    This module generates sinusoidal embeddings for input positions, commonly used
    in transformer architectures and diffusion models to provide temporal or
    positional information to the model.

    Parameters
    ----------
    num_channels : int
        Dimensionality of the embedding vectors. Must be even.
    max_positions : int, optional
        Maximum number of positions (or timesteps) for which embeddings are generated.
        Determines the frequency scaling. Default is 10000.
    endpoint : bool, optional
        If True, scales frequencies such that the last frequency is 1/2 of the first.
        If False, uses the standard scaling. Default is False.

    Attributes
    ----------
    num_channels : int
        Dimensionality of the embedding vectors.
    max_positions : int
        Maximum positions for frequency scaling.
    endpoint : bool
        Whether to use endpoint scaling.

    Notes
    -----
    - The embedding uses sine and cosine functions of different frequencies to
      create a unique encoding for each position.
    - The frequencies are computed as:
      freqs = (1 / max_positions) ** (2i / num_channels) for i in range(num_channels//2)
      or with endpoint adjustment.
    - The output embedding is the concatenation of [cos(x*freqs), sin(x*freqs)].
    - This implementation is based on the original Transformer positional encoding
      and the diffusion model timestep embedding.
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        """
        Initialize the PositionalEmbedding module.

        Parameters
        ----------
        num_channels : int
            Dimensionality of the embedding vectors.
        max_positions : int, optional
            Maximum number of positions for frequency scaling.
            Default is 10000.
        endpoint : bool, optional
            Whether to use endpoint scaling.
            Default is False.
        """
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        """
        Forward pass to generate positional embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of positions (or timesteps) of shape (batch_size,) or (n,).
            Values are typically integers in [0, max_positions-1].

        Returns
        -------
        torch.Tensor
            Positional embeddings of shape (len(x), num_channels).

        Notes
        -----
        - The input tensor `x` is typically a 1D tensor of position indices.
        - The output is a 2D tensor where each row corresponds to the embedding
          of the respective position.
        - The embedding uses the device and dtype of the input tensor `x`.
        """
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    """
    Random Fourier feature embedding for positional encoding.

    This module generates random Fourier features (RFF) for input positions or
    coordinates, mapping low-dimensional inputs to a higher-dimensional space
    using random frequency sampling. Commonly used in neural fields, kernel
    methods, and coordinate-based neural networks.

    Parameters
    ----------
    num_channels : int
        Dimensionality of the embedding vectors. Must be even.
    scale : float, optional
        Standard deviation for sampling the random frequencies.
        Determines the frequency distribution. Default is 16.

    Attributes
    ----------
    freqs : torch.Tensor (buffer)
        Random frequencies sampled from a normal distribution with mean 0
        and standard deviation `scale`. Shape: (num_channels // 2,).

    Notes
    -----
    - The frequencies are randomly initialized and fixed (non-learnable).
    - The embedding uses sine and cosine projections of the input multiplied
      by 2π times the random frequencies.
    - This technique approximates shift-invariant kernels via Bochner's theorem.
    - Unlike learned embeddings, this provides a fixed, deterministic mapping
      from input space to embedding space.
    """

    def __init__(self, num_channels, scale=16):
        """
        Initialize the FourierEmbedding module.

        Parameters
        ----------
        num_channels : int
            Dimensionality of the embedding vectors.
        scale : float, optional
            Standard deviation for frequency sampling.
            Default is 16.
        """
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        """
        Forward pass to generate Fourier feature embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size,) or (n,).
            Typically continuous values representing positions or coordinates.

        Returns
        -------
        torch.Tensor
            Fourier feature embeddings of shape (len(x), num_channels).

        Notes
        -----
        - The transformation is: x ↦ [cos(2π * freqs * x), sin(2π * freqs * x)].
        - The output dimension is twice the number of frequencies (num_channels).
        - This embedding is deterministic given the fixed random frequencies.
        """
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# DDPM++ and NCSN++ architectures


class SongUNet(torch.nn.Module):
    """
    U-Net architecture for diffusion models based on Song et al. (2020).

    This implementation supports both DDPM++ and NCSN++ architectures with
    configurable encoder/decoder types, attention mechanisms, and conditioning.
    It handles both square and rectangular input resolutions.

    Parameters
    ----------
    img_resolution : int or tuple
        Input image resolution. If int, assumes square images (img_resolution x img_resolution).
        If tuple, should be (height, width).
    in_channels : int
        Number of input color channels.
    out_channels : int
        Number of output color channels.
    label_dim : int, optional
        Number of class labels. Set to 0 for unconditional generation.
        Default is 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels (e.g., time-dependent augmentation).
        Set to 0 for no augmentation. Default is 0.
    model_channels : int, optional
        Base channel multiplier for the network.
        Default is 128.
    channel_mult : list of int, optional
        Channel multipliers for each resolution level.
        Default is [1, 2, 2, 2].
    channel_mult_emb : int, optional
        Multiplier for embedding dimensionality relative to model_channels.
        Default is 4.
    num_blocks : int, optional
        Number of residual blocks per resolution.
        Default is 4.
    attn_resolutions : list of int, optional
        List of resolutions (minimum dimension) to apply self-attention.
        Default is [16].
    dropout : float, optional
        Dropout probability for intermediate activations.
        Default is 0.10.
    label_dropout : float, optional
        Dropout probability for class labels (classifier-free guidance).
        Default is 0.
    embedding_type : str, optional
        Type of timestep embedding: 'positional' for DDPM++, 'fourier' for NCSN++.
        Default is 'positional'.
    channel_mult_noise : int, optional
        Multiplier for noise embedding dimensionality: 1 for DDPM++, 2 for NCSN++.
        Default is 1.
    encoder_type : str, optional
        Encoder architecture: 'standard' for DDPM++, 'skip' or 'residual' for NCSN++.
        Default is 'standard'.
    decoder_type : str, optional
        Decoder architecture: 'standard' for both, 'skip' for NCSN++.
        Default is 'standard'.
    resample_filter : list, optional
        Resampling filter coefficients: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        Default is [1,1].

    Attributes
    ----------
    img_resolution : tuple
        Input image resolution as (height, width).
    img_height : int
        Input image height.
    img_width : int
        Input image width.
    label_dropout : float
        Class label dropout probability.
    map_noise : PositionalEmbedding or FourierEmbedding
        Noise/timestep embedding module.
    map_label : Linear or None
        Class label embedding module.
    map_augment : Linear or None
        Augmentation label embedding module.
    map_layer0, map_layer1 : Linear
        Embedding transformation layers.
    enc : torch.nn.ModuleDict
        Encoder modules organized by resolution.
    dec : torch.nn.ModuleDict
        Decoder modules organized by resolution.

    Raises
    ------
    AssertionError
        If embedding_type is not 'fourier' or 'positional'.
        If encoder_type is not 'standard', 'skip', or 'residual'.
        If decoder_type is not 'standard' or 'skip'.
        If img_resolution tuple doesn't have exactly 2 elements.

    Notes
    -----
    - The architecture follows a U-Net structure with skip connections.
    - Supports multiple conditioning types: noise (timestep), class labels, augmentations.
    - Attention is applied at specified resolutions to capture long-range dependencies.
    - Different encoder/decoder types and embedding methods allow emulating DDPM++ or NCSN++.
    - Rectangular resolutions are supported by tracking height and width separately.

    References
    ----------
    - Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM)
    - Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (NCSN++)
    """

    def __init__(
        self,
        img_resolution,  # Image resolution as tuple (height, width)
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        """
        Initialize the SongUNet.

        Parameters
        ----------
        img_resolution : int or tuple
            Image resolution.
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        label_dim : int, optional
            Class label dimension.
        augment_dim : int, optional
            Augmentation label dimension.
        model_channels : int, optional
            Base channel multiplier.
        channel_mult : list, optional
            Channel multipliers per resolution.
        channel_mult_emb : int, optional
            Embedding channel multiplier.
        num_blocks : int, optional
            Blocks per resolution.
        attn_resolutions : list, optional
            Resolutions for attention.
        dropout : float, optional
            Dropout probability.
        label_dropout : float, optional
            Label dropout probability.
        embedding_type : str, optional
            Embedding type.
        channel_mult_noise : int, optional
            Noise embedding multiplier.
        encoder_type : str, optional
            Encoder type.
        decoder_type : str, optional
            Decoder type.
        resample_filter : list, optional
            Resampling filter coefficients.
        """
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        # Handle rectangular resolution
        if isinstance(img_resolution, (tuple, list)):
            assert (
                len(img_resolution) == 2
            ), "img_resolution must be a tuple/list (height, width)"
            self.img_resolution = img_resolution
            self.img_height, self.img_width = img_resolution
        else:
            self.img_resolution = (img_resolution, img_resolution)
            self.img_height = self.img_width = img_resolution

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = (
            Linear(in_features=label_dim, out_features=noise_channels, **init)
            if label_dim
            else None
        )
        self.map_augment = (
            Linear(
                in_features=augment_dim, out_features=noise_channels, bias=False, **init
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            # Calculate current resolution level
            res_h = self.img_height >> level
            res_w = self.img_width >> level
            res_key = f"{res_h}x{res_w}"

            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res_key}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res_key}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res_key}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res_key}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res_key}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                # Check attention for rectangular resolution
                attn = min(res_h, res_w) in attn_resolutions
                self.enc[f"{res_key}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            # Calculate current resolution level
            res_h = self.img_height >> level
            res_w = self.img_width >> level
            res_key = f"{res_h}x{res_w}"

            if level == len(channel_mult) - 1:
                self.dec[f"{res_key}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res_key}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res_key}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                # Check attention for rectangular resolution
                attn = idx == num_blocks and (min(res_h, res_w) in attn_resolutions)
                self.dec[f"{res_key}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res_key}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res_key}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res_key}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        """
        Forward pass through the U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).
        noise_labels : torch.Tensor
            Noise/timestep labels of shape (batch_size,).
        class_labels : torch.Tensor or None
            Class labels of shape (batch_size,) or (batch_size, label_dim).
            Can be None if label_dim is 0.
        augment_labels : torch.Tensor or None, optional
            Augmentation labels of shape (batch_size, augment_dim).
            Can be None if augment_dim is 0.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).

        Notes
        -----
        - The forward pass consists of three main stages:
          1. Embedding mapping: combines noise, class, and augmentation embeddings.
          2. Encoder: extracts hierarchical features with optional skip connections.
          3. Decoder: reconstructs output with skip connections from encoder.
        - Classifier-free guidance is supported via label_dropout.
        - The noise embedding uses sinusoidal (positional) or Fourier features.
        """
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


# ----------------------------------------------------------------------------
# ADM architecture


class DhariwalUNet(torch.nn.Module):
    """
    U-Net architecture based on Dhariwal & Nichol (2021) for diffusion models.

    This implementation follows the ADM (Ablated Diffusion Model) architecture
    with configurable attention mechanisms, conditioning, and rectangular resolution
    support. It features a U-Net structure with skip connections, group normalization,
    and optional conditioning on class labels and augmentation.

    Parameters
    ----------
    img_resolution : int or tuple
        Input image resolution. If int, assumes square images (img_resolution x img_resolution).
        If tuple, should be (height, width).
    in_channels : int
        Number of input color channels.
    out_channels : int
        Number of output color channels.
    label_dim : int, optional
        Number of class labels. Set to 0 for unconditional generation.
        Default is 0.
    augment_dim : int, optional
        Dimensionality of augmentation labels (e.g., time-dependent augmentation).
        Set to 0 for no augmentation. Default is 0.
    model_channels : int, optional
        Base channel multiplier for the network.
        Default is 128.
    channel_mult : list of int, optional
        Channel multipliers for each resolution level.
        Default is [1, 2, 3, 4].
    channel_mult_emb : int, optional
        Multiplier for embedding dimensionality relative to model_channels.
        Default is 4.
    num_blocks : int, optional
        Number of residual blocks per resolution.
        Default is 3.
    attn_resolutions : list of int, optional
        List of resolutions (minimum dimension) to apply self-attention.
        Default is [32, 16, 8].
    dropout : float, optional
        Dropout probability for intermediate activations.
        Default is 0.10.
    label_dropout : float, optional
        Dropout probability for class labels (classifier-free guidance).
        Default is 0.
    diffusion_model : bool, optional
        Whether to configure the network for diffusion models.
        If True, includes timestep embedding; if False, only uses label conditioning.
        Default is True.

    Attributes
    ----------
    img_resolution : tuple
        Input image resolution as (height, width).
    img_height : int
        Input image height.
    img_width : int
        Input image width.
    label_dropout : float
        Class label dropout probability.
    map_noise : PositionalEmbedding or None
        Noise/timestep embedding module (if diffusion_model=True).
    map_label : Linear or None
        Class label embedding module.
    map_augment : Linear or None
        Augmentation label embedding module.
    map_layer0, map_layer1 : Linear
        Embedding transformation layers.
    enc : torch.nn.ModuleDict
        Encoder modules organized by resolution.
    dec : torch.nn.ModuleDict
        Decoder modules organized by resolution.
    out_norm : GroupNorm
        Final group normalization layer.
    out_conv : Conv2d
        Final convolutional output layer.

    Raises
    ------
    AssertionError
        If img_resolution tuple doesn't have exactly 2 elements.

    Notes
    -----
    - The architecture is based on the U-Net from "Diffusion Models Beat GANs on Image Synthesis".
    - Features group normalization throughout and attention at multiple resolutions.
    - Supports classifier-free guidance via label_dropout.
    - Can be used for both diffusion models and other conditional generation tasks.
    - Rectangular resolutions are supported by tracking height and width separately.

    References
    ----------
    - Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", 2021.
    """

    def __init__(
        self,
        img_resolution,  # Image resolution as tuple (height, width)
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            3,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        diffusion_model=True,  # Whether to use the Unet for diffusion models.
    ):
        """
        Initialize the DhariwalUNet.

        Parameters
        ----------
        img_resolution : int or tuple
            Image resolution.
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        label_dim : int, optional
            Class label dimension.
        augment_dim : int, optional
            Augmentation label dimension.
        model_channels : int, optional
            Base channel multiplier.
        channel_mult : list, optional
            Channel multipliers per resolution.
        channel_mult_emb : int, optional
            Embedding channel multiplier.
        num_blocks : int, optional
            Blocks per resolution.
        attn_resolutions : list, optional
            Resolutions for attention.
        dropout : float, optional
            Dropout probability.
        label_dropout : float, optional
            Label dropout probability.
        diffusion_model : bool, optional
            Whether to configure for diffusion models.
        """
        # Handle rectangular resolution
        if isinstance(img_resolution, (tuple, list)):
            assert (
                len(img_resolution) == 2
            ), "img_resolution must be a tuple (height, width)"
            self.img_resolution = img_resolution
            self.img_height, self.img_width = img_resolution
        else:
            self.img_resolution = (img_resolution, img_resolution)
            self.img_height = self.img_width = img_resolution

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=model_channels)
            if diffusion_model
            else None
        )
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            # Calculate current resolution level
            res_h = self.img_height >> level
            res_w = self.img_width >> level
            res_key = f"{res_h}x{res_w}"

            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res_key}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res_key}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )

            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                # Check attention for rectangular resolution
                attn = min(res_h, res_w) in attn_resolutions
                self.enc[f"{res_key}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )

        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            # Calculate current resolution level
            res_h = self.img_height >> level
            res_w = self.img_width >> level
            res_key = f"{res_h}x{res_w}"

            if level == len(channel_mult) - 1:
                self.dec[f"{res_key}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res_key}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res_key}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )

            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                # Check attention for rectangular resolution
                attn = min(res_h, res_w) in attn_resolutions
                self.dec[f"{res_key}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )

        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
        )

    def forward(self, x, noise_labels=None, class_labels=None, augment_labels=None):
        """
        Forward pass through the Dhariwal U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).
        noise_labels : torch.Tensor or None
            Noise/timestep labels of shape (batch_size,).
            Required if diffusion_model=True, otherwise optional.
        class_labels : torch.Tensor or None
            Class labels of shape (batch_size,) or (batch_size, label_dim).
            Can be None if label_dim is 0.
        augment_labels : torch.Tensor or None, optional
            Augmentation labels of shape (batch_size, augment_dim).
            Can be None if augment_dim is 0.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).

        Notes
        -----
        - The forward pass combines conditioning embeddings (noise, class, augmentation)
          and processes through encoder-decoder with skip connections.
        - When diffusion_model=False, the noise_labels can be omitted.
        - Classifier-free guidance is implemented via label_dropout during training.
        """
        # Mapping.
        emb = torch.zeros([1, self.map_layer1.in_features], device=x.device)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = self.map_label(tmp)
        if self.map_noise is not None:
            emb_n = self.map_noise(noise_labels)
            emb_n = silu(self.map_layer0(emb_n))
            emb_n = self.map_layer1(emb_n)
            emb = emb + emb_n
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)

        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation


class VPPrecond(torch.nn.Module):
    """
    Variance Preserving (VP) preconditioning for diffusion models.

    This class implements preconditioning for the Variance Preserving formulation
    of diffusion models, as described in Song et al. (2020). It wraps a base U-Net
    model and applies the appropriate scaling and conditioning for VP SDEs.

    Parameters
    ----------
    img_resolution : int or tuple
        Input image resolution. If int, assumes square images.
        If tuple, should be (height, width).
    in_channels : int
        Number of input color channels.
    out_channels : int
        Number of output color channels.
    label_dim : int, optional
        Number of class labels. Set to 0 for unconditional generation.
        Default is 0.
    use_fp16 : bool, optional
        Whether to execute the underlying model at FP16 precision for speed.
        Default is False.
    beta_d : float, optional
        Extent of the noise level schedule. Controls the rate of noise increase.
        Default is 19.9.
    beta_min : float, optional
        Initial slope of the noise level schedule.
        Default is 0.1.
    M : int, optional
        Original number of timesteps in the DDPM formulation.
        Default is 1000.
    epsilon_t : float, optional
        Minimum t-value used during training. Prevents numerical issues.
        Default is 1e-5.
    model_type : str, optional
        Class name of the underlying U-Net model ('SongUNet' or 'DhariwalUNet').
        Default is 'SongUNet'.
    **model_kwargs : dict
        Additional keyword arguments passed to the underlying model.

    Attributes
    ----------
    img_resolution : tuple
        Input image resolution as (height, width).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    label_dim : int
        Number of class labels.
    use_fp16 : bool
        Whether to use FP16 precision.
    sigma_min : float
        Minimum noise level (sigma) corresponding to epsilon_t.
    sigma_max : float
        Maximum noise level (sigma) corresponding to t=1.
    model : torch.nn.Module
        The underlying U-Net model.

    Notes
    -----
    - The VP formulation maintains unit variance throughout the diffusion process.
    - The noise schedule follows: σ(t) = sqrt(exp(0.5*β_d*t² + β_min*t) - 1)
    - The preconditioning applies scaling factors: c_skip, c_out, c_in, c_noise
    - Supports conditional generation via class labels and condition images.
    - Implements the continuous-time formulation of diffusion models.

    References
    ----------
    - Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", 2020.
    """

    def __init__(
        self,
        img_resolution,  # Image resolution as tuple (height, width)
        in_channels,  # Number of color channels.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        beta_d=19.9,  # Extent of the noise level schedule.
        beta_min=0.1,  # Initial slope of the noise level schedule.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        epsilon_t=1e-5,  # Minimum t-value used during training.
        model_type="SongUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        """
        Initialize the VPPrecond module.

        Parameters
        ----------
        img_resolution : int or tuple
            Image resolution.
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        label_dim : int, optional
            Class label dimension.
        use_fp16 : bool, optional
            Use FP16 precision.
        beta_d : float, optional
            Noise schedule extent.
        beta_min : float, optional
            Initial noise schedule slope.
        M : int, optional
            Number of timesteps.
        epsilon_t : float, optional
            Minimum t-value.
        model_type : str, optional
            Underlying model class name.
        **model_kwargs : dict
            Additional model arguments.
        """
        super().__init__()
        # Store resolution for compatibility
        if isinstance(img_resolution, (tuple, list)):
            self.img_resolution = img_resolution
        else:
            self.img_resolution = (img_resolution, img_resolution)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=self.out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x,
        sigma,
        condition_img=None,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        """
        Forward pass with VP preconditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input noisy tensor of shape (batch_size, in_channels, height, width).
        sigma : torch.Tensor
            Noise level(s) of shape (batch_size,) or scalar.
        condition_img : torch.Tensor, optional
            Condition image tensor of same spatial dimensions as x.
            Default is None.
        class_labels : torch.Tensor, optional
            Class labels for conditioning of shape (batch_size,) or (batch_size, label_dim).
            Default is None.
        force_fp32 : bool, optional
            Force FP32 precision even if use_fp16 is True.
            Default is False.
        **model_kwargs : dict
            Additional keyword arguments passed to the underlying model.

        Returns
        -------
        torch.Tensor
            Denoised output of shape (batch_size, out_channels, height, width).

        Notes
        -----
        - Applies the preconditioning: D(x) = c_skip * x + c_out * F(c_in * x, c_noise)
        - Where F is the underlying U-Net model.
        - c_in, c_out, c_skip, c_noise are computed from sigma according to VP formulation.
        - Condition images are concatenated along the channel dimension.
        """
        in_img = (
            torch.cat([x, condition_img], dim=1) if condition_img is not None else x
        )
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([1, self.label_dim], device=in_img.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)

        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and in_img.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(
            (c_in * in_img).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        ).to(dtype)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        """
        Compute noise level sigma for given time t.

        Parameters
        ----------
        t : float or torch.Tensor
            Time value(s) in [epsilon_t, 1].

        Returns
        -------
        torch.Tensor
            Noise level sigma corresponding to t.

        Notes
        -----
        Formula: σ(t) = sqrt(exp(0.5*β_d*t² + β_min*t) - 1)
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        """
        Inverse function: compute time t for given noise level sigma.

        Parameters
        ----------
        sigma : float or torch.Tensor
            Noise level(s).

        Returns
        -------
        torch.Tensor
            Time t corresponding to sigma.

        Notes
        -----
        Formula: t = (sqrt(β_min² + 2*β_d*log(1+σ²)) - β_min) / β_d
        """
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma):
        """
        Round noise level(s) for compatibility with discrete schedules.

        Parameters
        ----------
        sigma : float or torch.Tensor
            Noise level(s).

        Returns
        -------
        torch.Tensor
            Rounded noise level(s).
        """
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation


class VEPrecond(torch.nn.Module):
    """
    Variance Exploding (VE) preconditioning for diffusion models.

    This class implements preconditioning for the Variance Exploding formulation
    of diffusion models, as described in Song et al. (2020). It wraps a base U-Net
    model and applies the appropriate scaling and conditioning for VE SDEs.

    Parameters
    ----------
    img_resolution : int or tuple
        Input image resolution. If int, assumes square images.
        If tuple, should be (height, width).
    in_channels : int
        Number of input color channels.
    out_channels : int
        Number of output color channels.
    label_dim : int, optional
        Number of class labels. Set to 0 for unconditional generation.
        Default is 0.
    use_fp16 : bool, optional
        Whether to execute the underlying model at FP16 precision for speed.
        Default is False.
    sigma_min : float, optional
        Minimum supported noise level.
        Default is 0.02.
    sigma_max : float, optional
        Maximum supported noise level.
        Default is 100.
    model_type : str, optional
        Class name of the underlying U-Net model ('SongUNet' or 'DhariwalUNet').
        Default is 'SongUNet'.
    **model_kwargs : dict
        Additional keyword arguments passed to the underlying model.

    Attributes
    ----------
    img_resolution : tuple
        Input image resolution as (height, width).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    label_dim : int
        Number of class labels.
    use_fp16 : bool
        Whether to use FP16 precision.
    sigma_min : float
        Minimum noise level.
    sigma_max : float
        Maximum noise level.
    model : torch.nn.Module
        The underlying U-Net model.

    Notes
    -----
    - The VE formulation uses a simple geometric noise schedule.
    - The preconditioning applies scaling factors: c_skip, c_out, c_in, c_noise
    - c_noise = 0.5 * log(sigma) maps noise levels to conditioning inputs.
    - Supports conditional generation via class labels and condition images.

    References
    ----------
    - Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", 2020.
    """

    def __init__(
        self,
        img_resolution,  # Image resolution as tuple (height, width)
        in_channels,  # Number of color channels.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=100,  # Maximum supported noise level.
        model_type="SongUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        """
        Initialize the VEPrecond module.

        Parameters
        ----------
        img_resolution : int or tuple
            Image resolution.
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        label_dim : int, optional
            Class label dimension.
        use_fp16 : bool, optional
            Use FP16 precision.
        sigma_min : float, optional
            Minimum noise level.
        sigma_max : float, optional
            Maximum noise level.
        model_type : str, optional
            Underlying model class name.
        **model_kwargs : dict
            Additional model arguments.
        """
        super().__init__()
        # Store resolution for compatibility
        if isinstance(img_resolution, (tuple, list)):
            self.img_resolution = img_resolution
        else:
            self.img_resolution = (img_resolution, img_resolution)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=self.out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x,
        sigma,
        condition_img=None,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        """
        Forward pass with VE preconditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input noisy tensor of shape (batch_size, in_channels, height, width).
        sigma : torch.Tensor
            Noise level(s) of shape (batch_size,) or scalar.
        condition_img : torch.Tensor, optional
            Condition image tensor of same spatial dimensions as x.
            Default is None.
        class_labels : torch.Tensor, optional
            Class labels for conditioning of shape (batch_size,) or (batch_size, label_dim).
            Default is None.
        force_fp32 : bool, optional
            Force FP32 precision even if use_fp16 is True.
            Default is False.
        **model_kwargs : dict
            Additional keyword arguments passed to the underlying model.

        Returns
        -------
        torch.Tensor
            Denoised output of shape (batch_size, out_channels, height, width).

        Notes
        -----
        - Applies the preconditioning: D(x) = c_skip * x + c_out * F(c_in * x, c_noise)
        - Where F is the underlying U-Net model.
        - For VE: c_skip = 1, c_out = sigma, c_in = 1, c_noise = 0.5 * log(sigma)
        - Condition images are concatenated along the channel dimension.
        """

        in_img = (
            torch.cat([x, condition_img], dim=1) if condition_img is not None else x
        )
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([1, self.label_dim], device=in_img.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)

        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and in_img.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model(
            (c_in * in_img).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        ).to(dtype)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        """
        Round noise level(s) for compatibility with discrete schedules.

        Parameters
        ----------
        sigma : float or torch.Tensor
            Noise level(s).

        Returns
        -------
        torch.Tensor
            Rounded noise level(s).
        """
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation


class iDDPMPrecond(torch.nn.Module):
    """
    Improved DDPM (iDDPM) preconditioning for diffusion models.

    This class implements the improved preconditioning scheme from the iDDPM paper,
    which refines the noise schedule and preconditioning for better sample quality.
    It provides a bridge between discrete-time DDPM formulations and continuous-time
    SDE formulations.

    Parameters
    ----------
    img_resolution : int or tuple
        Input image resolution. If int, assumes square images.
        If tuple, should be (height, width).
    in_channels : int
        Number of input color channels.
    out_channels : int
        Number of output color channels.
    label_dim : int, optional
        Number of class labels. Set to 0 for unconditional generation.
        Default is 0.
    use_fp16 : bool, optional
        Whether to execute the underlying model at FP16 precision for speed.
        Default is False.
    C_1 : float, optional
        Timestep adjustment parameter for low noise levels.
        Default is 0.001.
    C_2 : float, optional
        Timestep adjustment parameter for high noise levels.
        Default is 0.008.
    M : int, optional
        Original number of timesteps in the DDPM formulation.
        Default is 1000.
    model_type : str, optional
        Class name of the underlying U-Net model ('SongUNet' or 'DhariwalUNet').
        Default is 'DhariwalUNet'.
    **model_kwargs : dict
        Additional keyword arguments passed to the underlying model.

    Attributes
    ----------
    img_resolution : tuple
        Input image resolution as (height, width).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    label_dim : int
        Number of class labels.
    use_fp16 : bool
        Whether to use FP16 precision.
    sigma_min : float
        Minimum noise level (learned from schedule).
    sigma_max : float
        Maximum noise level (learned from schedule).
    u : torch.Tensor (buffer)
        Learned noise schedule of length M+1.
    model : torch.nn.Module
        The underlying U-Net model.

    Notes
    -----
    - The iDDPM formulation improves upon DDPM with a refined noise schedule.
    - The noise schedule is learned during initialization via backward recursion.
    - Uses alpha_bar schedule: ᾱ(j) = sin(π/2 * j/M/(C₂+1))²
    - Implements discrete-time preconditioning with improved numerical stability.

    References
    ----------
    - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", 2021.
    """

    def __init__(
        self,
        img_resolution,  # Image resolution as tuple (height, width)
        in_channels,  # Number of color channels.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        C_1=0.001,  # Timestep adjustment at low noise levels.
        C_2=0.008,  # Timestep adjustment at high noise levels.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        """
        Initialize the iDDPMPrecond module.

        Parameters
        ----------
        img_resolution : int or tuple
            Image resolution.
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        label_dim : int, optional
            Class label dimension.
        use_fp16 : bool, optional
            Use FP16 precision.
        C_1 : float, optional
            Low noise adjustment.
        C_2 : float, optional
            High noise adjustment.
        M : int, optional
            Number of timesteps.
        model_type : str, optional
            Underlying model class name.
        **model_kwargs : dict
            Additional model arguments.
        """
        super().__init__()
        # Store resolution for compatibility
        if isinstance(img_resolution, (tuple, list)):
            self.img_resolution = img_resolution
        else:
            self.img_resolution = (img_resolution, img_resolution)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=self.out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1)
                / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1)
                - 1
            ).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(
        self,
        x,
        sigma,
        condition_img=None,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        """
        Forward pass with iDDPM preconditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input noisy tensor of shape (batch_size, in_channels, height, width).
        sigma : torch.Tensor
            Noise level(s) of shape (batch_size,) or scalar.
        condition_img : torch.Tensor, optional
            Condition image tensor of same spatial dimensions as x.
            Default is None.
        class_labels : torch.Tensor, optional
            Class labels for conditioning of shape (batch_size,) or (batch_size, label_dim).
            Default is None.
        force_fp32 : bool, optional
            Force FP32 precision even if use_fp16 is True.
            Default is False.
        **model_kwargs : dict
            Additional keyword arguments passed to the underlying model.

        Returns
        -------
        torch.Tensor
            Denoised output of shape (batch_size, out_channels, height, width).

        Notes
        -----
        - Applies the preconditioning: D(x) = c_skip * x + c_out * F(c_in * x, c_noise)
        - Where F is the underlying U-Net model.
        - For iDDPM: c_skip = 1, c_out = -σ, c_in = 1/√(σ²+1)
        - Condition images are concatenated along the channel dimension.
        - c_noise maps sigma to discrete timesteps for the underlying model.
        """
        if condition_img is not None:
            in_img = torch.cat([x, condition_img], dim=1)  # [B, C + C_cond, H, W]
        else:
            in_img = x

        sigma = sigma.reshape(-1, 1, 1, 1)
        # Prepare class labels
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros(
                [in_img.shape[0], self.label_dim], device=in_img.device
            )
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)

        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and in_img.device.type == "cuda")
            else torch.float32
        )

        # Diffusion coefficients
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        # Noise label calculation for model
        c_noise = (
            self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        )

        # Forward pass through underlying model
        F_x = self.model(
            (c_in * in_img).to(dtype),
            noise_labels=c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        ).to(dtype)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, : self.in_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        """
        Compute alpha_bar for timestep j in the improved schedule.

        Parameters
        ----------
        j : int or torch.Tensor
            Timestep index (0 <= j <= M).

        Returns
        -------
        torch.Tensor
            ᾱ(j) = sin(π/2 * j/M/(C₂+1))²
        """
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        """
        Round noise level(s) to the nearest value in the learned schedule.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise level(s).
        return_index : bool, optional
            If True, return the index in the schedule instead of the value.
            Default is False.

        Returns
        -------
        torch.Tensor
            Rounded noise level(s) or indices.
        """
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
            self.u.reshape(1, -1, 1),
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


class EDMPrecond(torch.nn.Module):
    """
    EDM preconditioning for diffusion models.

    This class implements the EDM (Elucidating Diffusion Models) preconditioning
    scheme, which provides a unified framework for various diffusion formulations
    with optimized preconditioning coefficients.

    Parameters
    ----------
    img_resolution : int or tuple
        Input image resolution. If int, assumes square images.
        If tuple, should be (height, width).
    in_channels : int
        Number of input color channels.
    out_channels : int
        Number of output color channels.
    label_dim : int, optional
        Number of class labels. Set to 0 for unconditional generation.
        Default is 0.
    use_fp16 : bool, optional
        Whether to execute the underlying model at FP16 precision for speed.
        Default is False.
    sigma_min : float, optional
        Minimum supported noise level.
        Default is 0.
    sigma_max : float, optional
        Maximum supported noise level.
        Default is float('inf').
    sigma_data : float, optional
        Standard deviation of the training data.
        Default is 1.0.
    model_type : str, optional
        Class name of the underlying U-Net model ('SongUNet' or 'DhariwalUNet').
        Default is 'DhariwalUNet'.
    **model_kwargs : dict
        Additional keyword arguments passed to the underlying model.

    Attributes
    ----------
    img_resolution : tuple
        Input image resolution as (height, width).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    label_dim : int
        Number of class labels.
    use_fp16 : bool
        Whether to use FP16 precision.
    sigma_min : float
        Minimum noise level.
    sigma_max : float
        Maximum noise level.
    sigma_data : float
        Training data standard deviation.
    model : torch.nn.Module
        The underlying U-Net model.

    Notes
    -----
    - The EDM formulation provides a unified preconditioning scheme that
      generalizes VP, VE, and other diffusion formulations.
    - Preconditioning coefficients: c_skip = σ_data²/(σ²+σ_data²)
      c_out = σ·σ_data/√(σ²+σ_data²), c_in = 1/√(σ_data²+σ²)
    - c_noise = log(σ)/4 provides the noise conditioning input.
    - This formulation often yields better sample quality and training stability.

    References
    ----------
    - Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022.
    """

    def __init__(
        self,
        img_resolution,  # Image resolution.
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        label_dim=0,  # Number of class labels.
        use_fp16=False,  # FP16 execution?
        sigma_min=0,  # Min noise level.
        sigma_max=float("inf"),  # Max noise level.
        sigma_data=1.0,  # Training data std.
        model_type="DhariwalUNet",  # Underlying model class.
        **model_kwargs,  # Keyword args.
    ):
        """
        Initialize the EDMPrecond module.

        Parameters
        ----------
        img_resolution : int or tuple
            Image resolution.
        in_channels : int
            Input channels.
        out_channels : int
            Output channels.
        label_dim : int, optional
            Class label dimension.
        use_fp16 : bool, optional
            Use FP16 precision.
        sigma_min : float, optional
            Minimum noise level.
        sigma_max : float, optional
            Maximum noise level.
        sigma_data : float, optional
            Training data standard deviation.
        model_type : str, optional
            Underlying model class name.
        **model_kwargs : dict
            Additional model arguments.
        """
        super().__init__()
        # Store resolution for compatibility
        if isinstance(img_resolution, (tuple, list)):
            self.img_resolution = img_resolution
        else:
            self.img_resolution = (img_resolution, img_resolution)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        # keep names exactly the same
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x,
        sigma,
        condition_img=None,
        class_labels=None,
        force_fp32=True,
        **model_kwargs,
    ):
        """
        Forward pass with EDM preconditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input noisy tensor of shape (batch_size, in_channels, height, width).
        sigma : torch.Tensor
            Noise level(s) of shape (batch_size,) or scalar.
        condition_img : torch.Tensor, optional
            Condition image tensor of same spatial dimensions as x.
            Default is None.
        class_labels : torch.Tensor, optional
            Class labels for conditioning of shape (batch_size,) or (batch_size, label_dim).
            Default is None.
        force_fp32 : bool, optional
            Force FP32 precision even if use_fp16 is True.
            Default is True.
        **model_kwargs : dict
            Additional keyword arguments passed to the underlying model.

        Returns
        -------
        torch.Tensor
            Denoised output of shape (batch_size, out_channels, height, width).

        Notes
        -----
        - Applies the EDM preconditioning: D(x) = c_skip * x + c_out * F(c_in * x, c_noise)
        - Where F is the underlying U-Net model.
        - EDM coefficients: c_skip = σ_data²/(σ²+σ_data²)
          c_out = σ·σ_data/√(σ²+σ_data²), c_in = 1/√(σ_data²+σ²)
        - Condition images are concatenated along the channel dimension.
        - c_noise = log(σ)/4 provides the noise conditioning.
        """
        # -----------------------------
        # Input concatenation
        # -----------------------------
        if condition_img is not None:
            in_img = torch.cat([x, condition_img], dim=1)
        else:
            in_img = x

        sigma = sigma.reshape(-1, 1, 1, 1)

        # -----------------------------
        # Class labels (same variable name, UNet#2-compatible)
        # -----------------------------
        if self.label_dim == 0:
            class_labels = None
        else:
            if class_labels is None:
                class_labels = torch.zeros(
                    [in_img.shape[0], self.label_dim], device=in_img.device
                )
            else:
                class_labels = class_labels.to(torch.float32).reshape(
                    -1, self.label_dim
                )

        # -----------------------------
        # Precision logic
        # -----------------------------
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and in_img.device.type == "cuda")
            else torch.float32
        )

        # -----------------------------
        # EDM coefficients
        # -----------------------------
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_noise = sigma.log() / 4

        # -----------------------------
        # Call the UNet
        # -----------------------------
        F_x = self.model(
            (c_in * in_img).to(dtype),
            noise_labels=c_noise.flatten(),  # required by UNet
            class_labels=class_labels,  # UNet optional labels
            **model_kwargs,
        ).to(dtype)

        # -----------------------------
        # Output reconstruction
        # -----------------------------
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        """
        Round noise level(s) for compatibility with discrete schedules.

        Parameters
        ----------
        sigma : float or torch.Tensor
            Noise level(s).

        Returns
        -------
        torch.Tensor
            Rounded noise level(s).

        Notes
        -----
        In EDM, sigma is continuous, so rounding is typically a no-op
        unless implementing a discrete schedule variant.
        """
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
#
class TestDiffusionNetworks(unittest.TestCase):
    """Unit tests for diffusion network architectures."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = 3
        self.cond_channels = 2
        self.out_channels = 3
        self.label_dim = 2

        if self.logger:
            self.logger.info(f"Test setup complete - using device: {self.device}")

    def test_song_unet_square_resolution(self):
        """Test SongUNet with square resolution."""
        if self.logger:
            self.logger.info("Testing SongUNet with square resolution")

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
        class_labels = torch.randint(0, self.label_dim, (self.batch_size,)).to(
            self.device
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
        """Test SongUNet with rectangular resolution."""
        if self.logger:
            self.logger.info("Testing SongUNet with rectangular resolution")

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
        class_labels = torch.randint(0, self.label_dim, (self.batch_size,)).to(
            self.device
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

    def test_dhariwal_unet(self):
        """Test DhariwalUNet architecture."""
        if self.logger:
            self.logger.info("Testing DhariwalUNet")

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
        class_labels = (
            torch.randint(0, self.label_dim, (self.batch_size,)).to(self.device).float()
        )  # Convert to float

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
        sigma = torch.tensor([0.1, 0.5], device=self.device)
        class_labels = torch.randint(
            0, self.label_dim, (self.batch_size, 2), device=self.device
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
        sigma = torch.tensor([0.1, 0.5], device=self.device)
        class_labels = torch.randint(
            0, self.label_dim, (self.batch_size, 2), device=self.device
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
        sigma = torch.tensor([0.1, 0.5], device=self.device)
        class_labels = torch.randint(
            0, self.label_dim, (self.batch_size, 2), device=self.device
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
                {"model_channels": 32, "channel_mult": [1, 2]},
            ),
            (
                "SongUNet-Medium",
                SongUNet,
                {"model_channels": 64, "channel_mult": [1, 2, 2]},
            ),
            (
                "DhariwalUNet-Small",
                DhariwalUNet,
                {"model_channels": 32, "channel_mult": [1, 2]},
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


# ----------------------------------------------------------------------------
