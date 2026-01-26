Neural Architectures
====================

IPSL-AID relies on **UNet-based architectures**, adapted for climate data:

- ADM-style UNet
- Conditional UNet variants
- Support for static and dynamic covariates
- Flexible input / output channel definitions

Architectures are selected using runtime parameters, allowing rapid experimentation
without code changes.

Default Configuration
---------------------

The U-Net configuration includes:

- Base channel count: :math:`C_\mathrm{base} = 128`
- Channel multipliers per resolution: `[1, 2, 3, 4]`
- Residual blocks per resolution: 3
- Self-attention at resolutions: `[32, 16, 8]`
- Dropout probability: :math:`p = 0.10`
- Embedding dimension: :math:`C_\mathrm{emb} = 4 \times C_\mathrm{base}`

Architecture Components
-----------------------

**Encoder**
   Progressive downsampling with convolutional and residual blocks.
   At level :math:`l`, feature map has height :math:`H_{\mathrm l} = \lfloor H / 2^{\mathrm l} \rfloor`
   and width :math:`W_{\mathrm l} = \lfloor W / 2^{\mathrm l} \rfloor`.

**Decoder**
   Mirror of encoder with upsampling and skip connections from encoder.

**Attention**
   Multi-head self-attention at specific resolutions (64 channels per head):

   .. math::

      \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\Big({\mathbf{Q}^\top \mathbf{K}}/{\sqrt{d_{\mathrm k}}}\Big)\mathbf{V}

**Conditioning**
   Support for:
   - Noise level embeddings
   - Class conditioning (e.g., season, region)
   - Augmentation embeddings
   - Spatiotemporal context (latitude, longitude, time)

Embedding Layers
----------------

Noise levels :math:`\sigma` are represented using sinusoidal positional embedding:

.. math::

   \mathbf{e}_\sigma = \text{PE}(\sigma) \in \mathbb{R}^{C_\mathrm{base}}

Processed by two fully connected layers with SiLU activations.

Conditioning Strategies
-----------------------

1. **Spatial Conditioning**: Low-resolution inputs concatenated channel-wise
2. **Global Conditioning**: Scalar features projected and added to embeddings
3. **Adaptive Normalization**: Feature-wise modulation based on conditioning
4. **Cross-Attention**: Attention between features and conditioning vectors

Input/Output Specification
--------------------------

.. code-block:: python

   class UNet(nn.Module):
       def __init__(
         self,
         img_resolution,                     # Image resolution as tuple (height, width)
         in_channels,                        # Number of color channels at input.
         out_channels,                       # Number of color channels at output.
         label_dim           = 0,            # Number of class labels, 0 = unconditional.
         augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.
         model_channels      = 128,          # Base multiplier for the number of channels.
         channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
         channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
         num_blocks          = 3,            # Number of residual blocks per resolution.
         attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
         dropout             = 0.10,         # List of resolutions with self-attention.
         label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
         diffusion_model = True              # Whether to use the Unet for diffusion models.
         ):
         ...

Climate-Specific Adaptations
----------------------------

1. **Periodic Boundary Handling**: Special convolutions for longitude wrapping
2. **Spatial Context**: Incorporation of latitude/longitude grids
3. **Topography Integration**: Terrain elevation as conditioning input
4. **Land-Sea Masks**: Binary masks for land/ocean differentiation

Configuration Examples
----------------------

.. code-block:: yaml

   architecture:
     type: "DhariwalUNet"
     base_channels: 128
     channel_mult: [1, 2, 3, 4]
     num_res_blocks: 3
     attention_resolutions: [32, 16, 8]
     dropout: 0.1
     use_attention: true
     conditioning: "concat"

   input:
     variables: ["t2m", "u10", "v10"]
     static_features: ["lat", "lon", "z", "lsm"]
     dynamic_features: ["day_of_year", "hour_of_day"]

Performance Considerations
--------------------------

- **Memory**: Channel multipliers control memory usage
- **Speed**: Attention layers can be computationally expensive
- **Accuracy**: More channels/residual blocks generally improve quality
- **Overfitting**: Dropout and regularization important for small datasets
