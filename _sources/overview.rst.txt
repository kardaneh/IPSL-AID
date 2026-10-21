Overview
========

IPSL-AID enables:

- Global-to-regional climate downscaling
- Multi-variable spatio-temporal generation
- Diffusion-based generative modeling
- Scalable training and inference on multi-GPU systems
- HPC-friendly workflows using SLURM and batch scripts

The codebase is modular, extensively configurable, and designed to be explored
**incrementally**, starting from unit tests before full model training.

Key Features
------------

- **Multiple Diffusion Formulations**: VE, VP, EDM, iDDPM
- **UNet-based Architectures**: ADM-style and conditional variants
- **Global Training**: Random block sampling across the globe
- **Flexible Inference**: Global, regional, or domain-specific
- **HPC Integration**: SLURM-ready with full reproducibility
- **Comprehensive Testing**: Module-level tests for validation

Use Cases
---------

- High-resolution climate projections
- Regional climate impact assessments
- Ensemble generation for uncertainty quantification
- Data augmentation for climate studies
- Downscaling of global climate model outputs

Background
----------

Anthropogenic climate change poses substantial risks to critical socio-economic
sectors, such as agriculture, forestry, energy, and water supply. The development
of effective adaptation and mitigation strategies requires accessible, high-resolution
climate projections to anticipate impacts at both local and regional scales.

General circulation models (GCMs), which are widely used in climate research,
typically operate at spatial resolutions of approximately 150–200 km. This coarse
resolution is insufficient to capture essential fine-scale processes, especially
those shaped by topography, land-sea contrast, and surface heterogeneity, all of
which are vital to regional climate dynamics and extreme weather events. Therefore,
downscaling climate model outputs to finer resolutions is necessary to provide
relevant climate information at the local level.

IPSL-AID addresses this need by providing a diffusion-based generative model
for efficient climate downscaling with uncertainty estimation.
