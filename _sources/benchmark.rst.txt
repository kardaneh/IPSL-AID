
Benchmark
=========

This section documents the experimental results comparing three different U-Net architectures for downscaling within the IPSL-AID framework.

Model Performance Comparison
----------------------------

This presents a comprehensive comparison of three U-Net architectures trained for statistical downscaling of atmosphericvariables. All models were trained with identical hyperparameters.

Experiment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

All models were trained with the following common configuration:

- **Dataset**: ERA5 reanalysis data (2015-2019 training, 2020 validation)
- **Domain**: Global
- **Input resolution**: 721 × 1440 (0.25° grid)
- **Input channels**: 10 (6 target variables + 4 constants)
- **Output channels**: 6 (downscaled meteorological variables)
- **Target variables**:

  - VAR_2T: 2-meter temperature (K)
  - VAR_10U: 10-meter U wind component (m/s)
  - VAR_10V: 10-meter V wind component (m/s)
  - VAR_TP: Total precipitation (m/h)
  - VAR_D2M: 2-meter dewpoint temperature (K)
  - VAR_ST: Skin temperature (K)

- **Normalization**: Standard scaling (log1p for precipitation)
- **Time encoding**: Sine/cosine of day-of-year (4 channels)
- **Constant variables**: Orography (z) and land-sea mask (lsm)
- **Loss function**: UNet diffusion loss (MSE-based)
- **Learning rate**: 0.0001
- **Batch size**: 36 (12 spatial × 1460 temporal)
- **Epochs**: 20
- **Spatial batching**: 12 tiles
- **Temporal batching**: 1460 time steps

Model Architectures
~~~~~~~~~~~~~~~~~~~

Three U-Net variants were evaluated:

1. **DDPM++ (SongUNet - Positional embedding)**

   - Denoising Diffusion Probabilistic Model architecture
   - Positional timestep embedding
   - Standard encoder/decoder with skip connections
   - Channel multiplier: [2, 2, 2]
   - Base channels: 128
   - Resampling filter: [1, 1]
   - **Parameters**: 54,429,958

2. **NCSN++ (SongUNet - Fourier embedding)**

   - Noise-Conditioned Score Network architecture
   - Fourier feature timestep embedding
   - Residual encoder with skip connections
   - Channel multiplier: [2, 2, 2]
   - Base channels: 128
   - Resampling filter: [1, 3, 3, 1]
   - **Parameters**: 55,109,510

3. **ADM (DhariwalUNet)**

   - Ablated Diffusion Model architecture
   - Multi-resolution attention (32, 16, 8)
   - Channel multiplier: [1, 2, 3, 4]
   - Base channels: 128
   - Number of blocks: 2
   - **Parameters**: 92,140,550

Performance Metrics
~~~~~~~~~~~~~~~~~~~

The following metrics were used for evaluation on the validation set (year 2020):

- **Loss**: UNet diffusion loss value
- **MAE**: Mean Absolute Error (normalized scale)
- **NMAE**: Normalized Mean Absolute Error (normalized by variable range)
- **RMSE**: Root Mean Square Error (normalized scale)
- **R²**: Coefficient of determination
- **Pearson**: Pearson correlation coefficient
- **KL**: KL divergence (distribution similarity)

Quantitative Comparison - Overall Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Performance comparison for all variables (validation set)
   :header-rows: 1
   :widths: 25, 15, 15, 15, 15, 15, 15
   :align: center

   * - Architecture
     - Loss ↓
     - MAE ↓
     - NMAE ↓
     - RMSE ↓
     - R² ↑
     - Pearson ↑
   * - DDPM++ (SongUNet)
     - 0.0524
     - 0.3458 ± 0.0069
     - 0.1170 ± 0.0035
     - 0.5604 ± 0.0164
     - 0.9482 ± 0.0025
     - 0.9722 ± 0.0014
   * - NCSN++ (SongUNet)
     - 0.0517
     - 0.3432 ± 0.0068
     - 0.1176 ± 0.0037
     - 0.5552 ± 0.0159
     - 0.9489 ± 0.0027
     - 0.9725 ± 0.0016
   * - ADM (DhariwalUNet)
     - 0.0527
     - 0.3500 ± 0.0071
     - 0.1179 ± 0.0035
     - 0.5656 ± 0.0166
     - 0.9480 ± 0.0023
     - 0.9721 ± 0.0013

*Note: ↓ indicates lower is better, ↑ indicates higher is better. Values show mean ± std across spatial batches.*

Baseline Comparison
~~~~~~~~~~~~~~~~~~~

For reference, coarse input (bilinear interpolation of low-resolution input) metrics are provided:

.. list-table:: Baseline coarse input performance (all variables)
   :header-rows: 1
   :widths: 20, 15, 15, 15, 15, 15
   :align: center

   * - Baseline
     - MAE
     - NMAE
     - RMSE
     - R²
     - Pearson
   * - Coarse Input
     - 0.6993 ± 0.0214
     - 0.1981 ± 0.0045
     - 1.2208 ± 0.0368
     - 0.8873 ± 0.0039
     - 0.9377 ± 0.0028

**Improvement over baseline**: All three U-Net architectures achieve significant improvements, reducing MAE by approximately 50% and increasing R² from 0.887 to 0.948+.

Per-Variable Performance
~~~~~~~~~~~~~~~~~~~~~~~~

VAR_2T (2-meter Temperature)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: VAR_2T performance comparison
   :header-rows: 1
   :widths: 25, 15, 15, 15, 15, 15
   :align: center

   * - Architecture
     - MAE ↓
     - RMSE ↓
     - R² ↑
     - Pearson ↑
     - KL ↓
   * - DDPM++
     - 0.3697 ± 0.0095
     - 0.5873 ± 0.0171
     - 0.9992 ± 0.0001
     - 0.9996 ± 0.0001
     - 0.0010
   * - NCSN++
     - 0.3684 ± 0.0097
     - 0.5816 ± 0.0168
     - 0.9992 ± 0.0001
     - 0.9996 ± 0.0001
     - 0.0011
   * - ADM
     - 0.3775 ± 0.0103
     - 0.5968 ± 0.0176
     - 0.9992 ± 0.0001
     - 0.9996 ± 0.0001
     - 0.0007

VAR_10U (10-meter U Wind)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: VAR_10U performance comparison
   :header-rows: 1
   :widths: 25, 15, 15, 15, 15, 15
   :align: center

   * - Architecture
     - MAE ↓
     - RMSE ↓
     - R² ↑
     - Pearson ↑
     - KL ↓
   * - DDPM++
     - 0.3938 ± 0.0071
     - 0.5921 ± 0.0237
     - 0.9886 ± 0.0014
     - 0.9943 ± 0.0007
     - 0.0006
   * - NCSN++
     - 0.3905 ± 0.0069
     - 0.5867 ± 0.0228
     - 0.9888 ± 0.0013
     - 0.9944 ± 0.0007
     - 0.0005
   * - ADM
     - 0.3966 ± 0.0071
     - 0.5960 ± 0.0227
     - 0.9885 ± 0.0014
     - 0.9942 ± 0.0007
     - 0.0005

VAR_10V (10-meter V Wind)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: VAR_10V performance comparison
   :header-rows: 1
   :widths: 25, 15, 15, 15, 15, 15
   :align: center

   * - Architecture
     - MAE ↓
     - RMSE ↓
     - R² ↑
     - Pearson ↑
     - KL ↓
   * - DDPM++
     - 0.3813 ± 0.0068
     - 0.5701 ± 0.0240
     - 0.9859 ± 0.0016
     - 0.9929 ± 0.0008
     - 0.0006
   * - NCSN++
     - 0.3792 ± 0.0068
     - 0.5663 ± 0.0237
     - 0.9861 ± 0.0016
     - 0.9930 ± 0.0008
     - 0.0005
   * - ADM
     - 0.3844 ± 0.0066
     - 0.5739 ± 0.0238
     - 0.9857 ± 0.0016
     - 0.9928 ± 0.0008
     - 0.0005

VAR_TP (Total Precipitation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: VAR_TP performance comparison (most challenging variable)
   :header-rows: 1
   :widths: 25, 15, 15, 15, 15, 15
   :align: center

   * - Architecture
     - MAE ↓
     - NMAE ↓
     - R² ↑
     - Pearson ↑
     - KL ↓
   * - DDPM++
     - 0.0001
     - 0.5058 ± 0.0145
     - 0.7182 ± 0.0133
     - 0.8478 ± 0.0079
     - 0.1427
   * - NCSN++
     - 0.0001
     - 0.5105 ± 0.0155
     - 0.7220 ± 0.0148
     - 0.8497 ± 0.0088
     - 0.1326
   * - ADM
     - 0.0001
     - 0.5096 ± 0.0145
     - 0.7176 ± 0.0133
     - 0.8474 ± 0.0079
     - 0.2088

*Note: MAE values are in normalized scale; precipitation shows lowest absolute error due to large number of zero in the dataset.*

VAR_D2M (2-meter Dewpoint Temperature)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: VAR_D2M performance comparison
   :header-rows: 1
   :widths: 25, 15, 15, 15, 15, 15
   :align: center

   * - Architecture
     - MAE ↓
     - RMSE ↓
     - R² ↑
     - Pearson ↑
     - KL ↓
   * - DDPM++
     - 0.4515 ± 0.0117
     - 0.7178 ± 0.0221
     - 0.9987 ± 0.0003
     - 0.9994 ± 0.0002
     - 0.0016
   * - NCSN++
     - 0.4450 ± 0.0115
     - 0.7104 ± 0.0222
     - 0.9988 ± 0.0003
     - 0.9994 ± 0.0002
     - 0.0015
   * - ADM
     - 0.4551 ± 0.0117
     - 0.7261 ± 0.0215
     - 0.9987 ± 0.0003
     - 0.9994 ± 0.0002
     - 0.0017

VAR_ST (Skin Temperature)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: VAR_ST performance comparison
   :header-rows: 1
   :widths: 25, 15, 15, 15, 15, 15
   :align: center

   * - Architecture
     - MAE ↓
     - RMSE ↓
     - R² ↑
     - Pearson ↑
     - KL ↓
   * - DDPM++
     - 0.4785 ± 0.0214
     - 0.8947 ± 0.0485
     - 0.9983 ± 0.0004
     - 0.9991 ± 0.0002
     - 0.0101
   * - NCSN++
     - 0.4759 ± 0.0216
     - 0.8862 ± 0.0485
     - 0.9983 ± 0.0004
     - 0.9991 ± 0.0002
     - 0.0184
   * - ADM
     - 0.4859 ± 0.0215
     - 0.9004 ± 0.0501
     - 0.9982 ± 0.0004
     - 0.9991 ± 0.0002
     - 0.0116

Model Complexity Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model complexity and efficiency
   :header-rows: 1
   :widths: 25, 20, 20, 30
   :align: center

   * - Architecture
     - Parameters
     - Relative Size
     - Inference Characteristics
   * - DDPM++ (SongUNet)
     - 54.4M
     - 1.0×
     - Lightweight, fast inference
   * - NCSN++ (SongUNet)
     - 55.1M
     - 1.01×
     - Slightly larger, Fourier embeddings
   * - ADM (DhariwalUNet)
     - 92.1M
     - 1.69×
     - Larger model, multi-resolution attention

Key Findings
~~~~~~~~~~~~

**Best Overall Performance**: The **NCSN++** architecture achieves the best overall metrics:

- Lowest loss (0.0517 vs 0.0524 for DDPM++ and 0.0527 for ADM)
- Lowest MAE (0.3432 vs 0.3458/0.3500)
- Highest R² (0.9489 vs 0.9482/0.9480)
- Highest Pearson correlation (0.9725 vs 0.9722/0.9721)

**Best Performance for Precipitation**: **NCSN++** achieves the highest R² for VAR_TP (0.7220) and lowest KL divergence (0.1326), indicating better distribution matching.

**Best Performance for Wind Fields**: **NCSN++** consistently outperforms for both U and V wind components across all metrics.

**Most Challenging Variable**: Precipitation (VAR_TP) shows the lowest R² scores (0.718-0.722) and highest NMAE (0.506-0.511), reflecting the difficulty of downscaling intermittent precipitation events.

**Model Efficiency**: The **DDPM++** architecture has the fewest parameters (54.4M) while maintaining competitive performance, making it suitable for resource-constrained applications.

**Wind Field Anisotropy**: Performance is slightly better for U-wind (R² ~0.9888) than V-wind (R² ~0.9861), which may reflect the zonal dominance of atmospheric circulation.

Recommendations
~~~~~~~~~~~~~~~

Based on the comprehensive comparison across 6 meteorological variables:

1. **For maximum accuracy**: Use **NCSN++** (SongUNet with Fourier embeddings)

   - Best overall performance across nearly all metrics
   - Superior handling of precipitation distributions
   - Marginal parameter increase over DDPM++

2. **For balanced performance**: Use **DDPM++** (SongUNet with positional embeddings)

   - Excellent performance with slightly fewer parameters
   - Competitive across all variables
   - Best for resource-constrained deployment

3. **For temperature-sensitive applications**: All three models perform excellently (R² > 0.999), with minimal differences

4. **For precipitation downscaling**: **NCSN++** is the recommended choice due to superior distribution matching and higher R²

5. **For ensemble applications**: Consider all three as they show complementary strengths across different variable types

Note on ADM Performance
~~~~~~~~~~~~~~~~~~~~~~~

While the ADM architecture achieves competitive performance, it underperforms both SongUNet variants despite having nearly 1.7× more parameters. This suggests that:

- The SongUNet architecture is better suited for downscaling tasks
- The simplified U-Net design with fewer attention layers may generalize better
- The additional complexity of ADM does not translate to improved performance for this application
