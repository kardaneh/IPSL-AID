Climate Informatics 2026 – Hackathon
====================================

Problem: How do AI weather models "see" clouds?
-----------------------------------------------

This project was developed as part of the **Climate Informatics 2026 Hackathon**.
The goal was to infer **total cloud cover (TCC)** from atmospheric variables and
evaluate how well models generalize across regions and AI-generated datasets.

Datasets
-----------

We used two main datasets:

- **ERA5 (ECMWF global reanalysis dataset)**

  .. note::
     Dataset entity: :contentReference[oaicite:0]{index=0}

  - Resolution: 1.5°
  - Fully observed (used for training & evaluation)

- **AIMIP Phase 1 (AI Model Intercomparison Project dataset)**

  .. note::
     Dataset entity: :contentReference[oaicite:1]{index=1}

  - Resolution: 1.5°
  - AI-generated data with **masked cloud cover**

Inputs
---------

Atmospheric variables at **7 pressure levels**:

- Temperature (**T**)
- Specific humidity (**Q**)
- Zonal wind (**U**)
- Meridional wind (**V**)
- + Auxiliary variables

Target
---------

- **Total Cloud Cover (TCC)**

Spatial Setup
----------------

- **Region 1**: 64 × 64 grid
- **Region 2**: 64 × 64 grid

🧪 Data Split
-------------

+--------------+--------+----------+----------------------+
| Split        | Dataset| Region   | Time Period          |
+==============+========+==========+======================+
| Train        | ERA5   | Region 1 | 1979–2018            |
+--------------+--------+----------+----------------------+
| Validation   | ERA5   | Region 1 | 2019                 |
+--------------+--------+----------+----------------------+
| Test (obs)   | ERA5   | R1 + R2  | 2019                 |
+--------------+--------+----------+----------------------+
| Test (masked)| AIMIP  | R1 + R2  | masked               |
+--------------+--------+----------+----------------------+
| Test (masked)| ERA5   | R1 + R2  | masked               |
+--------------+--------+----------+----------------------+

Task
-------

- Train models on **ERA5 (Region 1)**
- Generalize to:

  - unseen region (**Region 2**)
  - unseen distribution (**AIMIP**)

- Predict **cloud cover on masked datasets**
- Use a **selection of 3 models**

Evaluation
-------------

**Deterministic metric**

- Mean Absolute Error (**MAE**)

  - ERA5 Region 1
  - ERA5 Region 2

**Probabilistic metric**

- Fair Continuous Ranked Probability Score (**CRPS**)

  - AIMIP Region 1
  - AIMIP Region 2

Scoring
----------

Final score is the **average skill score relative to a baseline**:

::

   Total Score = (1/4) * Σ (1 - Score_i / Score_baseline)

- Baseline: Sundqvist Scheme

  .. note::
     Baseline entity: :contentReference[oaicite:2]{index=2}

- Higher = better

Our Approach
---------------

We adapted **IPSL-AID**, a diffusion-based generative model, to infer cloud cover
from atmospheric states:

- **Architecture**: DhariwalUNet with the following configuration:

  .. code-block:: python

     DhariwalUNet(
         in_channels=30,
         out_channels=1,
         label_dim=0,
         dropout=0.2,
         model_channels=128,
         channel_mult=[1, 2, 3, 4],
         num_blocks=3,
     )

- **Training setup**:
  - Batch size: 32
  - Epochs: 15
  - Learning rate: 0.00015

- Repurposed for **diagnostic cloud prediction**
- Captured **spatial structure & uncertainty**
- Designed for **cross-dataset generalization**

Results – Leaderboard
------------------------

+-------------------+-------+----------------+----------------+--------------------+--------------------+
| Team Name         | Score | MAE ERA5 R1    | MAE ERA5 R2    | CRPS AIMIP R1      | CRPS AIMIP R2      |
+===================+=======+================+================+====================+====================+
| **IPSL AID**      | 0.266 | 0.079          | 0.106          | 0.125              | 0.137              |
+-------------------+-------+----------------+----------------+--------------------+--------------------+

💡 Key Challenges
-----------------

- Indirect learning of clouds (not explicitly modeled in inputs)
- Domain shift: ERA5 → AIMIP
- Spatial generalization (Region 1 → Region 2)
- Deterministic + probabilistic evaluation

Leaderboard
--------------

https://tobifinn-ci2026-hackathon-leaderboard.hf.space

Takeaway
-----------

Cloud representation remains one of the biggest uncertainties in climate modeling.
This challenge highlights how **machine learning + physics-informed approaches**
can help bridge that gap.
