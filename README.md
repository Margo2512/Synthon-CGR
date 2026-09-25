# Synthon-CGR: Interpretable prediction of cocrystal formation based on graph neural networks with explicit modeling of supramolecular syntons

**Synthon-CGR** is an interpretable graph neural network that predicts co-crystal formation and explicitly identifies the supramolecular synthons — the real intermolecular interactions driving co-crystallization — making each prediction chemically grounded.

---

## Model Overview

The model was trained on a curated dataset of **15,606** molecular pairs, 
augmented to **27,018** pairs to balance positive and negative classes + common test (700 pairs). 
Training followed a **two-stage** protocol:

- **Stage 1 (pretraining)** — 25,560 pairs **without** synthon masks
- **Stage 2 (finetuning)** — 1,458 pairs **with** HBAT-derived synthon annotations

**Performance on the held-out test set:**

| Task | Metric | Value |
|------|--------|-------|
| Co-crystal classification | F1 | **0.893** |
| Co-crystal classification | Balanced Accuracy | **0.873** |
| Co-crystal classification | ROC-AUC | **0.940** |
| Synthon ranking | HitRate@5 | **0.935** |
| Synthon ranking | MRR | **0.753** |

Pretrained weights for the main model and baseline versions (V0–V2) 
are available in `pretrained_models/`.

---

## Repository Structure

```
Synthon-CGR/
├── src/                          # Source code
|   ├── __init__.py
│   ├── config.py                 # Configurations for all model versions
│   ├── models.py                 # Model definitions (V0–V3)
│   ├── models_v3.py              # V3 model (Synthon-CGR)
│   ├── train.py                  # Training loop
│   ├── data_utils.py             # Data loading and preprocessing
│   ├── candidate_generator.py    # Candidate synthon generation
│   ├── candidate_initializer.py  # Candidate encoding
│   ├── synthon_reasoner.py       # Self-attention over candidates
│   ├── motif_encoder.py          # EFG motif encoding
│   ├── hbond_prior.py            # H-bond statistical prior
│   ├── graph_utils.py            # Molecular graph construction
│   ├── motif_extractor.py        # EFGs decomposition
│   ├── metrics.py                # Metrics and bootstrap CIs
│   └── losses.py                 # Loss functions
├── pretrained_models/            # Pretrained weights
├── data/                         # Data
├── checkpoints/                  # Saved models
├── logs/                         # Training logs
├── plots/                        # Training curves
├── cache/                        # Motif cache
├── requirements.txt
├── local_file.sh                 # Launch script
├── main.py                       # Entry point
├── predict.py
└── README.md
```

---

## Installation

The package can be installed with **conda** as follows:

```bash
conda create --name synthon-cgr python=3.10
conda activate synthon-cgr
git clone https://github.com/your-username/Synthon-CGR.git
cd Synthon-CGR
pip install -r requirements.txt
```


The EFGs library is required for functional group extraction.
Clone it from the official repository:
```bash
git clone https://github.com/bbu-imdea/efgs.git
```
---

## Data Format

Training data must be provided as a **CSV** file with **three columns**:

| Column | Description |
|--------|-------------|
| `SMILES1` | SMILES string of the first molecule |
| `SMILES2` | SMILES string of the second molecule |
| `result` | Binary label: `1` for co-crystal, `0` otherwise |

**Example:**

```csv
SMILES1,SMILES2,result
BrC#CC#CBr,O=C(NCc1cccnc1)C(=O)NCc1cccnc1,1
CC(CC(=O)O)C(O)C(=O)O,O=C(O)c1ccccc1,0
```

For **Stage 2** (finetuning with synthon masks), an additional column 
is required:

| Column | Description |
|--------|-------------|
| `observed_synthon_mask` | Binary matrix of observed synthon pairs (from HBAT) |

---

## Quick Start

### Full training

Configurations for each model version are defined in `src/config.py`. 
Select the desired config by editing `local_file.sh`:

```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python main.py --config ConfigV3_Stage2_CommonTest > logs/run_${TIMESTAMP}.out > logs/run_${TIMESTAMP}.err
```

Then run:

```bash
bash local_file.sh
```

### Inference with pretrained weights

```bash
python predict.py --smiles1 "CC(=O)c1ccccn1" --smiles2 "O=C(O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1"
```

---

## Model Versions

| Version | Description |
|---------|-------------|
| **V0** | GNN only (baseline) |
| **V1** | GNN + EFG-derived motifs |
| **V2** | + synthon channel with H-bond prior |
| **V3** | + two-stage training with HBAT synthon masks (main model) |

Configs for all versions are defined in `src/config.py`.

---

## Output and Interpretability

For each input pair, Synthon-CGR returns:

1. **Co-crystal probability** — scalar in `[0, 1]`
2. **Ranked synthon list** — top-N candidate synthons with scores, 
   each defined by a pair of functional groups (pseudo-SMILES)

**Example output:**

```
======================================================================
Synthon-CGR prediction
======================================================================
SMILES1 : CC(=O)c1ccccn1
SMILES2 : O=C(O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1
----------------------------------------------------------------------
Co-crystal probability: 0.8614
Prediction: CO-CRYSTAL
----------------------------------------------------------------------
Top-6 predicted synthons:
   1. O=C([R])[R]                         <-> O=C(O)[R]                           : 0.8924
   2. [Nar]                               <-> O=C(O)[R]                           : 0.8462
   3. O=C([R])[R]                         <-> O=[N+]([O-])[R]                     : 0.6575
   4. O=C([R])[R]                         <-> O=[N+]([O-])[R]                     : 0.6285
   5. [Nar]                               <-> O=[N+]([O-])[R]                     : 0.6089
   6. [Nar]                               <-> O=[N+]([O-])[R]                     : 0.5912
======================================================================
```

---

## Training Pipeline

**Stage 1** — pretraining of the GNN and `global_head` on the full 
balanced pool **without** synthon masks. The goal is to learn general 
co-crystallization patterns.

**Stage 2** — finetuning on the subset of positive pairs with non-empty 
HBAT masks. Includes:

- `SynthonReasoner` — self-attention between candidates
- `candidate_head` — candidate scoring
- `mask_loss` — direct supervision on masks

**Loss function for Stage 2:**

```
L = L_BCE + λ_syn · L_synthon + λ_mask · L_mask + λ_contr · L_contrastive
```

where:
- `L_BCE` — binary cross-entropy on the final logit
- `L_synthon` — focal loss on candidates
- `L_mask` — BCE on masks
- `L_contrastive` — contrastive loss between positive/negative candidates

---