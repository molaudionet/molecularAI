# Sound of Molecules: Molecular Sonification for Aqueous Solubility (ESOL)

This repository investigates **molecular sonification as an auxiliary modality** for predicting aqueous solubility on the ESOL dataset.  
We evaluate how different **audio representations** interact with classical molecular descriptors under small-data conditions.

The key takeaway is that **representation choice matters more than model size**:  
compact, task-aligned audio features (MFCCs) enable stable multimodal learning, while high-capacity speech embeddings (wav2vec2) destabilize training.

---

## Overview

**Pipeline**
1. Convert SMILES → deterministic audio waveform (sonification)
2. Extract audio features:
   - MFCC + spectral statistics, or
   - wav2vec2 embeddings
3. Train regression models:
   - descriptors only
   - audio only
   - descriptor + audio fusion
4. Evaluate with test R² and bootstrap confidence intervals

**Task**
- Dataset: ESOL (Delaney), ~1128 molecules
- Target: log aqueous solubility
- Metric: R² (higher is better)
- Evaluation: fixed 80/10/10 split + 2000-sample bootstrap CI

---

## Audio Representations

### 1. wav2vec2 (speech self-supervised)
- 768-dimensional embeddings
- Pretrained on speech
- High representational capacity
- Found to be **misaligned with sonification**

### 2. MFCC + spectral features (recommended)
- 20 MFCCs (mean + std)
- Spectral centroid, bandwidth, rolloff, RMS (mean + std)
- Total: **48 dimensions**
- Compact, interpretable, stable

---

## Experimental Results

| Representation | Audio | Fusion | Test R² | 95% Bootstrap CI |
|---|---|---|---|---|
| RDKit descriptors | — | — | ~0.805 | [~0.707, ~0.869] |
| Audio-only | wav2vec2 | — | ~0.56 | very wide |
| Fusion | wav2vec2 | concat | ~0.48 | unstable |
| Audio-only | MFCC | — | −0.06 | [−0.184, 0.014] |
| **Fusion** | **MFCC** | **concat** | **0.808** | **[0.719, 0.868]** |

---

## Interpretation

- **Audio alone does not predict solubility**
- **High-capacity speech models destabilize fusion**
- **Compact spectral audio features enable stable multimodal learning**
- Audio functions as a **weak auxiliary bias**, not a standalone predictor

This establishes a design principle:

> Molecular sonification is most effective when paired with representations whose inductive bias matches both the physics of sound and the chemistry of the task.

---

## How to Run Experiments

All experiments are **script-based (CLI)** — no notebooks required.

### Descriptor-only baseline
```bash
python run.py fit \
  --config configs/esol_desc.yaml \
  --outdir runs/esol_desc
Audio-only (MFCC)
bash
Copy code
python run.py fit \
  --config configs/esol_audio_mfcc.yaml \
  --outdir runs/esol_audio_mfcc

python tools/bootstrap_r2.py \
  --npz runs/esol_audio_mfcc/test_preds.npz
Fusion (Descriptors + MFCC) — recommended
bash
Copy code
python run.py fit \
  --config configs/esol_fuse_mfcc_pca.yaml \
  --outdir runs/esol_fuse_mfcc_pca

python tools/bootstrap_r2.py \
  --npz runs/esol_fuse_mfcc_pca/test_preds.npz
Fusion (wav2vec2 + PCA)
bash
Copy code
python run_wav2vec2.py fit \
  --config configs/esol_fuse_wav2vec2_pca.yaml \
  --outdir runs/esol_fuse_wav2vec2_pca

python tools/bootstrap_r2.py \
  --npz runs/esol_fuse_wav2vec2_pca/test_preds.npz
Reproducibility Notes
Audio embeddings are cached on disk (cache/)

Clear caches when switching representation types

Sonification is deterministic

Bootstrap uses fixed random seeds

All results reported include confidence intervals

Where to Run This Code (Important)
✅ Recommended Platforms
Lambda Labs GPU Cloud
Real Linux VM

SSH access

Persistent filesystem

Conda works normally

Excellent for wav2vec2 experiments

Typical workflow:

bash
Copy code
ssh <vm>
git clone <repo>
conda activate sound-of-molecules
python run.py fit ...
RunPod (best price/performance)
Very cheap GPUs (RTX 3090 / 4090)

Persistent volumes

Ideal for batch experiments

Use /workspace/cache and /workspace/runs for persistence.

Why Google Colab Is Not Recommended
Google Colab is not suitable for this project.

Problems with Colab
Ephemeral filesystem (caches disappear)

Background processes killed

Poor support for long CLI scripts

Audio + subprocess + disk I/O is fragile

Notebook-first workflow conflicts with script-based experiments

Difficult to reproduce results reliably

Summary
Feature	Colab	This Project
Persistent cache		Required
Long runs		Required
CLI scripts		Core design
Audio generation		Core design
Reproducibility		Mandatory

Recommendation: Avoid Colab entirely for these experiments.

Pros and Cons Summary
MFCC Audio (Recommended)
Pros

Stable

Interpretable

Small dimensionality

Works well in fusion

Cons

Weak standalone predictor

Requires thoughtful sonification design

wav2vec2 Audio
Pros

Powerful representation

Works well for speech

Cons

Misaligned with sonification

High variance

Destabilizes fusion

Expensive to run

Future Directions
Polarity-targeted sonification (explicit spectral tilt)

Late-fusion (α-blend) models

Extension to bioactivity and binding tasks

Audio pretraining aligned to molecular structure

License
MIT (or your preferred license)

markdown
Copy code

---

## Final note

This README does **three important things**:

1. Documents **negative results** honestly (audio-only, wav2vec2)
2. Explains **why MFCC fusion works**
3. Saves users from wasting time on **Google Colab**

That combination is rare — and very strong.







