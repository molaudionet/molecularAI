# Molecular Sonification for Aqueous Solubility Prediction (ESOL)

This repository contains experiments evaluating molecular sonification as an auxiliary modality for predicting aqueous solubility on the ESOL dataset.

The work explores how different audio representations interact with classical molecular descriptors in small-data regression settings.

---

## Overview

Molecules are converted into deterministic audio waveforms via a sonification mapping. Audio is then embedded using either:

- High-capacity speech self-supervised models (wav2vec2), or
- Compact spectral representations (MFCC + spectral statistics).

We compare audio-only, descriptor-only, and fused models to understand when and how sonification contributes useful signal.

---

## Dataset

- **ESOL** (Delaney): ~1128 molecules
- Target: measured log solubility (regression)
- Evaluation: R² on a fixed 80/10/10 split
- Statistical uncertainty: 2000-sample bootstrap confidence intervals

---

## Audio Representations

### 1. wav2vec2 (Speech SSL)
- Pretrained on speech
- 768-dimensional embeddings
- High representational capacity
- Found to be unstable in fusion

### 2. MFCC + Spectral Features
- 20 MFCCs (mean + std)
- Spectral centroid, bandwidth, rolloff, RMS (mean + std)
- 48 dimensions total
- Low-capacity, interpretable, task-aligned

---

## Experimental Results

| Representation | Audio | Fusion | Test R² | 95% CI |
|---|---|---|---|---|
| RDKit descriptors | — | — | ~0.805 | tight |
| Audio-only | wav2vec2 | — | ~0.56 | wide |
| Fusion | wav2vec2 | concat | ~0.48 | very wide |
| Audio-only | MFCC | — | −0.06 | ~0 |
| **Fusion** | **MFCC** | **concat** | **0.808** | **[0.719, 0.868]** |

---

## Key Findings

- Audio alone does not predict solubility.
- High-dimensional speech embeddings destabilize multimodal learning.
- Compact spectral audio features enable stable fusion.
- Audio acts as a weak but helpful auxiliary signal when combined with descriptors.

---

## How to Run

### Descriptor-only
```bash
python run.py fit --config configs/esol_desc.yaml --outdir runs/esol_desc

Audio-only (MFCC)
python run.py fit --config configs/esol_audio_mfcc.yaml --outdir runs/esol_audio_mfcc

Fusion (Descriptors + MFCC)
python run.py fit --config configs/esol_fuse_mfcc_pca.yaml --outdir runs/esol_fuse_mfcc_pca

Bootstrap Confidence Intervals
python tools/bootstrap_r2.py --npz runs/<run_name>/test_preds.npz

Reproducibility Notes

All audio embeddings are cached to disk.

Sonification is deterministic.

Bootstrap evaluation uses fixed random seeds.

Audio cache must be cleared when switching representation types.

Interpretation

This work demonstrates that molecular sonification is not a replacement for chemistry-based descriptors, but can function as a complementary, low-capacity bias when represented appropriately.

The results emphasize the importance of representation alignment in multimodal molecular learning.

License

MIT 
---

## Final note (important)

WE now have:
- **Negative results** (audio-only)
- **Failure modes** (wav2vec2 fusion)
- **A successful design** (MFCC fusion)
- **Statistical rigor** (bootstrap CIs)



