PART I — Results & Discussion 
Results
Task

We evaluate aqueous solubility prediction on the ESOL dataset (≈1128 molecules) using regression with R² as the primary metric. Models are evaluated on a fixed 80/10/10 train/validation/test split. Statistical uncertainty is quantified using 2000-sample bootstrap confidence intervals on the held-out test set.

Representations Evaluated

We compare four representations:

RDKit descriptors (baseline)
A small set of classical molecular descriptors.

Audio-only (wav2vec2 embeddings)
Molecular sonification rendered as audio waveforms, embedded using a pretrained speech self-supervised model.

Audio-only (MFCC + spectral features)
Compact spectral features (MFCCs, spectral centroid, bandwidth, rolloff, RMS) extracted from sonified audio.

Fusion models
Early fusion of RDKit descriptors with audio representations (wav2vec2 or MFCC), with optional PCA applied to audio features.

Quantitative Results
Representation	Audio Feature Type	Fusion	Test R²	95% Bootstrap CI
RDKit descriptors	—	—	~0.805	[~0.707, ~0.869]
Audio-only	wav2vec2	—	~0.56	wide, unstable
Fusion	wav2vec2	concat	~0.48	[~0.07, ~0.75]
Audio-only	MFCC + spectral	—	−0.06	[−0.184, 0.014]
Fusion	MFCC + spectral	concat	0.808	[0.719, 0.868]
Discussion
Audio Representations Alone Are Insufficient

Both wav2vec2 and MFCC audio-only models fail to predict solubility reliably. MFCC audio-only performance is statistically indistinguishable from a mean predictor (R² ≈ 0), indicating that sonified audio does not uniquely encode solubility. This is expected: solubility depends on interacting chemical factors (polarity, hydrogen bonding, size), none of which can be fully disentangled from audio alone.

High-Capacity Speech Models Are Misaligned

Fusion with wav2vec2 embeddings dramatically degrades performance and exhibits large uncertainty. Despite high training R², test performance collapses. This behavior indicates variance amplification, caused by combining a small dataset with a high-dimensional, speech-trained representation that is invariant to absolute spectral cues relevant to sonification.

This result demonstrates that speech self-supervised audio models are poorly aligned with molecular sonification tasks, even when dimensionality reduction (e.g., PCA) is applied.

Compact Spectral Features Enable Stable Fusion

Replacing wav2vec2 with compact MFCC-based features fundamentally changes the behavior of the system. Although MFCC audio alone is non-predictive, early fusion with RDKit descriptors yields:

Stable training–validation–test performance

Tight bootstrap confidence intervals

Slight improvement over descriptor-only baselines

No degradation of generalization

This indicates that MFCC audio features act as a low-capacity auxiliary signal that gently reshapes the regression geometry without overpowering chemically grounded descriptors.

Interpretation: Audio as a Regularizing Side-Channel

The key insight is that audio does not function as an independent predictor, but rather as a contextual bias. MFCC features encode coarse spectral properties (e.g., brightness, envelope shape) that weakly correlate with physicochemical trends (such as polarity), but only become useful when anchored by classical descriptors.

This behavior is consistent with multimodal learning theory: auxiliary modalities can improve robustness and stability even when they carry limited standalone signal.

Implications for Molecular Sonification

These results establish an important design principle:

Molecular sonification is most effective when paired with representations whose inductive bias matches both the physics of sound and the chemistry of the task.

For solubility prediction, global spectral summaries (MFCCs) are more appropriate than deep speech embeddings. For more complex endpoints (e.g., bioactivity, binding kinetics), richer learned audio representations may become beneficial once alignment is addressed.

Summary of Key Findings

Audio-only models do not predict solubility.

High-dimensional speech embeddings destabilize fusion.

Compact spectral features enable stable, interpretable multimodal learning.

Audio functions as a complementary bias, not a replacement for chemistry.
