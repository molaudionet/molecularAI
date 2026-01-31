from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _require_mfcc_stack():
    """
    Lazy import so the rest of the repo can run without librosa unless audio MFCC is used.
    """
    try:
        import librosa  # noqa: F401
        return librosa
    except Exception as e:
        raise RuntimeError(
            "MFCC audio embedding requires librosa.\n"
            "Install:\n"
            "  pip install librosa\n"
            "If you hit soundfile errors:\n"
            "  pip install soundfile\n"
        ) from e


def mfcc_features_from_wav(
    wav_path: str,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 256,
    roll_percent: float = 0.85,
) -> Optional[np.ndarray]:
    """
    Extract a compact, ESOL-friendly feature vector:
      - MFCC mean + std over time  -> 2*n_mfcc
      - spectral centroid mean/std -> 2
      - spectral bandwidth mean/std-> 2
      - spectral rolloff mean/std  -> 2
      - RMS mean/std              -> 2
    Total dims = 2*n_mfcc + 8  (e.g., 48 when n_mfcc=20)

    Returns:
        np.ndarray shape [D] float32 or None on failure
    """
    librosa = _require_mfcc_stack()

    try:
        y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
        if y is None or len(y) == 0:
            return None

        # Normalize amplitude (stable across mappings)
        maxv = float(np.max(np.abs(y)) + 1e-9)
        y = (y / maxv).astype(np.float32)

        # MFCCs
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr_loaded,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )  # [n_mfcc, T]
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr_loaded, n_fft=n_fft, hop_length=hop_length
        )  # [1, T]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr_loaded, n_fft=n_fft, hop_length=hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr_loaded, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent
        )
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

        feats = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                np.array([centroid.mean(), centroid.std()], dtype=np.float32),
                np.array([bandwidth.mean(), bandwidth.std()], dtype=np.float32),
                np.array([rolloff.mean(), rolloff.std()], dtype=np.float32),
                np.array([rms.mean(), rms.std()], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        # Replace any rare NaNs/Infs (e.g., near-silent waveforms)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feats

    except Exception as e:
        print(f"[WARN] mfcc_features_from_wav failed on {wav_path}: {e}")
        return None


class AudioFeaturizer:
    """
    Drop-in replacement for your wav2vec2 AudioFeaturizer.

    Expected config subtree: cfg['features']['audio']
      audio:
        enabled: true
        generate:
          sonify_script: "tools/sonify_smiles.py"
          tmp_dir: "cache/tmp_audio"
          sample_rate: 16000
          duration: 2.0
        embed:
          cache_path: "cache/esol_fuse/audio_embeddings.npy"   # optional (run.py handles caching)
          # MFCC params (optional)
          n_mfcc: 20
          n_fft: 1024
          hop_length: 256
          roll_percent: 0.85
    """

    def __init__(self, audio_cfg: dict):
        self.audio_cfg = audio_cfg or {}

        gen_cfg = (self.audio_cfg.get("generate") or self.audio_cfg.get("generation") or {})
        self.sonify_script = str(gen_cfg.get("sonify_script", "tools/sonify_smiles.py"))
        self.tmp_dir = str(gen_cfg.get("tmp_dir", "cache/tmp_audio"))
        self.sample_rate = int(gen_cfg.get("sample_rate", 16000))
        self.duration = float(gen_cfg.get("duration", 2.0))

        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)

        emb_cfg = (self.audio_cfg.get("embed") or {})
        self.n_mfcc = int(emb_cfg.get("n_mfcc", 20))
        self.n_fft = int(emb_cfg.get("n_fft", 1024))
        self.hop_length = int(emb_cfg.get("hop_length", 256))
        self.roll_percent = float(emb_cfg.get("roll_percent", 0.85))

    def _sonify_one(self, smiles: str, out_wav: str):
        """
        Calls your sonify script to produce a WAV at out_wav.
        Assumes your script supports --smiles, --out, --sr, --duration (your stub does).
        """
        cmd = [
            "python",
            self.sonify_script,
            "--smiles",
            smiles,
            "--out",
            out_wav,
            "--sr",
            str(self.sample_rate),
            "--duration",
            str(self.duration),
        ]
        subprocess.check_call(cmd)

    def featurize_list(self, smiles_list: Sequence[str]) -> np.ndarray:
        feats = []
        D_expected = 2 * self.n_mfcc + 8

        print(
            f" Audio MFCC Featurizer | sr={self.sample_rate} dur={self.duration}s "
            f"| n_mfcc={self.n_mfcc} => D={D_expected}"
        )

        for i, smi in enumerate(smiles_list, start=1):
            out_wav = str(Path(self.tmp_dir) / f"mol_{i:06d}.wav")

            try:
                self._sonify_one(smi, out_wav)
                f = mfcc_features_from_wav(
                    out_wav,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    roll_percent=self.roll_percent,
                )
            except Exception as e:
                print(f"[WARN] sonify+mfcc failed at i={i}: {e}")
                f = None

            if f is None:
                f = np.zeros((D_expected,), dtype=np.float32)

            # Hard guard: never let a mismatch slip through silently
            if f.shape[0] != D_expected:
                raise ValueError(f"MFCC feature dim mismatch: got {f.shape[0]} expected {D_expected}")

            feats.append(f)

            if i % 100 == 0:
                print(f"  Processed {i}/{len(smiles_list)}...")

        A = np.stack(feats).astype(np.float32)
        return A

