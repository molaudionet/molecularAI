from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import tempfile
from pathlib import Path

from typing import Optional
import numpy as np

def _require_audio_stack():
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        return torch, Wav2Vec2Model, Wav2Vec2Processor, wavfile, resample_poly
    except Exception as e:
        raise RuntimeError(
            "Audio embedding requires: transformers, torch, scipy.\n"
            "Install: pip install transformers scipy\n"
        ) from e

_MODEL = None
_PROC = None

def get_wav2vec2(model_name: str = "facebook/wav2vec2-base-960h", device: str = "cpu"):
    global _MODEL, _PROC
    torch, Wav2Vec2Model, Wav2Vec2Processor, _, _ = _require_audio_stack()

    if _MODEL is None:
        _PROC = Wav2Vec2Processor.from_pretrained(model_name)
        _MODEL = Wav2Vec2Model.from_pretrained(model_name).to(device)
        _MODEL.eval()
        for p in _MODEL.parameters():
            p.requires_grad = False
    return _MODEL, _PROC

def embed_wav(wav_path: str, device: str = "cpu",
              model_name: str = "facebook/wav2vec2-base-960h",
              target_sr: int = 16000) -> Optional[np.ndarray]:
    torch, _, _, wavfile, resample_poly = _require_audio_stack()
    model, proc = get_wav2vec2(model_name=model_name, device=device)

    try:
        sr, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32)

        # if stereo, take mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # normalize to [-1,1] roughly
        if np.max(np.abs(audio)) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # resample to 16k
        if sr != target_sr:
            # resample_poly is stable and fast
            audio = resample_poly(audio, target_sr, sr).astype(np.float32)
            sr = target_sr

        inputs = proc(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            out = model(input_values)
            # out.last_hidden_state: (B, T, C). mean pool over time
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return emb
    except Exception:
        return None

import torch

class AudioFeaturizer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Directory for temporary WAV files
        # Set via cfg['audio_dir'] (recommended per-dataset), otherwise defaults to system temp.
        self.audio_dir = cfg.get('audio_dir')
        if not self.audio_dir:
            self.audio_dir = str(Path(tempfile.gettempdir()) / 'molaudio_tmp_wavs')
        Path(self.audio_dir).mkdir(parents=True, exist_ok=True)

        # --- Dynamic Device Detection ---
        requested_device = cfg["embed"].get("device", "cpu")
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device
            
        print(f" Audio Featurizer using device: {self.device.upper()}")
        
        self.model_name = cfg["embed"].get("model_name", "facebook/wav2vec2-base-960h")
        # Load model to the detected device
        self.model, self.processor = get_wav2vec2(self.model_name, self.device)

    def featurize_list(self, smiles_list: list[str]) -> np.ndarray:
        from featurizers.sonify_smiles import smiles_to_wav  # Assuming this exists
        
        embeddings = []
        print(f" Processing {len(smiles_list)} molecules into audio embeddings...")
        
        for i, sm in enumerate(smiles_list):
            # 1. Convert SMILES to a temporary .wav file
            wav_path = os.path.join(self.audio_dir, f"temp_{i}.wav")
            smiles_to_wav(sm, wav_path)
            
            # 2. Use your existing embed_wav function
            emb = embed_wav(wav_path, device=self.device, model_name=self.model_name)
            
            # 3. Fallback if embedding fails
            if emb is None:
                emb = np.zeros(768, dtype=np.float32)
            
            embeddings.append(emb)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(smiles_list)}...")

        return np.stack(embeddings)
