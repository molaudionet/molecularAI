import argparse
import numpy as np
from pathlib import Path
import wave

def smiles_to_seed(smiles: str) -> int:
    return abs(hash(smiles)) % (2**31 - 1)

def synth_tone(freq: float, duration: float, sr: int = 16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

def write_wav(path: str, audio: np.ndarray, sr: int = 16000):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def sonify_smiles(smiles: str, sr: int = 16000, duration: float = 2.0) -> np.ndarray:
    seed = smiles_to_seed(smiles)
    base_freq = 220.0 + (seed % 880)  # [220, 1099]
    y = synth_tone(base_freq, duration, sr=sr)
    y += 0.35 * synth_tone(base_freq * 2.0, duration, sr=sr)
    y += 0.15 * synth_tone(base_freq * 3.0, duration, sr=sr)

    # simple attack envelope
    env = np.linspace(0.0, 1.0, len(y), endpoint=False)
    y *= env
    y = (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)
    return y

def smiles_to_wav(smiles: str, out_wav: str, sr: int = 16000, duration: float = 2.0):
    """
    Backward-compatible helper expected by featurizers/audio_featurizer.py.
    Writes a WAV file to out_wav for the given SMILES.
    """
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    y = sonify_smiles(smiles, sr=sr, duration=duration)
    write_wav(str(out), y, sr=sr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=2.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    y = sonify_smiles(args.smiles, sr=args.sr, duration=args.duration)
    write_wav(str(out), y, sr=args.sr)

if __name__ == "__main__":
    main()

