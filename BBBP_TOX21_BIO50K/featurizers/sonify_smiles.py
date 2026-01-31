from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.io import wavfile


def _note_freq(semitone_from_a4: int) -> float:
    # A4 = 440 Hz
    return 440.0 * (2.0 ** (semitone_from_a4 / 12.0))


def smiles_to_wav(
    smiles: str,
    out_path: str,
    duration_s: float = 5.0,
    sr: int = 16000,
) -> Optional[str]:
    """
    Deterministic, rule-based demo sonification:
    - Parse characters from SMILES
    - Map characters -> pitch classes
    - Render a monophonic note stream
    """
    if not smiles or not isinstance(smiles, str):
        return None

    # Basic character->semitone mapping (demo)
    # Keep it simple and stable.
    chars = list(smiles.strip())
    if len(chars) == 0:
        return None

    # Map SMILES chars to 0..11 pitch class
    # Use ASCII code mod 12; deterministic and universal for demo.
    pitch_classes = [(ord(c) % 12) for c in chars]

    # Timing: split total duration into N notes (cap N)
    max_notes = 64
    pitch_classes = pitch_classes[:max_notes]
    n = len(pitch_classes)
    note_dur = duration_s / n

    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    y = np.zeros_like(t, dtype=np.float32)

    # Synthesis params
    base_octave = 0  # shift up/down if desired
    amp = 0.2

    idx = 0
    for pc in pitch_classes:
        start = int(idx * sr * note_dur)
        end = int((idx + 1) * sr * note_dur)
        if end <= start:
            continue

        # Map pitch class to a musical semitone offset around A4
        # Put pitches roughly between 220-880 Hz for speech-audio compatibility
        semitone = (pc - 9) + 12 * base_octave  # pc 9 => A
        f = _note_freq(semitone)

        tt = np.arange(end - start, dtype=np.float32) / sr
        tone = np.sin(2 * np.pi * f * tt).astype(np.float32)

        # short fade to reduce clicks
        fade = int(0.005 * sr)
        if fade > 1 and len(tone) > 2 * fade:
            w = np.ones_like(tone)
            w[:fade] = np.linspace(0, 1, fade)
            w[-fade:] = np.linspace(1, 0, fade)
            tone *= w

        y[start:end] += amp * tone
        idx += 1

    # normalize
    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx * 0.9

    wavfile.write(out_path, sr, (y * 32767).astype(np.int16))
    return out_path

