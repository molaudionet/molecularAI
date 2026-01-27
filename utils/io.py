from __future__ import annotations
from pathlib import Path
import json
import joblib
from typing import Any, Dict
import numpy as np

def _json_default(o):
    # numpy scalars
    if isinstance(o, np.generic):
        return o.item()
    # numpy arrays
    if isinstance(o, np.ndarray):
        return o.tolist()
    # pathlib paths
    if isinstance(o, Path):
        return str(o)
    # fallback
    return str(o)

def save_artifacts(outdir, model, metrics, cfg):
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    # metrics.json
    (d / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default),
        encoding="utf-8"
    )

    # config.json (optional)
    (d / "config.json").write_text(
        json.dumps(cfg, indent=2, default=_json_default),
        encoding="utf-8"
    )

    # model.joblib etc... keep your existing code here

def load_model(path: str):
    return joblib.load(path)
