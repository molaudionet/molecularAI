from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json
import numpy as np
from joblib import dump, load

def cache_paths(cache_dir: str) -> Tuple[Path, Path, Path, Path, Path]:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d/"X.joblib", d/"Y.npy", d/"M.npy", d/"ids.json", d/"meta.json"

def save_cache(cache_dir: str, X: np.ndarray, Y: np.ndarray, M: np.ndarray, ids: List[str], meta: dict):
    x_path, y_path, m_path, ids_path, meta_path = cache_paths(cache_dir)
    dump(X, x_path)
    np.save(y_path, Y)
    np.save(m_path, M)
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def load_cache(cache_dir: str):
    x_path, y_path, m_path, ids_path, meta_path = cache_paths(cache_dir)
    if not x_path.exists():
        return None
    X = load(x_path)
    Y = np.load(y_path)
    M = np.load(m_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return X, Y, M, ids, meta
