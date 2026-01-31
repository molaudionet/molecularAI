import os
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.config import load_yaml
from datasets.csv_generic import CSVGenericDataset, parse_task_types
from featurizers import rdkit_desc
from utils.cache import load_cache, save_cache
from utils.splits import make_split_indices
from training.trainer import fit
from utils.io import save_artifacts

# For saving test predictions (bootstrap/reporting)
from models.head import predict_proba


def build_cache(cfg, limit: int):
    """
    Featurize molecules into descriptor features (X), labels (Y), and mask (M).

    Returns:
        X, Y, M, ids, smiles_list, meta
    """
    cache_dir = cfg["features"]["cache_dir"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Try to reuse cached descriptor arrays if available
    try:
        cached = load_cache(cache_dir)
    except Exception:
        cached = None

    if cached is not None:
        # expected: (X, Y, M, ids, meta) or similar
        if isinstance(cached, (tuple, list)) and len(cached) >= 5:
            X, Y, M, ids, meta = cached[:5]
            smiles_list = meta.get("smiles_list", [])
            return X, Y, M, ids, smiles_list, meta

    ds = CSVGenericDataset(cfg, limit=limit)

    # Normalize task_types config
    task_types = parse_task_types(cfg["dataset"].get("task_types", {}), ds.label_names)
    cfg["dataset"]["task_types"] = task_types

    ids, smiles_list, X_list, Y_list, M_list = [], [], [], [], []

    for i in tqdm(range(len(ds)), desc="Featurizing"):
        s = ds[i]
        x = rdkit_desc.featurize(s.smiles)
        if x is None:
            continue
        ids.append(str(s.id))
        smiles_list.append(s.smiles)
        X_list.append(x)
        Y_list.append(s.y)
        M_list.append(s.mask)

    if not X_list:
        raise RuntimeError("No valid molecules were featurized (X_list empty). Check SMILES parsing/featurizer.")

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    M = np.stack(M_list).astype(np.float32)

    meta = {
        "label_names": ds.label_names,
        "task_types": task_types,
        "X_shape": list(X.shape),
        # store smiles so we can generate audio embeddings on-demand
        "smiles_list": smiles_list,
    }

    save_cache(cache_dir, X, Y, M, ids, meta)
    return X, Y, M, ids, smiles_list, meta


def _ensure_audio_embeddings(cfg, X, smiles_list):
    """
    Load audio embeddings from cache_path; if missing, generate with AudioFeaturizer and save.
    Returns audio matrix A with shape [N, D] matching X rows.
    """
    audio_cfg = cfg.get("features", {}).get("audio", {})
    if not audio_cfg or not audio_cfg.get("enabled", False):
        raise ValueError("Audio embeddings requested but cfg.features.audio.enabled is not True.")

    cache_path = audio_cfg.get("embed", {}).get("cache_path")
    if not cache_path:
        raise ValueError("Missing config: features.audio.embed.cache_path")

    audio_path = Path(cache_path)
    print("DEBUG audio_path:", str(audio_path), "exists?", audio_path.exists())

    if audio_path.exists():
        A = np.load(audio_path)
    else:
        if not smiles_list:
            raise ValueError("Audio cache missing and smiles_list is empty; cannot generate audio embeddings.")
        print("🎸 Generating Audio Embeddings (this takes a moment)...")
        from featurizers.audio_featurizer import AudioFeaturizer

        af = AudioFeaturizer(audio_cfg)
        A = af.featurize_list(smiles_list)

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(audio_path, A.astype(np.float32))
        print(f"✅ Saved audio embeddings to {audio_path}")

    # Hard check: row alignment matters (avoid silent slicing/leakage)
    if A.shape[0] != X.shape[0]:
        raise ValueError(
            f"Row mismatch: X has {X.shape[0]} rows but audio has {A.shape[0]} rows. "
            f"Check that cache_path is dataset-specific and was generated with the same limit."
        )

    return A.astype(np.float32)


def _apply_modalities(cfg, X, Y, M, smiles_list):
    """
    Choose modality based on cfg.train_mode:
      - desc: use RDKit descriptors
      - audio: use audio embeddings
      - fuse/fusion: concatenate descriptors + audio
    """
    mode = (cfg.get("train_mode") or cfg.get("features", {}).get("mode", "desc"))
    mode = str(mode).lower()

    # Normalize naming
    if mode == "fuse":
        mode = "fusion"

    print("DEBUG train_mode:", cfg.get("train_mode"))
    print("DEBUG features.mode:", cfg.get("features", {}).get("mode"))
    print("DEBUG resolved mode:", mode)

    if mode in ("audio", "fusion"):
        A = _ensure_audio_embeddings(cfg, X, smiles_list)

        if mode == "audio":
            print(f"🎸 MODE: AUDIO | Features: {A.shape[1]}")
            return A, Y, M

        # fusion
        Xf = np.concatenate([X, A], axis=1)
        print(f"🚀 MODE: FUSION | Features: {Xf.shape[1]}")
        return Xf, Y, M

    # descriptors
    print(f"📊 MODE: DESC | Features: {X.shape[1]}")
    return X, Y, M


def cmd_fit(cfg, limit, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, Y, M, ids, smiles_list, meta = build_cache(cfg, limit)
    X_f, Y_f, M_f = _apply_modalities(cfg, X, Y, M, smiles_list=smiles_list)

    splits = make_split_indices(len(X_f), seed=cfg["run"]["seed"], split=cfg["run"]["split"])

    # Ensure label names / task types are populated for the trainer/model
    cfg["dataset"]["label_cols"] = meta["label_names"]
    cfg["dataset"]["task_types"] = meta["task_types"]

    results = fit(cfg, X_f, Y_f, M_f, splits)

    metrics = {
        "train": results["train"],
        "val": results["val"],
        "test": results["test"],
        "train_X_shape": list(X_f.shape),
    }

    save_artifacts(str(outdir), results["model"], metrics, cfg)

    # Save raw test preds for bootstrap / reporting
    test_idx = splits["test"]
    y_true = Y_f[test_idx]
    y_pred = predict_proba(results["model"], X_f[test_idx])

    np.savez(
        outdir / "test_preds.npz",
        y_true=y_true,
        y_pred=y_pred,
        task_names=np.array(cfg["dataset"]["label_cols"], dtype=object),
    )
    print(f"✅ Saved test preds for bootstrap: {outdir / 'test_preds.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "fit"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", default="runs/exp")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if args.cmd == "cache":
        # Build descriptor cache (smiles_list is stored in meta)
        X, Y, M, ids, smiles_list, meta = build_cache(cfg, args.limit)

        # If audio is enabled, also build audio cache now (so fusion can reuse it)
        if cfg.get("features", {}).get("audio", {}).get("enabled", False):
            _ = _ensure_audio_embeddings(cfg, X, smiles_list)
    else:
        cmd_fit(cfg, args.limit, args.outdir)


if __name__ == "__main__":
    main()

