#!/usr/bin/env bash
set -euo pipefail

# Clean run dirs? uncomment if you want a fresh start:
# rm -rf runs/bbbp_* runs/bioactive_* runs/tox21_*

# --- Tox21 (3) ---
python run.py fit --config configs/tox21_desc.yaml --outdir runs/tox21_desc_2000
python run.py fit --config configs/tox21_audio.yaml --outdir runs/tox21_audio_2000
python run.py fit --config configs/tox21_fuse.yaml --outdir runs/tox21_fuse_desc_audio_2000

# --- BBBP (3) ---
python run.py fit --config configs/bbbp_desc.yaml --outdir runs/bbbp_desc
python run.py fit --config configs/bbbp_audio.yaml --outdir runs/bbbp_audio
python run.py fit --config configs/bbbp_fuse.yaml --outdir runs/bbbp_fuse

# --- Bioactive50284 (3) ---
python run.py fit --config configs/bioactive_desc.yaml --outdir runs/bioactive_desc
python run.py fit --config configs/bioactive_audio.yaml --outdir runs/bioactive_audio
python run.py fit --config configs/bioactive_fuse.yaml --outdir runs/bioactive_fuse

# --- Consolidated summary ---
python tools/summarize_runs.py

