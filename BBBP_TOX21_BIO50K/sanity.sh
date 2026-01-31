#!/usr/bin/env bash
set -euo pipefail

python run.py fit --config configs/bbbp_audio.yaml --limit 100 --outdir runs/bbbp_audio
python run.py fit --config configs/bbbp_audio.yaml --limit 100 --outdir runs/bbbp_audio
python run.py fit --config configs/bbbp_fuse.yaml  --limit 100 --outdir runs/bbbp_fuse

