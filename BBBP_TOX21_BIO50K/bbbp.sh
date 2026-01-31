#!/usr/bin/env bash
set -euo pipefail

#python run.py fit --config configs/bbbp_desc.yaml --outdir runs/bbbp_desc
#python run.py fit --config configs/bbbp_audio.yaml --outdir runs/bbbp_audio
#python run.py fit --config configs/bbbp_fuse.yaml --outdir runs/bbbp_fuse
python run.py fit --config configs/bbbp_audio.yaml --limit 3000--outdir runs/bbbp_audio
python run.py fit --config configs/bbbp_audio.yaml --limit 3000 --outdir runs/bbbp_audio
python run.py fit --config configs/bbbp_fuse.yaml  --limit 3000 --outdir runs/bbbp_fuse

