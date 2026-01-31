#!/usr/bin/env bash
# Run ESOL (Aqueous Solubility) experiments
# Three modes: Descriptors only, Audio only, Fusion (Desc + Audio)

set -euo pipefail

echo "==============================================="
echo "ESOL Solubility Prediction Experiments"
echo "==============================================="

# 1. Descriptor-only baseline
echo ""
echo ">> Running ESOL - Descriptor Only..."
python run.py fit --config configs/esol_desc.yaml --outdir runs/esol_desc

# 2. Audio-only (molecular sonification)
echo ""
echo ">> Running ESOL - Audio Only..."
python run.py fit --config configs/esol_audio.yaml --outdir runs/esol_audio

# 3. Fusion (descriptors + audio)
echo ""
echo ">> Running ESOL - Fusion (Desc + Audio)..."
python run.py fit --config configs/esol_fuse.yaml --outdir runs/esol_fuse

echo ""
echo "==============================================="
echo " All ESOL experiments complete!"
echo "==============================================="
echo ""
echo "Results saved in:"
echo "  - runs/esol_desc/"
echo "  - runs/esol_audio/"
echo "  - runs/esol_fuse/"
echo ""
echo "Generate summary:"
echo "  python tools/summarize_runs.py"
