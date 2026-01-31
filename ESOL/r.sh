# 1. Backup current files
cp models/head.py models/head.py.bak
cp training/trainer.py training/trainer.py.bak

# 2. Replace with fixed versions (copy/paste from above)

# 3. CLEAR CACHE to force re-featurization with correct task types
rm -rf cache/esol_*

# 4. Run benchmarks
python run.py fit --config configs/esol_desc.yaml --outdir runs/esol_desc
python run.py fit --config configs/esol_audio.yaml --outdir runs/esol_audio
python run.py fit --config configs/esol_fuse.yaml --outdir runs/esol_fuse

# 5. Analyze results
python tools/analyze_esol.py
