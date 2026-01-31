# 1. GENERATE THE MISSING AUDIO DATA (Wait for this to finish!)
python run.py cache --config configs/tox21_audio.yaml --limit 2000

# 2. VERIFY THE FILE EXISTS
ls -lh cache/tox21_audio/audio_embeddings.npy

# 3. RUN THE EXPERIMENTS
rm -rf runs/*
python run.py fit --config configs/tox21_desc.yaml --outdir runs/tox21_desc_2000
python run.py fit --config configs/tox21_audio.yaml --outdir runs/tox21_audio_2000
python run.py fit --config configs/tox21_fuse.yaml --outdir runs/tox21_fuse_desc_audio_2000

# 4. VIEW RESULTS
python tools/summarize_runs.py
