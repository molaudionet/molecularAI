python run.py fit --config configs/tox21_desc.yaml --limit 8000 --outdir runs/tox21_desc_full
python run.py fit --config configs/tox21_audio.yaml --limit 8000 --outdir runs/tox21_audio_full
python run.py fit --config configs/tox21_fuse.yaml --limit 8000 --outdir runs/tox21_fuse_desc_audio_full

python tools/summarize_runs.py


