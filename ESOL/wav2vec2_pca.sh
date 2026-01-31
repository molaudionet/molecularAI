rm -f cache/esol_fuse_wav2vec2_pca/audio_embeddings.npy

python run_wav2vec2.py fit \
  --config configs/esol_fuse_wav2vec2_pca.yaml \
  --outdir runs/esol_fuse_wav2vec2_pca

python tools/bootstrap_r2.py \
  --npz runs/esol_fuse_wav2vec2_pca/test_preds.npz

