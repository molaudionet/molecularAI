
##Fresh run (recommended to avoid mixing cached wav2vec2 embeddings)
##Delete the old audio embeddings cache first, because it will have the wrong shape (768 vs ~48):
rm -f cache/esol_fuse_pca/audio_embeddings.npy

##Then run training:
#python run.py fit --config configs/esol_fuse_pca.yaml --outdir runs/esol_fuse_mfcc_pca

##Then bootstrap:
#python tools/bootstrap_r2.py --npz runs/esol_fuse_mfcc_pca/test_preds.npz

##What you should expect
##Audio-only MFCC: often more stable than wav2vec2
##Fusion: CI should shrink a lot, because audio is now low-dimensional and chemically “global”

python run.py fit  --config configs/esol_fuse_pca.yaml --outdir runs/esol_fuse_mfcc

python tools/bootstrap_r2.py --npz runs/esol_fuse_mfcc/test_preds.npz

