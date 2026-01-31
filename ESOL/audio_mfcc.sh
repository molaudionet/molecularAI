##Before running (important if you previously ran wav2vec2):
rm -f cache/esol_audio_mfcc/audio_embeddings.npy

##Then train:
python run_mfcc.py fit  --config configs/esol_audio_mfcc.yaml --outdir runs/esol_audio_mfcc

####Bootstrap:
python tools/bootstrap_r2.py  --npz runs/esol_audio_mfcc/test_preds.npz

