Here’s a complete “do-this-next” package:

Which GPU to rent (cheapest that works for your runs)

A 1-page “how to run on Lambda” guide

A 1-page “how to run on RunPod” guide

A Docker setup (Dockerfile + compose) so your repo runs identically anywhere

A realistic cost-per-experiment estimate

keep this focused on your workloads: wav2vec2 embedding, MFCC, PCA, Ridge training, caching.

1) GPU choice: what you should rent
If you want “works every time” + easy SSH VM

Lambda GPU Cloud VM (best sanity)

Pick: RTX 4090 (24GB) if available, otherwise A10 (24GB) or A100 40GB.

Why: you’ll run python run*.py ... like local, persistent disk, no notebook weirdness.

If you want cheapest/hour and can tolerate a little infra

RunPod (best price)

Pick: RTX 4090 24GB (~$0.34/hr) or RTX 3090 24GB (~$0.22/hr)

Your code will run fine on either. 4090 is faster; 3090 is cheaper.

Rule of thumb for your repo

MFCC experiments: GPU not required (CPU is fine).

wav2vec2 embedding: GPU helps a lot; CPU is slow/painful.

2) Lambda VM: “do this exactly” guide (SSH workflow)
Create the instance

Choose Ubuntu + a GPU (4090/A10/A100 depending on availability)

On the VM
# 1) system basics
sudo apt-get update
sudo apt-get install -y git wget tmux

# 2) miniconda (recommended)
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 3) clone repo
git clone <YOUR_REPO_URL>
cd ESOL   # (or your repo root)

# 4) create env
conda create -n sound-of-molecules python=3.10 -y
conda activate sound-of-molecules

# 5) install deps (tune as needed)
pip install -U pip
pip install numpy scikit-learn pyyaml tqdm librosa soundfile
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate

Run like local (recommended with tmux)
tmux new -s esol

# MFCC fusion
python run.py fit --config configs/esol_fuse_mfcc_pca.yaml --outdir runs/esol_fuse_mfcc_pca
python tools/bootstrap_r2.py --npz runs/esol_fuse_mfcc_pca/test_preds.npz

# wav2vec2 + PCA (after your smiles_to_wav fix)
python run_wav2vec2.py fit --config configs/esol_fuse_wav2vec2_pca.yaml --outdir runs/esol_fuse_wav2vec2_pca
python tools/bootstrap_r2.py --npz runs/esol_fuse_wav2vec2_pca/test_preds.npz

3) RunPod: easiest “pod + persistent volume” setup

RunPod pricing reference: GPUs from $0.34/hr (RTX 4090) and storage billed separately.

Recommended setup

Choose a RunPod “Pod” (on-demand GPU) with RTX 4090 or RTX 3090

Attach a persistent volume (so caches survive). Storage billed monthly (network volume rates documented).

Inside the pod (same as Lambda)

You’ll typically SSH or open web terminal, then run the same commands as above.

Important: Put caches and runs on the persistent volume path, e.g.

/workspace/cache/...

/workspace/runs/...

So your YAML becomes:

features:
  cache_dir: "/workspace/cache/esol_fuse_mfcc_pca"

4) Dockerize it so it runs anywhere identically

This avoids “works on my machine” issues and makes GitHub supplementation stronger.

Dockerfile (GPU-capable, works on Lambda/RunPod/local)

Create Dockerfile at repo root:

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git wget ffmpeg libsndfile1 \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps
RUN pip3 install --no-cache-dir -U pip

# If you have requirements.txt, copy it first for caching
# COPY requirements.txt /app/requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt

# Otherwise install minimal deps here:
RUN pip3 install --no-cache-dir \
    numpy scikit-learn pyyaml tqdm librosa soundfile \
    transformers accelerate \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy repo
COPY . /app

# Default shell
CMD ["/bin/bash"]

Run it (GPU)
docker build -t esol-audio .
docker run --rm -it --gpus all \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/runs:/app/runs \
  esol-audio


Then inside container:

python run.py fit --config configs/esol_fuse_mfcc_pca.yaml --outdir runs/esol_fuse_mfcc_pca

Notes

libsndfile1 is included to prevent audio decoding issues.

Using -v mounts keeps caches/results outside the container (persistent).

5) Cost estimate (realistic, with current prices)
GPU hourly prices (RunPod reference)

RTX 4090: ~$0.34/hr

RTX 3090: ~$0.22/hr

Storage (RunPod network volume)

Network volumes billed monthly (docs list rates; treat as small add-on unless you store huge data).

Your experiment costs (rough but useful)

MFCC runs: basically CPU-bound and fast → you can run on cheap CPU or minimal GPU time (near $0).

wav2vec2 embedding cost driver: embedding 1128 short clips.

On a 4090, expect minutes-level to tens-of-minutes depending on implementation/batching.

Total “full run” (embed + train + bootstrap) often fits within ~0.3–1.0 GPU-hours.

So ballpark:

RTX 4090: 0.3–1.0 hr × $0.34/hr → $0.10–$0.34 per run

RTX 3090: 0.3–1.0 hr × $0.22/hr → $0.07–$0.22 per run

Even if you’re slower and it takes 2 GPU-hours:

4090: ~$0.68

3090: ~$0.44

That’s why I’m pushing RunPod for iteration.

Quick “what should I do tomorrow morning?”

Use RunPod RTX 4090 for wav2vec2/PCA experiments; keep /workspace/cache persistent.

Use your local machine (or cheap CPU) for MFCC experiments.

Add the Dockerfile so every run is reproducible and paper-supplement friendly.
