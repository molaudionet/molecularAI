from __future__ import annotations

import os
# Important for macOS stability: set before importing torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from typing import Optional
import numpy as np

def _require_torch():
    try:
        import torch
        import torchvision
        from torchvision import transforms

        # Set threads ONCE (safe): only if torch hasn't been initialized fully yet
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        return torch, torchvision, transforms
    except Exception as e:
        raise RuntimeError(
            "Torch/torchvision not available. Install with:\n"
            "  conda install -c pytorch pytorch torchvision\n"
            "or\n"
            "  pip install torch torchvision\n"
        ) from e

_MODEL = None
_TRANSFORM = None

def get_resnet18_encoder(device: str = "cpu"):
    """
    Returns a model that maps an image -> 512-d embedding (pre-logits).
    """
    global _MODEL, _TRANSFORM
    torch, torchvision, transforms = _require_torch()

    # macOS stability: keep threads low , too late to set now
    #torch.set_num_threads(1)
    #torch.set_num_interop_threads(1)

    if _MODEL is None:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            model = torchvision.models.resnet18(weights=weights)
            preprocess = weights.transforms()
        except Exception:
            model = torchvision.models.resnet18(pretrained=True)
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # replace classifier with identity to output 512-d embedding
        model.fc = torch.nn.Identity()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        _MODEL = model.to(device)
        _TRANSFORM = preprocess

    return _MODEL, _TRANSFORM


def embed_png(png_path: str, device: str = "cpu") -> Optional[np.ndarray]:
    """
    Load PNG and return 512-d float32 embedding, or None if fails.
    """
    torch, _, _ = _require_torch()
    from PIL import Image

    model, preprocess = get_resnet18_encoder(device=device)

    try:
        img = Image.open(png_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).detach().cpu().numpy().reshape(-1).astype(np.float32)
        return emb
    except Exception:
        return None

