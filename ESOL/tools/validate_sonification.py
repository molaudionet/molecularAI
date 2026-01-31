# tools/validate_sonification.py
import numpy as np
from rdkit import Chem
from featurizers.sonify_smiles import smiles_to_wav
from featurizers.audio_featurizer import embed_wav
import tempfile, os

# Chemically similar pairs SHOULD have high cosine similarity
pairs = [
    ("ethanol", "CCO", "methanol", "CO"),          # Both alcohols → should sound similar
    ("benzene", "c1ccccc1", "toluene", "Cc1ccccc1"), # Methylbenzene → similar
    ("water", "O", "hexane", "CCCCCC"),            # Polar vs nonpolar → should sound DIFFERENT
]

print("🧪 Sonification Quality Check")
print("="*70)

for name1, sm1, name2, sm2 in pairs:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
        wav1 = f1.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
        wav2 = f2.name
    
    try:
        smiles_to_wav(sm1, wav1)
        smiles_to_wav(sm2, wav2)
        emb1 = embed_wav(wav1)
        emb2 = embed_wav(wav2)
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        status = "" if ("ethanol" in name1 and sim > 0.7) or \
                      ("water" in name1 and sim < 0.4) else ""
        print(f"{status} {name1:12s} vs {name2:12s} | Cosine similarity: {sim:.3f}")
    finally:
        os.unlink(wav1); os.unlink(wav2)

print("="*70)
print(" GOOD: Similar molecules >0.7 similarity, dissimilar <0.4")
print(" BAD: Random similarities → sonification lacks chemical meaning")
