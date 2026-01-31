# test_sonification.py
from featurizers.sonify_smiles import smiles_to_wav
import tempfile, os

# Test with chemically similar molecules
pairs = [
    ("ethanol", "CCO", "methanol", "CO"),
    ("benzene", "c1ccccc1", "toluene", "Cc1ccccc1"),
]

print("🧪 Testing sonification quality...")
for name1, sm1, name2, sm2 in pairs:
    f1 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    f2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    try:
        smiles_to_wav(sm1, f1)
        smiles_to_wav(sm2, f2)
        print(f" Generated audio for {name1} and {name2}")
        print(f"   Files: {os.path.basename(f1)}, {os.path.basename(f2)}")
    finally:
        os.unlink(f1); os.unlink(f2)

print("\n Listen to outputs with: afplay /tmp/test.wav (macOS) or play test.wav (Linux)")
