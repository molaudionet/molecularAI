from __future__ import annotations
from typing import Optional, List
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

DESC_FNS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("HBA", Descriptors.NumHAcceptors),
    ("HBD", Descriptors.NumHDonors),
    ("RotB", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
]

def featurize(smiles: str) -> Optional[np.ndarray]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    feats = [float(fn(m)) for _, fn in DESC_FNS]
    return np.asarray(feats, dtype=np.float32)

def feature_names() -> List[str]:
    return [n for n, _ in DESC_FNS]
