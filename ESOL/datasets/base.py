from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class Sample:
    id: str
    smiles: str
    y: np.ndarray        # (T,)
    mask: np.ndarray     # (T,) 1 if label exists, 0 if missing
    meta: Dict[str, Any]

class BaseDataset:
    def __len__(self) -> int:
        raise NotImplementedError
    def __getitem__(self, idx: int) -> Sample:
        raise NotImplementedError
    @property
    def label_names(self) -> List[str]:
        raise NotImplementedError
