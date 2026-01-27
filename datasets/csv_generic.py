from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .base import BaseDataset, Sample

def _is_missing(v, missing_values) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip() in set(str(x) for x in missing_values):
        return True
    return False

def parse_task_types(task_types_cfg, label_cols: List[str]) -> Dict[str, str]:
    # supports "*:classification" shorthand
    if isinstance(task_types_cfg, str) and task_types_cfg.startswith("*:"):
        t = task_types_cfg.split(":", 1)[1]
        return {c: t for c in label_cols}
    if isinstance(task_types_cfg, dict):
        return {c: task_types_cfg.get(c, "regression") for c in label_cols}
    return {c: "regression" for c in label_cols}

class CSVGenericDataset(BaseDataset):
    def __init__(self, cfg: Dict[str, Any], limit: Optional[int] = None):
        ds = cfg["dataset"]
        self.path = ds["path"]
        self.id_col = ds.get("id_col")
        self.smiles_col = ds["smiles_col"]
        self._label_cols = list(ds["label_cols"])
        self.missing_values = ds.get("missing_values", ["", "NaN", None])
        self.task_types = parse_task_types(ds.get("task_types", {}), self._label_cols)

        df = pd.read_csv(self.path)
        if limit is not None:
            df = df.head(limit).copy()

        # fallback id
        if not self.id_col or self.id_col not in df.columns:
            df["_id"] = [str(i) for i in range(len(df))]
            self.id_col = "_id"

        self.df = df.reset_index(drop=True)

    @property
    def label_names(self) -> List[str]:
        return self._label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        sid = str(row[self.id_col])
        smiles = str(row[self.smiles_col])

        y = np.zeros((len(self._label_cols),), dtype=np.float32)
        mask = np.zeros((len(self._label_cols),), dtype=np.float32)

        for j, col in enumerate(self._label_cols):
            v = row[col] if col in self.df.columns else None
            if _is_missing(v, self.missing_values):
                mask[j] = 0.0
                y[j] = 0.0
            else:
                mask[j] = 1.0
                if self.task_types[col] == "classification":
                    y[j] = float(int(v))
                else:
                    y[j] = float(v)

        meta = {"task_types": self.task_types}
        return Sample(id=sid, smiles=smiles, y=y, mask=mask, meta=meta)
