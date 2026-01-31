from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float | None:
    # AUROC requires both classes
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, prob))

def classification_metrics(y_true: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    m = mask.astype(bool)
    n = int(m.sum())
    if n == 0:
        return {"auc": None, "acc": None, "n": 0}

    yt = y_true[m].astype(int)
    pr = prob[m]

    auc = _safe_auc(yt, pr)
    pred = (pr >= 0.5).astype(int)
    acc = float(accuracy_score(yt, pred))
    return {"auc": auc, "acc": acc, "n": n}

def multitask_summary(per_task: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    aucs = [v["auc"] for v in per_task.values() if v.get("auc") is not None]
    accs = [v["acc"] for v in per_task.values() if v.get("acc") is not None]
    ns   = [v["n"]   for v in per_task.values() if v.get("n") is not None]

    return {
        "macro_auc": None if len(aucs) == 0 else float(np.mean(aucs)),
        "macro_acc": None if len(accs) == 0 else float(np.mean(accs)),
        "total_labeled": int(np.sum(ns)) if len(ns) else 0,
        "tasks_with_auc": int(len(aucs)),
        "tasks_total": int(len(per_task)),
    }

