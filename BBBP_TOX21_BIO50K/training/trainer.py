from __future__ import annotations
from typing import Any, Dict
import numpy as np

from models.head import train_logreg_multitask, predict_proba
from training.metrics import classification_metrics, multitask_summary

def evaluate(model, X, Y, M, return_preds: bool = False) -> Dict[str, Any]:
    prob = predict_proba(model, X)
    per_task = {}
    for j, name in enumerate(model.label_names):
        per_task[name] = classification_metrics(Y[:, j], prob[:, j], M[:, j])

    out = {"per_task": per_task, "summary": multitask_summary(per_task)}

    # add this
    if return_preds:
        # store raw arrays for downstream bootstrap CIs / task-wise AUC
        out["y_pred"] = prob  # shape [N, T]
        out["y_true"] = Y     # keep original 0/1 labels
        out["mask"] = M       # 0/1 mask for missing labels

    return out

def fit(cfg: Dict[str, Any], X: np.ndarray, Y: np.ndarray, M: np.ndarray, splits: Dict[str, np.ndarray]) -> Dict[str, Any]:
    max_iter = int(cfg.get("train", {}).get("max_iter", 2000))
    label_names = cfg["dataset"]["label_cols"]
    task_types = cfg["dataset"]["task_types"]

    model = train_logreg_multitask(
        X[splits["train"]], Y[splits["train"]], M[splits["train"]],
        label_names=label_names, task_types=task_types, max_iter=max_iter
    )

    return {
        "model": model,
        "train": evaluate(model, X[splits["train"]], Y[splits["train"]], M[splits["train"]]),
        "val":   evaluate(model, X[splits["val"]],   Y[splits["val"]],   M[splits["val"]]),
        # only test needs preds (keeps files small)
        "test":  evaluate(model, X[splits["test"]],  Y[splits["test"]],  M[splits["test"]], return_preds=True),
    }

