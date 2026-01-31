from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class TrainedModel:
    label_names: List[str]
    task_types: Dict[str, str]
    models: Dict[str, Optional[Any]]          # per-label model (logistic regression or None)
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    default_proba: Dict[str, float]           # fallback prob per task (e.g., prevalence)


def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def train_logreg_multitask(
    X: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    label_names: List[str],
    task_types: Dict[str, str],
    max_iter: int = 2000,
    min_n: int = 10,
) -> TrainedModel:
    """
    Train per-task logistic regression models for multitask classification.

    Robustness:
      - If a task has too few labeled samples (valid < min_n) -> model=None
      - If a task's labeled training data has only one class -> model=None
      - Store a default probability per task (prevalence if computable, else 0.5)
    """
    mean, std = standardize_fit(X)
    Xs = standardize_apply(X, mean, std)

    models: Dict[str, Optional[Any]] = {}
    default_proba: Dict[str, float] = {}

    for j, name in enumerate(label_names):
        valid = M[:, j] > 0.5
        n_valid = int(valid.sum())

        # Too few labeled points
        if n_valid < min_n:
            models[name] = None
            default_proba[name] = 0.5
            continue

        yj = Y[valid, j].astype(int)
        xj = Xs[valid]

        # If only one class present, sklearn can't fit
        uniq = np.unique(yj)
        if uniq.size < 2:
            models[name] = None
            # prevalence is either 0.0 or 1.0 here; for prediction,
            # using prevalence is honest but can be extreme; still fine.
            default_proba[name] = float(yj.mean()) if yj.size else 0.5
            print(f" Skipping task {name}: only one class in train (class={int(uniq[0])}, n={n_valid})")
            continue

        # Use prevalence as a reasonable default if model fails later
        default_proba[name] = float(yj.mean())

        clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
        clf.fit(xj, yj)
        models[name] = clf

    return TrainedModel(
        label_names=label_names,
        task_types=task_types,
        models=models,
        scaler_mean=mean,
        scaler_std=std,
        default_proba=default_proba,
    )


def predict_proba(model: TrainedModel, X: np.ndarray) -> np.ndarray:
    Xs = standardize_apply(X, model.scaler_mean, model.scaler_std)
    N = X.shape[0]
    T = len(model.label_names)
    prob = np.zeros((N, T), dtype=np.float32)

    for j, name in enumerate(model.label_names):
        m = model.models.get(name)

        if m is None:
            # Fill with default probability (prevalence or 0.5)
            prob[:, j] = np.float32(model.default_proba.get(name, 0.5))
            continue

        prob[:, j] = m.predict_proba(Xs)[:, 1].astype(np.float32)

    return prob

