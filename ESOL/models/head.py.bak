"""
models/head.py
Multi-task model head with support for BOTH classification and regression.
Patent-compliant implementation for Molecular Sonification framework (USP 9,018,506).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge  #  ADDED Ridge for regression

@dataclass
class TrainedModel:
    label_names: List[str]
    task_types: Dict[str, str]          # "classification" or "regression" per task
    models: Dict[str, Optional[Any]]    # Per-task model (LogisticRegression/Ridge/None)
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    default_value: Dict[str, float]     # Fallback: prevalence (class) or mean (regression)

def standardize_fit(X: np.ndarray):
    """Compute mean/std for feature standardization."""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # Avoid division by zero
    return mean, std

def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Apply standardization to features."""
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
    Train per-task models (classification OR regression) with robust fallbacks.
    
    Args:
        X: Feature matrix [n_samples, n_features]
        Y: Label matrix [n_samples, n_tasks] (continuous for regression, binary for classification)
        M: Mask matrix [n_samples, n_tasks] (>0.5 = labeled)
        label_names: List of task names (e.g., ["ESOL"])
        task_types: Dict mapping task name → "classification" or "regression"
        max_iter: Max iterations for sklearn solvers
        min_n: Minimum labeled samples required to train a task-specific model
    
    Returns:
        TrainedModel dataclass with per-task models and fallback values
    """
    # Standardize features globally
    mean, std = standardize_fit(X)
    Xs = standardize_apply(X, mean, std)

    models: Dict[str, Optional[Any]] = {}
    default_value: Dict[str, float] = {}

    for j, name in enumerate(label_names):
        # Identify valid (labeled) samples for this task
        valid = M[:, j] > 0.5
        n_valid = int(valid.sum())

        # Too few labeled points → use fallback
        if n_valid < min_n:
            task_type = task_types.get(name, "classification")
            models[name] = None
            default_value[name] = 0.5 if task_type == "classification" else 0.0
            print(f" Skipping task '{name}': insufficient labels (n={n_valid} < {min_n})")
            continue

        # Extract valid labels/features
        yj = Y[valid, j]
        xj = Xs[valid]
        task_type = task_types.get(name, "classification")

        # Check for single-value data (cannot learn meaningful model)
        if task_type == "regression":
            uniq = np.unique(np.round(yj, 6))  # Tolerance for float equality
        else:
            uniq = np.unique(yj.astype(int))
        
        if uniq.size < 2:
            models[name] = None
            default_value[name] = float(yj.mean()) if yj.size else (0.5 if task_type == "classification" else 0.0)
            print(f" Skipping task '{name}': single value in train (value={uniq[0]:.3f}, n={n_valid})")
            continue

        #  TRAIN APPROPRIATE MODEL PER TASK TYPE
        try:
            if task_type == "regression":
                # Ridge regression for continuous targets (ESOL solubility)
                model = Ridge(alpha=1.0, max_iter=max_iter)
                model.fit(xj, yj)
            else:
                # Logistic regression for binary classification (BBBP/Tox21)
                model = LogisticRegression(max_iter=max_iter, n_jobs=-1)
                model.fit(xj, yj.astype(int))
            
            models[name] = model
            default_value[name] = float(yj.mean())
        except Exception as e:
            print(f" Training failed for task '{name}': {e}")
            models[name] = None
            default_value[name] = float(yj.mean())

    return TrainedModel(
        label_names=label_names,
        task_types=task_types,
        models=models,
        scaler_mean=mean,
        scaler_std=std,
        default_value=default_value,
    )

def predict_proba(model: TrainedModel, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions for all tasks.
    
    For regression tasks: returns raw predictions (continuous values).
    For classification tasks: returns class probabilities (P(class=1)).
    
    Args:
        model: TrainedModel instance
        X: Feature matrix [n_samples, n_features]
    
    Returns:
        Prediction matrix [n_samples, n_tasks]
    """
    # Standardize input features
    Xs = standardize_apply(X, model.scaler_mean, model.scaler_std)
    N = X.shape[0]
    T = len(model.label_names)
    prob = np.zeros((N, T), dtype=np.float32)

    for j, name in enumerate(model.label_names):
        m = model.models.get(name)
        task_type = model.task_types.get(name, "classification")

        if m is None:
            # Use fallback value when no model trained
            prob[:, j] = np.float32(model.default_value.get(name, 0.5))
            continue

        if task_type == "regression":
            #  RAW PREDICTION for regression (e.g., logS solubility)
            prob[:, j] = m.predict(Xs).astype(np.float32)
        else:
            # Class probability for binary classification
            prob[:, j] = m.predict_proba(Xs)[:, 1].astype(np.float32)

    return prob
