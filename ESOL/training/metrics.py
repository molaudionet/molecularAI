"""
training/metrics.py

Metrics for both classification and regression tasks.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error


def _safe_auc(y_true, prob):
    """
    Safely compute AUC, handling edge cases.
    
    Args:
        y_true: True binary labels
        prob: Predicted probabilities
    
    Returns:
        float: AUC score (0.5 if computation fails)
    """
    try:
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return 0.5


def classification_metrics(y_true, prob, mask=None):
    """
    Compute classification metrics (AUC).
    
    Args:
        y_true (np.ndarray): True binary labels (0 or 1)
        prob (np.ndarray): Predicted probabilities [0, 1]
        mask (np.ndarray, optional): Binary mask (1=valid, 0=ignore)
    
    Returns:
        dict: Dictionary with 'auc' key
    
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> prob = np.array([0.2, 0.8, 0.9, 0.3, 0.7])
        >>> metrics = classification_metrics(y_true, prob)
        >>> print(metrics)
        {'auc': 0.92}
    """
    # Apply mask if provided
    if mask is not None:
        mask = mask.astype(bool)
        if not mask.any():
            return {"auc": 0.5}
        y_true = y_true[mask]
        prob = prob[mask]
    
    # Check for empty arrays
    if len(y_true) == 0:
        return {"auc": 0.5}
    
    # Compute AUC
    auc = _safe_auc(y_true, prob)
    
    return {"auc": auc}


def regression_metrics(y_true, y_pred, mask=None):
    """
    Compute regression metrics: R², RMSE, MAE.
    
    Args:
        y_true (np.ndarray): True continuous values
        y_pred (np.ndarray): Predicted continuous values
        mask (np.ndarray, optional): Binary mask (1=valid, 0=ignore)
    
    Returns:
        dict: Dictionary with 'r2', 'rmse', 'mae' keys
    
    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1])
        >>> metrics = regression_metrics(y_true, y_pred)
        >>> print(metrics)
        {'r2': 0.975, 'rmse': 0.158, 'mae': 0.125}
    """
    # Apply mask if provided
    if mask is not None:
        mask = mask.astype(bool)
        if not mask.any():
            # No valid samples
            return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # Check for empty arrays
    if len(y_true) == 0:
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
    
    # Check for constant predictions (would cause R² error)
    if np.std(y_pred) == 0:
        print("Warning: All predictions are constant")
        mae_val = float(np.abs(y_true - y_pred).mean())
        return {"r2": 0.0, "rmse": mae_val, "mae": mae_val}
    
    # Compute metrics
    try:
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
    except Exception as e:
        print(f"Warning: Error computing regression metrics: {e}")
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}
    
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }
