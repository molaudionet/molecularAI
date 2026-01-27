import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
from sklearn.metrics import roc_auc_score


def _bootstrap_auc_1d(y_true_1d, y_pred_1d, rng, n_boot=1000, min_pos=5, min_neg=5):
    """
    Bootstrap ROC-AUC for 1D arrays. Returns dict with mean/ci_low/ci_high.
    Suppresses UndefinedMetricWarning and skips low-support class balances.
    """
    y_true_1d = np.asarray(y_true_1d)
    y_pred_1d = np.asarray(y_pred_1d)

    n = len(y_true_1d)
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_eff": 0, "n_boot_ok": 0}

    # overall support check (reduces invalid bootstraps a lot on rare tasks)
    pos = np.sum(y_true_1d == 1)
    neg = np.sum(y_true_1d == 0)
    if pos < min_pos or neg < min_neg:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_eff": int(n), "n_boot_ok": 0}

    aucs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)

        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt = y_true_1d[idx]
            yp = y_pred_1d[idx]

            # fast skip for single-class bootstrap sample
            if np.all(yt == 0) or np.all(yt == 1):
                continue

            try:
                aucs.append(roc_auc_score(yt, yp))
            except ValueError:
                continue

    if len(aucs) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_eff": int(n), "n_boot_ok": 0}

    aucs = np.asarray(aucs)
    return {
        "mean": float(aucs.mean()),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
        "n_eff": int(n),
        "n_boot_ok": int(len(aucs)),
    }

def _bootstrap_auc_1d2(y_true_1d, y_pred_1d, rng, n_boot=1000):
    """
    Bootstrap ROC-AUC for 1D arrays. Returns dict with mean/ci_low/ci_high.
    If no valid bootstrap samples exist (e.g., always single-class), returns NaNs.
    """
    y_true_1d = np.asarray(y_true_1d)
    y_pred_1d = np.asarray(y_pred_1d)

    n = len(y_true_1d)
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_eff": 0, "n_boot_ok": 0}

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true_1d[idx]
        yp = y_pred_1d[idx]
        # roc_auc_score fails if yt is single-class
        try:
            aucs.append(roc_auc_score(yt, yp))
        except ValueError:
            continue

    if len(aucs) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_eff": n, "n_boot_ok": 0}

    aucs = np.asarray(aucs)
    return {
        "mean": float(aucs.mean()),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
        "n_eff": int(n),
        "n_boot_ok": int(len(aucs)),
    }


def bootstrap_auc(y_true, y_pred, n_boot=1000, seed=42):
    """
    Backwards-compatible: bootstrap ROC-AUC for 1D y_true/y_pred.
    """
    rng = np.random.default_rng(seed)
    out = _bootstrap_auc_1d(y_true, y_pred, rng=rng, n_boot=n_boot)
    # Keep the old keys for compatibility
    return {"mean": out["mean"], "ci_low": out["ci_low"], "ci_high": out["ci_high"]}


def bootstrap_taskwise_auc(y_true, y_pred, task_names, n_boot=1000, seed=42, min_n=10):
    """
    Task-wise ROC-AUC with bootstrap CIs for multi-task arrays shaped [N, T].
    Handles missing labels via NaNs in y_true.

    Returns:
      dict: {task_name: {"mean":..., "ci_low":..., "ci_high":..., "n_eff":..., "n_boot_ok":...}, ...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("bootstrap_taskwise_auc expects y_true and y_pred as 2D arrays [N, T].")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if len(task_names) != y_true.shape[1]:
        raise ValueError(f"task_names length {len(task_names)} != num tasks {y_true.shape[1]}")

    rng = np.random.default_rng(seed)
    results = {}

    for i, task in enumerate(task_names):
        mask = ~np.isnan(y_true[:, i])
        n_eff = int(mask.sum())

        # Too few labeled points
        if n_eff < min_n:
            results[task] = {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_eff": n_eff, "n_boot_ok": 0}
            continue

        yt = y_true[mask, i]
        yp = y_pred[mask, i]

        # If single-class overall, AUC undefined
        uniq = np.unique(yt)
        if uniq.size < 2:
            results[task] = {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_eff": n_eff, "n_boot_ok": 0}
            continue

        #results[task] = _bootstrap_auc_1d(yt, yp, rng=rng, n_boot=n_boot)
        results[task] = _bootstrap_auc_1d(yt, yp, rng=rng, n_boot=n_boot, min_pos=5, min_neg=5)

    return results


def bootstrap_taskwise_auc_simple(y_true, y_pred, task_names, n_boot=1000, seed=42, min_n=10):
    """
    Convenience wrapper that matches your earlier `taskwise_auc` output style
    but includes CI tuples:
      {task: (mean, ci_low, ci_high)}
    """
    out = bootstrap_taskwise_auc(y_true, y_pred, task_names, n_boot=n_boot, seed=seed, min_n=min_n)
    return {k: (v["mean"], v["ci_low"], v["ci_high"]) for k, v in out.items()}

