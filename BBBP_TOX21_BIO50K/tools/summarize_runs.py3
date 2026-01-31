import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

# Make sure we can import tools/bootstrap_auc.py regardless of where we run from
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from bootstrap_auc import bootstrap_taskwise_auc  # noqa: E402


RUNS = [
    ("Tox21 Descriptors", "runs/tox21_desc_2000/metrics.json"),
    ("Tox21 Audio", "runs/tox21_audio_2000/metrics.json"),
    ("Tox21 Fusion", "runs/tox21_fuse_desc_audio_2000/metrics.json"),
]

# Candidate filenames your training might have produced.
PRED_FILES = [
    "test_preds.npz",
    "preds_test.npz",
    "test_outputs.npz",
    "test_predictions.npz",
    "outputs_test.npz",
]

# Use smaller during iteration; bump to 2000 for final reporting
N_BOOT = 500
SEED = 0
MIN_N = 10


def load_test_npz(run_dir: Path):
    """
    Looks for an NPZ in the run_dir containing y_true/y_pred (and optionally task_names).
    Returns (y_true, y_pred, task_names) or (None, None, None) if not found.
    """
    for fn in PRED_FILES:
        p = run_dir / fn
        if p.exists():
            z = np.load(p, allow_pickle=True)

            y_true = z["y_true"] if "y_true" in z else (z["y_test"] if "y_test" in z else None)
            y_pred = z["y_pred"] if "y_pred" in z else (z["p_test"] if "p_test" in z else None)

            task_names = None
            if "task_names" in z:
                task_names = list(z["task_names"])
            elif "tasks" in z:
                task_names = list(z["tasks"])

            return y_true, y_pred, task_names

    # fallback: any npz
    any_npz = list(run_dir.glob("*.npz"))
    if any_npz:
        p = any_npz[0]
        z = np.load(p, allow_pickle=True)
        y_true = z["y_true"] if "y_true" in z else (z["y_test"] if "y_test" in z else None)
        y_pred = z["y_pred"] if "y_pred" in z else (z["p_test"] if "p_test" in z else None)
        task_names = list(z["task_names"]) if "task_names" in z else None
        return y_true, y_pred, task_names

    return None, None, None


def macro_auc_once(y_true, y_pred, task_names, min_n=10):
    """
    Compute macro AUC once (no bootstrap), skipping tasks with too few labels or single-class labels.
    Missing labels must be NaN in y_true.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for j, _tname in enumerate(task_names):
            mask = ~np.isnan(y_true[:, j])
            if int(mask.sum()) < min_n:
                continue
            yt = y_true[mask, j]
            yp = y_pred[mask, j]
            if np.unique(yt).size < 2:
                continue
            try:
                vals.append(roc_auc_score(yt, yp))
            except ValueError:
                continue

    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def bootstrap_macro_auc(y_true, y_pred, task_names, n_boot=1000, seed=0, min_n=10):
    """
    Bootstrap CI for macro AUC by resampling molecules (rows).
    For each bootstrap sample, compute per-task AUCs (skipping undefined) and average them.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = y_true.shape[0]
    macros = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = macro_auc_once(y_true[idx], y_pred[idx], task_names, min_n=min_n)
        if np.isfinite(m):
            macros.append(m)

    if len(macros) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot_ok": 0}

    macros = np.asarray(macros)
    return {
        "mean": float(macros.mean()),
        "ci_low": float(np.percentile(macros, 2.5)),
        "ci_high": float(np.percentile(macros, 97.5)),
        "n_boot_ok": int(len(macros)),
    }


print(f"\n{'Run Name':25s} | {'Features':>8s} | {'Train':>6s} | {'Val':>6s} | {'Test':>6s}")
print("-" * 65)

for name, metrics_path in RUNS:
    p = Path(metrics_path)
    if not p.exists():
        print(f"{name:25s} | MISSING")
        continue

    run_dir = p.parent

    with open(p) as f:
        data = json.load(f)

    f_count = data.get("train_X_shape", [0, "??"])[1]
    tr = data["train"]["summary"]["macro_auc"]
    va = data["val"]["summary"]["macro_auc"]
    te = data["test"]["summary"]["macro_auc"]

    print(f"{name:25s} | {f_count:8} | {tr:6.3f} | {va:6.3f} | {te:6.3f}")

    # ---- Load test predictions ----
    y_true, y_pred, task_names = load_test_npz(run_dir)

    if y_true is None or y_pred is None:
        print(f"  ↳ No test predictions found in {run_dir} (expected one of {PRED_FILES} or any .npz).")
        continue

    if task_names is None:
        task_names = [f"task_{i}" for i in range(y_true.shape[1])]

    # ---- Task-wise bootstrap CIs ----
    task_ci = bootstrap_taskwise_auc(
        y_true, y_pred, task_names,
        n_boot=N_BOOT, seed=SEED, min_n=MIN_N
    )

    # ---- Macro AUC CI (bootstrap over molecules) ----
    macro_ci = bootstrap_macro_auc(
        y_true, y_pred, task_names,
        n_boot=N_BOOT, seed=SEED, min_n=MIN_N
    )

    # Point estimate macro from the (task-wise) boot means
    vals = np.asarray([d["mean"] for d in task_ci.values()], dtype=float)
    macro_point = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

    print(f"  ↳ Task-wise bootstrap computed (n_boot={N_BOOT}). "
          f"Macro(point)={macro_point:.3f}")
    print(f"  ↳ Macro AUC (bootstrap over molecules): {macro_ci['mean']:.3f} "
          f"[{macro_ci['ci_low']:.3f}, {macro_ci['ci_high']:.3f}] "
          f"(boot_ok={macro_ci['n_boot_ok']})")

    # ---- Best/Worst tasks without overlap ----
    rows = [(t, d["mean"], d["ci_low"], d["ci_high"], d["n_eff"], d["n_boot_ok"]) for t, d in task_ci.items()]
    rows = [r for r in rows if np.isfinite(r[1])]
    rows.sort(key=lambda x: x[1], reverse=True)

    if not rows:
        print("  ↳ No finite task AUCs (likely too many single-class tasks or missing labels).")
        continue

    k = min(5, max(1, len(rows) // 2))
    best = rows[:k]
    worst = rows[-k:] if len(rows) > k else []
    best_tasks = {r[0] for r in best}
    worst = [r for r in worst if r[0] not in best_tasks]

    print(f"  ↳ Best {len(best)} tasks:")
    for t, m, lo, hi, n_eff, n_ok in best:
        print(f"     {t:25s} {m:.3f} [{lo:.3f}, {hi:.3f}]  n={n_eff} boot_ok={n_ok}")

    print(f"  ↳ Worst {len(worst)} tasks:")
    for t, m, lo, hi, n_eff, n_ok in worst:
        print(f"     {t:25s} {m:.3f} [{lo:.3f}, {hi:.3f}]  n={n_eff} boot_ok={n_ok}")

