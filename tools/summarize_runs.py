import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

# Ensure we can import bootstrap_auc no matter where we run from
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
from bootstrap_auc import bootstrap_taskwise_auc  # optional, but kept available


# --------- CONFIG ---------
N_BOOT = 200      # bump to 2000 for final paper numbers
#N_BOOT = 500      # bump to 2000 for final paper numbers
SEED = 0
MIN_N = 10        # min labeled points per task in a bootstrap sample

RUNS = [
    # Tox21
    ("Tox21", "Desc",  "runs/tox21_desc_2000/metrics.json"),
    ("Tox21", "Audio", "runs/tox21_audio_2000/metrics.json"),
    ("Tox21", "Fuse",  "runs/tox21_fuse_desc_audio_2000/metrics.json"),

    # BBBP
    ("BBBP", "Desc",  "runs/bbbp_desc/metrics.json"),
    ("BBBP", "Audio", "runs/bbbp_audio/metrics.json"),
    ("BBBP", "Fuse",  "runs/bbbp_fuse/metrics.json"),

    # Bioactive50284
    ("Bioactive", "Desc",  "runs/bioactive_desc/metrics.json"),
    ("Bioactive", "Audio", "runs/bioactive_audio/metrics.json"),
    ("Bioactive", "Fuse",  "runs/bioactive_fuse/metrics.json"),
]

PRED_FILES = [
    "test_preds.npz",
    "preds_test.npz",
    "test_outputs.npz",
    "test_predictions.npz",
    "outputs_test.npz",
]


def load_test_npz(run_dir: Path):
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
    return None, None, None

def macro_auc_once(y_true, y_pred, task_names, min_n=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)

        for j in range(len(task_names)):
            mask = ~np.isnan(y_true[:, j])
            if int(mask.sum()) < min_n:
                continue

            yt = y_true[mask, j].astype(float)
            yp = y_pred[mask, j].astype(float)

            # remove any non-finite rows
            fin = np.isfinite(yt) & np.isfinite(yp)
            yt = yt[fin]
            yp = yp[fin]
            if yt.size < min_n:
                continue

            # ensure binary labels
            u = np.unique(yt)
            if u.size < 2:
                continue
            # if labels are not 0/1 but still binary (e.g. -1/1), roc_auc_score is fine,
            # but if there are >2 unique values, skip.
            if u.size > 2:
                continue

            try:
                vals.append(roc_auc_score(yt, yp))
            except Exception:
                continue

    return float(np.mean(vals)) if vals else np.nan

def macro_auc_once2(y_true, y_pred, task_names, min_n=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for j, _ in enumerate(task_names):
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


def fmt(x):
    return "  nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:6.3f}"


print("\nConsolidated Results (macro AUC with 95% CI on Test)")
print(f"(macro CI computed by bootstrapping molecules; n_boot={N_BOOT}, seed={SEED})\n")

hdr = (
    f"{'Dataset':10s} | {'Mode':5s} | {'Feat':>5s} | "
    f"{'Train':>6s} | {'Val':>6s} | {'Test':>6s} | "
    f"{'TestCI':>20s} | {'boot':>4s}"
)
print(hdr)
print("-" * len(hdr))

for dataset, mode, metrics_path in RUNS:
    p = Path(metrics_path)
    if not p.exists():
        print(f"{dataset:10s} | {mode:5s} | {'MISSING':>5s}")
        continue

    run_dir = p.parent
    with open(p) as f:
        data = json.load(f)

    feat = data.get("train_X_shape", [0, "??"])[1]
    tr = data["train"]["summary"]["macro_auc"]
    va = data["val"]["summary"]["macro_auc"]
    te = data["test"]["summary"]["macro_auc"]

    # CI from raw test preds (if present)
    y_true, y_pred, task_names = load_test_npz(run_dir)
    if y_true is None or y_pred is None:
        ci_str = "      (no preds)"
        boot_ok = ""
    else:
        if task_names is None:
            task_names = [f"task_{i}" for i in range(y_true.shape[1])]
        ci = bootstrap_macro_auc(y_true, y_pred, task_names, n_boot=N_BOOT, seed=SEED, min_n=MIN_N)
        ci_str = f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        boot_ok = str(ci["n_boot_ok"])

    print(
        f"{dataset:10s} | {mode:5s} | {str(feat):>5s} | "
        f"{fmt(tr)} | {fmt(va)} | {fmt(te)} | "
        f"{ci_str:>20s} | {boot_ok:>4s}"
    )

print("")

