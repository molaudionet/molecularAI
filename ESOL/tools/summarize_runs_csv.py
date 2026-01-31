import json
from pathlib import Path
import sys
import warnings
import csv

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

# Ensure we can import bootstrap_auc regardless of cwd
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
from bootstrap_auc import bootstrap_taskwise_auc  # noqa: F401


# ---------------- CONFIG ----------------
N_BOOT = 500       # use 2000 for final paper
SEED = 0
MIN_N = 10

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


# ---------------- HELPERS ----------------
def load_test_npz(run_dir: Path):
    for fn in PRED_FILES:
        p = run_dir / fn
        if p.exists():
            z = np.load(p, allow_pickle=True)
            y_true = z["y_true"] if "y_true" in z else (z["y_test"] if "y_test" in z else None)
            y_pred = z["y_pred"] if "y_pred" in z else (z["p_test"] if "p_test" in z else None)
            task_names = list(z["task_names"]) if "task_names" in z else None
            return y_true, y_pred, task_names
    return None, None, None


def macro_auc_once(y_true, y_pred, task_names, min_n=10):
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for j in range(len(task_names)):
            mask = ~np.isnan(y_true[:, j])
            if mask.sum() < min_n:
                continue
            yt = y_true[mask, j]
            yp = y_pred[mask, j]
            if np.unique(yt).size < 2:
                continue
            try:
                vals.append(roc_auc_score(yt, yp))
            except ValueError:
                continue
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_macro_auc(y_true, y_pred, task_names, n_boot=1000, seed=0, min_n=10):
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    macros = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = macro_auc_once(y_true[idx], y_pred[idx], task_names, min_n=min_n)
        if np.isfinite(m):
            macros.append(m)

    if not macros:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n_boot_ok=0)

    macros = np.asarray(macros)
    return dict(
        mean=float(macros.mean()),
        ci_low=float(np.percentile(macros, 2.5)),
        ci_high=float(np.percentile(macros, 97.5)),
        n_boot_ok=len(macros),
    )


def fmt(x):
    return "  nan" if not np.isfinite(x) else f"{x:6.3f}"


# ---------------- MAIN ----------------
print("\nConsolidated Results (macro AUC with 95% CI on Test)")
print(f"(bootstrap over molecules; n_boot={N_BOOT})\n")

header = (
    f"{'Dataset':10s} | {'Mode':5s} | {'Feat':>5s} | "
    f"{'Train':>6s} | {'Val':>6s} | {'Test':>6s} | "
    f"{'Test CI':>20s}"
)
print(header)
print("-" * len(header))

rows_csv = []

for dataset, mode, metrics_path in RUNS:
    p = Path(metrics_path)
    if not p.exists():
        print(f"{dataset:10s} | {mode:5s} | MISSING")
        continue

    run_dir = p.parent
    with open(p) as f:
        data = json.load(f)

    feat = data.get("train_X_shape", [0, "??"])[1]
    tr = data["train"]["summary"]["macro_auc"]
    va = data["val"]["summary"]["macro_auc"]
    te = data["test"]["summary"]["macro_auc"]

    y_true, y_pred, task_names = load_test_npz(run_dir)

    if y_true is None or y_pred is None:
        ci = dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n_boot_ok=0)
        ci_str = "(no preds)"
    else:
        if task_names is None:
            task_names = [f"task_{i}" for i in range(y_true.shape[1])]
        ci = bootstrap_macro_auc(
            y_true, y_pred, task_names,
            n_boot=N_BOOT, seed=SEED, min_n=MIN_N
        )
        ci_str = f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"

    print(
        f"{dataset:10s} | {mode:5s} | {str(feat):>5s} | "
        f"{fmt(tr)} | {fmt(va)} | {fmt(te)} | {ci_str:>20s}"
    )

    rows_csv.append(dict(
        dataset=dataset,
        mode=mode,
        features=feat,
        train_auc=tr,
        val_auc=va,
        test_auc=te,
        test_macro_auc_mean=ci["mean"],
        test_macro_auc_ci_low=ci["ci_low"],
        test_macro_auc_ci_high=ci["ci_high"],
        n_boot_ok=ci["n_boot_ok"],
    ))

# ---------------- CSV EXPORT ----------------
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / "consolidated_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_csv[0].keys())
    writer.writeheader()
    for r in rows_csv:
        writer.writerow(r)

print(f"\n CSV written to: {csv_path.resolve()}\n")

