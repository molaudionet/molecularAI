import argparse
import numpy as np

def r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to test_preds.npz")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.npz)
    print("Keys:", list(data.keys()))

    # Common conventions — adjust if your keys differ
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    rng = np.random.default_rng(args.seed)
    n = len(y_true)

    stats = []
    for _ in range(args.n):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        stats.append(r2(y_true[idx], y_pred[idx]))

    stats = np.sort(np.array(stats))
    point = r2(y_true, y_pred)
    lo = np.percentile(stats, 2.5)
    hi = np.percentile(stats, 97.5)

    print(f"Test R2: {point:.3f}")
    print(f"95% bootstrap CI: [{lo:.3f}, {hi:.3f}]  (n={args.n})")

if __name__ == "__main__":
    main()

