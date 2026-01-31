#!/usr/bin/env python
"""Validate that a saved model produces reasonable predictions."""
import pickle
import numpy as np
from models.head import predict_proba

def validate(run_dir="runs/esol_fuse"):
    # Load model
    with open(f"{run_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load test features (if cached)
    try:
        X_test = np.load(f"{run_dir}/X_test.npy")
    except:
        print(" X_test.npy not found - skipping prediction validation")
        return
    
    # Predict
    y_pred = predict_proba(model, X_test)
    
    # Check prediction sanity
    print(f"Model: {run_dir}")
    print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}] logS")
    print(f"Mean prediction: {y_pred.mean():.2f} logS")
    
    # ESOL ground truth range: ~[-11, +2] logS
    if y_pred.min() < -15 or y_pred.max() > 5:
        print(" WARNING: Predictions outside chemically plausible range!")
    else:
        print(" Predictions look chemically plausible")
    
    # Load metrics to check R²
    import json
    with open(f"{run_dir}/metrics.json") as f:
        metrics = json.load(f)
    r2 = metrics["test"]["macro"].get("r2", None)
    if r2 is not None:
        status = "" if r2 >= 0.85 else ""
        print(f"{status} Test R²: {r2:.3f} (target ≥0.85)")

if __name__ == "__main__":
    validate("runs/esol_fuse")
