"""
training/trainer.py
Minimal trainer for sklearn-based multi-modal models.
NO PyTorch dependencies -- single-shot training matches paper methodology.
Patent-compliant implementation for Molecular Sonification framework (USP 9,018,506).
"""
import numpy as np
from models.head import train_logreg_multitask, predict_proba
from training.metrics import classification_metrics, regression_metrics

def evaluate(model, X, Y, M, cfg):
    """
    Evaluate model on data (classification + regression aware).
    
    Args:
        model: TrainedModel from head.py
        X: Feature matrix [n_samples, n_features]
        Y: Label matrix [n_samples, n_tasks]
        M: Mask matrix [n_samples, n_tasks]
        cfg: Configuration dictionary with task_types
    
    Returns:
        dict with 'tasks' (per-task metrics) and 'macro' (averaged metrics)
    """
    # Get predictions
    prob = predict_proba(model, X)
    per_task = {}
    
    # Extract task metadata from config
    label_cols = cfg["dataset"]["label_cols"]
    task_types = cfg.get("dataset", {}).get("task_types", {})

    # Evaluate each task
    for j, name in enumerate(label_cols):
        task_type = task_types.get(name, "classification")
        valid_mask = M[:, j] > 0.5
        
        if not np.any(valid_mask):
            continue  # Skip tasks with no valid labels
            
        if task_type == "regression":
            # Regression: prob[:, j] contains raw predictions
            pred = prob[:, j]
            per_task[name] = regression_metrics(Y[valid_mask, j], pred[valid_mask], M[valid_mask, j])
        else:
            # Classification: prob[:, j] contains class probabilities
            per_task[name] = classification_metrics(Y[valid_mask, j], prob[valid_mask, j], M[valid_mask, j])

    # Compute macro-average metrics across tasks
    if not per_task:
        return {"tasks": {}, "macro": {}}
    
    # Collect all metric names
    all_metrics = set()
    for task_metrics in per_task.values():
        all_metrics.update(task_metrics.keys())
    
    # Average each metric across tasks
    macro = {}
    for metric in all_metrics:
        values = [m.get(metric, 0.0) for m in per_task.values() if metric in m]
        if values:
            macro[metric] = float(np.mean(values))

    return {"tasks": per_task, "macro": macro}

def fit(cfg, X, Y, M, splits):
    """
    Train model in ONE SHOT (sklearn style) -- NO epochs/batches needed.
    Matches paper's methodology: simple task-specific models with fallbacks.
    
    Args:
        cfg: Configuration dictionary
        X: Feature matrix [n_samples, n_features]
        Y: Label matrix [n_samples, n_tasks]
        M: Mask matrix [n_samples, n_tasks]
        splits: Dict with 'train', 'val', 'test' index arrays
    
    Returns:
        dict with train/val/test metrics and trained model
    """
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    #  SINGLE-SHOT TRAINING (paper methodology)
    print("Training multi-task model...")
    model = train_logreg_multitask(
        X=X[train_idx],
        Y=Y[train_idx],
        M=M[train_idx],
        label_names=cfg["dataset"]["label_cols"],
        task_types=cfg.get("dataset", {}).get("task_types", {}),
        max_iter=cfg.get("train", {}).get("max_iter", 2000),
        min_n=cfg.get("train", {}).get("min_n", 10)
    )
    print(" Training complete")

    # Evaluate on all splits
    results = {
        "train": evaluate(model, X[train_idx], Y[train_idx], M[train_idx], cfg),
        "val": evaluate(model, X[val_idx], Y[val_idx], M[val_idx], cfg),
        "test": evaluate(model, X[test_idx], Y[test_idx], M[test_idx], cfg),
        "model": model,
    }

    # Print summary with task-appropriate metrics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Determine primary metric based on task type
    primary_metric = "r2" if any(
        cfg.get("dataset", {}).get("task_types", {}).get(name) == "regression"
        for name in cfg["dataset"]["label_cols"]
    ) else "auc"
    
    for split in ["train", "val", "test"]:
        metric_val = results[split]["macro"].get(primary_metric)
        if metric_val is not None:
            status = "" if split == "test" and primary_metric == "r2" and metric_val >= 0.85 else ""
            print(f"{split.capitalize():<5} {primary_metric.upper()}: {metric_val:.3f} {status}")
    
    print("=" * 60 + "\n")

    return results
