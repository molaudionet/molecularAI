"""
training/trainer.py
Trainer for scikit-learn logistic regression (no PyTorch needed).
Compatible with models.head.train_logreg_multitask API.
"""
import numpy as np
from models.head import train_logreg_multitask, predict_proba
from training.metrics import classification_metrics, regression_metrics


def evaluate(model, X, Y, M, cfg):
    """
    Evaluate model on data.
    Supports both classification and regression tasks.
    
    Args:
        model: TrainedModel from head.py
        X (np.ndarray): Feature matrix [n_samples, n_features]
        Y (np.ndarray): Label matrix [n_samples, n_tasks]
        M (np.ndarray): Mask matrix [n_samples, n_tasks]
        cfg (dict): Configuration dictionary with task_types

    Returns:
        dict: Dictionary with 'tasks' (per-task metrics) and 'macro' (macro-averaged metrics)
    """
    # Get predictions from model
    prob = predict_proba(model, X)

    # Initialize results
    per_task = {}

    # Get label columns and task types from config
    label_cols = cfg["dataset"]["label_cols"]
    task_types = cfg.get("dataset", {}).get("task_types", {})

    # Evaluate each task
    for j, name in enumerate(label_cols):
        # Determine task type (default to classification)
        task_type = task_types.get(name, "classification")
        
        if task_type == "regression":
            # For regression: prob[:, j] contains predictions (not probabilities)
            pred = prob[:, j]
            per_task[name] = regression_metrics(Y[:, j], pred, M[:, j])
        else:
            # For classification: prob[:, j] contains class probabilities
            per_task[name] = classification_metrics(Y[:, j], prob[:, j], M[:, j])

    # Aggregate metrics across tasks (macro average)
    if not per_task:
        return {"tasks": {}, "macro": {}}

    # Collect all unique metric names across all tasks
    all_metrics = set()
    for task_metrics in per_task.values():
        all_metrics.update(task_metrics.keys())

    # Compute macro average for each metric
    macro = {}
    for metric in all_metrics:
        # Get values for this metric from all tasks that have it
        values = [m.get(metric, 0.0) for m in per_task.values() if metric in m]
        if values:
            macro[metric] = float(np.mean(values))

    return {
        "tasks": per_task,
        "macro": macro
    }


def fit(cfg, X, Y, M, splits):
    """
    Train logistic regression model on the given data.
    No epochs/batches needed -- sklearn trains in one shot.
    
    Args:
        cfg (dict): Configuration dictionary
        X (np.ndarray): Feature matrix [n_samples, n_features]
        Y (np.ndarray): Label matrix [n_samples, n_tasks]
        M (np.ndarray): Mask matrix [n_samples, n_tasks]
        splits (dict): Dictionary with 'train', 'val', 'test' indices

    Returns:
        dict: Results with train/val/test metrics and trained model
    """
    # Extract split indices
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # Get training data
    X_train, Y_train, M_train = X[train_idx], Y[train_idx], M[train_idx]
    
    #  TRAIN LOGISTIC REGRESSION IN ONE SHOT (no epochs/batches)
    print("Training logistic regression model...")
    model = train_logreg_multitask(
        X=X_train,
        Y=Y_train,
        M=M_train,
        label_names=cfg["dataset"]["label_cols"],
        task_types=cfg.get("dataset", {}).get("task_types", {}),
        max_iter=cfg.get("train", {}).get("max_iter", 2000),
        min_n=cfg.get("train", {}).get("min_n", 10)
    )
    print(" Model training complete")

    #  EVALUATE ON ALL SPLITS
    results = {
        "train": evaluate(model, X[train_idx], Y[train_idx], M[train_idx], cfg),
        "val": evaluate(model, X[val_idx], Y[val_idx], M[val_idx], cfg),
        "test": evaluate(model, X[test_idx], Y[test_idx], M[test_idx], cfg),
        "model": model,
    }

    # Print final results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Train metrics: {results['train']['macro']}")
    print(f"Val metrics:   {results['val']['macro']}")
    print(f"Test metrics:  {results['test']['macro']}")
    print("=" * 50 + "\n")

    return results