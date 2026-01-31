#!/usr/bin/env python
"""
ESOL Benchmark Analyzer
Extracts and compares R² scores from existing ESOL runs (desc/audio/fuse).
No training required -- analyzes cached results only.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

RUN_DIRS = {
    "Desc": "runs/esol_desc",
    "Audio": "runs/esol_audio",
    "Fuse": "runs/esol_fuse"
}

def load_json(path: str) -> Optional[Dict]:
    """Safely load JSON file or return None if missing."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {path} ({e})")
        return None

def extract_r2(metrics: Dict) -> Optional[float]:
    """Extract R² from metrics structure (handles both old/new formats)."""
    # Try multiple paths since format may vary
    paths = [
        ('test', 'macro', 'r2'),
        ('test', 'r2'),
        ('test', 'macro', 'R2'),
    ]
    for path in paths:
        d = metrics
        for key in path:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                d = None
                break
        if isinstance(d, (int, float)):
            return float(d)
    return None

def extract_features(config: Dict) -> str:
    """Extract feature dimension from config."""
    # Try multiple locations
    if 'train_X_shape' in config and isinstance(config['train_X_shape'], list):
        return str(config['train_X_shape'][1])
    if 'dataset' in config and 'n_features' in config['dataset']:
        return str(config['dataset']['n_features'])
    return "N/A"

def analyze_esol():
    print("\n" + "="*70)
    print(" ESOL BENCHMARK ANALYSIS (Aqueous Solubility Prediction)")
    print("="*70)
    print("Task Type: Regression | Metric: R² (higher = better)")
    print("Paper Target (Fuse mode): R² ≥ 0.85")
    print("-"*70)
    print(f"{'Mode':<10} | {'Features':<10} | {'Test R²':<12} | {'Status'}")
    print("-"*70)
    
    results = {}
    best_r2 = -1.0
    best_mode = None
    
    for mode_name, run_dir in RUN_DIRS.items():
        metrics_path = Path(run_dir) / "metrics.json"
        config_path = Path(run_dir) / "config.json"
        
        # Check if run exists
        if not metrics_path.exists():
            print(f"{mode_name:<10} | {'MISSING':<10} | {'--':<12} | Run directory not found")
            continue
        
        # Load data
        metrics = load_json(metrics_path)
        config = load_json(config_path) if config_path.exists() else {}
        
        # Extract values
        r2 = extract_r2(metrics) if metrics else None
        feats = extract_features(config) if config else "N/A"
        
        # Determine status
        if r2 is None:
            status = "  No R² found"
            r2_str = "N/A"
        else:
            results[mode_name] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_mode = mode_name
            
            # Status relative to paper target
            if mode_name == "Fuse" and r2 >= 0.85:
                status = " TARGET ACHIEVED"
            elif mode_name == "Fuse" and r2 >= 0.82:
                status = "🟡 Close to target"
            elif r2 >= 0.80:
                status = "🟢 Good"
            else:
                status = " Low"
            r2_str = f"{r2:.3f}"
        
        print(f"{mode_name:<10} | {feats:<10} | {r2_str:<12} | {status}")
    
    print("-"*70)
    
    # Summary insights
    print("\n Key Insights:")
    if best_mode and best_r2 >= 0.85 and "Fuse" in results:
        print(f"    Fuse mode achieved R² = {best_r2:.3f} ≥ 0.85 (paper target validated!)")
    elif "Fuse" in results and results["Fuse"] > results.get("Desc", 0):
        gap = results["Fuse"] - results["Desc"]
        print(f"    Fuse (+{gap:.3f} R²) outperforms Desc -- sonification adds signal")
    elif "Audio" in results and results["Audio"] > results.get("Desc", 0):
        print(f"    Audio alone beats Desc -- acoustic encoding captures solubility patterns")
    
    # Scientific interpretation
    print("\n Scientific Interpretation:")
    if "Fuse" in results and "Desc" in results:
        delta = results["Fuse"] - results["Desc"]
        if delta > 0.02:
            print(f"   • Audio embeddings add >0.02 R² gain → sonification captures")
            print(f"     physicochemical properties missed by 7D descriptors")
        elif delta > 0:
            print(f"   • Modest gain ({delta:.3f} R²) suggests complementary signal")
        else:
            print(f"   •   Audio didn't improve prediction -- check sonification mapping quality")
    
    print("\n Next Steps:")
    print("   1. If Fuse R² < 0.85: Apply trainer.py/head.py fixes for proper regression support")
    print("   2. Validate sonification: Do similar molecules produce similar audio embeddings?")
    print("   3. Run bootstrap CI: python tools/bootstrap_auc.py --task esol --mode fuse")
    print("="*70 + "\n")
    
    return results

if __name__ == "__main__":
    analyze_esol()
