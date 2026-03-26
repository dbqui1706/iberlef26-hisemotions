"""
Ensemble Inference: Average logits from multiple saved models,
then run per-class threshold optimization.

Usage:
    python scripts/ensemble_eval.py \
        --models models/exp01_beto_wbce_baseline/final \
                 models/exp02_beto_wbce_sampler/final \
                 models/exp03_robertuito_wbce/final \
        --configs configs/exp01_beto_wbce_baseline.yaml \
                  configs/exp02_beto_wbce_sampler.yaml \
                  configs/exp03_robertuito_wbce.yaml \
        --dev_path data/processed/dev.csv
"""
import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import torch

# ── Torchvision hotfix ────────────────────────────────────────────────────────
try:
    from torchvision.io import VideoReader  # noqa: F401
except ImportError:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _DummyVR: pass
        _tvio.VideoReader = _DummyVR

# ── Load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.data_deps.preprocessing import prepare_dataset
from src.evaluate.metrics import find_optimal_thresholds

LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]


def get_logits(model_dir: str, config_path: str, dev_df: pd.DataFrame) -> np.ndarray:
    """Load a saved model, tokenize dev_df using its config, and return raw logits."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]

    # Prepare dataset with the model's own tokenizer
    ds = prepare_dataset(dev_df.copy(), model_name, LABEL_COLS, max_length=max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    args = TrainingArguments(
        output_dir="tmp_ensemble",
        per_device_eval_batch_size=cfg["training"].get("per_device_eval_batch_size", 8),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args)
    preds = trainer.predict(ds)
    return preds.predictions, preds.label_ids


def main():
    parser = argparse.ArgumentParser(description="Ensemble model evaluation")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to saved model directories",
    )
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Paths to YAML configs (same order as --models)",
    )
    parser.add_argument(
        "--dev_path", type=str, default="data/processed/dev.csv",
        help="Path to dev CSV",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Optional per-model weights for weighted average (default: equal)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save ensemble results JSON",
    )
    args = parser.parse_args()

    assert len(args.models) == len(args.configs), \
        "--models and --configs must have equal length"

    n = len(args.models)
    weights = args.weights or [1.0 / n] * n
    assert len(weights) == n, "--weights must match number of models"
    # Normalize
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    dev_df = pd.read_csv(args.dev_path)
    print(f"Dev set: {len(dev_df)} samples\n")

    all_logits = []
    labels = None
    for i, (model_dir, cfg_path) in enumerate(zip(args.models, args.configs)):
        print(f"[{i+1}/{n}] Loading {os.path.basename(model_dir)} ...")
        logits, lbl = get_logits(model_dir, cfg_path, dev_df)
        all_logits.append(logits)
        labels = lbl
        print(f"  Logits shape: {logits.shape}")

    # ── Weighted average of logits ────────────────────────────────────────────
    print(f"\nEnsemble weights: {dict(zip([os.path.basename(m) for m in args.models], weights))}")
    ensemble_logits = sum(w * l for w, l in zip(weights, all_logits))

    # ── Threshold optimization ────────────────────────────────────────────────
    # Competition uses 6 classes (no neutral), so optimize for 6-class macro-F1
    EVAL_INDICES = list(range(6))  # [0,1,2,3,4,5] = anger..hope, excluding neutral
    EVAL_NAMES = LABEL_COLS[:6]
    print("\nOptimizing per-class thresholds (6-class competition metric)...\n")
    optimal_thresholds, optimized_f1, class_report, conf_matrix = find_optimal_thresholds(
        ensemble_logits, labels, LABEL_COLS, eval_class_indices=EVAL_INDICES
    )

    print(f"{'='*60}")
    print(f"  ENSEMBLE Optimized Macro-F1 (6-class): {optimized_f1:.4f}")
    print(f"{'='*60}\n")
    print(f"Per-class report:\n{class_report}")
    print("Confusion Matrix:")
    for name, matrix in zip(EVAL_NAMES, conf_matrix):
        print(f"--- {name} ---")
        print(matrix)

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "models": args.models,
        "weights": weights,
        "optimized_macro_f1": optimized_f1,
        "optimal_thresholds": optimal_thresholds,
        "per_class_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
    }

    out_path = args.output or "models/slm_checkpoints/ensemble_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
