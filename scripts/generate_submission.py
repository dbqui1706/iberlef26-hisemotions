"""
Generate CodaBench submission from single or ensemble model predictions.

Supports both SLM (AutoModelForSequenceClassification) and Pooling models.

Usage (single model):
    python scripts/generate_submission.py \
        --models models/exp02_beto_wbce_sampler/final \
        --configs configs/exp02_beto_wbce_sampler.yaml \
        --test_path data/raw/dev.csv

Usage (ensemble):
    python scripts/generate_submission.py \
        --models models/exp01_beto_wbce_baseline/final \
                 models/exp02_beto_wbce_sampler/final \
                 models/exp03_robertuito_wbce/final \
        --configs configs/exp01_beto_wbce_baseline.yaml \
                  configs/exp02_beto_wbce_sampler.yaml \
                  configs/exp03_robertuito_wbce.yaml \
        --test_path data/raw/dev.csv \
        --name ensemble_3models

Usage (pooling model):
    python scripts/generate_submission.py \
        --pooling_model models/exp04_pooling_mean_wbce/final/model.pt \
        --pooling_type mean \
        --test_path data/raw/dev.csv \
        --name exp04_pooling_mean_wbce
"""
import argparse
import os
import sys
import zipfile

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

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.data_deps.preprocessing import prepare_dataset
from src.models.pooling_models import MeanPoolingClassifier, AttentionPoolingClassifier

# Competition uses 6 emotions (no neutral)
SUBMIT_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope"]
ALL_LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]


def get_slm_logits(model_dir, config_path, test_df):
    """Get logits from a HuggingFace SLM model."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    ds = prepare_dataset(test_df.copy(), cfg["model"]["name"], ALL_LABEL_COLS, cfg["model"]["max_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=len(ALL_LABEL_COLS),
        problem_type="multi_label_classification", ignore_mismatched_sizes=True,
    ).to(device)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="tmp_sub", per_device_eval_batch_size=4, report_to="none"),
    )
    preds = trainer.predict(ds)
    return preds.predictions


def get_pooling_logits(model_path, pooling_type, test_df, model_name="dccuchile/bert-base-spanish-wwm-uncased", max_length=256):
    """Get logits from a Pooling model (.pt checkpoint)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(ALL_LABEL_COLS)

    if pooling_type == "mean":
        model = MeanPoolingClassifier(model_name, num_labels)
    elif pooling_type == "attention":
        model = AttentionPoolingClassifier(model_name, num_labels)
    else:
        raise ValueError(f"Unknown pooling_type: {pooling_type}")

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    ds = prepare_dataset(test_df.copy(), model_name, ALL_LABEL_COLS, max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="tmp_sub", per_device_eval_batch_size=8, report_to="none"),
    )
    preds = trainer.predict(ds)
    return preds.predictions


def optimize_thresholds(probs, gt_df):
    """Optimize per-class thresholds on dev set if labels available."""
    from sklearn.metrics import f1_score
    gt = gt_df[SUBMIT_COLS].values
    thresholds = {}
    for i, col in enumerate(SUBMIT_COLS):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            pred = (probs[:, i] >= t).astype(int)
            f = f1_score(gt[:, i], pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, round(float(t), 2)
        thresholds[col] = best_t
        print(f"    {col:12s}: threshold={best_t:.2f}  F1={best_f1:.4f}")
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    # SLM ensemble
    parser.add_argument("--models", nargs="+", default=None, help="SLM model dirs")
    parser.add_argument("--configs", nargs="+", default=None, help="Config YAMLs for SLM models")
    # Pooling model
    parser.add_argument("--pooling_model", default=None, help="Pooling model .pt path")
    parser.add_argument("--pooling_type", default="mean", help="mean or attention")
    # Common
    parser.add_argument("--test_path", default="data/raw/dev.csv", help="Test data for predictions")
    parser.add_argument("--dev_path", default="data/processed/dev.csv", help="Dev with labels for threshold optimization")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None, help="Manual thresholds (6 values)")
    parser.add_argument("--name", default=None, help="Submission name (creates submissions/{name}/)")
    args = parser.parse_args()

    # Determine submission name
    if args.name:
        sub_name = args.name
    elif args.models and len(args.models) == 1:
        sub_name = os.path.basename(os.path.dirname(args.models[0]))
    elif args.pooling_model:
        sub_name = os.path.basename(os.path.dirname(os.path.dirname(args.pooling_model)))
    else:
        sub_name = "ensemble"

    # Create submission directory
    sub_dir = os.path.join("submissions", sub_name)
    os.makedirs(sub_dir, exist_ok=True)

    # Load test data
    test_df = pd.read_csv(args.test_path)
    print(f"Test samples: {len(test_df)}")

    # Ensure required columns exist
    for col in ALL_LABEL_COLS:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df["text"] = test_df["text"].fillna("").astype(str)

    # Get logits
    if args.pooling_model:
        print(f"\nPooling model: {args.pooling_model}")
        all_logits = [get_pooling_logits(args.pooling_model, args.pooling_type, test_df)]
    elif args.models:
        assert args.configs and len(args.models) == len(args.configs)
        all_logits = []
        for i, (m, c) in enumerate(zip(args.models, args.configs)):
            print(f"[{i+1}/{len(args.models)}] {os.path.basename(os.path.dirname(m))}")
            all_logits.append(get_slm_logits(m, c, test_df))
    else:
        print("Error: Provide --models+--configs or --pooling_model")
        sys.exit(1)

    # Average logits and apply sigmoid
    ensemble_logits = np.mean(all_logits, axis=0)
    probs = 1 / (1 + np.exp(-ensemble_logits))

    # Thresholds
    if args.thresholds:
        thresholds = dict(zip(SUBMIT_COLS, args.thresholds[:6]))
        print(f"Manual thresholds: {thresholds}")
    elif os.path.exists(args.dev_path) and SUBMIT_COLS[0] in pd.read_csv(args.dev_path, nrows=1).columns:
        print("\nOptimizing thresholds on dev set...")
        dev_df = pd.read_csv(args.dev_path)
        # Get dev logits for threshold optimization
        if args.pooling_model:
            dev_logits_list = [get_pooling_logits(args.pooling_model, args.pooling_type, dev_df)]
        else:
            dev_logits_list = []
            for m, c in zip(args.models, args.configs):
                dev_logits_list.append(get_slm_logits(m, c, dev_df))
        dev_logits = np.mean(dev_logits_list, axis=0)
        dev_probs = 1 / (1 + np.exp(-dev_logits))
        thresholds = optimize_thresholds(dev_probs, dev_df)
    else:
        thresholds = {c: 0.5 for c in SUBMIT_COLS}
        print("Using default thresholds: 0.5")

    # Binary predictions (6 columns only)
    submission = pd.DataFrame()
    for i, col in enumerate(SUBMIT_COLS):
        submission[col] = (probs[:, i] >= thresholds[col]).astype(int)

    # Save
    csv_path = os.path.join(sub_dir, "predictions.csv")
    submission.to_csv(csv_path, index=False)
    print(f"\nPredictions saved: {csv_path}")
    print(f"Shape: {submission.shape}")
    print(f"Per-class positive counts:")
    for c in SUBMIT_COLS:
        print(f"  {c:12s}: {int(submission[c].sum())}")

    # Create zip
    zip_path = os.path.join(sub_dir, "predictions.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "predictions.csv")
    print(f"\nZIP created: {zip_path}")
    print(f"Ready to submit on CodaBench!")


if __name__ == "__main__":
    main()
