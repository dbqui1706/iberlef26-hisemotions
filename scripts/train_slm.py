"""
Train SLM (Small Language Model) for HISEMOTIONS multi-label classification.

Uses AutoModelForSequenceClassification with config-driven loss and sampler.

Usage:
    python scripts/train_slm.py --config configs/exp01_beto_wbce_baseline.yaml
"""
import argparse
import yaml
import os
import sys
import shutil

# Load .env (HF_TOKEN, etc.)
from dotenv import load_dotenv
load_dotenv()

# Fix nightly torchvision compatibility
try:
    from torchvision.io import VideoReader  # noqa: F401
except ImportError:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _DummyVideoReader:
            pass
        _tvio.VideoReader = _DummyVideoReader

# Set wandb project name
os.environ["WANDB_PROJECT"] = "hisemotions_2026"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from src.data_deps.preprocessing import prepare_dataset
from src.evaluate.metrics import compute_multilabel_metrics, find_optimal_thresholds
from src.losses.asl import AsymmetricLoss
from src.trainer import HisemotionTrainer
from src.experiment_logger import log_experiment
import numpy as np


LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_loss_fn(config: dict, train_df: pd.DataFrame, label_cols: list, device: torch.device):
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "").lower()

    if loss_type == "asl":
        gamma_neg = float(loss_cfg.get("gamma_neg", 4.0))
        gamma_pos = float(loss_cfg.get("gamma_pos", 0.0))
        clip      = float(loss_cfg.get("clip", 0.05))
        print(f"  Loss: AsymmetricLoss(gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip})")
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    elif loss_type == "weighted_bce":
        max_weight = float(loss_cfg.get("max_weight", 10.0))
        weights = _calculate_pos_weights(train_df, label_cols, max_weight).to(device)
        print(f"  Loss: WeightedBCE  pos_weights={weights.tolist()}")
        return nn.BCEWithLogitsLoss(pos_weight=weights)

    else:
        print(f"  Loss: BCEWithLogitsLoss (default)")
        return nn.BCEWithLogitsLoss()


def _calculate_pos_weights(df: pd.DataFrame, labels: list, max_weight: float = 10.0) -> torch.Tensor:
    weights = []
    for label in labels:
        pos = df[label].sum()
        neg = len(df) - pos
        weights.append(min(neg / (pos + 1e-6), max_weight))
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SLM for HISEMOTIONS")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment_name"]

    # ── Directory structure: models/{exp_name}/ ──────────────────────────
    exp_model_dir = os.path.join("models", exp_name)
    checkpoint_dir = os.path.join(exp_model_dir, "checkpoints")
    final_dir = os.path.join(exp_model_dir, "final")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Copy config for reproducibility
    shutil.copy2(args.config, os.path.join(exp_model_dir, "config.yaml"))

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Model      : {config['model']['name']}")
    print(f"Save to    : {exp_model_dir}")
    print(f"{'='*60}\n")

    # Initialize wandb if configured
    report_to = config.get("report_to", config.get("training", {}).get("report_to", "none"))
    if report_to == "wandb":
        try:
            import wandb
            wandb.init(project="hisemotions_2026", name=exp_name, config=config, reinit=True)
        except ImportError:
            print("Warning: wandb not installed, disabling")
            report_to = "none"

    # 1. Load data
    train_df = pd.read_csv(config["data"]["train_path"])
    dev_df   = pd.read_csv(config["data"]["dev_path"])

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 3. Loss function
    loss_fn = build_loss_fn(config, train_df, LABEL_COLS, device)

    # 4. Prepare datasets
    train_dataset = prepare_dataset(train_df, config["model"]["name"], LABEL_COLS, max_length=config["model"]["max_length"])
    dev_dataset = prepare_dataset(dev_df, config["model"]["name"], LABEL_COLS, max_length=config["model"]["max_length"])

    # 5. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 6. Training args
    t_cfg = config["training"]
    total_steps = (len(train_dataset) // t_cfg["per_device_train_batch_size"]) * t_cfg["num_train_epochs"]
    warmup_ratio = t_cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        learning_rate=float(t_cfg["learning_rate"]),
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        num_train_epochs=t_cfg["num_train_epochs"],
        eval_strategy=t_cfg.get("eval_strategy", "epoch"),
        save_strategy=t_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        save_total_limit=t_cfg.get("save_total_limit", 1),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "macro_f1"),
        report_to=report_to,
        run_name=exp_name,
        logging_steps=10,
        fp16=(t_cfg.get("precision", "fp32") == "fp16" and torch.cuda.is_available()),
        max_grad_norm=t_cfg.get("max_grad_norm", 1.0),
        warmup_steps=warmup_steps,
        weight_decay=t_cfg.get("weight_decay", 0.01),
    )

    # 7. Sampler
    labels_array = None
    sampler_cfg = config.get("data", {}).get("sampler", None)
    if sampler_cfg is not None:
        print("\nUsing HisemotionSampler for dynamic Class Balancing...")
        labels_array = np.array(train_dataset["labels"])

    trainer = HisemotionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_multilabel_metrics,
        loss_fn=loss_fn,
        labels_array=labels_array,
        sampler_cfg=sampler_cfg,
    )

    # 8. Train
    print("Training...\n")
    trainer.train()

    # 9. Eval default threshold
    print("\nEvaluating (threshold = 0.5)...")
    eval_results = trainer.evaluate()
    macro_f1_default = eval_results.get("eval_macro_f1", 0.0)
    print(f"Default Macro-F1: {macro_f1_default:.4f}")

    # 10. Optimize per-class thresholds
    print("\nOptimizing per-class thresholds...")
    dev_preds = trainer.predict(dev_dataset)
    optimal_thresholds, optimized_f1, class_report, conf_matrix = find_optimal_thresholds(
        dev_preds.predictions, dev_preds.label_ids, LABEL_COLS
    )
    print(f"Optimized Macro-F1: {optimized_f1:.4f}")
    print(f"Per-class report:\n{class_report}")

    # 11. Save final model
    trainer.save_model(final_dir)
    with open(os.path.join(final_dir, "optimal_thresholds.json"), "w") as f:
        json.dump({
            "thresholds": optimal_thresholds,
            "labels": LABEL_COLS,
            "optimized_macro_f1": optimized_f1,
            "per_class_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "loss_type": config.get("loss", {}).get("type", "bce"),
            "experiment": exp_name,
        }, f, indent=2)
    print(f"\nModel saved to: {final_dir}")

    # 12. Log experiment
    log_experiment(
        experiment_name=exp_name,
        config=config,
        macro_f1_default=macro_f1_default,
        macro_f1_optimized=optimized_f1,
        thresholds=optimal_thresholds,
        per_class_report=class_report,
        label_cols=LABEL_COLS,
        save_dir=final_dir,
    )

    print(f"\n{'='*60}")
    print(f"  DONE: {exp_name}")
    print(f"  Default  Macro-F1: {macro_f1_default:.4f}")
    print(f"  Optimized Macro-F1: {optimized_f1:.4f}")
    print(f"  Model: {final_dir}")
    print(f"  Log:   experiments/{exp_name}.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
